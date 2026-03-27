import os
import time
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .base_trainer import BaseTrainer
from utils.lr_schedulers import get_scheduler
from utils.metrics import compute_dice_per_class
from dataloaders.dataset_CSANet import TripleSliceDataset, SingleSliceDataset
from dataloaders.augment import RandomGenerator_new
from .evaluator import Evaluator

_EPS = 1e-8   # numerical stability constant (hardcoded)


class AdaMITrainer(BaseTrainer):
    """
    Trainer for AdaMI: Source-Free Domain Adaptation for Image Segmentation.

    Bateson et al., Medical Image Analysis 2022 (https://arxiv.org/abs/2108.03152)

    Core idea
    ---------
    Without access to source images during adaptation, AdaMI minimises a
    combined loss over **unlabeled target-domain images**:

        L = L_entropy + λ · KL(τ̂(t,θ,·) ‖ τ_e(t,·))

    where:
      - L_entropy : per-pixel weighted Shannon entropy of softmax predictions,
                    encouraging confident (low-uncertainty) outputs.
      - L_KL      : KL(predicted class-ratio ‖ prior) — prevents trivial
                    single-class collapse (under-segmentation).

    The class-ratio prior τ_e and class weights ν_k are automatically derived
    from the **source-domain training masks** before adaptation begins.  Only
    the source mask statistics are used — no source images are accessed during
    the adaptation loop.

    KL direction: AdaMI uses KL(τ̂ ‖ τ_e), which has bounded gradients near
    zero and avoids the instability of the original AdaEnt formulation
    KL(τ_e ‖ τ̂) (see Sec. 2.2 of the paper).

    Adaptation protocol
    -------------------
    1. Load source-pretrained weights.
    2. Scan source-domain training masks → compute τ_e and ν_k.
    3. All network parameters are optimised (no freezing, unlike Tent).
    4. Each training step:
       a. Forward pass on a target-domain batch → logits.
       b. Weighted Shannon entropy loss (Eq. 2).
       c. KL class-ratio regulariser (Eq. 3 / L2 in Sec. 2.2).
       d. Backprop total loss.
    5. Volume-level evaluation after every epoch.

    Extra args
    ----------
    --adami_lambda : weight λ for the KL term (default 1.0)
    """

    def __init__(self, args, metadata, model, device):
        super().__init__(args, metadata, model, device)

        self.adami_lambda   = getattr(args, 'adami_lambda', 1.0)
        self.val_interval   = getattr(args, 'adami_val_interval', 5)

        # Prior and weights are computed lazily in train() once the
        # source-domain slice directory is known.
        self.class_ratio_prior = None
        self.class_weights      = None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def train(self):
        self.is_25d = self.args.is_25d

        # ---------------------------------------------------------- #
        # 1. Load source-pretrained weights
        # ---------------------------------------------------------- #
        source_checkpoint = torch.load(
            self.args.source_pretrain_path, map_location=self.device
        )
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(source_checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(source_checkpoint['model_state_dict'])
        self.logger.info(
            f"Loaded source-pretrained weights from {self.args.source_pretrain_path}"
        )

        # ---------------------------------------------------------- #
        # 2. Compute class-ratio prior and class weights from source
        #    domain training masks (masks only, no source images used)
        # ---------------------------------------------------------- #
        self._compute_prior_from_source()
        self.logger.info(
            f"AdaMI — λ={self.adami_lambda} | "
            f"prior={self.class_ratio_prior.tolist()} | "
            f"class_weights={self.class_weights.tolist()}"
        )

        # ---------------------------------------------------------- #
        # 3. DataLoaders (target-domain train / test)
        # ---------------------------------------------------------- #
        if self.is_25d:
            db_train = TripleSliceDataset(
                base_dir=self.args.data_dir,
                domain_name=self.args.target,
                split='train',
                metadata=self.metadata,
                transform=transforms.Compose([
                    RandomGenerator_new(
                        output_size=(self.args.img_size, self.args.img_size),
                        phase='train'
                    )
                ])
            )
        else:
            db_train = SingleSliceDataset(
                base_dir=self.args.data_dir,
                domain_name=self.args.target,
                split='train',
                metadata=self.metadata,
                transform=transforms.Compose([
                    RandomGenerator_new(
                        output_size=(self.args.img_size, self.args.img_size),
                        phase='train'
                    )
                ])
            )
        self.logger.info(f"Number of target training slices: {len(db_train)}")
        self.train_loader = DataLoader(
            db_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        if self.is_25d:
            self.db_val = TripleSliceDataset(
                base_dir=self.args.data_dir,
                domain_name=self.args.target,
                split='test',
                metadata=self.metadata,
                transform=transforms.Compose([
                    RandomGenerator_new(
                        output_size=(self.args.img_size, self.args.img_size),
                        phase='val'
                    )
                ])
            )
        else:
            self.db_val = SingleSliceDataset(
                base_dir=self.args.data_dir,
                domain_name=self.args.target,
                split='test',
                metadata=self.metadata,
                transform=transforms.Compose([
                    RandomGenerator_new(
                        output_size=(self.args.img_size, self.args.img_size),
                        phase='val'
                    )
                ])
            )
        self.logger.info(f"Number of val slices: {len(self.db_val)}")

        # ---------------------------------------------------------- #
        # 4. Optimizer & scheduler  (all parameters, unlike Tent)
        # ---------------------------------------------------------- #
        if self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")

        eta_min = (
            self.args.min_lr
            if self.args.min_lr is not None
            else 1e-3 * self.args.base_lr
        )
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.args.scheduler,
            num_epochs=self.args.num_epochs,
            eta_min=eta_min,
            warmup_epochs=self.args.num_epochs // 10,
        )

        # ---------------------------------------------------------- #
        # 5. Evaluator
        # ---------------------------------------------------------- #
        self.evaluator = Evaluator(
            args=self.args,
            metadata=self.metadata,
            model=self.model,
            device=self.device,
            db_eval=self.db_val,
            logger=self.logger,
        )

        # ---------------------------------------------------------- #
        # 6. Training loop  (val + ckpt triggered every val_interval steps)
        # ---------------------------------------------------------- #
        self.current_epoch = 0
        self.global_step   = 0
        self.best_metric   = 0.0

        self.logger.info(
            f"AdaMI: val & checkpoint every {self.val_interval} step(s)."
        )

        start_time = time.time()
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.logger.info("-" * 60)
            self.logger.info(f"AdaMI Epoch {epoch}/{self.args.num_epochs}")
            self.logger.info("-" * 60)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning rate: {current_lr:.8f}")

            self.current_epoch = epoch

            # val + save happen inside _train_one_epoch at every val_interval steps
            train_metrics = self._train_one_epoch()

            # Log train metrics and LR using global_step as the wandb x-axis,
            # so they are consistent with the val metrics logged in _evaluate_and_save.
            metrics_str = ' | '.join(
                [f"{k}: {v:.4f}" for k, v in train_metrics.items()]
            )
            self.logger.info(f"Epoch {epoch} [train] {metrics_str}")
            if not self.args.disable_wandb:
                log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
                log_dict['learning_rate'] = current_lr
                wandb.log(log_dict, step=self.global_step)

            self.scheduler.step()

        total_time = time.time() - start_time
        self.logger.info(
            f"AdaMI adaptation completed in {total_time / 3600:.2f} hours"
        )
        self.logger.info(f"Best test Dice: {self.best_metric:.4f}")

        if not self.args.disable_wandb:
            wandb.finish()

    # ------------------------------------------------------------------ #
    # One epoch of AdaMI adaptation
    # ------------------------------------------------------------------ #

    def _train_one_epoch(self):
        """
        Iterate over target-domain slices and minimise the AdaMI loss:

            L = (1/|Ω_t|) Σ_i ℓ_ent(p_t(i,θ))  +  λ · KL(τ̂(t,θ,·) ‖ τ_e(t,·))

        Ground-truth labels are used **only** for batch-level Dice monitoring;
        they are never used in the loss computation.

        Every ``val_interval`` global steps, a full volume-level evaluation is
        triggered and the best checkpoint is saved.
        """
        self.model.train()

        # Move prior / weights to device once per epoch (no-op if already there)
        prior   = self.class_ratio_prior.to(self.device)   # (K,)
        weights = self.class_weights.to(self.device)       # (K,)

        total_loss     = 0.0
        total_ent_loss = 0.0
        total_kl_loss  = 0.0
        total_dice     = 0.0
        num_batches    = 0

        pbar = tqdm(self.train_loader, desc=f'AdaMI Epoch {self.current_epoch}')
        for batch in pbar:
            image_batch = batch['image'].to(self.device)
            label_batch = batch['mask'].to(self.device)   # monitoring only
            if self.is_25d:
                next_image_batch = batch['next_image'].to(self.device)
                prev_image_batch = batch['prev_image'].to(self.device)

            # ---- Forward pass ------------------------------------------ #
            self.optimizer.zero_grad()
            if self.is_25d:
                logits = self.model(prev_image_batch, image_batch, next_image_batch)
            else:
                logits = self.model(image_batch)
            # logits: (B, K, H, W)

            # ---- Entropy loss  L_entropy (Eq. 2) ----------------------- #
            ent_loss = self._weighted_entropy_loss(logits, weights)

            # ---- KL loss  L_KL = KL(τ̂ ‖ τ_e)  (Eq. 3 / L2 in Sec 2.2) #
            kl_loss = self._class_ratio_kl_loss(logits, prior)

            loss = ent_loss + self.adami_lambda * kl_loss

            # ---- Backward pass ----------------------------------------- #
            loss.backward()
            self.optimizer.step()

            # ---- Batch-level Dice monitoring (GT not used in loss) ------ #
            with torch.no_grad():
                pred_batch  = torch.argmax(logits, dim=1)
                dice_scores = compute_dice_per_class(
                    pred_batch.cpu().numpy(),
                    label_batch.cpu().numpy(),
                    num_classes=self.num_classes,
                    include_background=False,
                )
                batch_dice = (
                    sum(dice_scores.values()) / len(dice_scores)
                    if dice_scores else 0.0
                )

            total_loss     += loss.item()
            total_ent_loss += ent_loss.item()
            total_kl_loss  += kl_loss.item()
            total_dice     += batch_dice
            num_batches    += 1
            self.global_step += 1

            pbar.set_postfix({
                'step':    self.global_step,
                'loss':    f'{loss.item():.4f}',
                'ent':     f'{ent_loss.item():.4f}',
                'kl':      f'{kl_loss.item():.4f}',
                'dice':    f'{batch_dice:.4f}',
            })

            # ---- Step-level val + checkpoint ---------------------------- #
            if self.global_step % self.val_interval == 0:
                self._evaluate_and_save()
                self.model.train()   # restore train mode after evaluation

        return {
            'loss':         total_loss     / num_batches,
            'entropy_loss': total_ent_loss / num_batches,
            'kl_loss':      total_kl_loss  / num_batches,
            'dice':         total_dice     / num_batches,
        }

    # ------------------------------------------------------------------ #
    # Step-level evaluation + checkpoint
    # ------------------------------------------------------------------ #

    def _evaluate_and_save(self) -> None:
        """Run full volume-level evaluation and save checkpoint if improved.

        Called every ``val_interval`` global steps from within
        ``_train_one_epoch``.  The caller is responsible for restoring
        ``model.train()`` mode afterwards.
        """
        self.logger.info(
            f"[Step {self.global_step}] Running evaluation ..."
        )
        val_metrics = self.evaluator.evaluate(isotropic_spacing=True)

        # Log val metrics keyed by global_step (not epoch)
        metrics_str = ' | '.join(
            [f"{k}: {v:.4f}" for k, v in val_metrics.items()]
        )
        self.logger.info(
            f"[Step {self.global_step}] [val] {metrics_str}"
        )
        if not self.args.disable_wandb:
            wandb.log(
                {f'val/{k}': v for k, v in val_metrics.items()},
                step=self.global_step,
            )

        current_metric = val_metrics.get('dice_mean', 0.0)
        is_best = current_metric > self.best_metric
        if is_best:
            self.best_metric = current_metric
            self.logger.info(
                f"[Step {self.global_step}] New best model! Dice: {current_metric:.4f}"
            )

        self.save_checkpoint(current_metric=current_metric, is_best=is_best)

    # ------------------------------------------------------------------ #
    # Step-based checkpoint saving (overrides BaseTrainer)
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, current_metric: float, is_best: bool = False) -> None:
        """Save checkpoint with step-based filename.

        Overrides ``BaseTrainer.save_checkpoint`` so that filenames reflect
        the global step rather than the epoch number, matching the step-level
        monitoring cadence of AdaMI.

        Every val trigger writes:
          - ``last_checkpoint.pth``       — always overwritten
          - ``best_checkpoint.pth``       — only when a new best is reached
          - ``step_{N}_dice_{M}.pth``     — named snapshot at every val trigger
        """
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'epoch':             self.current_epoch,
            'global_step':       self.global_step,
            'model_state_dict':  model_state,
            'current_metric':    current_metric,
            'best_metric':       self.best_metric,
            'config':            self.all_configs,
        }

        last_path = os.path.join(self.checkpoint_path, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_path, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)

        # Named snapshot at every val trigger so every step is traceable
        name = f'step_{self.global_step}_dice_{current_metric:.4f}.pth'
        torch.save(checkpoint, os.path.join(self.checkpoint_path, name))

    # ------------------------------------------------------------------ #
    # AdaMI loss helpers
    # ------------------------------------------------------------------ #

    def _weighted_entropy_loss(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted mean per-pixel Shannon entropy (Eq. 2 in the paper).

        Parameters
        ----------
        logits  : (B, K, H, W)
        weights : (K,)  — per-class weight ν_k

        Returns
        -------
        Scalar loss.  Minimising this pushes predictions to be more confident.
        """
        prob     = F.softmax(logits, dim=1)           # (B, K, H, W)
        log_prob = F.log_softmax(logits, dim=1)       # (B, K, H, W)

        # Per-class weighted entropy: ν_k * p_k * log(p_k)
        # weights: (K,) → (1, K, 1, 1) for broadcasting
        w = weights.view(1, -1, 1, 1)
        # Sum over classes → per-pixel entropy (B, H, W)
        entropy = -(w * prob * log_prob).sum(dim=1)
        return entropy.mean()

    def _class_ratio_kl_loss(
        self,
        logits: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """KL(τ̂(t,θ,·) ‖ τ_e(t,·))  — the AdaMI regulariser (Eq. 3 / L2).

        τ̂(t,k,θ) = (1/|Ω_t|) Σ_i p^k_t(i,θ)   (predicted class ratio)
        τ_e(t,k)  = prior[k]                     (anatomical class ratio)

        The KL is computed per-image and then averaged over the batch.

        Parameters
        ----------
        logits : (B, K, H, W)
        prior  : (K,)

        Returns
        -------
        Scalar loss.
        """
        prob = F.softmax(logits, dim=1)          # (B, K, H, W)

        # Predicted class ratio per image: mean over spatial dims
        # τ̂: (B, K)
        tau_hat = prob.mean(dim=(-2, -1))        # (B, K)

        # Clamp for numerical stability
        tau_hat = tau_hat.clamp(min=_EPS)
        tau_e   = prior.clamp(min=_EPS)          # (K,)

        # KL(τ̂ ‖ τ_e) = Σ_k τ̂_k * log(τ̂_k / τ_e_k)
        # τ_e is broadcast over the batch dimension
        kl = (tau_hat * (tau_hat.log() - tau_e.log())).sum(dim=1)   # (B,)
        return kl.mean()

    # ------------------------------------------------------------------ #
    # Source-domain statistics
    # ------------------------------------------------------------------ #

    def _compute_prior_from_source(self) -> None:
        """Scan source-domain training masks to compute τ_e and ν_k.

        Only the ``label`` arrays inside the .npz slice files are read —
        no source images are loaded, keeping this source-free at adaptation time.

        Sets
        ----
        self.class_ratio_prior : torch.Tensor, shape (K,)
            Per-class pixel proportion averaged across all source training slices.
        self.class_weights : torch.Tensor, shape (K,)
            ν_k = (τ̄_k)^{-1} / Σ_k (τ̄_k)^{-1}  (paper Sec. 3.1.6).
        """
        K = self.num_classes
        source_slice_dir = os.path.join(
            self.args.data_dir, self.args.source, 'slices'
        )

        # Collect slice paths belonging to the source training split
        slice_paths = []
        if "splits" in self.metadata and self.args.source in self.metadata["splits"]:
            train_case_ids = set(
                self.metadata["splits"][self.args.source].get("train", [])
            )
            for fname in sorted(os.listdir(source_slice_dir)):
                if fname.endswith('.npz'):
                    case_id = fname.split('_')[1]
                    if case_id in train_case_ids:
                        slice_paths.append(os.path.join(source_slice_dir, fname))
        else:
            for fname in sorted(os.listdir(source_slice_dir)):
                if fname.endswith('.npz'):
                    slice_paths.append(os.path.join(source_slice_dir, fname))

        if not slice_paths:
            self.logger.warning(
                "AdaMI: no source training slices found — falling back to uniform prior."
            )
            prior   = [1.0 / K] * K
            weights = [1.0 / K] * K
        else:
            self.logger.info(
                f"AdaMI: computing class-ratio prior from {len(slice_paths)} "
                f"source training slices ({self.args.source}) ..."
            )
            pixel_counts = np.zeros(K, dtype=np.int64)
            for path in slice_paths:
                label = np.load(path)['label'].astype(np.int64).ravel()
                for k in range(K):
                    pixel_counts[k] += int((label == k).sum())

            total = pixel_counts.sum()
            prior = (pixel_counts / (total + _EPS)).tolist()

            # ν_k = (τ̄_k)^{-1} / Σ_k (τ̄_k)^{-1}
            inv     = [1.0 / (p + _EPS) for p in prior]
            inv_sum = sum(inv)
            weights = [v / inv_sum for v in inv]

        self.class_ratio_prior = torch.tensor(prior,   dtype=torch.float32)
        self.class_weights      = torch.tensor(weights, dtype=torch.float32)

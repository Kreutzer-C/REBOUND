import time
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
from dataloaders.dataset_CSANet import CSANet_SliceDataset, CSANet_VolumeDataset, RandomGenerator, RandomGenerator_new
from trainer.evaluator import Evaluator


class TentTrainer(BaseTrainer):
    """
    Trainer for Tent: Fully Test-Time Adaptation by Entropy Minimization.

    Wang et al., ICLR 2021 (https://arxiv.org/abs/2006.10726)

    Core idea
    ---------
    At test time (here: target-domain adaptation), only the **affine parameters
    (weight γ and bias β) of all BatchNorm layers** are updated.  All other
    parameters remain frozen.  The sole supervision signal is the **Shannon
    entropy** of the model's own softmax predictions — no target labels are
    required.

    Adaptation protocol
    -------------------
    1. Load source-pretrained weights.
    2. Configure the model: BN layers → train mode; all other layers → eval mode.
       Only BN affine parameters require gradients.
    3. For each epoch, iterate over **target-domain slices** and minimise the
       per-pixel mean entropy of the softmax predictions.
    4. Evaluate on the full target-domain test volumes after every epoch.
    """

    def __init__(self, args, metadata, model, device):
        super().__init__(args, metadata, model, device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self):
        # ----------------------------------------------------------
        # 1. Load source-pretrained weights
        # ----------------------------------------------------------
        source_checkpoint = torch.load(self.args.source_pretrain_path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(source_checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(source_checkpoint['model_state_dict'])
        self.logger.info(f"Loaded pre-train weight from {self.args.source_pretrain_path}")

        # ----------------------------------------------------------
        # 2. Configure model for Tent adaptation
        #    • BN layers  → train mode  (re-estimate batch statistics)
        #    • Everything else → eval mode (frozen)
        #    • Only BN affine params (γ, β) require gradients
        # ----------------------------------------------------------
        tent_params = self._configure_model_for_tent()
        bn_count = len(tent_params) // 2  # each BN contributes weight + bias
        self.logger.info(
            f"Tent: {bn_count} BatchNorm layer(s) will be adapted "
            f"({len(tent_params)} trainable parameter tensors)."
        )

        # ----------------------------------------------------------
        # 3. DataLoaders
        #    target-domain train split  →  entropy optimisation (no labels used)
        #    target-domain test  split  →  volume-level evaluation
        # ----------------------------------------------------------
        db_train = CSANet_SliceDataset(
            base_dir=self.args.data_dir,
            domain_name=self.args.target,
            split='train',
            metadata=self.metadata,
            transform=transforms.Compose(
                [RandomGenerator(output_size=(self.args.img_size, self.args.img_size), phase='train')]
            ),
        )
        self.logger.info(f"Number of target training slices: {len(db_train)}")
        self.train_loader = DataLoader(
            db_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.db_val = CSANet_SliceDataset(
            base_dir=self.args.data_dir,
            domain_name=self.args.target,
            split='test',
            metadata=self.metadata,
            transform=transforms.Compose(
                [RandomGenerator_new(output_size=(self.args.img_size, self.args.img_size), phase='val')])
        )
        self.logger.info(f"Number of val slices: {len(self.db_val)}")

        # ----------------------------------------------------------
        # 4. Optimizer  (only BN affine params)
        # ----------------------------------------------------------
        if self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                tent_params, lr=self.args.base_lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                tent_params, lr=self.args.base_lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                tent_params, lr=self.args.base_lr, weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")

        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.args.scheduler,
            num_epochs=self.args.num_epochs,
            eta_min=1e-6,
            warmup_epochs=self.args.num_epochs // 10,
        )

        # ----------------------------------------------------------
        # 5. Evaluator (db_val pre-loaded once here)
        # ----------------------------------------------------------
        self.evaluator = Evaluator(
            args=self.args,
            metadata=self.metadata,
            model=self.model,
            device=self.device,
            db_eval=self.db_val,
            logger=self.logger,
        )

        # ----------------------------------------------------------
        # 6. Training loop
        # ----------------------------------------------------------
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        start_time = time.time()
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.logger.info("-" * 60)
            self.logger.info(f"Tent Epoch {epoch}/{self.args.num_epochs}")
            self.logger.info("-" * 60)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning rate: {current_lr:.6f}")
            if not self.args.disable_wandb:
                wandb.log({'learning_rate': current_lr}, step=epoch)

            self.current_epoch = epoch

            # entropy-minimisation step
            train_metrics = self._train_one_epoch()
            self._log_metrics(train_metrics, prefix='train', epoch=epoch)

            # volume-level evaluation
            val_metrics = self.evaluator.evaluate(isotropic_spacing=True)
            self._log_metrics(val_metrics, prefix='val', epoch=epoch)

            current_metric = val_metrics.get('dice_mean', 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.logger.info(f"New best model! Dice: {current_metric:.4f}")

            self.save_checkpoint(is_best=is_best)
            self.scheduler.step()

        total_time = time.time() - start_time
        self.logger.info(f"Tent adaptation completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Best test Dice: {self.best_metric:.4f}")

        if not self.args.disable_wandb:
            wandb.finish()

    # ------------------------------------------------------------------
    # One epoch of entropy minimisation
    # ------------------------------------------------------------------

    def _train_one_epoch(self):
        """Iterate over target-domain slices and minimise prediction entropy.

        Labels are loaded for monitoring (Dice) only — they are **never** used
        in the loss computation, keeping this a fully unsupervised adaptation.
        """
        # BN layers stay in train mode; all other sub-modules stay in eval mode.
        # _configure_model_for_tent() already set this up, but re-apply here to
        # be safe in case evaluator.evaluate() called model.eval() internally.
        self._set_bn_train_mode()

        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Tent Epoch {self.current_epoch}')
        for batch in pbar:
            image_batch      = batch['image'].to(self.device)
            label_batch      = batch['mask'].to(self.device)          # monitoring only
            next_image_batch = batch['next_image'].to(self.device)
            prev_image_batch = batch['prev_image'].to(self.device)

            # Forward pass  →  logits (B, C, H, W)
            self.optimizer.zero_grad()
            logits = self.model(prev_image_batch, image_batch, next_image_batch)

            # Tent loss: mean per-pixel Shannon entropy of softmax predictions
            loss = self._entropy_loss(logits)

            # Backward pass  →  update only BN affine params
            loss.backward()
            self.optimizer.step()

            # Monitoring: Dice against ground-truth (not used for optimisation)
            with torch.no_grad():
                pred_batch = torch.argmax(logits, dim=1)
                dice_scores = compute_dice_per_class(
                    pred_batch.cpu().numpy(),
                    label_batch.cpu().numpy(),
                    num_classes=self.num_classes,
                    include_background=False,
                )
                batch_dice = (
                    sum(dice_scores.values()) / len(dice_scores) if dice_scores else 0.0
                )

            total_loss  += loss.item()
            total_dice  += batch_dice
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'entropy': f'{loss.item():.4f}',
                'dice':    f'{batch_dice:.4f}',
            })

        return {
            'entropy_loss': total_loss / num_batches,
            'dice':         total_dice / num_batches,
        }

    # ------------------------------------------------------------------
    # Tent-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        """Mean per-pixel Shannon entropy of the softmax distribution.

        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W)
            Raw (unnormalised) class scores from the segmentation head.

        Returns
        -------
        torch.Tensor
            Scalar entropy value (nats).  Lower → more confident predictions.
        """
        prob = F.softmax(logits, dim=1)                  # (B, C, H, W)
        log_prob = F.log_softmax(logits, dim=1)          # (B, C, H, W)
        # Entropy per pixel: -sum_c p_c * log(p_c)
        entropy = -(prob * log_prob).sum(dim=1)          # (B, H, W)
        return entropy.mean()                            # scalar

    def _configure_model_for_tent(self) -> list:
        """Prepare the model for Tent adaptation.

        Steps
        -----
        1. Put the entire model in eval mode (disables Dropout, etc.).
        2. Re-enable **train mode** on every BatchNorm layer so that running
           statistics are estimated from the current (target-domain) batch.
        3. Freeze **all** parameters (``requires_grad = False``).
        4. Unfreeze only the **affine parameters** (weight γ and bias β) of
           each BatchNorm layer.

        Returns
        -------
        list
            The parameter tensors that should be passed to the optimizer.
        """
        # Work on the inner module when DataParallel is used
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Step 1: full eval
        module.eval()

        # Step 2 & 3 & 4: iterate over all sub-modules
        params = []
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()                          # re-estimate batch statistics
                m.requires_grad_(False)            # freeze running_mean / running_var
                if m.affine:
                    m.weight.requires_grad_(True)  # γ  (scale)
                    m.bias.requires_grad_(True)    # β  (shift)
                    params += [m.weight, m.bias]
            else:
                # Freeze all non-BN layers
                for p in m.parameters(recurse=False):
                    p.requires_grad_(False)

        if not params:
            self.logger.warning(
                "Tent: no BatchNorm layers with affine=True found in the model. "
                "All parameters will remain frozen and no adaptation will occur."
            )
        return params

    def _set_bn_train_mode(self) -> None:
        """Re-apply train mode to all BatchNorm layers.

        Called at the start of each epoch because ``Evaluator.evaluate()``
        internally calls ``model.eval()``, which resets BN layers to eval mode.
        """
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()

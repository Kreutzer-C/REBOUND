import time
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
from utils.loss_functions import DiceLoss

from .base_trainer import BaseTrainer
from utils.lr_schedulers import get_scheduler
from utils.metrics import compute_dice_per_class
from dataloaders.dataset_CSANet import CSANet_SliceDataset, CSANet_VolumeDataset, RandomGenerator
from trainer.evaluator import Evaluator


class OracleTrainer(BaseTrainer):
    """
    Trainer for target domain oracle training (use ground-truth labels in target domain, as upper bound)
    """
    def __init__(self, args, metadata, model, device):
        super().__init__(args, metadata, model, device)


    def train(self):
        # load model pre-train weight
        source_checkpoint = torch.load(self.args.source_pretrain_path)
        self.model.load_from(weights=source_checkpoint['model_state_dict'])
        self.logger.info(f"Loaded pre-train weight from {self.args.source_pretrain_path}")

        # set dataloader
        db_train = CSANet_SliceDataset(
            base_dir=self.args.data_dir,
            domain_name=self.args.target,
            split='train',
            metadata=self.metadata,
            transform=transforms.Compose(
                [RandomGenerator(output_size=(self.args.img_size, self.args.img_size), phase='train')])
        )
        self.logger.info(f"Number of training slices: {len(db_train)}")
        self.train_loader = DataLoader(db_train, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.db_val = CSANet_VolumeDataset(
            base_dir=self.args.data_dir,
            domain_name=self.args.target,
            split='test',
            metadata=self.metadata,
        )
        self.logger.info(f"Number of val volumes: {len(self.db_val)}")

        # set optimizer & scheduler
        if self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")
        
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.args.scheduler,
            num_epochs=self.args.num_epochs,
            eta_min=1e-6,
            warmup_epochs=self.args.num_epochs // 10,
        )

        # set loss function
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=self.num_classes)

        # set evaluator
        self.evaluator = Evaluator(
            args=self.args,
            metadata=self.metadata,
            model=self.model,
            device=self.device,
            logger=self.logger,
        )

        # set training variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # train
        start_time = time.time()
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.logger.info("-" * 60)
            self.logger.info(f"Training Epoch {epoch}/{self.args.num_epochs}")
            self.logger.info("-" * 60)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning rate: {current_lr:.6f}")
            if not self.args.disable_wandb:
                wandb.log({'learning_rate': current_lr}, step=epoch)

            self.current_epoch = epoch

            # train on epoch
            train_metrics = self._train_one_epoch()
            self._log_metrics(train_metrics, prefix='train', epoch=epoch)

            # validate
            val_metrics = self.evaluator.evaluate(db_eval=self.db_val)
            self._log_metrics(val_metrics, prefix='val', epoch=epoch)

            # check best metric
            current_metric = val_metrics.get('dice_mean', 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.logger.info(f"New best model! Dice: {current_metric:.4f}")
            
            # save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Updata learning rate
            self.scheduler.step()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best test Dice: {self.best_metric:.4f}")

        # Close logging
        if not self.args.disable_wandb:
            wandb.finish()
            

    def _train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch in pbar:
            image_batch, label_batch = batch['image'].to(self.device), batch['mask'].to(self.device)    
            next_image_batch, prev_image_batch = batch['next_image'].to(self.device), batch['prev_image'].to(self.device)

            # forward pass
            self.optimizer.zero_grad()
            outputs = self.model(prev_image_batch, image_batch, next_image_batch)

            # compute loss
            ce_loss = self.ce_loss(outputs, label_batch)
            dice_loss = self.dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * ce_loss + 0.5 * dice_loss

            # backward pass
            loss.backward()
            self.optimizer.step()

            # compute metrics
            with torch.no_grad():
                pred_batch = torch.argmax(outputs, dim=1)
                dice_scores = compute_dice_per_class(
                    pred_batch.cpu().numpy(), 
                    label_batch.cpu().numpy(), 
                    num_classes=self.num_classes, 
                    include_background=False)
                batch_dice = sum(dice_scores.values()) / len(dice_scores) if dice_scores else 0.0
            
            total_loss += loss.item()
            total_dice += batch_dice
            num_batches += 1
            self.global_step += 1

            # update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{batch_dice:.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
        }

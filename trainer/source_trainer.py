import os
import json
import time
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
from utils.loss_functions import DiceLoss

from utils.simple_tools import get_logger, convert_namespace_to_dict
from utils.lr_schedulers import get_scheduler
from utils.metrics import compute_dice_per_class
from dataloaders.dataset_CSANet import CSANet_SliceDataset, CSANet_VolumeDataset, RandomGenerator
from trainer.evaluator import Evaluator


class SourceTrainer:
    def __init__(self, args, metadata, model, device):
        self.args = args
        self.metadata = metadata
        self.model = model
        self.device = device
        self.num_classes = metadata['num_classes']

        with open(self.args.model_config, 'r') as f:
            self.model_config = json.load(f)

        # collect all configs & args
        self.all_configs = {
            'args': convert_namespace_to_dict(self.args),
            'model_config': self.model_config,
            'metadata': self.metadata,
        }

        # set relative paths
        self.config_backup_path = os.path.join(self.args.result_dir, 'configs_backup')
        os.makedirs(self.config_backup_path, exist_ok=True)
        self.checkpoint_path = os.path.join(self.args.result_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # set logger
        log_file = os.path.join(self.args.result_dir, 'train.log')
        self.logger = get_logger('Trainer', log_file)

        # set wandb
        if not self.args.disable_wandb:
            wandb.init(
                project="REBOUND",
                name=self.args.exp,
                config=self.all_configs,
                dir=self.args.result_dir,
            )
            wandb.watch(self.model)
        else:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("All configs:")
            for key, value in self.all_configs.items():
                self.logger.info(f"{key}: {value}")
            self.logger.info("=" * 60)

        # backup configs
        with open(os.path.join(self.config_backup_path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=4)
        with open(os.path.join(self.config_backup_path, 'args.json'), 'w') as f:
            json.dump(convert_namespace_to_dict(self.args), f, indent=4)
        with open(os.path.join(self.config_backup_path, 'model_config.json'), 'w') as f:
            json.dump(self.model_config, f, indent=4)


    def train(self):
        # set dataloader
        processed_dir = os.path.join(self.args.data_dir, self.args.dataset, 'processed')

        db_train = CSANet_SliceDataset(
            base_dir=processed_dir,
            domain_name=self.args.source,
            split='train',
            metadata=self.metadata,
            transform=transforms.Compose(
                [RandomGenerator(output_size=(self.args.img_size, self.args.img_size), phase='train')])
        )
        self.logger.info(f"Number of training slices: {len(db_train)}")
        self.train_loader = DataLoader(db_train, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.db_val = CSANet_VolumeDataset(
            base_dir=processed_dir,
            domain_name=self.args.source,
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


    def _log_metrics(self, metrics, prefix, epoch):
        """Log metrics to wandb and console."""
        # Console logging
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} [{prefix}] {metrics_str}")
        
        # Wandb
        if not self.args.disable_wandb:
            wandb_metrics = {f'{prefix}/{name}': value for name, value in metrics.items()}
            wandb.log(wandb_metrics, step=epoch)


    def save_checkpoint(self, is_best=False, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'epoch_{self.current_epoch}_dice_{self.best_metric:.4f}.pth'
        
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.all_configs,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save last checkpoint
        last_path = os.path.join(self.checkpoint_path, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_path, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        if self.current_epoch % self.args.save_every == 0:
            periodic_path = os.path.join(self.checkpoint_path, filename)
            torch.save(checkpoint, periodic_path)


    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")

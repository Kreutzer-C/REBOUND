import os
import json
import time
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.simple_tools import get_logger, convert_namespace_to_dict


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    Provides common functionality for logging, checkpointing.
    """
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
                name=f"{self.args.exp}-{self.args.source[:3]}_to_{self.args.target[:3]}-{self.args.dataset[:3]}",
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


    @abstractmethod
    def train(self):
        """
        Training Procedure, including set dataloader, optimizer, scheduler, loss function, and training loop.
        Must be implemented by subclass.
        """
        pass
        
            
    @abstractmethod
    def _train_one_epoch(self):
        """
        Train for one epoch.
        Must be implemented by subclass.
        """
        pass


    def _log_metrics(self, metrics, prefix, epoch):
        """Log metrics to wandb and console."""
        # Console logging
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} [{prefix}] {metrics_str}")
        
        # Wandb
        if not self.args.disable_wandb:
            wandb_metrics = {f'{prefix}/{name}': value for name, value in metrics.items()}
            wandb.log(wandb_metrics, step=epoch)


    def save_checkpoint(self, current_metric, is_best=False):
        """
        Save model checkpoint.
        (Considering no need for resume training, we comment out the optimizer and scheduler state dict saving)
        """
        filename = f'epoch_{self.current_epoch}_dice_{current_metric:.4f}.pth'
        
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            'current_metric': current_metric,
            'best_metric': self.best_metric,
            'config': self.all_configs,
        }
        
        # if self.scheduler is not None:
        #     checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
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

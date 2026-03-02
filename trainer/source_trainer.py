import os
import json
import wandb
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
from utils.loss_functions import DiceLoss

from utils.simple_tools import get_logger, convert_namespace_to_dict
from utils.lr_schedulers import get_scheduler
from dataloaders.dataset_CSANet import CSANet_SliceDataset, RandomGenerator

class SourceTrainer:
    def __init__(self, args, metadata, model, device):
        self.args = args
        self.metadata = metadata
        self.model = model
        self.device = device

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
        log_file = os.path.join(self.result_dir, 'train.log')
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
        db_train = CSANet_SliceDataset(
            base_dir=os.path.join(self.args.data_dir, self.args.dataset, 'processed'),
            domain_name=self.args.source,
            split='train',
            metadata=self.metadata,
            transform=transforms.Compose(
                [RandomGenerator(output_size=self.args.img_size, phase='train')])
        )
        self.logger.info(f"Number of training slices: {len(db_train)}")
        self.train_loader = DataLoader(db_train, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

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
        self.dice_loss = DiceLoss(n_classes=self.metadata['num_classes'])

        # use _train_one_epoch
        # use _validate
        # log losses & metrics via logger and wandb
        # save checkpoints
        pass
    
    def _train_one_epoch(self):
        pass
    
    def _validate(self):
        # use method in evaluator.py
        pass
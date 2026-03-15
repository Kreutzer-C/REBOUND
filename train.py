import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='REBOUND: REliable BOundary expansion and structural UNderstanding via Diffusion-distillation for 2.5D Source-Free Medical Image Segmentation')

    # Method
    parser.add_argument('--method', type=str, default='source_pretrain',
                        help='Training method')
    
    # Dataset
    parser.add_argument('--dataset', '-d', type=str, default='ABDOMINAL',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Data directory')
    parser.add_argument('--processed_dir', '-pd', type=str, default='processed_DDFP',
                        help='Processed data directory')
    parser.add_argument('--source', '-src', type=str, default='BTCV',
                        help='Source domain name')
    parser.add_argument('--target', '-tgt', type=str, default='CHAOST2',
                        help='Target domain name')
    
    # Model
    parser.add_argument('--model', '-m', type=str, default='CSANet',
                        help='Model name')
    parser.add_argument('--is_25d', default=True,
                        help='Whether the model is 2.5D (auto-determined according to model name)')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Model configuration file (auto-generated if not provided)')
    parser.add_argument('--source_pretrain_path', '-sp', type=str, default=None,
                        help='Source domain pretrained model path')
    
    # Training
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--num_epochs', '-e', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--base_lr', '-lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='Minimum learning rate for scheduler (1e-3 * base_lr if not provided)')                    
    parser.add_argument('--weight_decay', '-wd', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['step', 'cosine', 'cosine_warmup'],
                        help='Schedule type')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'Adam', 'SGD'],
                        help='Optimizer type')
    
    # Logging
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Result directory (auto-generated if not provided)')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='REBOUND',
                        help='W&B project name')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_device(gpu_ids):
    """Set the device to use for training. (Supports multiple GPUs)"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            print(">>> Device Info:")
            print(" " * 4 + f"GPU {i}: {gpu_properties.name} ({gpu_properties.total_memory / 1024 ** 2}MB)")

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu_id) for gpu_id in gpu_ids)      
        return torch.device(f'cuda:{gpu_ids[0]}')
    else:
        return torch.device('cpu')


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("REBOUND - Training")
    print("=" * 60)

    # Step1: set random seed and device
    set_seed(args.seed)
    device = set_device(args.gpu_ids)

    # Step2: set experiment name
    if args.exp is None:
        args.exp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f">>> Experiment: {args.exp}")

    # Step3: set data directory
    args.data_dir = os.path.join(args.data_dir, args.dataset, args.processed_dir)
    assert os.path.exists(args.data_dir), f"Processed data directory does not exist: {args.data_dir}"
    print(f">>> Processed data directory: {args.data_dir}")

    # Step4: set dataset metadata
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert args.source in metadata['domains'], f"Source domain {args.source} not found in metadata: {metadata['domains']}"
    assert args.target in metadata['domains'], f"Target domain {args.target} not found in metadata: {metadata['domains']}"

    # Step5: set model configuration
    if args.model_config is None:
        if args.is_25d:
            args.model_config = './networks/R50_ViTB16_config.json'
        else:
            args.model_config = './networks/unet_config.json'
    print(f">>> Model configuration: {args.model_config}")

    # Step6: set result directory
    if args.result_dir is None:
        args.result_dir = os.path.join('./results', args.dataset, f"{args.source}_to_{args.target}", args.exp)
    if os.path.exists(args.result_dir):
        raise ValueError(f"Result directory already exists: {args.result_dir}, please use a different exp name or delete the existing directory.")
    os.makedirs(args.result_dir, exist_ok=True)
    print(f">>> Result directory: {args.result_dir}")

    # Step7: set model
    if args.model == 'CSANet':
        from networks.csanet_modeling import CSANet
        model = CSANet(args.model_config, args.img_size, metadata['num_classes']).to(device)
    elif args.model == 'CSANet_V2':
        from networks.csanet_modeling_v2 import CSANet_V2
        model = CSANet_V2(args.model_config, args.img_size, metadata['num_classes']).to(device)
    elif args.model == 'CSANet_V3':
        from networks.csanet_modeling_v3 import CSANet_V3
        model = CSANet_V3(args.model_config, args.img_size, metadata['num_classes']).to(device)
    elif args.model == 'UNet':
        from networks.unet_modeling import build_unet
        args.is_25d = False
        model = build_unet(args.model_config, args.img_size, metadata['num_classes']).to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f">>> Model: {args.model}")
    print(f">>> Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step8: start trainer
    if args.method == 'source_pretrain':
        from trainer import SourceTrainer
        trainer = SourceTrainer(args, metadata, model, device)
        trainer.train()
    elif args.method == 'oracle':
        from trainer import OracleTrainer
        trainer = OracleTrainer(args, metadata, model, device)
        trainer.train()
    elif args.method == 'self_train':
        from trainer import SelfTrainer
        trainer = SelfTrainer(args, metadata, model, device)
        trainer.train()
    elif args.method == 'tent':
        from trainer import TentTrainer
        trainer = TentTrainer(args, metadata, model, device)
        trainer.train()
    else:
        raise ValueError(f"Invalid method: {args.method}")

    
if __name__ == '__main__':
    main()
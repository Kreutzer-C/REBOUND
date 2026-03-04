import os
import argparse
import json
import torch
import numpy as np

from dataloaders.dataset_CSANet import CSANet_VolumeDataset
from networks.csanet_modeling import CSANet
from trainer.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='REBOUND: REliable BOundary expansion and structural UNderstanding via Diffusion-distillation for 2.5D Source-Free Medical Image Segmentation')

    # Dataset
    parser.add_argument('--dataset', '-d', type=str, default='ABDOMINAL',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Data directory')
    parser.add_argument('--processed_dir', type=str, default='processed',
                        help='Processed data directory')
    parser.add_argument('--source', '-src', type=str, default='BTCV',
                        help='Source domain name')
    parser.add_argument('--target', '-tgt', type=str, default='CHAOST2',
                        help='Target domain name')
    
    # Model
    parser.add_argument('--model', '-m', type=str, default='CSANet',
                        help='Model name')
    parser.add_argument('--model_config', type=str, default='./networks/R50_ViTB16_config.json',
                        help='Model configuration file')

    # Checkpoint
    parser.add_argument('--exp_dir', '-ed', type=str, default=None,
                        help='Experiment base directory')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None,
                        help='Specify checkpoint file (auto: best)')
    
    # Inference
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')

    # Output options
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction results as nii.gz files')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save prediction results (auto generated if not provided)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
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


def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print("REBOUND - Testing")
    print("=" * 60)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Device: {device}")
    print(f">>> Dataset: {args.dataset}")
    print(f">>> Domain: {args.target}")

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), f"Checkpoint file does not exist: {args.checkpoint}"
    else:
        assert args.exp_dir is not None, "Experiment directory is required if checkpoint is not provided"
        if os.path.exists(args.exp_dir):
            checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
            best_checkpoint = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            if os.path.exists(best_checkpoint):
                args.checkpoint = best_checkpoint
            else:
                raise ValueError(f"Best checkpoint not found in {checkpoint_dir}")
        else:
            raise ValueError(f"Experiment directory does not exist: {args.exp_dir}")
    print(f">>> Loading checkpoint: {args.checkpoint}")

    args.data_dir = os.path.join(args.data_dir, args.dataset, args.processed_dir)
    assert os.path.exists(args.data_dir), f"Processed data directory does not exist: {args.data_dir}"
    print(f">>> Processed data directory: {args.data_dir}")

    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert args.target in metadata['domains'], f"Target domain {args.target} not found in metadata: {metadata['domains']}"
    assert args.source in metadata['domains'], f"Source domain {args.source} not found in metadata: {metadata['domains']}"
    
    model = CSANet(args.model_config, args.img_size, metadata['num_classes']).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    print(f">>> Model: {args.model}")
    print(f">>> Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = os.path.join(args.exp_dir, 'predictions')
        os.makedirs(args.save_dir, exist_ok=True)
    print(f">>> Saving predictions to: {args.save_dir}")

    # BEGIN TESTING
    evaluator = Evaluator(
        args=args,
        metadata=metadata,
        model=model,
        device=device
        )
    db_test = CSANet_VolumeDataset(
        base_dir=args.data_dir,
        domain_name=args.target,
        split='test',
        metadata=metadata,
        )
    print(f">>> Number of test volumes: {len(db_test)}")

    evaluator.evaluate(db_eval=db_test, show_details=True, save_predictions=args.save_predictions, save_dir=args.save_dir)

    print(f">>> Testing completed.")

if __name__ == '__main__':
    main()
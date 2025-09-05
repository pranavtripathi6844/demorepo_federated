#!/usr/bin/env python3
"""
Centralized training experiment script.
Runs baseline centralized training for comparison with federated learning.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.vision_transformer import DINOBackboneClassifier
from src.data.dataset_loader import CIFAR100DataManager
from src.training.centralized_training import train_centralized_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run centralized training experiment')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Vision Transformer model size (not used with DINO backbone)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze the backbone and only train the head (default: True)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--scheduler', type=str, default='cosine',
                   choices=['cosine', 'step', 'none'],
                   help='Learning rate scheduler type')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"Config file {config_path} not found, using defaults")
        return {}


def main():
    """Main experiment function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.update({
        'model_size': args.model_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'freeze_backbone': args.freeze_backbone,
        'batch_size': args.batch_size,
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'data_dir': args.data_dir,
        'val_split': args.val_split,
        'scheduler': args.scheduler
    })
    
    print("=== Centralized Training Experiment ===")
    print(f"Configuration: {config}")
    
    # Set device
    def get_device(device_preference: str = 'auto'):
        """Get the best available device for training."""
        if device_preference == 'auto':
            # Try MPS (Apple Silicon) first
            if torch.backends.mps.is_available():
                return torch.device('mps')
            # Try CUDA next
            elif torch.cuda.is_available():
                return torch.device('cuda')
            # Fall back to CPU
            else:
                return torch.device('cpu')
        elif device_preference == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("MPS not available, falling back to CPU")
                return torch.device('cpu')
        elif device_preference == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("CUDA not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')

    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Create data manager and loaders
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=4,
        download=True
    )
    
    train_loader, val_loader, test_loader = data_manager.get_centralized_loaders(
        val_split=config['val_split']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model - Using DINO backbone with frozen backbone (head-only training)
    print(f"Creating DINO Backbone Classifier (frozen backbone: {config['freeze_backbone']})...")
    model = DINOBackboneClassifier(
        num_classes=100,
        freeze_backbone=config['freeze_backbone']
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Train model
    print("Starting centralized training...")
    training_history = train_centralized_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0001),
        momentum=config.get('momentum', 0.9),
        scheduler_type=config.get('scheduler', 'cosine'),  # Add this line
        checkpoint_dir=config['checkpoint_dir'],
        model_name=f"centralized_dino_{config['model_size']}"
    )
    
    print("=== Training Complete ===")
    print(f"Final training accuracy: {training_history['train_accuracies'][-1]:.2f}%")
    print(f"Final validation accuracy: {training_history['val_accuracies'][-1]:.2f}%")
    
    # Save training history
    import json
    history_file = os.path.join(config['checkpoint_dir'], 'centralized_training_history.json')
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_file}")


if __name__ == '__main__':
    main()

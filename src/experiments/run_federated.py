#!/usr/bin/env python3
"""
Federated learning experiment script.
Runs federated learning with model editing capabilities.
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

from src.models.vision_transformer import create_vision_transformer
from src.data.dataset_loader import CIFAR100DataManager
from src.training.federated_training import train_federated_model_editing
from src.training.model_editing import create_client_masks


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run federated learning experiment')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Vision Transformer model size')
    parser.add_argument('--num_rounds', type=int, default=100,
                       help='Number of federated rounds')
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of federated clients')
    parser.add_argument('--client_fraction', type=float, default=0.1,
                       help='Fraction of clients per round')
    parser.add_argument('--num_client_steps', type=int, default=4,
                       help='Local training steps per client')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--non_iid_degree', type=float, default=0.0,
                       help='Degree of non-IID distribution (0.0 = IID)')
    parser.add_argument('--target_sparsity', type=float, default=0.9,
                       help='Target sparsity for model editing')
    parser.add_argument('--num_mask_iterations', type=int, default=10,
                       help='Number of mask generation iterations')
    parser.add_argument('--soft_zero_value', type=float, default=0.01,
                       help='Soft zero value for mask generation')
    # Checkpointing & resume support
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                       help='Save a checkpoint every N federated rounds')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a checkpoint .pth file to resume from')
    
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
        'num_rounds': args.num_rounds,
        'num_clients': args.num_clients,
        'client_fraction': args.client_fraction,
        'num_client_steps': args.num_client_steps,
        'batch_size': args.batch_size,
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'data_dir': args.data_dir,
        'non_iid_degree': args.non_iid_degree,
        'target_sparsity': args.target_sparsity,
        'num_mask_iterations': args.num_mask_iterations,
        'soft_zero_value': args.soft_zero_value,
        'checkpoint_interval': args.checkpoint_interval,
        'resume_from': args.resume_from
    })
    
    print("=== Federated Learning Experiment ===")
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
    
    # Create data manager
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=4,
        download=True
    )
    
    # Create federated datasets
    print(f"Creating federated datasets for {config['num_clients']} clients...")
    client_datasets = data_manager.create_federated_datasets(
        num_clients=config['num_clients'],
        non_iid_degree=config['non_iid_degree']
    )
    
    # Create validation loader
    _, val_loader, _ = data_manager.get_centralized_loaders(val_split=0.1)
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    
    # Create model
    print(f"Creating Vision Transformer ({config['model_size']})...")
    model = create_vision_transformer(
        model_size=config['model_size'],
        num_classes=100
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Generate client masks with adaptive stratified sampling
    print("Generating client masks for model editing...")
    print(f"Using adaptive stratified sampling for classes_per_client={config['classes_per_client']}")
    client_masks = create_client_masks(
        model=model,
        client_datasets=client_datasets,
        classes_per_client=config['classes_per_client'],
        batch_size=config['batch_size'],
        target_sparsity=config['target_sparsity'],
        num_iterations=config['num_mask_iterations'],
        soft_zero_value=config['soft_zero_value'],
        max_samples=25,
        debug_mode=True
    )
    
    print(f"Generated masks for {len(client_masks)} clients")
    
    # Optimizer configuration
    optimizer_config = {
        'lr': config.get('learning_rate', 0.01),
        'momentum': config.get('momentum', 0.9),
        'weight_decay': config.get('weight_decay', 0.0001)
    }
    
    # Train federated model
    print("Starting federated learning training...")
    training_history = train_federated_model_editing(
        model=model,
        client_datasets=client_datasets,
        client_masks=client_masks,
        optimizer_config=optimizer_config,
        device=device,
        num_rounds=config['num_rounds'],
        num_clients=config['num_clients'],
        client_fraction=config['client_fraction'],
        num_client_steps=config['num_client_steps'],
        batch_size=config['batch_size'],
        validation_loader=val_loader,
        checkpoint_path=config['checkpoint_dir'],
        model_name=f"federated_{config['model_size']}",
        checkpoint_interval=config['checkpoint_interval'],
        resume_from=config['resume_from']
    )
    
    print("=== Training Complete ===")
    print(f"Final client accuracy: {training_history['round_accuracies'][-1]:.2f}%")
    print(f"Final validation accuracy: {training_history['validation_accuracies'][-1]:.2f}%")
    
    # Save training history
    import json
    history_file = os.path.join(config['checkpoint_dir'], 'federated_training_history.json')
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_file}")


if __name__ == '__main__':
    main()

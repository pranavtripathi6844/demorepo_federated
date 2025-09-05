#!/usr/bin/env python3
"""
Model editing experiment script.
Runs model editing experiments using TaLoS methodology.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.vision_transformer import create_vision_transformer
from src.data.dataset_loader import CIFAR100DataManager
from src.training.model_editing import IterativeMaskGenerator
from src.training.sparse_optimizer import SparseSGDWithMomentum
from src.training.centralized_training import CentralizedTrainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run model editing experiment')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Vision Transformer model size')
    parser.add_argument('--target_sparsity', type=float, default=0.9,
                       help='Target sparsity for model editing')
    parser.add_argument('--num_iterations', type=int, default=10,
                       help='Number of mask generation iterations')
    parser.add_argument('--soft_zero_value', type=float, default=0.01,
                       help='Soft zero value for mask generation')
    parser.add_argument('--num_epochs', type=int, default=40,
                       help='Number of training epochs after editing')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--num_classes', type=int, default=100,
                       help='Number of classes for stratification')
    parser.add_argument('--samples_per_class', type=int, default=1,
                       help='Samples per class for stratification')
    
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
        'target_sparsity': args.target_sparsity,
        'num_iterations': args.num_iterations,
        'soft_zero_value': args.soft_zero_value,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'data_dir': args.data_dir,
        'num_classes': args.num_classes,
        'samples_per_class': args.samples_per_class
    })
    
    print("=== Model Editing Experiment ===")
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
        val_split=0.1
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating Vision Transformer ({config['model_size']})...")
    model = create_vision_transformer(
        model_size=config['model_size'],
        num_classes=100
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create stratified loader for mask generation
    print("Creating stratified data loader for mask generation...")
    stratified_loader = data_manager.get_stratified_loader(
        train_loader,
        num_classes=config['num_classes'],
        samples_per_class=config['samples_per_class']
    )
    
    print(f"Stratified samples: {len(stratified_loader.dataset)}")
    
    # Generate pruning mask
    print("Generating pruning mask using iterative Fisher-based pruning...")
    mask_generator = IterativeMaskGenerator(
        target_sparsity=config['target_sparsity'],
        num_iterations=config['num_iterations'],
        soft_zero_value=config['soft_zero_value']
    )
    
    start_time = time.time()
    pruning_mask = mask_generator.generate_mask(
        model=model,
        dataloader=stratified_loader,
        device=device,
        enable_visualization=False,
        debug_mode=True
    )
    end_time = time.time()
    
    print(f"Mask generation completed in {end_time - start_time:.2f} seconds")
    
    # Calculate mask statistics
    total_mask_params = sum(m.numel() for m in pruning_mask.values())
    masked_params = sum((m == 0).sum().item() for m in pruning_mask.values())
    actual_sparsity = masked_params / total_mask_params
    
    print(f"Target sparsity: {config['target_sparsity']:.2f}")
    print(f"Actual sparsity: {actual_sparsity:.4f}")
    print(f"Active parameters: {total_mask_params - masked_params:,}")
    print(f"Masked parameters: {masked_params:,}")
    
    # Train model with mask
    print("Training model with generated mask...")
    
    # Create sparse optimizer
    optimizer = SparseSGDWithMomentum(
        model.named_parameters(),
        pruning_mask,
        learning_rate=config['learning_rate'],
        momentum_factor=0.9,
        weight_decay_factor=0.0001
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    best_val_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        
        # Training
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = total_loss / total if total > 0 else 0.0
        train_accuracy = 100 * correct / total if total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mask': pruning_mask,
                'val_accuracy': val_accuracy,
                'sparsity': actual_sparsity
            }, os.path.join(config['checkpoint_dir'], 'best_model_editing.pth'))
    
    print("=== Model Editing Experiment Complete ==="")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final sparsity achieved: {actual_sparsity:.4f}")
    
    # Test final model
    print("Testing final model...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0.0
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == '__main__':
    main()

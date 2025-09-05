#!/usr/bin/env python3
"""
Hyperparameter Search for Centralized Training with Head-Only Training
Compares different learning rates and weight decays with frozen backbone.
Supports both StepLR and CosineAnnealingLR schedulers.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vision_transformer import DINOBackboneClassifier
from data.dataset_loader import CIFAR100DataManager
from utils.device_utils import get_device


class HyperparameterSearch:
    """
    Hyperparameter search for centralized training with head-only training.
    """
    
    def __init__(self, device: str = 'auto', freeze_backbone: bool = True, val_split: float = None, scheduler_type: str = 'StepLR'):
        """
        Initialize hyperparameter search.
        
        Args:
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
            freeze_backbone: Whether to freeze backbone and train only head
            val_split: Validation split ratio (overrides config if provided)
            scheduler_type: Type of scheduler ('StepLR' or 'CosineAnnealingLR')
        """
        self.device = get_device(device)
        self.freeze_backbone = freeze_backbone
        self.scheduler_type = scheduler_type
        
        # Load configuration
        with open('configs/default_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize data manager
        self.data_manager = CIFAR100DataManager(
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers']
        )
        
        # Get data loaders with custom validation split if provided
        if val_split is not None:
            self.train_loader, self.val_loader, self.test_loader = \
                self.data_manager.get_centralized_loaders(val_split=val_split)
        else:
            self.train_loader, self.val_loader, self.test_loader = \
                self.data_manager.get_centralized_loaders()
        
        print(f"✅ Data loaded: {len(self.train_loader.dataset)} train, "
              f"{len(self.val_loader.dataset)} val, {len(self.test_loader.dataset)} test samples")
        print(f"✅ Device: {self.device}")
        print(f"✅ Backbone frozen: {freeze_backbone}")
        print(f"✅ Scheduler: {scheduler_type}")
    
    def create_model(self):
        """Create model: DINO hub backbone + trainable head (head-only when frozen)."""
        model = DINOBackboneClassifier(
            num_classes=self.config['model']['num_classes'],
            freeze_backbone=self.freeze_backbone
        ).to(self.device)
        return model
    
    def create_scheduler(self, optimizer, num_epochs: int):
        """Create scheduler based on type."""
        if self.scheduler_type == 'StepLR':
            return StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_type == 'CosineAnnealingLR':
            return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def train_single_config(self, 
                           learning_rate: float, 
                           weight_decay: float,
                           momentum: float = 0.8,
                           num_epochs: int = 30) -> Tuple[float, List[float]]:
        """
        Train model with single hyperparameter configuration.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
            momentum: SGD momentum
            num_epochs: Number of training epochs
            
        Returns:
            Tuple of (best_val_accuracy, validation_accuracies_history)
        """
        # Create fresh model
        model = self.create_model()
        
        # Setup optimizer and scheduler
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )
        
        # Create scheduler based on type
        scheduler = self.create_scheduler(optimizer, num_epochs)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        val_accuracies = []
        best_val_acc = 0.0
        
        print(f"\n🔄 Training with LR={learning_rate}, WD={weight_decay}, Scheduler={self.scheduler_type}, Momentum={momentum}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calculate accuracies
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            val_accuracies.append(val_acc)
            
            # Update best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Step scheduler
            scheduler.step()
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1:2d}/{num_epochs}: "
                      f"Train Acc: {train_acc:5.2f}%, "
                      f"Val Acc: {val_acc:5.2f}%, "
                      f"LR: {current_lr:.6f}")
        
        print(f"  ✅ Best Validation Accuracy: {best_val_acc:.2f}%")
        return best_val_acc, val_accuracies
    
    def run_hyperparameter_search(self) -> Dict:
        """
        Run hyperparameter search across all combinations.
        
        Returns:
            Dictionary with search results
        """
        # Fixed values
        epochs = 30  # T_max = 30
        eta_min = 0.0  # Default minimum LR
        
        # Tuned values - same search space for both schedulers
        learning_rates = [0.001, 0.005, 0.01, 0.03]  # Initial LR
        weight_decays = [0.0001, 0.001]  # L2 penalty
        momentum = 0.8  # Fixed as requested
        
        print("\n" + "="*60)
        print("🔍 HYPERPARAMETER SEARCH: HEAD-ONLY TRAINING")
        print("="*60)
        print(f"Learning Rates: {learning_rates}")
        print(f"Weight Decays: {weight_decays}")
        print(f"Scheduler: {self.scheduler_type}")
        print(f"Momentum: {momentum}")
        print(f"Epochs: {epochs}")
        
        if self.scheduler_type == 'StepLR':
            print(f"StepLR: decay factor 0.1 every 10 epochs")
        elif self.scheduler_type == 'CosineAnnealingLR':
            print(f"CosineAnnealingLR: T_max={epochs}, eta_min={eta_min}")
        
        print(f"Total Combinations: {len(learning_rates) * len(weight_decays)}")
        print("="*60)
        
        # Results storage
        results = []
        best_config = None
        best_val_acc = 0.0
        
        # Grid search
        for lr in learning_rates:
            for wd in weight_decays:
                print(f"\n{'='*50}")
                print(f"Testing: LR={lr}, WD={wd}")
                print(f"{'='*50}")
                
                # Train with this configuration
                val_acc, val_history = self.train_single_config(
                    learning_rate=lr,
                    weight_decay=wd,
                    momentum=momentum,
                    num_epochs=epochs
                )
                
                # Store results
                result = {
                    'learning_rate': lr,
                    'weight_decay': wd,
                    'scheduler_type': self.scheduler_type,
                    'momentum': momentum,
                    'best_val_accuracy': val_acc,
                    'val_history': val_history
                }
                results.append(result)
                
                # Update best configuration
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = result
                
                print(f"📊 Result: LR={lr}, WD={wd} → Best Val Acc: {val_acc:.2f}%")
        
        # Print summary
        self._print_results_summary(results, best_config)
        
        return {
            'results': results,
            'best_config': best_config,
            'best_val_accuracy': best_val_acc
        }
    
    def _print_results_summary(self, results: List[Dict], best_config: Dict):
        """Print comprehensive results summary."""
        print("\n" + "="*80)
        print("📊 HYPERPARAMETER SEARCH RESULTS")
        print("="*80)
        
        # Create results table
        print(f"{'LR':<10} {'WD':<10} {'Best Val Acc':<15} {'Final Val Acc':<15}")
        print("-" * 60)
        
        for result in results:
            lr = result['learning_rate']
            wd = result['weight_decay']
            best_acc = result['best_val_accuracy']
            final_acc = result['val_history'][-1]
            
            print(f"{lr:<10.3f} {wd:<10.4f} {best_acc:<15.2f}% {final_acc:<15.2f}%")
        
        print("\n" + "="*80)
        print("🎯 BEST CONFIGURATION")
        print("="*80)
        print(f"Learning Rate: {best_config['learning_rate']}")
        print(f"Weight Decay: {best_config['weight_decay']}")
        print(f"Scheduler: {best_config['scheduler_type']}")
        print(f"Momentum: {best_config['momentum']}")
        print(f"Best Validation Accuracy: {best_config['best_val_accuracy']:.2f}%")
        print("="*80)
    
    def test_best_configuration(self, best_config: Dict) -> Tuple[float, float, Dict[str, List[float]]]:
        """
        Test the best configuration on test set.
        
        Args:
            best_config: Best hyperparameter configuration
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        print("\n" + "="*60)
        print("🧪 FINAL TEST WITH BEST CONFIGURATION")
        print("="*60)
        
        # Create model with best configuration
        model = self.create_model()
        
        # Setup optimizer and scheduler
        optimizer = optim.SGD(
            model.parameters(),
            lr=best_config['learning_rate'],
            weight_decay=best_config['weight_decay'],
            momentum=best_config['momentum']
        )
        
        scheduler = self.create_scheduler(optimizer, 30)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 30 epochs and record per-epoch test metrics
        print(f"Training for 30 epochs with best configuration...")
        test_loss_history: List[float] = []
        test_accuracy_history: List[float] = []
        for epoch in range(30):
            # Training
            model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Evaluate on test set this epoch
            model.eval()
            epoch_test_loss = 0.0
            epoch_test_correct = 0
            epoch_test_total = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    epoch_test_loss += loss.item()
                    _, predicted = output.max(1)
                    epoch_test_total += target.size(0)
                    epoch_test_correct += predicted.eq(target).sum().item()
            avg_epoch_test_loss = epoch_test_loss / len(self.test_loader)
            epoch_test_accuracy = 100. * epoch_test_correct / epoch_test_total if epoch_test_total > 0 else 0.0
            test_loss_history.append(avg_epoch_test_loss)
            test_accuracy_history.append(epoch_test_accuracy)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1:2d}/30 | Test Loss: {avg_epoch_test_loss:.4f} | Test Acc: {epoch_test_accuracy:.2f}% | LR: {current_lr:.6f}")
        
        # Test on test set
        print("Testing on test set...")
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        # Calculate final metrics
        avg_test_loss = test_loss / len(self.test_loader)
        test_accuracy = 100. * test_correct / test_total
        
        print(f"\n🏆 FINAL TEST RESULTS")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print("="*60)
        
        return avg_test_loss, test_accuracy, {
            'test_loss_history': test_loss_history,
            'test_accuracy_history': test_accuracy_history
        }


def main():
    """Main function to run hyperparameter search."""
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Centralized Training')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and train only head')
    parser.add_argument('--val_split', type=float, default=None,
                       help='Validation split ratio (e.g., 0.2 for 20% validation)')
    parser.add_argument('--scheduler', type=str, default='StepLR',
                       choices=['StepLR', 'CosineAnnealingLR'],
                       help='Scheduler type to use')
    
    args = parser.parse_args()
    
    print("🚀 Starting Hyperparameter Search...")
    print(f"Device: {args.device}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Scheduler: {args.scheduler}")
    if args.val_split is not None:
        print(f"Validation Split: {args.val_split}")
    
    # Create search instance
    search = HyperparameterSearch(
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        val_split=args.val_split,
        scheduler_type=args.scheduler
    )
    
    # Run hyperparameter search
    search_results = search.run_hyperparameter_search()
    
    # Test best configuration
    if search_results['best_config']:
        test_loss, test_accuracy, test_hist = search.test_best_configuration(
            search_results['best_config']
        )
        
        # Save results with scheduler type in filename
        results_filename = f'hyperparameter_search_results_{args.scheduler.lower()}.yaml'
        results_summary = {
            'scheduler_type': args.scheduler,
            'best_config': search_results['best_config'],
            'best_val_accuracy': search_results['best_val_accuracy'],
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_history': test_hist,
            'all_results': search_results['results']
        }
        
        print(f"\n💾 Results saved to '{results_filename}'")
        with open(results_filename, 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False)
    
    print("\n✅ Hyperparameter search completed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Standard Non-IID Federated Learning Experiments
Tests the 5 standard non-IID scenarios used in federated learning research:
- IID: classes_per_client = 100 (all classes per client)
- Non-IID(1): classes_per_client = 1 (extreme non-IID - each client sees only 1 class)
- Non-IID(5): classes_per_client = 5 (moderate non-IID)
- Non-IID(10): classes_per_client = 10 (mild non-IID)
- Non-IID(50): classes_per_client = 50 (slight non-IID)
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys
import time
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset_loader import CIFAR100DataManager
from src.training.baseline_federated_training import BaselineFederatedTrainer
from src.models.vision_transformer import DINOBackboneClassifier


def get_device(pref: str = 'auto'):
    """Get the best available device for training."""
    if pref == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if pref == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run standard non-IID federated learning experiments')
    
    # Experiment parameters
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--num_clients', type=int, default=100, help='Number of federated clients')
    parser.add_argument('--client_fraction', type=float, default=0.1, help='Fraction of clients per round')
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of federated rounds')
    parser.add_argument('--num_client_steps', type=int, default=4, help='Local training steps per client')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, help='Freeze backbone')
    
    # Non-IID scenarios to test
    parser.add_argument('--scenarios', nargs='+', default=['iid', 'non_iid_1', 'non_iid_5', 'non_iid_10', 'non_iid_50'],
                       help='Non-IID scenarios to test')
    parser.add_argument('--output_dir', type=str, default='./results/standard_non_iid',
                       help='Directory to save results')
    
    return parser.parse_args()


def get_classes_per_client(scenario: str) -> int:
    """Get classes_per_client for a given scenario."""
    scenario_map = {
        'iid': 100,
        'non_iid_1': 1,
        'non_iid_5': 5,
        'non_iid_10': 10,
        'non_iid_50': 50
    }
    return scenario_map.get(scenario, 100)


def run_single_scenario(scenario: str, args, device) -> Dict:
    """Run federated learning for a single non-IID scenario."""
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario.upper()}")
    print(f"{'='*60}")
    
    classes_per_client = get_classes_per_client(scenario)
    print(f"Classes per client: {classes_per_client}")
    
    # Create data manager
    data_manager = CIFAR100DataManager(batch_size=args.batch_size, download=True)
    
    # Create federated datasets using standard shard-based approach
    print(f"Creating federated datasets for {args.num_clients} clients...")
    client_datasets = data_manager.create_federated_datasets(
        num_clients=args.num_clients, 
        classes_per_client=classes_per_client,
        val_split=args.val_split
    )
    
    # Create validation loader
    _, val_loader, _ = data_manager.get_centralized_loaders(val_split=args.val_split)
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Client dataset sizes: {[len(dataset) for dataset in client_datasets[:5]]}...")  # Show first 5
    
    # Create model
    model = DINOBackboneClassifier(
        num_classes=100, 
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer configuration
    optimizer_config = {
        'lr': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }
    
    # Create trainer
    trainer = BaselineFederatedTrainer(
        model=model,
        client_datasets=client_datasets,
        optimizer_config=optimizer_config,
        device=str(device),
        num_clients=args.num_clients,
        client_fraction=args.client_fraction,
        num_client_steps=args.num_client_steps,
        batch_size=args.batch_size
    )
    
    # Train the model
    print(f"Starting federated training for {args.num_rounds} rounds...")
    start_time = time.time()
    
    history = trainer.train_federated_rounds(
        num_rounds=args.num_rounds,
        validation_loader=val_loader,
        model_name=f"standard_{scenario}"
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Extract results
    best_val_acc = max(history['validation_accuracies']) if history['validation_accuracies'] else 0.0
    final_val_acc = history['validation_accuracies'][-1] if history['validation_accuracies'] else 0.0
    
    print(f"\nScenario {scenario.upper()} Results:")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Final validation accuracy: {final_val_acc:.2f}%")
    print(f"  Training time: {training_time:.2f} seconds")
    
    return {
        'scenario': scenario,
        'classes_per_client': classes_per_client,
        'best_val_accuracy': best_val_acc,
        'final_val_accuracy': final_val_acc,
        'training_time': training_time,
        'history': history
    }


def main():
    """Main experiment function."""
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("STANDARD NON-IID FEDERATED LEARNING EXPERIMENTS")
    print("="*80)
    print(f"Scenarios to test: {args.scenarios}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Client fraction: {args.client_fraction}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Local steps per client: {args.num_client_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print("="*80)
    
    # Run experiments for each scenario
    all_results = []
    
    for scenario in args.scenarios:
        try:
            result = run_single_scenario(scenario, args, device)
            all_results.append(result)
        except Exception as e:
            print(f"Error running scenario {scenario}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<15} {'Classes/Client':<12} {'Best Val Acc':<12} {'Final Val Acc':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['scenario']:<15} {result['classes_per_client']:<12} "
              f"{result['best_val_accuracy']:<12.2f} {result['final_val_accuracy']:<12.2f} "
              f"{result['training_time']:<10.2f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'standard_non_iid_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    
    print(f"\nResults saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()

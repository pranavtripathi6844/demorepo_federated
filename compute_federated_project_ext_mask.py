#!/usr/bin/env python3
"""
Compute federated masks for all strategies with command line sparsity support.

This script generates masks for all 5 parameter selection strategies:
1. least_sensitive (current/baseline approach)
2. most_sensitive (opposite approach)
3. lowest_magnitude (classic magnitude pruning)
4. highest_magnitude (counter-intuitive approach)
5. random (baseline for comparison)

Usage:
    python compute_federated_masks_all_strategies.py --sparsity 0.78
    python compute_federated_masks_all_strategies.py --sparsity 0.68 --device cuda
    python compute_federated_masks_all_strategies.py --sparsity 0.88 --num_clients_test 5
"""

import argparse
import torch
import os
import time
from typing import List, Dict
from torch.utils.data import DataLoader

# Import the new mask computation functions
from src.training.model_editing import (
    compute_mask,                    # Original (least-sensitive)
    compute_mask_most_sensitive,     # New
    compute_mask_lowest_magnitude,   # New  
    compute_mask_highest_magnitude,  # New
    compute_mask_random             # New
)

# Import existing functionality
from src.models.linear_flexible_dino import LinearFlexibleDino
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits
from compute_federated_masks_stratified import get_n_examples_per_class_loader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute federated masks for all strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute masks with 78% sparsity
  python compute_federated_masks_all_strategies.py --sparsity 0.78
  
  # Compute masks with 68% sparsity on GPU
  python compute_federated_masks_all_strategies.py --sparsity 0.68 --device cuda
  
  # Test with fewer clients first
  python compute_federated_masks_all_strategies.py --sparsity 0.88 --num_clients_test 5
        """
    )
    
    parser.add_argument('--sparsity', type=float, required=True,
                       help='Target sparsity (e.g., 0.78 for 78% sparsity)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for computation (default: cpu)')
    parser.add_argument('--R', type=int, default=3,
                       help='Number of iterative refinement rounds (default: 3)')
    parser.add_argument('--soft_zero', type=float, default=0.01,
                       help='Soft zero value for Fisher computation (default: 0.01)')
    parser.add_argument('--num_examples', type=int, default=30,
                       help='Number of examples per client for Fisher calculation (default: 30)')
    parser.add_argument('--output_dir', type=str, default='masks/strategy_experiments',
                       help='Output directory for masks (default: masks/strategy_experiments)')
    parser.add_argument('--num_clients_test', type=int, default=None,
                       help='Number of clients to test with (default: all as per nc_configs)')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['least_sensitive', 'most_sensitive', 'lowest_magnitude', 'highest_magnitude', 'random'],
                       default=['least_sensitive', 'most_sensitive', 'lowest_magnitude', 'highest_magnitude', 'random'],
                       help='Strategies to compute (default: all)')
    
    return parser.parse_args()


def compute_all_strategy_masks(
    sparsity: float,
    strategies: List[str],
    R: int = 3,
    soft_zero: float = 0.01,
    num_examples: int = 30,
    device: str = 'cpu',
    output_dir: str = 'masks/strategy_experiments',
    num_clients_test: int = None
):
    """
    Compute federated masks for all strategies following the same nc_configs pattern.
    
    Args:
        sparsity: Target sparsity (e.g., 0.78 for 78% sparsity)
        strategies: List of strategies to compute
        R: Number of iterative refinement rounds
        soft_zero: Soft zero value for Fisher computation
        num_examples: Number of examples per client for Fisher calculation
        device: Device to use for computation
        output_dir: Output directory for masks
        num_clients_test: Number of clients to test with (None = use all as per nc_configs)
    """
    
    # Define strategy functions
    strategy_functions = {
        'least_sensitive': compute_mask,                    # Your current approach
        'most_sensitive': compute_mask_most_sensitive,      # New
        'lowest_magnitude': compute_mask_lowest_magnitude,   # New
        'highest_magnitude': compute_mask_highest_magnitude, # New
        'random': compute_mask_random                       # New
    }
    
    # Same configuration as current approach
    nc_configs = [
        {"nc": 1,   "num_clients": 100, "num_classes": 1,   "n_per_class": 100},
        {"nc": 5,   "num_clients": 20,  "num_classes": 5,   "n_per_class": 20},
        {"nc": 10,  "num_clients": 10,  "num_classes": 10,  "n_per_class": 10},
        {"nc": 50,  "num_clients": 2,   "num_classes": 50,  "n_per_class": 2},
        {"nc": 100, "num_clients": 1,   "num_classes": 100, "n_per_class": 1}
    ]
    
    print("🧪 Computing federated masks for ALL strategies")
    print(f"📊 Configuration: sparsity={sparsity}, R={R}, soft_zero={soft_zero}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Strategies: {strategies}")
    
    # Load model and data (SAME for all strategies)
    print("\n📚 Loading model and data...")
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
    model.eval()
    
    # CRITICAL: Unfreeze all backbone blocks for mask generation
    print("🔓 Unfreezing all backbone blocks for mask generation...")
    model.freeze(0)  # Unfreeze all backbone blocks
    
    # Load data
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders(val_split=0.2)
    
    # Create strategy-specific output directory
    strategy_dir = f"{output_dir}/sparsity_{int(sparsity*100):03d}"
    os.makedirs(strategy_dir, exist_ok=True)
    
    # For EACH strategy
    for strategy_name in strategies:
        if strategy_name not in strategy_functions:
            print(f"❌ Unknown strategy: {strategy_name}")
            continue
            
        compute_function = strategy_functions[strategy_name]
        print(f"\n{'='*60}")
        print(f"🎯 Computing masks for strategy: {strategy_name.upper()}")
        print(f"📊 Sparsity: {sparsity}")
        print(f"{'='*60}")
        
        # For EACH Nc configuration
        for config in nc_configs:
            nc = config["nc"]
            num_clients = config["num_clients"]
            num_classes = config["num_classes"]
            n_per_class = config["n_per_class"]
            
            # Override number of clients for testing if specified
            if num_clients_test is not None:
                actual_clients = min(num_clients_test, num_clients)
            else:
                actual_clients = num_clients
            
            print(f"\n  === Processing Nc={nc} with {strategy_name} ===")
            print(f"    - Will compute {actual_clients} masks (out of {num_clients} total)")
            print(f"    - Classes per client: {num_classes}")
            print(f"    - Examples per class: {n_per_class}")
            
            # Create non-IID splits (SAME as current)
            client_datasets = create_non_iid_splits(
                train_loader.dataset,
                num_clients=100,  # Always create 100, use first num_clients
                classes_per_client=nc
            )
            
            # Compute masks for this Nc configuration
            client_masks = []
            start_time = time.time()
            
            for client_id, client_dataset in enumerate(client_datasets[:actual_clients]):
                print(f"    Client {client_id+1}/{actual_clients} - {strategy_name}", end="\r")
                
                # Create DataLoader for the client dataset
                client_loader = DataLoader(
                    client_dataset, 
                    batch_size=128, 
                    shuffle=True, 
                    num_workers=0
                )
                
                # Create stratified loader (SAME as current)
                stratified_loader = get_n_examples_per_class_loader(
                    client_loader, 
                    num_classes=num_classes, 
                    n_per_class=n_per_class
                )
                
                # ONLY DIFFERENCE: Use different compute function
                mask = compute_function(
                    model=model,
                    dataloader=stratified_loader,
                    sparsity_target=sparsity,
                    R=R,
                    soft_zero=soft_zero,
                    num_examples=num_examples,
                    device=device,
                    enable_plot=False,
                    debug=False
                )
                
                client_masks.append(mask)
            
            elapsed_time = time.time() - start_time
            print(f"\n    ⏱️  Completed in {elapsed_time:.2f} seconds")
            
            # Save masks with SAME naming pattern as current system
            save_path = f"{strategy_dir}/federated_masks_nc{nc}_R{R}_sz{soft_zero}_{strategy_name}.pth"
            torch.save(client_masks, save_path)
            print(f"    💾 Saved {len(client_masks)} masks to {save_path}")
            
            # Quick analysis
            analyze_mask_properties(client_masks, strategy_name, nc)
    
    print(f"\n🎉 All {len(strategies)} strategy masks computed successfully!")
    print(f"📁 Results saved in: {strategy_dir}")
    print(f"\n📋 Summary:")
    print(f"   • Sparsity: {sparsity}")
    print(f"   • Strategies: {len(strategies)}")
    print(f"   • Nc configurations: {len(nc_configs)}")
    print(f"   • Total mask files: {len(strategies) * len(nc_configs)}")


def analyze_mask_properties(client_masks: List[Dict], strategy_name: str, nc: int):
    """Quick analysis of mask properties."""
    if not client_masks:
        return
        
    # Analyze first client's mask
    mask = client_masks[0]
    total_params = sum(m.numel() for m in mask.values())
    pruned_params = sum((m == 0).sum().item() for m in mask.values())
    actual_sparsity = pruned_params / total_params
    
    print(f"    📊 Actual sparsity: {actual_sparsity:.4f}")
    print(f"    🔢 Total params: {total_params:,}, Pruned: {pruned_params:,}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🚀 Federated Mask Generation - All Strategies")
    print("=" * 50)
    print(f"Target sparsity: {args.sparsity}")
    print(f"Device: {args.device}")
    print(f"R (iterations): {args.R}")
    print(f"Soft zero: {args.soft_zero}")
    print(f"Number of examples: {args.num_examples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Strategies: {args.strategies}")
    if args.num_clients_test:
        print(f"Testing with: {args.num_clients_test} clients per configuration")
    print("=" * 50)
    
    # Validate sparsity
    if not (0.0 < args.sparsity < 1.0):
        raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {args.sparsity}")
    
    # Compute masks for all strategies
    compute_all_strategy_masks(
        sparsity=args.sparsity,
        strategies=args.strategies,
        R=args.R,
        soft_zero=args.soft_zero,
        num_examples=args.num_examples,
        device=args.device,
        output_dir=args.output_dir,
        num_clients_test=args.num_clients_test
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute federated masks for highest and lowest magnitude parameter strategies.

This script generates masks for magnitude-based parameter selection:
1. lowest_magnitude (classic magnitude pruning - removes smallest weights)
2. highest_magnitude (counter-intuitive approach - removes largest weights)

Maintains the same format as existing system:
- nc=1: 100 masks (one per client)
- nc=5: 20 masks
- nc=10: 10 masks  
- nc=50: 2 masks
- nc=100: 1 mask
Total: 133 masks

Usage:
    python compute_magnitude_masks.py --sparsity 0.78
    python compute_magnitude_masks.py --sparsity 0.68 --device cuda
    python compute_magnitude_masks.py --sparsity 0.88 --strategies lowest_magnitude
"""

import torch
import os
import time
import argparse
from typing import List, Dict
from torch.utils.data import DataLoader

from src.models.vision_transformer import LinearFlexibleDino
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits


def compute_mask_lowest_magnitude_direct(model: torch.nn.Module, 
                                       sparsity_target: float = 0.78,
                                       device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Compute mask by pruning LOWEST-MAGNITUDE parameters directly (no Fisher Information).
    
    Args:
        model: Model to generate masks for
        sparsity_target: Target sparsity (e.g., 0.78 for 78% sparsity)
        device: Device to use for computation
        
    Returns:
        Dictionary of binary masks for each parameter
    """
    print("📏 Computing mask with LOWEST-MAGNITUDE parameter selection (direct)")
    
    # Initialize mask with all parameters active
    mask = {}
    all_magnitudes = []
    param_info = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Get parameter magnitudes (absolute values)
            param_magnitudes = torch.abs(param.data).view(-1)
            mask[name] = torch.ones_like(param.data, device=device)
            all_magnitudes.append(param_magnitudes)
            param_info.append((name, param_magnitudes, param.data.shape))
    
    # Concatenate all magnitudes
    all_magnitudes = torch.cat(all_magnitudes)
    
    # Calculate threshold for target sparsity
    total_params = len(all_magnitudes)
    target_pruned = int(total_params * sparsity_target)
    
    if target_pruned > 0:
        # Sort magnitudes in ascending order (smallest first)
        sorted_magnitudes, _ = torch.sort(all_magnitudes)
        threshold = sorted_magnitudes[target_pruned - 1].item()
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Target pruned: {target_pruned:,}")
        print(f"  - Threshold: {threshold:.6f}")
        
        # Apply threshold to each parameter
        for name, param_magnitudes, shape in param_info:
            param_mask = mask[name]
            param_magnitudes_reshaped = torch.abs(model.state_dict()[name])
            pruning_indices = param_magnitudes_reshaped <= threshold
            param_mask[pruning_indices] = 0
            
            pruned_count = (param_mask == 0).sum().item()
            print(f"  - {name}: {pruned_count:,} pruned out of {param_mask.numel():,}")
    
    return mask


def compute_mask_highest_magnitude_direct(model: torch.nn.Module, 
                                        sparsity_target: float = 0.78,
                                        device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Compute mask by pruning HIGHEST-MAGNITUDE parameters directly (no Fisher Information).
    
    Args:
        model: Model to generate masks for
        sparsity_target: Target sparsity (e.g., 0.78 for 78% sparsity)
        device: Device to use for computation
        
    Returns:
        Dictionary of binary masks for each parameter
    """
    print("🎯 Computing mask with HIGHEST-MAGNITUDE parameter selection (direct)")
    
    # Initialize mask with all parameters active
    mask = {}
    all_magnitudes = []
    param_info = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Get parameter magnitudes (absolute values)
            param_magnitudes = torch.abs(param.data).view(-1)
            mask[name] = torch.ones_like(param.data, device=device)
            all_magnitudes.append(param_magnitudes)
            param_info.append((name, param_magnitudes, param.data.shape))
    
    # Concatenate all magnitudes
    all_magnitudes = torch.cat(all_magnitudes)
    
    # Calculate threshold for target sparsity
    total_params = len(all_magnitudes)
    target_pruned = int(total_params * sparsity_target)
    
    if target_pruned > 0:
        # Sort magnitudes in descending order (largest first)
        sorted_magnitudes, _ = torch.sort(all_magnitudes, descending=True)
        threshold = sorted_magnitudes[target_pruned - 1].item()
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Target pruned: {target_pruned:,}")
        print(f"  - Threshold: {threshold:.6f}")
        
        # Apply threshold to each parameter
        for name, param_magnitudes, shape in param_info:
            param_mask = mask[name]
            param_magnitudes_reshaped = torch.abs(model.state_dict()[name])
            pruning_indices = param_magnitudes_reshaped >= threshold
            param_mask[pruning_indices] = 0
            
            pruned_count = (param_mask == 0).sum().item()
            print(f"  - {name}: {pruned_count:,} pruned out of {param_mask.numel():,}")
    
    return mask


def get_n_examples_per_class_loader(client_loader: DataLoader, 
                                  num_classes: int, 
                                  n_per_class: int) -> DataLoader:
    """
    Create stratified loader with exactly n_per_class examples from each class.
    
    Args:
        client_loader: Client's data loader
        num_classes: Number of classes this client sees
        n_per_class: Number of examples per class to select
        
    Returns:
        Stratified data loader with balanced class representation
    """
    # Collect samples by class
    class_samples = {}
    for batch_idx, (data, targets) in enumerate(client_loader):
        for sample_idx, target in enumerate(targets):
            target_item = target.item()
            if target_item not in class_samples:
                class_samples[target_item] = []
            class_samples[target_item].append((data[sample_idx], target))
    
    # Select balanced samples
    selected_samples = []
    for class_label in range(min(num_classes, len(class_samples))):
        if class_label in class_samples:
            # Randomly select n_per_class samples from this class
            class_data = class_samples[class_label]
            if len(class_data) >= n_per_class:
                selected_indices = torch.randperm(len(class_data))[:n_per_class]
                for idx in selected_indices:
                    selected_samples.append(class_data[idx])
            else:
                # If not enough samples, take all available
                selected_samples.extend(class_data)
    
    # Create new dataset and loader
    if selected_samples:
        data_list, target_list = zip(*selected_samples)
        data_tensor = torch.stack(data_list)
        target_tensor = torch.stack(target_list)
        
        # Create a custom dataset
        class StratifiedDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        stratified_dataset = StratifiedDataset(data_tensor, target_tensor)
        return DataLoader(stratified_dataset, batch_size=32, shuffle=False, num_workers=0)
    else:
        # Fallback to original dataloader if no samples found
        return client_loader


def compute_magnitude_mask_clients(model: torch.nn.Module,
                                 client_datasets: List,
                                 strategy: str,
                                 num_examples: int = 30,
                                 num_classes: int = 1,
                                 n_per_class: int = 100,
                                 final_sparsity: float = 0.78,
                                 R: int = 3,
                                 soft_zero: float = 0.01,
                                 device: str = 'cpu') -> List[Dict[str, torch.Tensor]]:
    """
    Compute magnitude-based masks for all clients using DIRECT parameter values (no Fisher).
    """
    # Select the appropriate magnitude function
    if strategy == 'lowest_magnitude':
        compute_function = compute_mask_lowest_magnitude_direct
    elif strategy == 'highest_magnitude':
        compute_function = compute_mask_highest_magnitude_direct
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    client_masks = []
    
    for client_id, client_dataset in enumerate(client_datasets):
        print(f"  Client {client_id+1}/{len(client_datasets)} - {strategy}", end="\r")
        
        # For magnitude-based strategies, we don't need dataloader
        # We work directly with the model parameters
        mask = compute_function(
            model=model, 
            sparsity_target=final_sparsity,
            device=device
        )
        
        client_masks.append(mask)
    
    return client_masks


def compute_federated_magnitude_masks(sparsity=0.78, 
                                    strategies=['lowest_magnitude', 'highest_magnitude'],
                                    output_dir="masks/magnitude_strategies",
                                    device='cpu'):
    """Compute federated masks using magnitude-based strategies."""
    print("🎯 Computing federated masks with MAGNITUDE-BASED strategies...")
    print(f"⚙️  Configuration: R=3, sparsity={sparsity}, soft_zero=0.01")
    print(f"🎯 Strategies: {strategies}")
    print(f"🖥️  Device: {device}")
    print(f"🚀 Using DIRECT parameter values (NO Fisher Information)")
    
    # Load model
    print("🤖 Loading LinearFlexibleDino...")
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
    model.eval()
    
    # CRITICAL: Unfreeze all backbone blocks for mask generation
    print("🔓 Unfreezing all backbone blocks for mask generation...")
    model.freeze(0)  # Unfreeze all backbone blocks
    
    # Load data
    print("📚 Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders(val_split=0.2)
    
    # Configuration
    R = 3
    soft_zero = 0.01
    num_examples = 30
    
    # Nc values and corresponding parameters - REORDERED for quick testing (descending)
    nc_configs = [
        {"nc": 100, "num_clients": 1, "num_classes": 100, "n_per_class": 1},   # 1 mask - fastest test
        {"nc": 50, "num_clients": 2, "num_classes": 50, "n_per_class": 2},     # 2 masks
        {"nc": 10, "num_clients": 10, "num_classes": 10, "n_per_class": 10},   # 10 masks
        {"nc": 5, "num_clients": 20, "num_classes": 5, "n_per_class": 20},     # 20 masks
        {"nc": 1, "num_clients": 100, "num_classes": 1, "n_per_class": 100}    # 100 masks - slowest
    ]
    
    print(f"📊 Using 40k/10k/10k split (train/val/test)")
    print(f"⚙️  Configuration: R={R}, sparsity={sparsity}, soft_zero={soft_zero}")
    print(f"📊 Stratified sampling: {num_examples} examples per client")
    
    # Create masks directory
    strategy_dir = f"{output_dir}/sparsity_{int(sparsity*100):03d}"
    os.makedirs(strategy_dir, exist_ok=True)
    
    total_masks = 0
    
    # For each strategy
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"🎯 Computing masks for strategy: {strategy.upper()}")
        print(f"📊 Sparsity: {sparsity}")
        print(f"{'='*60}")
        
        # For each Nc configuration
        for config in nc_configs:
            nc = config["nc"]
            num_clients = config["num_clients"]
            num_classes = config["num_classes"]
            n_per_class = config["n_per_class"]
            
            print(f"\n  === Processing Nc={nc} with {strategy} ===")
            print(f"    - Will compute {num_clients} masks")
            print(f"    - Classes per client: {num_classes}")
            print(f"    - Examples per class: {n_per_class}")
            print(f"    - Total examples per client: {num_classes * n_per_class}")
            
            # Create non-IID splits
            print(f"    Creating non-IID splits: {num_clients} clients, {nc} classes per client")
            client_datasets = create_non_iid_splits(
                train_loader.dataset,
                num_clients=num_clients, 
                classes_per_client=nc
            )
            
            # Compute masks for each client
            print(f"    Computing {strategy} masks with DIRECT parameter values...")
            start_time = time.time()
            
            client_masks = compute_magnitude_mask_clients(
                model=model,
                client_datasets=client_datasets,
                strategy=strategy,
                num_examples=num_examples,
                num_classes=num_classes,
                n_per_class=n_per_class,
                final_sparsity=sparsity,
                R=R,
                soft_zero=soft_zero,
                device=device
            )
            
            elapsed_time = time.time() - start_time
            print(f"\n    ⏱️  Completed in {elapsed_time:.2f} seconds")
            
            # Calculate mask statistics
            if len(client_masks) > 0:
                total_params = sum(m.numel() for m in client_masks[0].values())
                avg_pruned_params = 0
                for client_mask in client_masks:
                    pruned_params = sum((m == 0).sum().item() for m in client_mask.values())
                    avg_pruned_params += pruned_params
                avg_pruned_params /= len(client_masks)
                avg_sparsity = avg_pruned_params / total_params
                
                print(f"    📊 Mask statistics:")
                print(f"      - Total parameters: {total_params:,}")
                print(f"      - Average pruned parameters: {avg_pruned_params:.0f}")
                print(f"      - Average sparsity: {avg_sparsity:.4f} ({avg_sparsity*100:.1f}%)")
                print(f"      - Average active parameters: {total_params - avg_pruned_params:.0f}")
                
                # Verify masks are not all zeros
                all_zeros = all((m == 0).all().item() for m in client_masks[0].values())
                if all_zeros:
                    print(f"      ❌ WARNING: All masks are zeros! {strategy} mask generation failed.")
                else:
                    print(f"      ✅ {strategy} masks generated successfully with proper sparsity.")
            
            # Save masks for this Nc configuration
            save_path = f"{strategy_dir}/federated_masks_nc{nc}_R{R}_sz{soft_zero}_{strategy}.pth"
            torch.save(client_masks, save_path)
            print(f"    💾 Saved {len(client_masks)} masks to {save_path}")
            
            # Verify the saved masks
            try:
                loaded_masks = torch.load(save_path, map_location='cpu')
                if len(loaded_masks) == len(client_masks):
                    print(f"    ✅ Verification: {len(loaded_masks)} masks loaded successfully")
                else:
                    print(f"    ❌ Verification failed: Expected {len(client_masks)}, got {len(loaded_masks)}")
            except Exception as e:
                print(f"    ❌ Verification failed: {e}")
            
            total_masks += len(client_masks)
    
    print(f"\n🎉 All {len(strategies)} magnitude strategy masks computed successfully!")
    print(f"📁 Results saved in: {strategy_dir}")
    print(f"📊 Total masks generated: {total_masks}")
    print(f"⚙️  Configuration: R={R}, sparsity={sparsity}, soft_zero={soft_zero}")
    print(f"📊 Used DIRECT parameter values (NO Fisher Information)")
    
    # List all generated mask files
    print(f"\n📋 Generated mask files:")
    mask_files = [f for f in os.listdir(strategy_dir) if f.startswith("federated_masks")]
    for mask_file in sorted(mask_files):
        print(f"  - {strategy_dir}/{mask_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute federated masks with magnitude-based strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute both magnitude strategies with 78% sparsity
  python compute_magnitude_masks.py --sparsity 0.78
  
  # Compute only lowest magnitude with 68% sparsity on GPU
  python compute_magnitude_masks.py --sparsity 0.68 --strategies lowest_magnitude --device cuda
  
  # Test with fewer clients first
  python compute_magnitude_masks.py --sparsity 0.88 --num_clients_test 5
        """
    )
    
    parser.add_argument('--sparsity', type=float, required=True,
                       help='Target sparsity (e.g., 0.78 for 78% sparsity)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for computation (default: cpu)')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['lowest_magnitude', 'highest_magnitude'],
                       default=['lowest_magnitude', 'highest_magnitude'],
                       help='Magnitude strategies to compute (default: both)')
    parser.add_argument('--output_dir', type=str, default='masks/magnitude_strategies',
                       help='Output directory for masks (default: masks/magnitude_strategies)')
    parser.add_argument('--num_clients_test', type=int, default=None,
                       help='Number of clients to test with (default: all as per nc_configs)')
    
    args = parser.parse_args()
    
    # Validate sparsity
    if not (0.0 < args.sparsity < 1.0):
        raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {args.sparsity}")
    
    compute_federated_magnitude_masks(
        sparsity=args.sparsity,
        strategies=args.strategies,
        output_dir=args.output_dir,
        device=args.device
    )

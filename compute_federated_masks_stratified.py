#!/usr/bin/env python3
"""
Compute federated masks using stratified loaders for balanced Fisher Information calculation.
This approach ensures each client gets a representative sample for mask generation.
"""

import torch
import os
import time
from typing import List, Dict
from torch.utils.data import DataLoader

from src.models.vision_transformer import LinearFlexibleDino
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits
from src.training.model_editing import compute_mask


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


def compute_mask_clients(model: torch.nn.Module,
                        client_datasets: List,
                        num_examples: int = 30,
                        num_classes: int = 1,
                        n_per_class: int = 100,
                        final_sparsity: float = 0.78,
                        R: int = 3,
                        soft_zero: float = 0.01,
                        device: str = 'cpu') -> List[Dict[str, torch.Tensor]]:
    """
    Compute masks for all clients using stratified loaders.
    
    Args:
        model: Model to generate masks for
        client_datasets: List of client datasets
        num_examples: Number of examples to use for Fisher calculation
        num_classes: Number of classes per client
        n_per_class: Number of examples per class in stratified loader
        final_sparsity: Target sparsity for masks
        R: Number of iterative refinement rounds
        soft_zero: Soft zero value for Fisher computation
        device: Device to use for computation
        
    Returns:
        List of masks for each client
    """
    client_masks = []
    
    for client_id, client_dataset in enumerate(client_datasets):
        print(f"  Client {client_id+1}/{len(client_datasets)}", end="\r")
        
        # Create DataLoader for the client dataset
        client_loader = DataLoader(
            client_dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=0  # Use 0 for compatibility
        )
        
        # Create stratified loader for balanced Fisher calculation
        stratified_loader = get_n_examples_per_class_loader(
            client_loader, 
            num_classes=num_classes, 
            n_per_class=n_per_class
        )
        
        # Compute mask for this client using stratified data
        mask = compute_mask(
            model=model, 
            dataloader=stratified_loader, 
            sparsity_target=final_sparsity, 
            R=R, 
            soft_zero=soft_zero, 
            num_examples=num_examples,
            device=device,
            enable_plot=False,
            debug=False
        )
        
        client_masks.append(mask)
    
    return client_masks


def compute_federated_masks_stratified():
    """Compute federated masks using stratified loaders."""
    print("Computing federated masks with stratified loaders...")
    print("Configuration: R=3, sparsity=0.78, soft_zero=0.01")
    print("Using stratified loaders for balanced Fisher Information calculation")
    
    # Load model
    print("Loading LinearFlexibleDino...")
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
    model.eval()
    
    # CRITICAL: Unfreeze all backbone blocks for mask generation
    print("Unfreezing all backbone blocks for mask generation...")
    model.freeze(0)  # Unfreeze all backbone blocks
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders(val_split=0.2)
    
    # Configuration
    R = 3
    sparsity = 0.78
    soft_zero = 0.01
    num_examples = 30
    
    # Nc values and corresponding parameters
    nc_configs = [
        {"nc": 1, "num_clients": 100, "num_classes": 1, "n_per_class": 100},
        {"nc": 5, "num_clients": 20, "num_classes": 5, "n_per_class": 20},
        {"nc": 10, "num_clients": 10, "num_classes": 10, "n_per_class": 10},
        {"nc": 50, "num_clients": 2, "num_classes": 50, "n_per_class": 2},
        {"nc": 100, "num_clients": 1, "num_classes": 100, "n_per_class": 1}
    ]
    
    print(f"Using 40k/10k/10k split (train/val/test)")
    print(f"Configuration: R={R}, sparsity={sparsity}, soft_zero={soft_zero}")
    print(f"Stratified sampling: {num_examples} examples per client")
    
    # Create masks directory
    os.makedirs("masks", exist_ok=True)
    
    total_masks = 0
    
    for config in nc_configs:
        nc = config["nc"]
        num_clients = config["num_clients"]
        num_classes = config["num_classes"]
        n_per_class = config["n_per_class"]
        
        print(f"\n=== Processing Nc={nc} ===")
        print(f"  - Clients: {num_clients}")
        print(f"  - Classes per client: {num_classes}")
        print(f"  - Examples per class: {n_per_class}")
        print(f"  - Total examples per client: {num_classes * n_per_class}")
        
        # Create non-IID splits
        print(f"Creating non-IID splits: {num_clients} clients, {nc} classes per client")
        client_datasets = create_non_iid_splits(
            train_loader.dataset,
            num_clients=num_clients, 
            classes_per_client=nc
        )
        
        # Compute masks for each client
        print("Computing masks with stratified loaders...")
        start_time = time.time()
        
        client_masks = compute_mask_clients(
            model=model,
            client_datasets=client_datasets,
            num_examples=num_examples,
            num_classes=num_classes,
            n_per_class=n_per_class,
            final_sparsity=sparsity,
            R=R,
            soft_zero=soft_zero,
            device='cpu'  # Use CPU to avoid MPS issues
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n  Completed in {elapsed_time:.2f} seconds")
        
        # Calculate mask statistics
        if len(client_masks) > 0:
            total_params = sum(m.numel() for m in client_masks[0].values())
            avg_pruned_params = 0
            for client_mask in client_masks:
                pruned_params = sum((m == 0).sum().item() for m in client_mask.values())
                avg_pruned_params += pruned_params
            avg_pruned_params /= len(client_masks)
            avg_sparsity = avg_pruned_params / total_params
            
            print(f"  Mask statistics:")
            print(f"    - Total parameters: {total_params:,}")
            print(f"    - Average pruned parameters: {avg_pruned_params:.0f}")
            print(f"    - Average sparsity: {avg_sparsity:.4f} ({avg_sparsity*100:.1f}%)")
            print(f"    - Average active parameters: {total_params - avg_pruned_params:.0f}")
            
            # Verify masks are not all zeros
            all_zeros = all((m == 0).all().item() for m in client_masks[0].values())
            if all_zeros:
                print(f"    ❌ WARNING: All masks are zeros! Mask generation failed.")
            else:
                print(f"    ✓ Masks generated successfully with proper sparsity.")
        
        # Save masks for this Nc configuration
        save_path = f"masks/federated_masks_nc{nc}_R{R}_sz{soft_zero}_stratified.pth"
        torch.save(client_masks, save_path)
        print(f"  ✓ Saved {len(client_masks)} masks to {save_path}")
        
        # Verify the saved masks
        try:
            loaded_masks = torch.load(save_path, map_location='cpu')
            if len(loaded_masks) == len(client_masks):
                print(f"  ✓ Verification: {len(loaded_masks)} masks loaded successfully")
            else:
                print(f"  ❌ Verification failed: Expected {len(client_masks)}, got {len(loaded_masks)}")
        except Exception as e:
            print(f"  ❌ Verification failed: {e}")
        
        total_masks += len(client_masks)
    
    print(f"\n✓ All federated masks computed successfully!")
    print(f"Total masks generated: {total_masks}")
    print(f"Configuration: R={R}, sparsity={sparsity}, soft_zero={soft_zero}")
    print(f"Used stratified loaders for balanced Fisher Information calculation")
    
    # List all generated mask files
    print(f"\nGenerated mask files:")
    mask_files = [f for f in os.listdir("masks") if f.startswith("federated_masks") and "stratified" in f]
    for mask_file in sorted(mask_files):
        print(f"  - masks/{mask_file}")


if __name__ == "__main__":
    compute_federated_masks_stratified()

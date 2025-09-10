#!/usr/bin/env python3
"""
Compute federated masks for the best configuration only.
Uses R=3, sparsity=0.78, soft_zero=0.01 for Nc={1,5,10,50,100}
"""

import torch
import os
import time
from src.models.vision_transformer import LinearFlexibleDino
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits
from src.training.model_editing import compute_mask

def compute_federated_masks_best_config():
    """Compute federated masks for the best configuration only."""
    print("Computing federated masks for best configuration...")
    print("Configuration: R=3, sparsity=0.78, soft_zero=0.01")
    print("Nc values: [1, 5, 10, 50, 100]")
    
    # Load model
    print("Loading LinearFlexibleDino...")
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
    model.eval()
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders(val_split=0.2)  # 40k/10k/10k split
    
    # Best configuration parameters
    R = 3
    sparsity = 0.78
    soft_zero = 0.01
    Nc_values = [1, 5, 10, 50, 100]
    num_examples = 25
    
    print(f"Using 40k/10k/10k split (train/val/test)")
    print(f"Configuration: R={R}, sparsity={sparsity}, soft_zero={soft_zero}")
    
    for Nc in Nc_values:
        print(f"\n=== Processing Nc={Nc} ===")
        
        # Create non-IID splits - FIXED: Pass the dataset, not data_manager
        print(f"Creating non-IID splits: 100 clients, {Nc} classes per client")
        client_datasets = create_non_iid_splits(
            train_loader.dataset,  # Pass the actual dataset, not data_manager
            num_clients=100, 
            classes_per_client=Nc
        )
        
        # Compute masks for each client
        client_masks = []
        start_time = time.time()
        
        for client_id, client_dataset in enumerate(client_datasets):
            print(f"  Client {client_id+1}/100", end="\r")
            
            # Create DataLoader for the client dataset
            from torch.utils.data import DataLoader
            client_loader = DataLoader(
                client_dataset, 
                batch_size=32, 
                shuffle=False, 
                num_workers=0
            )
            
            # Compute mask for this client
            mask = compute_mask(
                model=model, 
                dataloader=client_loader, 
                sparsity_target=sparsity, 
                R=R, 
                soft_zero=soft_zero, 
                num_examples=num_examples,
                device='mps',  # Use Apple GPU with Metal Performance Shaders
                enable_plot=False,
                debug=False
            )
            
            client_masks.append(mask)
        
        elapsed_time = time.time() - start_time
        print(f"\n  Completed in {elapsed_time:.2f} seconds")
        
        # Save masks for this Nc configuration
        save_path = f"masks/federated_masks_nc{Nc}_R{R}_sz{soft_zero}.pth"
        os.makedirs("masks", exist_ok=True)
        torch.save(client_masks, save_path)
        print(f"  ✓ Saved {len(client_masks)} masks to {save_path}")
    
    print("\n✓ All federated masks computed successfully!")

if __name__ == "__main__":
    compute_federated_masks_best_config()

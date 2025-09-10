#!/usr/bin/env python3
"""
Debug script to check mask structure
"""

import torch

# Check the federated mask structure
mask_file = 'masks/federated_masks_nc100_R3_sz0.01.pth'
print(f"Loading mask file: {mask_file}")

try:
    masks = torch.load(mask_file, map_location='cpu')
    print(f"Type: {type(masks)}")
    print(f"Length: {len(masks)}")
    
    if len(masks) > 0:
        print(f"First mask type: {type(masks[0])}")
        print(f"First mask keys: {list(masks[0].keys())}")
        
        # Check a specific parameter
        first_key = list(masks[0].keys())[0]
        print(f"\nFirst parameter: {first_key}")
        print(f"Shape: {masks[0][first_key].shape}")
        print(f"Unique values: {torch.unique(masks[0][first_key])}")
        print(f"Sum of mask: {masks[0][first_key].sum().item()}")
        print(f"Total elements: {masks[0][first_key].numel()}")
        print(f"Active elements (1s): {(masks[0][first_key] == 1).sum().item()}")
        print(f"Pruned elements (0s): {(masks[0][first_key] == 0).sum().item()}")
        
        # Calculate sparsity for first client
        total_params = sum(m.numel() for m in masks[0].values())
        pruned_params = sum((m == 0).sum().item() for m in masks[0].values())
        active_params = sum((m == 1).sum().item() for m in masks[0].values())
        sparsity = pruned_params / total_params if total_params > 0 else 0
        
        print(f"\nFirst client statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Pruned parameters: {pruned_params:,}")
        print(f"  Active parameters: {active_params:,}")
        print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.1f}%)")
        
        # Check if all masks are the same (all zeros)
        all_zeros = all((m == 0).all().item() for m in masks[0].values())
        print(f"  All masks are zeros: {all_zeros}")
        
except Exception as e:
    print(f"Error loading masks: {e}")

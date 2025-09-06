#!/usr/bin/env python3
"""
Stage 1: Compute masks for different R and soft_zero values.
This script generates all possible mask combinations for systematic testing.
"""

import torch
import os
import time
from typing import Dict, List

# Import project modules
from src.models.vision_transformer import LinearFlexibleDino  # Changed from DINOBackboneClassifier
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits
from src.training.model_editing import compute_mask, create_client_masks
from src.utils.visualization import (
    plot_sparsity_comparison_r_values,
    plot_layer_wise_sparsity_comparison,
    plot_sparsity_heatmap,
    plot_computation_time_comparison
)


def compute_centralized_mask():
    """Compute centralized mask with different R and soft_zero values."""
    print("=== Computing Centralized Masks ===")
    
    # Load model - Using LinearFlexibleDino
    print("Loading LinearFlexibleDino...")
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)  # Freeze all backbone blocks
    model.eval()
    
    # Save the model used in Stage 1
    print("Saving Stage 1 model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/stage1_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 100,
            'num_layers_to_freeze': 12,
            'model_type': 'LinearFlexibleDino'
        }
    }, model_path)
    print(f"✓ Stage 1 model saved to {model_path}")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders()
    
    # Create stratified loader (1 example per class for IID setting)
    print("Creating stratified loader (1 example per class)...")
    stratified_loader = data_manager.get_stratified_loader(
        train_loader, num_classes=100, samples_per_class=1
    )
    
    # Test different R and soft_zero values
    R_values = [3, 5, 10]
    soft_zero_values = [0.1, 0.01, 0.001]
    
    # Create masks directory
    os.makedirs("masks", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Data collection for plotting
    sparsity_data = {}
    time_data = {}
    mask_data = {}
    
    total_combinations = len(R_values) * len(soft_zero_values)
    current_combination = 0
    
    for R in R_values:
        for sz in soft_zero_values:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Computing mask: R={R}, soft_zero={sz}")
            
            start_time = time.time()
            
            mask = compute_mask(
                model=model,
                dataloader=stratified_loader,
                sparsity_target=0.9,
                R=R,
                soft_zero=sz,
                device='cuda',
                enable_plot=False,
                debug=False
            )
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Save mask
            filename = f"masks/centralized_mask_R{R}_sz{sz}.pth"
            torch.save(mask, filename)
            
            # Calculate actual sparsity
            total_params = sum(m.numel() for m in mask.values())
            pruned_params = sum((m == 0).sum().item() for m in mask.values())
            actual_sparsity = pruned_params / total_params
            
            # Collect data for plotting
            R_str = str(R)
            sz_str = str(sz)
            
            if R_str not in sparsity_data:
                sparsity_data[R_str] = {}
            if R_str not in time_data:
                time_data[R_str] = {}
            
            sparsity_data[R_str][sz_str] = actual_sparsity
            time_data[R_str][sz_str] = computation_time
            mask_data[f"R{R}_sz{sz}"] = mask
            
            print(f"  ✓ Saved: {filename}")
            print(f"  ✓ Computation time: {computation_time:.2f}s")
            print(f"  ✓ Actual sparsity: {actual_sparsity:.4f} ({pruned_params}/{total_params} params pruned)")

    # Generate comparison plots
    print("\n=== Generating Comparison Plots ===")
    
    # Plot 1: Sparsity comparison across R and soft zero values
    print("Creating sparsity comparison plots...")
    plot_sparsity_comparison_r_values(
        sparsity_data, 
        save_path="plots/centralized_sparsity_comparison.png"
    )
    
    # Plot 2: Sparsity heatmap
    print("Creating sparsity heatmap...")
    plot_sparsity_heatmap(
        sparsity_data,
        save_path="plots/centralized_sparsity_heatmap.png"
    )
    
    # Plot 3: Computation time comparison
    print("Creating computation time comparison plots...")
    plot_computation_time_comparison(
        time_data,
        save_path="plots/centralized_time_comparison.png"
    )
    
    # Plot 4: Layer-wise sparsity comparison (all layers)
    print("Creating layer-wise sparsity comparison (all layers)...")
    plot_layer_wise_sparsity_comparison(
        mask_data,
        layer_filter=None,  # All layers
        save_path="plots/centralized_layer_wise_all.png"
    )
    
    # Plot 5: Layer-wise sparsity comparison (blocks 8-11 only)
    print("Creating layer-wise sparsity comparison (blocks 8-11)...")
    plot_layer_wise_sparsity_comparison(
        mask_data,
        layer_filter=[8, 9, 10, 11],  # Only blocks 8-11
        save_path="plots/centralized_layer_wise_8_11.png"
    )
    
    print("✓ All comparison plots generated successfully!")


def compute_federated_masks():
    """Compute federated masks for different Nc values."""
    print("\n=== Computing Federated Masks ===")
    
    # Load model - Using LinearFlexibleDino
    print("Loading LinearFlexibleDino...")
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)  # Freeze all backbone blocks
    model.eval()
    
    # Save the model used in Stage 1 (if not already saved)
    if not os.path.exists("models/stage1_model.pth"):
        print("Saving Stage 1 model...")
        os.makedirs("models", exist_ok=True)
        model_path = "models/stage1_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_classes': 100,
                'num_layers_to_freeze': 12,
                'model_type': 'LinearFlexibleDino'
            }
        }, model_path)
        print(f"✓ Stage 1 model saved to {model_path}")
    else:
        print("✓ Stage 1 model already saved")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders(val_split=0.2)  # 40k train, 10k val, 10k test
    
    # Test different Nc values
    nc_values = [1, 5, 10, 50, 100]
    R_values = [3, 5, 10]
    soft_zero_values = [0.1, 0.01, 0.001]
    
    # Data collection for federated plotting
    federated_sparsity_data = {}
    federated_time_data = {}
    federated_mask_data = {}
    
    total_combinations = len(nc_values) * len(R_values) * len(soft_zero_values)
    current_combination = 0
    
    for nc in nc_values:
        print(f"\n--- Computing masks for Nc={nc} ---")
        
        # Create non-IID client datasets
        print(f"Creating non-IID client datasets (Nc={nc})...")
        client_datasets = create_non_iid_splits(
            train_loader.dataset, 
            num_clients=100, 
            classes_per_client=nc
        )
        
        for R in R_values:
            for sz in soft_zero_values:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Nc={nc}, R={R}, soft_zero={sz}")
                
                start_time = time.time()
                
                client_masks = create_client_masks(
                    model=model,
                    client_datasets=client_datasets,
                    classes_per_client=nc,
                    target_sparsity=0.9,
                    num_iterations=R,
                    soft_zero_value=sz,
                    max_samples=25,
                    debug_mode=False  # Changed from debug=False
                )
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                # Save masks
                filename = f"masks/federated_masks_nc{nc}_R{R}_sz{sz}.pth"
                torch.save(client_masks, filename)
                
                # Calculate average sparsity across clients
                total_params = sum(m.numel() for m in client_masks[0].values())
                avg_pruned_params = 0
                for client_mask in client_masks:
                    pruned_params = sum((m == 0).sum().item() for m in client_mask.values())
                    avg_pruned_params += pruned_params
                avg_pruned_params /= len(client_masks)
                avg_sparsity = avg_pruned_params / total_params
                
                # Collect data for plotting (using first client mask as representative)
                config_key = f"nc{nc}_R{R}_sz{sz}"
                R_str = str(R)
                sz_str = str(sz)
                
                if R_str not in federated_sparsity_data:
                    federated_sparsity_data[R_str] = {}
                if R_str not in federated_time_data:
                    federated_time_data[R_str] = {}
                
                federated_sparsity_data[R_str][sz_str] = avg_sparsity
                federated_time_data[R_str][sz_str] = computation_time
                federated_mask_data[config_key] = client_masks[0]  # Use first client as representative
                
                print(f"  ✓ Saved: {filename}")
                print(f"  ✓ Computation time: {computation_time:.2f}s")
                print(f"  ✓ Average sparsity: {avg_sparsity:.4f} ({avg_pruned_params:.0f}/{total_params} params pruned)")

    # Generate federated comparison plots
    print("\n=== Generating Federated Comparison Plots ===")
    
    # Plot 1: Federated sparsity comparison across R and soft zero values
    print("Creating federated sparsity comparison plots...")
    plot_sparsity_comparison_r_values(
        federated_sparsity_data, 
        save_path="plots/federated_sparsity_comparison.png"
    )
    
    # Plot 2: Federated sparsity heatmap
    print("Creating federated sparsity heatmap...")
    plot_sparsity_heatmap(
        federated_sparsity_data,
        save_path="plots/federated_sparsity_heatmap.png"
    )
    
    # Plot 3: Federated computation time comparison
    print("Creating federated computation time comparison plots...")
    plot_computation_time_comparison(
        federated_time_data,
        save_path="plots/federated_time_comparison.png"
    )
    
    # Plot 4: Federated layer-wise sparsity comparison (all layers)
    print("Creating federated layer-wise sparsity comparison (all layers)...")
    plot_layer_wise_sparsity_comparison(
        federated_mask_data,
        layer_filter=None,  # All layers
        save_path="plots/federated_layer_wise_all.png"
    )
    
    # Plot 5: Federated layer-wise sparsity comparison (blocks 8-11 only)
    print("Creating federated layer-wise sparsity comparison (blocks 8-11)...")
    plot_layer_wise_sparsity_comparison(
        federated_mask_data,
        layer_filter=[8, 9, 10, 11],  # Only blocks 8-11
        save_path="plots/federated_layer_wise_8_11.png"
    )
    
    print("✓ All federated comparison plots generated successfully!")


def print_summary():
    """Print summary of generated masks."""
    print("\n=== Mask Generation Summary ===")
    
    if not os.path.exists("masks"):
        print("No masks directory found.")
        return
    
    mask_files = [f for f in os.listdir("masks") if f.endswith('.pth')]
    
    centralized_masks = [f for f in mask_files if f.startswith('centralized_')]
    federated_masks = [f for f in mask_files if f.startswith('federated_')]
    
    print(f"Generated {len(centralized_masks)} centralized masks:")
    for mask_file in sorted(centralized_masks):
        print(f"  - {mask_file}")
    
    print(f"\nGenerated {len(federated_masks)} federated mask sets:")
    for mask_file in sorted(federated_masks):
        print(f"  - {mask_file}")
    
    print(f"\nTotal: {len(mask_files)} mask files generated")
    
    # Check if model was saved
    if os.path.exists("models/stage1_model.pth"):
        print(f"\n✓ Stage 1 model saved: models/stage1_model.pth")
    
    # Check if plots were generated
    if os.path.exists("plots"):
        plot_files = [f for f in os.listdir("plots") if f.endswith('.png')]
        if plot_files:
            print(f"\n✓ Generated {len(plot_files)} comparison plots:")
            for plot_file in sorted(plot_files):
                print(f"  - plots/{plot_file}")
    
    print("\nYou can now use these masks with stage2_train_with_masks.py")


def main():
    """Main function to compute all masks."""
    print("TALOS 2-Stage Mask Calibration - Stage 1: Mask Computation")
    print("=" * 60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU (will be slower).")
        device = 'cpu'
    else:
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
        device = 'cuda'
    
    try:
        # Compute centralized masks
        compute_centralized_mask()
        
        # Compute federated masks
        compute_federated_masks()
        
        # Print summary
        print_summary()
        
        print("\n" + "=" * 60)
        print("✅ Stage 1 Complete: All masks generated successfully!")
        print("Next step: Run stage2_train_with_masks.py to test different configurations")
        
    except Exception as e:
        print(f"\n❌ Error during mask computation: {e}")
        raise


if __name__ == "__main__":
    main()

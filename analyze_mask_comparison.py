#!/usr/bin/env python3
"""
Analyze and compare mask patterns for R=3 vs R=5 with soft_zero=0.01
Focus on backbone layers 8-12 and only show layers with active weights.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_mask(filepath):
    """Load a mask file."""
    return torch.load(filepath, map_location='cpu')

def analyze_mask_patterns(mask, layer_filter=None, min_active_ratio=0.0):
    """
    Analyze mask patterns for specific layers.
    
    Args:
        mask: Dictionary of masks
        layer_filter: List of layer numbers to include (e.g., [8, 9, 10, 11, 12])
        min_active_ratio: Minimum ratio of active weights to include layer
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    for name, mask_tensor in mask.items():
        # Check if this is a backbone block layer
        if 'backbone.blocks' in name:
            # Extract layer number
            try:
                layer_num = int(name.split('.')[2])  # backbone.blocks.X.attn -> X
            except:
                continue
                
            # Apply layer filter
            if layer_filter is not None and layer_num not in layer_filter:
                continue
                
            # Calculate statistics
            total_params = mask_tensor.numel()
            active_params = (mask_tensor == 1).sum().item()
            active_ratio = active_params / total_params
            
            # Only include if meets minimum active ratio
            if active_ratio >= min_active_ratio:
                results[name] = {
                    'layer_num': layer_num,
                    'total_params': total_params,
                    'active_params': active_params,
                    'active_ratio': active_ratio,
                    'sparsity': 1 - active_ratio,
                    'shape': mask_tensor.shape
                }
    
    return results

def create_comparison_plot(mask1_results, mask2_results, title1, title2):
    """Create comparison plot for two masks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    layers1 = sorted([r['layer_num'] for r in mask1_results.values()])
    active_ratios1 = [mask1_results[f'backbone.blocks.{layer}.attn.qkv.weight']['active_ratio'] 
                     for layer in layers1 if f'backbone.blocks.{layer}.attn.qkv.weight' in mask1_results]
    
    layers2 = sorted([r['layer_num'] for r in mask2_results.values()])
    active_ratios2 = [mask2_results[f'backbone.blocks.{layer}.attn.qkv.weight']['active_ratio'] 
                     for layer in layers2 if f'backbone.blocks.{layer}.attn.qkv.weight' in mask2_results]
    
    # Plot 1: R=3
    ax1.bar(layers1, active_ratios1, alpha=0.7, color='blue')
    ax1.set_title(f'{title1}\nLayers with Active Weights (8-12)')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Active Weight Ratio')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R=5
    ax2.bar(layers2, active_ratios2, alpha=0.7, color='red')
    ax2.set_title(f'{title2}\nLayers with Active Weights (8-12)')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Active Weight Ratio')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_detailed_comparison(mask1_results, mask2_results, title1, title2):
    """Print detailed comparison of mask results."""
    print(f"\n{'='*80}")
    print(f"DETAILED MASK COMPARISON: {title1} vs {title2}")
    print(f"{'='*80}")
    
    # Get all unique layers
    all_layers = set()
    for results in [mask1_results, mask2_results]:
        for name, data in results.items():
            all_layers.add(data['layer_num'])
    
    all_layers = sorted(all_layers)
    
    print(f"\n{'Layer':<8} {'Component':<20} {'R=3 Active':<12} {'R=5 Active':<12} {'Difference':<12}")
    print(f"{'-'*80}")
    
    for layer in all_layers:
        layer_printed = False
        
        # Check different components
        components = ['attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']
        
        for comp in components:
            name1 = f'backbone.blocks.{layer}.{comp}'
            name2 = f'backbone.blocks.{layer}.{comp}'
            
            if name1 in mask1_results and name2 in mask2_results:
                active1 = mask1_results[name1]['active_ratio']
                active2 = mask2_results[name2]['active_ratio']
                diff = active1 - active2
                
                if not layer_printed:
                    print(f"{layer:<8} {comp:<20} {active1:<12.4f} {active2:<12.4f} {diff:<12.4f}")
                    layer_printed = True
                else:
                    print(f"{'':<8} {comp:<20} {active1:<12.4f} {active2:<12.4f} {diff:<12.4f}")

def main():
    """Main analysis function."""
    print("Loading masks...")
    
    # Load the two masks
    mask_r3 = load_mask('masks/centralized_mask_R3_sz0.01.pth')
    mask_r5 = load_mask('masks/centralized_mask_R5_sz0.01.pth')
    
    print("Analyzing mask patterns...")
    
    # Analyze both masks for layers 8-12
    r3_results = analyze_mask_patterns(mask_r3, layer_filter=[8, 9, 10, 11, 12], min_active_ratio=0.001)
    r5_results = analyze_mask_patterns(mask_r5, layer_filter=[8, 9, 10, 11, 12], min_active_ratio=0.001)
    
    print(f"\nR=3 Results: {len(r3_results)} layers with active weights")
    print(f"R=5 Results: {len(r5_results)} layers with active weights")
    
    # Print detailed comparison
    print_detailed_comparison(r3_results, r5_results, "R=3 (sparsity=0.78)", "R=5 (sparsity=1.0)")
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    fig = create_comparison_plot(r3_results, r5_results, "R=3 (sparsity=0.78)", "R=5 (sparsity=1.0)")
    
    # Save plot
    plt.savefig('plots/mask_comparison_r3_vs_r5_layers_8_12.png', dpi=300, bbox_inches='tight')
    print("Plot saved as: plots/mask_comparison_r3_vs_r5_layers_8_12.png")
    
    # Show plot
    plt.show()
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY STATISTICS")
    print(f"{'='*50}")
    
    r3_layers = [r['layer_num'] for r in r3_results.values()]
    r5_layers = [r['layer_num'] for r in r5_results.values()]
    
    print(f"R=3 active layers: {sorted(set(r3_layers))}")
    print(f"R=5 active layers: {sorted(set(r5_layers))}")
    
    if r3_layers:
        r3_avg_active = np.mean([r['active_ratio'] for r in r3_results.values()])
        print(f"R=3 average active ratio: {r3_avg_active:.4f}")
    
    if r5_layers:
        r5_avg_active = np.mean([r['active_ratio'] for r in r5_results.values()])
        print(f"R=5 average active ratio: {r5_avg_active:.4f}")

if __name__ == "__main__":
    main()

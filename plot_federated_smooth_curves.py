#!/usr/bin/env python3
"""
Generate smooth, professional curves for federated learning comparison.
Creates curves similar to the reference image style.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

def load_results(filepath):
    """Load results from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def smooth_curve(x, y, num_points=300):
    """Create smooth interpolated curve."""
    if len(x) < 4:  # Need at least 4 points for cubic spline
        return x, y
    
    # Create smooth interpolation
    x_smooth = np.linspace(x[0], x[-1], num_points)
    spl = make_interp_spline(x, y, k=3)  # Cubic spline
    y_smooth = spl(x_smooth)
    
    return x_smooth, y_smooth

def create_smooth_federated_plots():
    """Create smooth, professional federated learning plots."""
    
    # Define result files
    results_files = {
        'IID': {
            'j=4': 'results/federated_iid_results_20250906_052506.yaml',
            'j=8': 'results/federated_iid_results_20250905_233436.yaml', 
            'j=16': 'results/federated_iid_results_20250906_014547.yaml'
        },
        'Non-IID (nc=1)': {
            'j=4': 'results/federated_non_iid_1_results_20250906_041107.yaml',
            'j=8': 'results/federated_non_iid_1_results_20250905_232608.yaml',
            'j=16': 'results/federated_non_iid_1_results_20250905_221354.yaml'
        }
    }
    
    # Professional color scheme and styling
    colors = {'j=4': '#1f77b4', 'j=8': '#ff7f0e', 'j=16': '#2ca02c'}
    labels = {'j=4': 'j = 4', 'j=8': 'j = 8', 'j=16': 'j = 16'}
    
    # Set professional matplotlib style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black'
    })
    
    # Plot 1: IID Server Validation Accuracy
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for j_val, filepath in results_files['IID'].items():
        data = load_results(filepath)
        val_acc = data['training_history']['validation_accuracies']
        rounds = np.array(range(1, len(val_acc) + 1))
        val_acc = np.array(val_acc)
        
        # Create smooth curve
        rounds_smooth, val_smooth = smooth_curve(rounds, val_acc)
        
        ax.plot(rounds_smooth, val_smooth, color=colors[j_val], 
               linewidth=3, label=labels[j_val], alpha=0.9)
        
        # Add subtle markers at actual data points (every 20th point)
        marker_indices = range(0, len(rounds), max(1, len(rounds)//10))
        ax.scatter(rounds[marker_indices], val_acc[marker_indices], 
                  color=colors[j_val], s=30, alpha=0.7, zorder=5)
    
    ax.set_xlabel('Communication Rounds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Server Val Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('(a) IID case', fontsize=16, fontweight='bold', loc='left')
    
    # Professional legend
    legend = ax.legend(fontsize=12, loc='lower right', 
                      bbox_to_anchor=(0.98, 0.02))
    legend.get_frame().set_linewidth(1.2)
    
    # Set axis limits and ticks
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('plots/iid_server_validation_smooth.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 1 saved: plots/iid_server_validation_smooth.png")
    plt.show()
    
    # Plot 2: Non-IID nc=1 Server Validation Accuracy
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for j_val, filepath in results_files['Non-IID (nc=1)'].items():
        data = load_results(filepath)
        val_acc = data['training_history']['validation_accuracies']
        rounds = np.array(range(1, len(val_acc) + 1))
        val_acc = np.array(val_acc)
        
        # Create smooth curve
        rounds_smooth, val_smooth = smooth_curve(rounds, val_acc)
        
        ax.plot(rounds_smooth, val_smooth, color=colors[j_val], 
               linewidth=3, label=labels[j_val], alpha=0.9)
        
        # Add subtle markers at actual data points
        marker_indices = range(0, len(rounds), max(1, len(rounds)//10))
        ax.scatter(rounds[marker_indices], val_acc[marker_indices], 
                  color=colors[j_val], s=30, alpha=0.7, zorder=5)
    
    ax.set_xlabel('Communication Rounds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Server Val Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('(b) Non-IID case (nc=1)', fontsize=16, fontweight='bold', loc='left')
    
    # Professional legend
    legend = ax.legend(fontsize=12, loc='lower right', 
                      bbox_to_anchor=(0.98, 0.02))
    legend.get_frame().set_linewidth(1.2)
    
    # Set axis limits
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('plots/non_iid_nc1_server_validation_smooth.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 2 saved: plots/non_iid_nc1_server_validation_smooth.png")
    plt.show()
    
    # Plot 3: IID Client Training Accuracy
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for j_val, filepath in results_files['IID'].items():
        data = load_results(filepath)
        client_acc = data['training_history']['round_accuracies']
        rounds = np.array(range(1, len(client_acc) + 1))
        client_acc = np.array(client_acc)
        
        # Create smooth curve
        rounds_smooth, acc_smooth = smooth_curve(rounds, client_acc)
        
        ax.plot(rounds_smooth, acc_smooth, color=colors[j_val], 
               linewidth=3, label=labels[j_val], alpha=0.9)
        
        # Add subtle markers
        marker_indices = range(0, len(rounds), max(1, len(rounds)//10))
        ax.scatter(rounds[marker_indices], client_acc[marker_indices], 
                  color=colors[j_val], s=30, alpha=0.7, zorder=5)
    
    ax.set_xlabel('Communication Rounds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Client Training Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('(c) IID Client Training', fontsize=16, fontweight='bold', loc='left')
    
    # Professional legend
    legend = ax.legend(fontsize=12, loc='lower right', 
                      bbox_to_anchor=(0.98, 0.02))
    legend.get_frame().set_linewidth(1.2)
    
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('plots/iid_client_training_smooth.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 3 saved: plots/iid_client_training_smooth.png")
    plt.show()
    
    # Plot 4: Non-IID nc=1 Client Training Accuracy
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for j_val, filepath in results_files['Non-IID (nc=1)'].items():
        data = load_results(filepath)
        client_acc = data['training_history']['round_accuracies']
        rounds = np.array(range(1, len(client_acc) + 1))
        client_acc = np.array(client_acc)
        
        # Create smooth curve
        rounds_smooth, acc_smooth = smooth_curve(rounds, client_acc)
        
        ax.plot(rounds_smooth, acc_smooth, color=colors[j_val], 
               linewidth=3, label=labels[j_val], alpha=0.9)
        
        # Add subtle markers
        marker_indices = range(0, len(rounds), max(1, len(rounds)//10))
        ax.scatter(rounds[marker_indices], client_acc[marker_indices], 
                  color=colors[j_val], s=30, alpha=0.7, zorder=5)
    
    ax.set_xlabel('Communication Rounds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Client Training Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('(d) Non-IID Client Training (nc=1)', fontsize=16, fontweight='bold', loc='left')
    
    # Professional legend
    legend = ax.legend(fontsize=12, loc='lower right', 
                      bbox_to_anchor=(0.98, 0.02))
    legend.get_frame().set_linewidth(1.2)
    
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('plots/non_iid_nc1_client_training_smooth.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 4 saved: plots/non_iid_nc1_client_training_smooth.png")
    plt.show()
    
    print("\n🎉 All 4 smooth professional plots created!")
    print("📂 Files saved in plots/ directory:")
    print("   1. iid_server_validation_smooth.png")
    print("   2. non_iid_nc1_server_validation_smooth.png") 
    print("   3. iid_client_training_smooth.png")
    print("   4. non_iid_nc1_client_training_smooth.png")

if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    create_smooth_federated_plots()

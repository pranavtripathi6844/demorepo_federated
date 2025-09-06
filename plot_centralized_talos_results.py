#!/usr/bin/env python3
"""
Plot training and validation accuracy curves for centralized TALOS runs.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

def load_training_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_centralized_talos_results():
    """Plot comparison of centralized TALOS results across different epochs."""
    
    # Find all centralized TALOS training history files
    pattern = "checkpoints/centralized_masked_centralized_mask_R3_sz0.01_epoch_*.json"
    files = sorted(glob(pattern))
    
    if not files:
        print("No centralized TALOS training history files found!")
        print("Looking for files matching:", pattern)
        return
    
    print(f"Found {len(files)} training history files:")
    for f in files:
        print(f"  - {f}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different epoch runs
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot data for each run
    for i, filepath in enumerate(files):
        # Extract epoch count from filename
        filename = os.path.basename(filepath)
        epoch_count = filename.split('_')[-1].replace('.json', '')
        
        # Load data
        data = load_training_history(filepath)
        
        epochs = data['epochs']
        train_acc = data['train_accuracy']
        val_acc = data['val_accuracy']
        
        color = colors[i % len(colors)]
        
        # Plot training accuracy
        ax1.plot(epochs, train_acc, 
                label=f'{epoch_count} epochs (train)', 
                color=color, linestyle='-', linewidth=2, alpha=0.8)
        
        # Plot validation accuracy
        ax2.plot(epochs, val_acc, 
                label=f'{epoch_count} epochs (val)', 
                color=color, linestyle='-', linewidth=2, alpha=0.8)
    
    # Customize training accuracy plot - Focus on 85-100%
    ax1.set_title('Centralized TALOS - Training Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(85, 100)  # Focus on 85-100% range
    ax1.set_yticks(np.arange(85, 101, 2))  # Tick marks every 2%
    
    # Customize validation accuracy plot - Focus on 50-70%
    ax2.set_title('Centralized TALOS - Validation Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(50, 70)  # Focus on 50-70% range
    ax2.set_yticks(np.arange(50, 71, 2))  # Tick marks every 2%
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('plots/centralized_talos_accuracy_curves_zoomed.png', dpi=300, bbox_inches='tight')
    print("Zoomed plot saved as: plots/centralized_talos_accuracy_curves_zoomed.png")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CENTRALIZED TALOS RESULTS SUMMARY")
    print("="*80)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        epoch_count = filename.split('_')[-1].replace('.json', '')
        
        data = load_training_history(filepath)
        final_train_acc = data['train_accuracy'][-1]
        final_val_acc = data['val_accuracy'][-1]
        best_val_acc = max(data['val_accuracy'])
        
        print(f"{epoch_count:>2} epochs: Train={final_train_acc:6.2f}%, Val={final_val_acc:6.2f}%, Best_Val={best_val_acc:6.2f}%")
    
    # Create a combined plot showing both training and validation on same axes
    plt.figure(figsize=(12, 8))
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        epoch_count = filename.split('_')[-1].replace('.json', '')
        
        data = load_training_history(filepath)
        epochs = data['epochs']
        train_acc = data['train_accuracy']
        val_acc = data['val_accuracy']
        
        color = colors[i % len(colors)]
        
        # Plot both training and validation
        plt.plot(epochs, train_acc, 
                label=f'{epoch_count} epochs (train)', 
                color=color, linestyle='-', linewidth=2, alpha=0.8)
        plt.plot(epochs, val_acc, 
                label=f'{epoch_count} epochs (val)', 
                color=color, linestyle='--', linewidth=2, alpha=0.8)
    
    plt.title('Centralized TALOS - Training vs Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(45, 100)  # Show both ranges but focus on the important parts
    
    plt.tight_layout()
    plt.savefig('plots/centralized_talos_combined_accuracy_zoomed.png', dpi=300, bbox_inches='tight')
    print("Combined zoomed plot saved as: plots/centralized_talos_combined_accuracy_zoomed.png")
    plt.show()

if __name__ == "__main__":
    plot_centralized_talos_results()

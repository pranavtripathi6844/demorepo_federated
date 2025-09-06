"""
Visualization utilities for training results and model analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional
import seaborn as sns


def plot_training_history(training_history: Dict[str, List[float]], 
                         save_path: Optional[str] = None):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        training_history: Dictionary containing training metrics
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    if 'train_losses' in training_history and 'val_losses' in training_history:
        epochs = range(1, len(training_history['train_losses']) + 1)
        ax1.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
    
    # Plot accuracies
    if 'train_accuracies' in training_history and 'val_accuracies' in training_history:
        epochs = range(1, len(training_history['train_accuracies']) + 1)
        ax2.plot(epochs, training_history['train_accuracies'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, training_history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_federated_training_history(training_history: Dict[str, List[float]],
                                  save_path: Optional[str] = None):
    """
    Plot federated learning training history.
    
    Args:
        training_history: Dictionary containing federated training metrics
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot client vs server metrics
    if 'round_accuracies' in training_history and 'validation_accuracies' in training_history:
        rounds = range(1, len(training_history['round_accuracies']) + 1)
        ax1.plot(rounds, training_history['round_accuracies'], 'b-', label='Client Average Accuracy')
        ax1.plot(rounds, training_history['validation_accuracies'], 'r-', label='Server Validation Accuracy')
        ax1.set_title('Federated Learning: Client vs Server Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
    
    # Plot losses
    if 'round_losses' in training_history and 'validation_losses' in training_history:
        rounds = range(1, len(training_history['round_losses']) + 1)
        ax2.plot(rounds, training_history['round_losses'], 'b-', label='Client Average Loss')
        ax2.plot(rounds, training_history['validation_losses'], 'r-', label='Server Validation Loss')
        ax2.set_title('Federated Learning: Client vs Server Loss')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_mask_sparsity_analysis(mask: Dict[str, torch.Tensor],
                               save_path: Optional[str] = None):
    """
    Plot analysis of mask sparsity across different layers.
    
    Args:
        mask: Dictionary containing parameter masks
        save_path: Optional path to save the plot
    """
    layer_names = []
    sparsity_values = []
    parameter_counts = []
    
    for name, mask_tensor in mask.items():
        layer_names.append(name.split('.')[-1])  # Extract layer name
        sparsity = (mask_tensor == 0).float().mean().item()
        sparsity_values.append(sparsity)
        parameter_counts.append(mask_tensor.numel())
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot sparsity per layer
    bars1 = ax1.bar(range(len(layer_names)), sparsity_values)
    ax1.set_title('Sparsity per Layer')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Sparsity Ratio')
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, sparsity_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot parameter count per layer
    bars2 = ax2.bar(range(len(layer_names)), parameter_counts)
    ax2.set_title('Parameter Count per Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Parameters')
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, parameter_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_client_selection_history(selected_clients: List[List[int]],
                                save_path: Optional[str] = None):
    """
    Plot client selection history across federated rounds.
    
    Args:
        selected_clients: List of selected client indices for each round
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create a matrix showing which clients were selected in each round
    max_clients = max(max(clients) for clients in selected_clients) + 1
    num_rounds = len(selected_clients)
    
    # Create selection matrix
    selection_matrix = np.zeros((max_clients, num_rounds))
    for round_idx, clients in enumerate(selected_clients):
        for client_idx in clients:
            selection_matrix[client_idx, round_idx] = 1
    
    # Plot heatmap
    sns.heatmap(selection_matrix, 
                cmap='Blues', 
                cbar_kws={'label': 'Selected (1) / Not Selected (0)'},
                xticklabels=range(1, num_rounds + 1),
                yticklabels=range(max_clients))
    
    plt.title('Client Selection History Across Federated Rounds')
    plt.xlabel('Round')
    plt.ylabel('Client ID')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_comparison(results: Dict[str, List[float]],
                                 param_name: str,
                                 save_path: Optional[str] = None):
    """
    Plot comparison of different hyperparameter values.
    
    Args:
        results: Dictionary mapping hyperparameter values to results
        param_name: Name of the hyperparameter being compared
        save_path: Optional path to save the plot
    """
    param_values = list(results.keys())
    performance_metrics = list(results.values())
    
    plt.figure(figsize=(10, 6))
    
    # Plot performance vs hyperparameter
    plt.plot(param_values, performance_metrics, 'bo-', linewidth=2, markersize=8)
    plt.xlabel(param_name)
    plt.ylabel('Performance Metric')
    plt.title(f'Performance vs {param_name}')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for x, y in zip(param_values, performance_metrics):
        plt.text(x, y + max(performance_metrics) * 0.01, f'{y:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sparsity_comparison_r_values(sparsity_data: Dict[str, Dict[str, float]],
                                    save_path: Optional[str] = None):
    """
    Plot sparsity comparison across different R values.
    
    Args:
        sparsity_data: Dictionary with structure {R_value: {soft_zero: sparsity}}
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    R_values = sorted(sparsity_data.keys(), key=lambda x: int(x))
    soft_zero_values = sorted(sparsity_data[R_values[0]].keys(), key=lambda x: float(x))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Sparsity vs R values for each soft zero
    for sz in soft_zero_values:
        sparsities = [sparsity_data[R][sz] for R in R_values]
        ax1.plot(R_values, sparsities, 'o-', label=f'soft_zero={sz}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('R Values (Iterations)')
    ax1.set_ylabel('Achieved Sparsity')
    ax1.set_title('Sparsity vs R Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Sparsity vs Soft Zero values for each R
    for R in R_values:
        sparsities = [sparsity_data[R][sz] for sz in soft_zero_values]
        ax2.plot(soft_zero_values, sparsities, 's-', label=f'R={R}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Soft Zero Values')
    ax2.set_ylabel('Achieved Sparsity')
    ax2.set_title('Sparsity vs Soft Zero Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_layer_wise_sparsity_comparison(mask_data: Dict[str, Dict[str, torch.Tensor]],
                                       layer_filter: Optional[List[int]] = None,
                                       save_path: Optional[str] = None):
    """
    Plot layer-wise sparsity comparison for different R and soft zero combinations.
    
    Args:
        mask_data: Dictionary with structure {config_name: mask_dict}
        layer_filter: Optional list of layer indices to include (e.g., [8, 9, 10, 11])
        save_path: Optional path to save the plot
    """
    # Extract layer information
    config_names = list(mask_data.keys())
    first_mask = list(mask_data.values())[0]
    
    # Get layer names and filter if needed
    layer_names = []
    for name in first_mask.keys():
        if 'backbone.blocks' in name:
            # Extract block number
            block_num = int(name.split('.')[2])  # backbone.blocks.X
            if layer_filter is None or block_num in layer_filter:
                layer_names.append(f'Block {block_num}')
    
    # Create subplots
    n_configs = len(config_names)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each configuration
    for idx, (config_name, mask) in enumerate(mask_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Calculate sparsity for each layer
        sparsities = []
        filtered_layer_names = []
        
        for name, mask_tensor in mask.items():
            if 'backbone.blocks' in name:
                block_num = int(name.split('.')[2])
                if layer_filter is None or block_num in layer_filter:
                    sparsity = (mask_tensor == 0).float().mean().item()
                    sparsities.append(sparsity)
                    filtered_layer_names.append(f'Block {block_num}')
        
        # Plot bar chart
        bars = ax.bar(range(len(filtered_layer_names)), sparsities, alpha=0.7)
        ax.set_title(f'{config_name}')
        ax.set_xlabel('Transformer Block')
        ax.set_ylabel('Sparsity')
        ax.set_xticks(range(len(filtered_layer_names)))
        ax.set_xticklabels(filtered_layer_names, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sparsities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_configs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sparsity_heatmap(sparsity_data: Dict[str, Dict[str, float]],
                         save_path: Optional[str] = None):
    """
    Plot sparsity heatmap for R vs Soft Zero values.
    
    Args:
        sparsity_data: Dictionary with structure {R_value: {soft_zero: sparsity}}
        save_path: Optional path to save the plot
    """
    # Extract data
    R_values = sorted(sparsity_data.keys(), key=lambda x: int(x))
    soft_zero_values = sorted(sparsity_data[R_values[0]].keys(), key=lambda x: float(x))
    
    # Create matrix
    matrix = []
    for R in R_values:
        row = [sparsity_data[R][sz] for sz in soft_zero_values]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
                xticklabels=soft_zero_values,
                yticklabels=R_values,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Achieved Sparsity'})
    
    plt.xlabel('Soft Zero Values')
    plt.ylabel('R Values (Iterations)')
    plt.title('Sparsity Heatmap: R vs Soft Zero Values')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_computation_time_comparison(time_data: Dict[str, Dict[str, float]],
                                   save_path: Optional[str] = None):
    """
    Plot computation time comparison across different R and soft zero values.
    
    Args:
        time_data: Dictionary with structure {R_value: {soft_zero: time}}
        save_path: Optional path to save the plot
    """
    # Extract data
    R_values = sorted(time_data.keys(), key=lambda x: int(x))
    soft_zero_values = sorted(time_data[R_values[0]].keys(), key=lambda x: float(x))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time vs R values for each soft zero
    for sz in soft_zero_values:
        times = [time_data[R][sz] for R in R_values]
        ax1.plot(R_values, times, 'o-', label=f'soft_zero={sz}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('R Values (Iterations)')
    ax1.set_ylabel('Computation Time (seconds)')
    ax1.set_title('Computation Time vs R Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time vs Soft Zero values for each R
    for R in R_values:
        times = [time_data[R][sz] for sz in soft_zero_values]
        ax2.plot(soft_zero_values, times, 's-', label=f'R={R}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Soft Zero Values')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time vs Soft Zero Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

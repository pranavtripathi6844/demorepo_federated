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

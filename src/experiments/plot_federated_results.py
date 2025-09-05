#!/usr/bin/env python3
"""
Plot federated learning results from saved YAML files.
Supports both single scenario results and multi-scenario comparisons.
"""

import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from typing import Dict, List, Any


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def plot_single_scenario(results: Dict[str, Any], output_dir: str = "plots"):
    """Plot results for a single scenario."""
    os.makedirs(output_dir, exist_ok=True)
    
    config = results['experiment_config']
    history = results['training_history']
    
    # Extract data
    rounds = list(range(1, len(history['validation_accuracies']) + 1))
    val_acc = history['validation_accuracies']
    client_acc = history['round_accuracies']
    val_loss = history['validation_losses']
    client_loss = history['round_losses']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Validation vs Client Accuracy
    ax1.plot(rounds, val_acc, 'b-', label='Validation Accuracy', linewidth=2)
    ax1.plot(rounds, client_acc, 'r-', label='Client Accuracy', linewidth=2)
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Accuracy Comparison - {config["scenario"].upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation vs Client Loss
    ax2.plot(rounds, val_loss, 'b-', label='Validation Loss', linewidth=2)
    ax2.plot(rounds, client_loss, 'r-', label='Client Loss', linewidth=2)
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Loss Comparison - {config["scenario"].upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy Only
    ax3.plot(rounds, val_acc, 'b-', linewidth=2)
    ax3.set_xlabel('Communication Rounds')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title(f'Validation Accuracy - {config["scenario"].upper()}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Client Accuracy Only
    ax4.plot(rounds, client_acc, 'r-', linewidth=2)
    ax4.set_xlabel('Communication Rounds')
    ax4.set_ylabel('Client Accuracy (%)')
    ax4.set_title(f'Client Accuracy - {config["scenario"].upper()}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    scenario_name = config['scenario']
    plot_filename = f"federated_{scenario_name}_plots.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Single scenario plot saved to: {plot_path}")
    
    # Print summary
    print(f"\n{scenario_name.upper()} Results Summary:")
    print(f"  Best Validation Accuracy: {results['best_validation_accuracy']:.2f}%")
    print(f"  Final Validation Accuracy: {results['final_validation_accuracy']:.2f}%")
    print(f"  Best Client Accuracy: {results['best_client_accuracy']:.2f}%")
    print(f"  Final Client Accuracy: {results['final_client_accuracy']:.2f}%")


def plot_multi_scenario_comparison(result_files: List[str], output_dir: str = "plots"):
    """Plot comparison across multiple scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_results = []
    for filepath in result_files:
        results = load_results(filepath)
        all_results.append(results)
    
    # Sort by classes_per_client for consistent ordering
    all_results.sort(key=lambda x: x['experiment_config']['classes_per_client'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, results in enumerate(all_results):
        config = results['experiment_config']
        history = results['training_history']
        
        rounds = list(range(1, len(history['validation_accuracies']) + 1))
        val_acc = history['validation_accuracies']
        client_acc = history['round_accuracies']
        
        scenario_name = config['scenario']
        color = colors[i % len(colors)]
        
        # Plot validation accuracy
        ax1.plot(rounds, val_acc, color=color, label=scenario_name.upper(), 
                linewidth=2, alpha=0.8)
        
        # Plot client accuracy
        ax2.plot(rounds, client_acc, color=color, label=scenario_name.upper(), 
                linewidth=2, alpha=0.8)
    
    # Format validation accuracy plot
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format client accuracy plot
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Client Accuracy (%)')
    ax2.set_title('Client Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = "federated_multi_scenario_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-scenario comparison plot saved to: {plot_path}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("SCENARIO COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<15} {'Classes/Client':<12} {'Best Val Acc':<12} {'Final Val Acc':<12}")
    print("-" * 80)
    
    for results in all_results:
        config = results['experiment_config']
        print(f"{config['scenario']:<15} {config['classes_per_client']:<12} "
              f"{results['best_validation_accuracy']:<12.2f} {results['final_validation_accuracy']:<12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Plot federated learning results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result YAML files')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific result file to plot (single scenario)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple scenarios')
    parser.add_argument('--pattern', type=str, default='federated_*_results_*.yaml',
                       help='Pattern to match result files for comparison')
    
    args = parser.parse_args()
    
    if args.file:
        # Plot single scenario
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        results = load_results(args.file)
        plot_single_scenario(results, args.output_dir)
        
    elif args.compare:
        # Plot multiple scenarios comparison
        pattern = os.path.join(args.results_dir, args.pattern)
        result_files = glob.glob(pattern)
        
        if not result_files:
            print(f"No result files found matching pattern: {pattern}")
            return
        
        print(f"Found {len(result_files)} result files:")
        for f in result_files:
            print(f"  - {f}")
        
        plot_multi_scenario_comparison(result_files, args.output_dir)
        
    else:
        # Auto-detect: if multiple files, compare; if single file, plot single
        pattern = os.path.join(args.results_dir, args.pattern)
        result_files = glob.glob(pattern)
        
        if not result_files:
            print(f"No result files found in {args.results_dir}")
            print("Usage examples:")
            print("  python plot_federated_results.py --file results/federated_iid_results_20231201_120000.yaml")
            print("  python plot_federated_results.py --compare")
            return
        
        if len(result_files) == 1:
            # Single file - plot single scenario
            results = load_results(result_files[0])
            plot_single_scenario(results, args.output_dir)
        else:
            # Multiple files - compare scenarios
            print(f"Found {len(result_files)} result files, creating comparison plot:")
            for f in result_files:
                print(f"  - {f}")
            plot_multi_scenario_comparison(result_files, args.output_dir)


if __name__ == "__main__":
    main()

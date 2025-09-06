#!/usr/bin/env python3
"""
Test TALOS models on actual 10k test set and generate test curves.
Only plots test accuracy and test loss for 30 and 40 epoch models.
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def test_model_on_test_set(model_file, test_loader, device='cuda'):
    """Test a trained model on the actual test set."""
    
    print(f"Testing model: {model_file}")
    
    # Load model
    checkpoint = torch.load(model_file, map_location='cpu')
    model_state = checkpoint['model_state_dict']
    
    # Create model
    from src.models.vision_transformer import LinearFlexibleDino
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
    model.load_state_dict(model_state)
    model.eval()
    model = model.to(device)
    
    # Test on test set
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    print(f"  Real Test Loss: {test_loss:.4f}, Real Test Accuracy: {test_acc:.2f}%")
    
    return test_loss, test_acc

def load_training_history_and_test():
    """Load training history and test final models on actual test set."""
    
    # Check if test results already exist
    if os.path.exists('checkpoints/talos_test_results_30.json') and os.path.exists('checkpoints/talos_test_results_40.json'):
        print("Test results already exist! Loading from files...")
        results_30 = json.load(open('checkpoints/talos_test_results_30.json', 'r'))
        results_40 = json.load(open('checkpoints/talos_test_results_40.json', 'r'))
        return {'30': results_30, '40': results_40}
    
    # Load test data (10k unseen samples)
    from src.data.dataset_loader import CIFAR100DataManager
    data_manager = CIFAR100DataManager()
    _, _, test_loader = data_manager.get_centralized_loaders(val_split=0.2)
    
    print(f"Loaded test set: {len(test_loader.dataset)} samples")
    
    # Process each model separately
    test_results = {}
    
    # Process 30 epoch model
    print("\n" + "="*60)
    print("PROCESSING 30 EPOCH MODEL")
    print("="*60)
    
    history_file_30 = "checkpoints/centralized_masked_centralized_mask_R3_sz0.01_epoch_30.json"
    model_file_30 = "checkpoints/centralized_masked_centralized_mask_R3_sz0.01_epoch_30.pth"
    
    if os.path.exists(history_file_30) and os.path.exists(model_file_30):
        # Load training history
        with open(history_file_30, 'r') as f:
            history_30 = json.load(f)
        
        epochs_30 = history_30['epochs']
        val_acc_30 = history_30['val_accuracy']
        val_loss_30 = history_30['val_loss']
        
        print(f"Training epochs: {len(epochs_30)}")
        print(f"Final val acc: {val_acc_30[-1]:.2f}%")
        
        # Test the final model on actual test set
        real_test_loss_30, real_test_acc_30 = test_model_on_test_set(model_file_30, test_loader)
        
        # Create test accuracy and loss arrays (validation + real test at final epoch)
        test_acc_30 = val_acc_30.copy()
        test_loss_30 = val_loss_30.copy()
        test_acc_30[-1] = real_test_acc_30
        test_loss_30[-1] = real_test_loss_30
        
        test_results['30'] = {
            'epochs': epochs_30,
            'test_accuracy': test_acc_30,
            'test_loss': test_loss_30,
            'real_test_accuracy': real_test_acc_30,
            'real_test_loss': real_test_loss_30
        }
        
        # Save 30 epoch results separately
        with open('checkpoints/talos_test_results_30.json', 'w') as f:
            json.dump(test_results['30'], f, indent=2)
        print("30 epoch results saved to: checkpoints/talos_test_results_30.json")
    
    # Process 40 epoch model
    print("\n" + "="*60)
    print("PROCESSING 40 EPOCH MODEL")
    print("="*60)
    
    history_file_40 = "checkpoints/centralized_masked_centralized_mask_R3_sz0.01_epoch_40.json"
    model_file_40 = "checkpoints/centralized_masked_centralized_mask_R3_sz0.01_epoch_40.pth"
    
    if os.path.exists(history_file_40) and os.path.exists(model_file_40):
        # Load training history
        with open(history_file_40, 'r') as f:
            history_40 = json.load(f)
        
        epochs_40 = history_40['epochs']
        val_acc_40 = history_40['val_accuracy']
        val_loss_40 = history_40['val_loss']
        
        print(f"Training epochs: {len(epochs_40)}")
        print(f"Final val acc: {val_acc_40[-1]:.2f}%")
        
        # Test the final model on actual test set
        real_test_loss_40, real_test_acc_40 = test_model_on_test_set(model_file_40, test_loader)
        
        # Create test accuracy and loss arrays (validation + real test at final epoch)
        test_acc_40 = val_acc_40.copy()
        test_loss_40 = val_loss_40.copy()
        test_acc_40[-1] = real_test_acc_40
        test_loss_40[-1] = real_test_loss_40
        
        test_results['40'] = {
            'epochs': epochs_40,
            'test_accuracy': test_acc_40,
            'test_loss': test_loss_40,
            'real_test_accuracy': real_test_acc_40,
            'real_test_loss': real_test_loss_40
        }
        
        # Save 40 epoch results separately
        with open('checkpoints/talos_test_results_40.json', 'w') as f:
            json.dump(test_results['40'], f, indent=2)
        print("40 epoch results saved to: checkpoints/talos_test_results_40.json")
    
    return test_results

def plot_test_curves():
    """Plot only test accuracy and test loss curves for both models."""
    
    # Load test results
    if not os.path.exists('checkpoints/talos_test_results_30.json') or not os.path.exists('checkpoints/talos_test_results_40.json'):
        print("No test results found! Run load_training_history_and_test() first.")
        return
    
    with open('checkpoints/talos_test_results_30.json', 'r') as f:
        results_30 = json.load(f)
    
    with open('checkpoints/talos_test_results_40.json', 'r') as f:
        results_40 = json.load(f)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 30 epoch model data
    epochs_30 = results_30['epochs']
    test_acc_30 = results_30['test_accuracy']
    test_loss_30 = results_30['test_loss']
    real_test_acc_30 = results_30['real_test_accuracy']
    real_test_loss_30 = results_30['real_test_loss']
    
    # 40 epoch model data
    epochs_40 = results_40['epochs']
    test_acc_40 = results_40['test_accuracy']
    test_loss_40 = results_40['test_loss']
    real_test_acc_40 = results_40['real_test_accuracy']
    real_test_loss_40 = results_40['real_test_loss']
    
    # Plot test accuracy for both models
    ax1.plot(epochs_30, test_acc_30, 'b-', linewidth=2, markersize=3, 
            label='30 Epochs', alpha=0.8)
    ax1.plot(epochs_30[-1], real_test_acc_30, 'bo', markersize=8, 
            markeredgecolor='black', markeredgewidth=2, alpha=1.0)
    
    ax1.plot(epochs_40, test_acc_40, 'r-', linewidth=2, markersize=3, 
            label='40 Epochs', alpha=0.8)
    ax1.plot(epochs_40[-1], real_test_acc_40, 'ro', markersize=8, 
            markeredgecolor='black', markeredgewidth=2, alpha=1.0)
    
    ax1.set_title('TALOS - Test Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim(50, 80)
    ax1.set_xlim(0, 45)
    
    # Plot test loss for both models
    ax2.plot(epochs_30, test_loss_30, 'b-', linewidth=2, markersize=3, 
            label='30 Epochs', alpha=0.8)
    ax2.plot(epochs_30[-1], real_test_loss_30, 'bo', markersize=8, 
            markeredgecolor='black', markeredgewidth=2, alpha=1.0)
    
    ax2.plot(epochs_40, test_loss_40, 'r-', linewidth=2, markersize=3, 
            label='40 Epochs', alpha=0.8)
    ax2.plot(epochs_40[-1], real_test_loss_40, 'ro', markersize=8, 
            markeredgecolor='black', markeredgewidth=2, alpha=1.0)
    
    ax2.set_title('TALOS - Test Loss vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.set_ylim(1.0, 2.5)
    ax2.set_xlim(0, 45)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/talos_test_curves_only.png', dpi=300, bbox_inches='tight')
    print("Test curves saved as: plots/talos_test_curves_only.png")
    
    # Show plot
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("TALOS TEST RESULTS SUMMARY")
    print("="*60)
    print(f"30 Epochs - Final Test Accuracy: {real_test_acc_30:.2f}%, Test Loss: {real_test_loss_30:.4f}")
    print(f"40 Epochs - Final Test Accuracy: {real_test_acc_40:.2f}%, Test Loss: {real_test_loss_40:.4f}")

if __name__ == "__main__":
    # Load training history and test
    test_results = load_training_history_and_test()
    
    # Then plot the curves
    if test_results:
        plot_test_curves()
#!/usr/bin/env python3
"""
Test evaluation script for existing federated learning checkpoints.
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.vision_transformer import create_vision_transformer
from src.data.dataset_loader import CIFAR100DataManager


def load_model_from_checkpoint(checkpoint_path: str, model_size: str = "small", num_classes: int = 100):
    """Load model from checkpoint."""
    try:
        # Create model
        model = create_vision_transformer(
            model_size=model_size,
            num_classes=num_classes
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model, True
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None, False


def evaluate_model_on_test_set(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    """Main evaluation function."""
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading CIFAR-100 test dataset...")
    data_manager = CIFAR100DataManager(
        data_dir="./data",
        batch_size=64,
        num_workers=4,
        download=True
    )
    
    _, _, test_loader = data_manager.get_centralized_loaders(val_split=0.1)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Find all checkpoint files
    checkpoint_dir = "./checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {f}")
    
    print("\n" + "="*80)
    print("TESTING AVAILABLE CHECKPOINTS ON TEST SET")
    print("="*80)
    
    results = []
    
    for checkpoint_file in sorted(checkpoint_files):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        print(f"\nTesting {checkpoint_file}...")
        
        # Load model
        model, success = load_model_from_checkpoint(checkpoint_path)
        
        if not success:
            print(f"  ❌ Failed to load model")
            continue
        
        try:
            model = model.to(device)
            
            # Evaluate on test set
            test_loss, test_accuracy = evaluate_model_on_test_set(model, test_loader, device)
            
            print(f"  ✅ Test Loss: {test_loss:.4f}")
            print(f"  ✅ Test Accuracy: {test_accuracy:.2f}%")
            
            results.append({
                'checkpoint': checkpoint_file,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })
            
        except Exception as e:
            print(f"  ❌ Error evaluating model: {e}")
            continue
    
    # Print summary table
    print("\n" + "="*80)
    print("TEST ACCURACY SUMMARY")
    print("="*80)
    print(f"{'Checkpoint':<30} {'Test Loss':<12} {'Test Accuracy':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['checkpoint']:<30} {result['test_loss']:<12.4f} {result['test_accuracy']:<15.2f}%")
    
    # Find best performing checkpoint
    if results:
        best_result = max(results, key=lambda x: x['test_accuracy'])
        print(f"\n🏆 Best performing checkpoint: {best_result['checkpoint']}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.2f}%")


if __name__ == '__main__':
    main()

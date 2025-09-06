#!/usr/bin/env python3
"""
Test evaluation script for the 5 specific federated models:
j=4 with Nc={1,5,10,50,100}
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime

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
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
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
    
    # Define your 5 specific models
    my_models = [
        {
            'nc': 1,
            'j': 4,
            'checkpoint': 'federated_nc1_j4_round_100.pth',
            'description': 'Non-IID: 1 class per client'
        },
        {
            'nc': 5,
            'j': 4,
            'checkpoint': 'federated_nc5_j4_round_100.pth',
            'description': 'Non-IID: 5 classes per client'
        },
        {
            'nc': 10,
            'j': 4,
            'checkpoint': 'federated_nc10_j4_round_100.pth',
            'description': 'Non-IID: 10 classes per client'
        },
        {
            'nc': 50,
            'j': 4,
            'checkpoint': 'federated_nc50_j4_round_100.pth',
            'description': 'Non-IID: 50 classes per client'
        },
        {
            'nc': 100,
            'j': 4,
            'checkpoint': 'federated_nc100_j4_round_100.pth',
            'description': 'IID: 100 classes per client'
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING YOUR 5 FEDERATED MODELS (j=4) ON TEST SET")
    print("="*80)
    print("Models to test:")
    for model in my_models:
        print(f"  - Nc={model['nc']}, j={model['j']}: {model['checkpoint']} ({model['description']})")
    
    results = []
    
    # Test each model
    for i, model_config in enumerate(my_models, 1):
        nc = model_config['nc']
        j = model_config['j']
        checkpoint_file = model_config['checkpoint']
        description = model_config['description']
        
        checkpoint_path = os.path.join("./checkpoints", checkpoint_file)
        
        print(f"\n[{i}/5] Testing Nc={nc}, j={j} ({description})...")
        print(f"       Checkpoint: {checkpoint_file}")
        
        if not os.path.exists(checkpoint_path):
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            continue
        
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
                'nc': nc,
                'j': j,
                'checkpoint': checkpoint_file,
                'description': description,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })
            
        except Exception as e:
            print(f"  ❌ Error evaluating model: {e}")
            continue
    
    # Print summary table
    print("\n" + "="*80)
    print("YOUR 5 MODELS TEST ACCURACY SUMMARY")
    print("="*80)
    print(f"{'Nc':<5} {'j':<3} {'Test Loss':<12} {'Test Accuracy':<15} {'Description':<25}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['nc']:<5} {result['j']:<3} {result['test_loss']:<12.4f} {result['test_accuracy']:<15.2f}% {result['description']:<25}")
    
    # Find best performing model
    if results:
        best_result = max(results, key=lambda x: x['test_accuracy'])
        print(f"\n🏆 Best performing model:")
        print(f"   Nc={best_result['nc']}, j={best_result['j']}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.2f}%")
        print(f"   Description: {best_result['description']}")
    
    # Save results to file
    results_file = f"./checkpoints/test_results_j4_nc_variations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\n📊 Summary:")
    print(f"   - Tested {len(results)}/5 models successfully")
    print(f"   - All models use j=4 (4 client steps)")
    print(f"   - Test set: 10,000 samples (untouched)")
    print(f"   - Nc values: 1, 5, 10, 50, 100")
    print(f"   - Results saved to: {results_file}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test script for adaptive stratified sampling implementation.
Verifies that the stratified sampling works correctly for different Nc values.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset_loader import CIFAR100DataManager
from src.training.model_editing import create_client_masks
from src.models.vision_transformer import DINOBackboneClassifier
import torch

def test_adaptive_stratified_sampling():
    """Test adaptive stratified sampling for different Nc values."""
    
    print("🧪 Testing Adaptive Stratified Sampling Implementation")
    print("=" * 60)
    
    # Initialize data manager
    data_manager = CIFAR100DataManager(batch_size=32, download=True)
    
    # Test different Nc values
    test_cases = [
        (100, "IID", "100 classes × 1 sample = 100 examples"),
        (50, "Non-IID", "50 classes × 2 samples = 100 examples"),
        (10, "Non-IID", "10 classes × 10 samples = 100 examples"),
        (5, "Non-IID", "5 classes × 20 samples = 100 examples"),
        (1, "Non-IID", "1 class × 100 samples = 100 examples")
    ]
    
    for nc, setting_type, description in test_cases:
        print(f"\n📊 Testing Nc={nc} ({setting_type}): {description}")
        print("-" * 50)
        
        try:
            # Create federated datasets
            client_datasets = data_manager.create_federated_datasets(
                num_clients=5,  # Small number for testing
                classes_per_client=nc,
                val_split=0.2
            )
            
            # Create a dummy model for testing
            model = DINOBackboneClassifier(num_classes=100, freeze_backbone=True)
            
            # Test adaptive stratified loader directly
            print(f"  Testing adaptive stratified loader...")
            test_loader = torch.utils.data.DataLoader(
                client_datasets[0], batch_size=32, shuffle=True
            )
            
            stratified_loader = data_manager.get_adaptive_stratified_loader(
                test_loader, classes_per_client=nc
            )
            
            # Count samples and classes
            samples = []
            classes = set()
            for batch_data, batch_labels in stratified_loader:
                for label in batch_labels:
                    samples.append(label.item())
                    classes.add(label.item())
            
            print(f"  ✅ Samples collected: {len(samples)}")
            print(f"  ✅ Classes represented: {len(classes)}")
            print(f"  ✅ Expected classes: {nc}")
            print(f"  ✅ Expected samples: {100}")
            
            # Verify the sampling is correct
            if len(samples) == 100 and len(classes) == nc:
                print(f"  ✅ PASS: Correct stratified sampling for Nc={nc}")
            else:
                print(f"  ❌ FAIL: Incorrect sampling for Nc={nc}")
                print(f"      Expected: {nc} classes, 100 samples")
                print(f"      Got: {len(classes)} classes, {len(samples)} samples")
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
    
    print("\n🎯 Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_adaptive_stratified_sampling()

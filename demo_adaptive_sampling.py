#!/usr/bin/env python3
"""
Demonstration of adaptive stratified sampling behavior.
Shows exactly how the sampling works for different Nc values.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset_loader import CIFAR100DataManager
import torch

def demonstrate_adaptive_sampling():
    """Demonstrate adaptive stratified sampling behavior."""
    
    print("🎯 Adaptive Stratified Sampling Demonstration")
    print("=" * 60)
    print("This shows exactly how stratified sampling works for different Nc values:")
    print()
    
    # Initialize data manager
    data_manager = CIFAR100DataManager(batch_size=32, download=True)
    
    # Create a sample client dataset for demonstration
    client_datasets = data_manager.create_federated_datasets(
        num_clients=1,  # Just one client for demo
        classes_per_client=5,  # Non-IID with 5 classes
        val_split=0.2
    )
    
    client_loader = torch.utils.data.DataLoader(
        client_datasets[0], batch_size=32, shuffle=True
    )
    
    # Test different Nc values
    test_cases = [
        (100, "IID Setting"),
        (50, "Non-IID (50 classes)"),
        (10, "Non-IID (10 classes)"),
        (5, "Non-IID (5 classes)"),
        (1, "Non-IID (1 class)")
    ]
    
    for nc, description in test_cases:
        print(f"📊 {description} (Nc={nc})")
        print("-" * 40)
        
        # Get adaptive stratified loader
        stratified_loader = data_manager.get_adaptive_stratified_loader(
            client_loader, classes_per_client=nc
        )
        
        # Analyze the sampling
        samples = []
        class_counts = {}
        
        for batch_data, batch_labels in stratified_loader:
            for label in batch_labels:
                label_val = label.item()
                samples.append(label_val)
                class_counts[label_val] = class_counts.get(label_val, 0) + 1
        
        print(f"  Total samples: {len(samples)}")
        print(f"  Classes represented: {len(class_counts)}")
        print(f"  Expected classes: {nc}")
        print(f"  Expected samples: {100}")
        
        # Show class distribution
        if len(class_counts) <= 10:  # Only show if manageable number
            print(f"  Class distribution: {dict(sorted(class_counts.items()))}")
        
        # Calculate samples per class
        if len(class_counts) > 0:
            samples_per_class = len(samples) // len(class_counts)
            print(f"  Samples per class: {samples_per_class}")
        
        print()

if __name__ == "__main__":
    demonstrate_adaptive_sampling()

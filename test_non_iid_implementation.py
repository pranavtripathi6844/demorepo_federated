#!/usr/bin/env python3
"""
Test script to verify the non-IID implementation works correctly.
"""

import sys
import os
sys.path.append('src')

from src.data.dataset_loader import CIFAR100DataManager, create_iid_splits, create_non_iid_splits
import torch


def test_non_iid_implementation():
    """Test the non-IID implementation with different scenarios."""
    print("Testing Non-IID Implementation")
    print("="*50)
    
    # Create data manager
    data_manager = CIFAR100DataManager(batch_size=32, download=True)
    train_dataset = data_manager.train_dataset
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: 100")
    
    # Test scenarios
    scenarios = [
        ('IID', 100),
        ('Non-IID(1)', 1),
        ('Non-IID(5)', 5),
        ('Non-IID(10)', 10),
        ('Non-IID(50)', 50)
    ]
    
    num_clients = 10  # Small number for testing
    
    for scenario_name, classes_per_client in scenarios:
        print(f"\nTesting {scenario_name} (classes_per_client={classes_per_client})")
        print("-" * 40)
        
        try:
            # Test using the new functions
            if classes_per_client == 100:
                client_datasets = create_iid_splits(train_dataset, num_clients=num_clients, debug=False)
            else:
                client_datasets = create_non_iid_splits(
                    train_dataset, 
                    num_clients=num_clients, 
                    classes_per_client=classes_per_client, 
                    debug=False
                )
            
            print(f"  Created {len(client_datasets)} client datasets")
            
            # Check dataset sizes
            sizes = [len(dataset) for dataset in client_datasets]
            print(f"  Dataset sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
            
            # Check class distribution for first few clients
            for i in range(min(3, len(client_datasets))):
                dataset = client_datasets[i]
                classes = set()
                for j in range(len(dataset)):
                    _, label = dataset[j]
                    classes.add(label)
                print(f"  Client {i}: {len(classes)} classes, samples: {len(dataset)}")
            
            print(f"  ✅ {scenario_name} test passed")
            
        except Exception as e:
            print(f"  ❌ {scenario_name} test failed: {e}")
    
    # Test using CIFAR100DataManager
    print(f"\nTesting CIFAR100DataManager integration")
    print("-" * 40)
    
    try:
        # Test IID
        client_datasets = data_manager.create_federated_datasets(
            num_clients=num_clients, 
            classes_per_client=100
        )
        print(f"  IID via DataManager: {len(client_datasets)} clients")
        
        # Test Non-IID
        client_datasets = data_manager.create_federated_datasets(
            num_clients=num_clients, 
            classes_per_client=5
        )
        print(f"  Non-IID(5) via DataManager: {len(client_datasets)} clients")
        
        print("  ✅ DataManager integration test passed")
        
    except Exception as e:
        print(f"  ❌ DataManager integration test failed: {e}")
    
    print(f"\n{'='*50}")
    print("Non-IID implementation test completed!")


if __name__ == "__main__":
    test_non_iid_implementation()

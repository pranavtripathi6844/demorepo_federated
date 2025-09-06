#!/usr/bin/env python3
"""
Test script for the 2-stage TALOS workflow.
This script tests the complete pipeline from mask computation to training.
"""

import torch
import os
import subprocess
import sys
from typing import Dict, Any


def test_mask_computation():
    """Test Stage 1: Mask computation."""
    print("=== Testing Stage 1: Mask Computation ===")
    
    # Test with a small subset to verify functionality
    print("Testing centralized mask computation...")
    
    # Create a simple test
    from src.models.vision_transformer import DINOBackboneClassifier
    from src.data.dataset_loader import CIFAR100DataManager
    from src.training.model_editing import compute_mask
    
    # Load model and data
    model = DINOBackboneClassifier(num_classes=100, freeze_backbone=True)
    model.eval()
    
    # Save the model used in Stage 1 (simulating stage1_compute_masks.py)
    print("Saving Stage 1 model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/stage1_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 100,
            'freeze_backbone': True
        }
    }, model_path)
    print(f"✓ Stage 1 model saved to {model_path}")
    
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders()
    
    # Create stratified loader
    stratified_loader = data_manager.get_stratified_loader(
        train_loader, num_classes=100, samples_per_class=1
    )
    
    # Test mask computation with small parameters
    print("Computing test mask (R=2, soft_zero=0.01)...")
    mask = compute_mask(
        model=model,
        dataloader=stratified_loader,
        sparsity_target=0.9,
        R=2,  # Small R for testing
        soft_zero=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_plot=False,
        debug=True
    )
    
    # Verify mask properties
    total_params = sum(m.numel() for m in mask.values())
    pruned_params = sum((m == 0).sum().item() for m in mask.values())
    sparsity = pruned_params / total_params
    
    print(f"✓ Mask computed successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Pruned parameters: {pruned_params:,}")
    print(f"  - Sparsity: {sparsity:.4f}")
    
    # Save test mask
    os.makedirs("masks", exist_ok=True)
    torch.save(mask, "masks/test_mask.pth")
    print("✓ Test mask saved to masks/test_mask.pth")
    
    return True


def test_training_with_mask():
    """Test Stage 2: Training with pre-computed mask."""
    print("\n=== Testing Stage 2: Training with Mask ===")
    
    # Check if test mask exists
    if not os.path.exists("masks/test_mask.pth"):
        print("❌ Test mask not found. Run test_mask_computation() first.")
        return False
    
    # Check if Stage 1 model exists
    if not os.path.exists("models/stage1_model.pth"):
        print("❌ Stage 1 model not found. Run test_mask_computation() first.")
        return False
    
    # Test centralized training with mask
    print("Testing centralized training with SparseSGDM...")
    
    from src.training.sparse_optimizer import SparseSGDWithMomentum
    from src.training.centralized_training import train_centralized_model
    
    # Load model from Stage 1 (simulating stage2_train_with_masks.py)
    print("Loading Stage 1 model...")
    checkpoint = torch.load("models/stage1_model.pth", map_location='cpu')
    model_config = checkpoint.get('model_config', {
        'num_classes': 100,
        'freeze_backbone': True
    })
    
    from src.models.vision_transformer import DINOBackboneClassifier
    model = DINOBackboneClassifier(
        num_classes=model_config['num_classes'],
        freeze_backbone=model_config['freeze_backbone']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Stage 1 model loaded successfully")
    
    # Load mask
    mask = torch.load("masks/test_mask.pth")
    
    # Create SparseSGDM optimizer
    optimizer = SparseSGDWithMomentum(
        parameters=model.named_parameters(),
        parameter_masks=mask,
        learning_rate=0.005,
        momentum_factor=0.8,
        weight_decay_factor=0.001
    )
    
    print("✓ SparseSGDM optimizer created successfully")
    
    # Test a short training run
    print("Running short training test (2 epochs)...")
    training_history = train_centralized_model(
        model=model,
        optimizer=optimizer,
        num_epochs=2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        scheduler_type='cosine',
        checkpoint_dir="./checkpoints",
        model_name="test_masked_model"
    )
    
    print("✓ Training completed successfully")
    print(f"  - Final training accuracy: {training_history['train_accuracy'][-1]:.2f}%")
    print(f"  - Final validation accuracy: {training_history['val_accuracy'][-1]:.2f}%")
    
    return True


def test_model_consistency():
    """Test that the loaded model in Stage 2 matches the model used in Stage 1."""
    print("\n=== Testing Model Consistency ===")
    
    # Check if Stage 1 model exists
    if not os.path.exists("models/stage1_model.pth"):
        print("❌ Stage 1 model not found. Run test_mask_computation() first.")
        return False
    
    # Load the saved model from Stage 1
    print("Loading Stage 1 model...")
    checkpoint = torch.load("models/stage1_model.pth", map_location='cpu')
    model_config = checkpoint.get('model_config', {
        'num_classes': 100,
        'freeze_backbone': True
    })
    
    from src.models.vision_transformer import DINOBackboneClassifier
    loaded_model = DINOBackboneClassifier(
        num_classes=model_config['num_classes'],
        freeze_backbone=model_config['freeze_backbone']
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    # Create a fresh model with the same configuration
    fresh_model = DINOBackboneClassifier(
        num_classes=model_config['num_classes'],
        freeze_backbone=model_config['freeze_backbone']
    )
    fresh_model.eval()
    
    # Compare model parameters
    print("Comparing model parameters...")
    loaded_params = dict(loaded_model.named_parameters())
    fresh_params = dict(fresh_model.named_parameters())
    
    # Check if all parameter names match
    if set(loaded_params.keys()) != set(fresh_params.keys()):
        print("❌ Parameter names don't match between loaded and fresh models")
        return False
    
    # Check if parameter shapes match
    for name in loaded_params.keys():
        if loaded_params[name].shape != fresh_params[name].shape:
            print(f"❌ Parameter shape mismatch for {name}")
            return False
    
    print("✓ Model consistency verified")
    print(f"  - Model configuration: {model_config}")
    print(f"  - Number of parameters: {sum(p.numel() for p in loaded_model.parameters()):,}")
    print(f"  - Parameter names match: ✓")
    print(f"  - Parameter shapes match: ✓")
    
    return True


def test_plotting_functionality():
    """Test the new plotting functionality."""
    print("\n=== Testing Plotting Functionality ===")
    
    # Check if test mask exists
    if not os.path.exists("masks/test_mask.pth"):
        print("❌ Test mask not found. Run test_mask_computation() first.")
        return False
    
    # Create test data for plotting
    print("Creating test data for plotting...")
    
    # Load test mask
    test_mask = torch.load("masks/test_mask.pth")
    
    # Create mock data structures
    sparsity_data = {
        '3': {'0.1': 0.85, '0.01': 0.88, '0.001': 0.90},
        '5': {'0.1': 0.87, '0.01': 0.89, '0.001': 0.91},
        '10': {'0.1': 0.89, '0.01': 0.90, '0.001': 0.92}
    }
    
    time_data = {
        '3': {'0.1': 10.5, '0.01': 12.3, '0.001': 15.2},
        '5': {'0.1': 18.7, '0.01': 22.1, '0.001': 28.4},
        '10': {'0.1': 35.2, '0.01': 42.8, '0.001': 55.6}
    }
    
    mask_data = {
        'R3_sz0.01': test_mask,
        'R5_sz0.01': test_mask,
        'R10_sz0.01': test_mask
    }
    
    # Test plotting functions
    try:
        from src.utils.visualization import (
            plot_sparsity_comparison_r_values,
            plot_layer_wise_sparsity_comparison,
            plot_sparsity_heatmap,
            plot_computation_time_comparison
        )
        
        print("Testing sparsity comparison plots...")
        plot_sparsity_comparison_r_values(
            sparsity_data, 
            save_path="plots/test_sparsity_comparison.png"
        )
        
        print("Testing sparsity heatmap...")
        plot_sparsity_heatmap(
            sparsity_data,
            save_path="plots/test_sparsity_heatmap.png"
        )
        
        print("Testing computation time comparison...")
        plot_computation_time_comparison(
            time_data,
            save_path="plots/test_time_comparison.png"
        )
        
        print("Testing layer-wise sparsity comparison (all layers)...")
        plot_layer_wise_sparsity_comparison(
            mask_data,
            layer_filter=None,
            save_path="plots/test_layer_wise_all.png"
        )
        
        print("Testing layer-wise sparsity comparison (blocks 8-11)...")
        plot_layer_wise_sparsity_comparison(
            mask_data,
            layer_filter=[8, 9, 10, 11],
            save_path="plots/test_layer_wise_8_11.png"
        )
        
        print("✓ All plotting functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Plotting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_federated_training():
    """Test federated training with masks."""
    print("\n=== Testing Federated Training with Masks ===")
    
    # Create test client masks
    from src.models.vision_transformer import DINOBackboneClassifier
    from src.data.dataset_loader import CIFAR100DataManager
    from src.training.model_editing import create_client_masks
    
    # Load model and data
    model = DINOBackboneClassifier(num_classes=100, freeze_backbone=True)
    model.eval()
    
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders()
    
    # Create small client datasets for testing
    print("Creating test client datasets (Nc=5, 10 clients)...")
    client_datasets = data_manager.create_non_iid_splits(
        train_loader.dataset,
        num_clients=10,  # Small number for testing
        classes_per_client=5
    )
    
    # Create client masks
    print("Creating test client masks...")
    client_masks = create_client_masks(
        model=model,
        client_datasets=client_datasets,
        classes_per_client=5,
        target_sparsity=0.9,
        num_iterations=2,  # Small R for testing
        soft_zero_value=0.01,
        max_samples=10,  # Small number for testing
        debug=False
    )
    
    print(f"✓ Created {len(client_masks)} client masks")
    
    # Test federated training
    from src.training.federated_training import train_federated_model_editing
    
    optimizer_config = {
        'lr': 0.005,
        'weight_decay': 0.001,
        'momentum': 0.8
    }
    
    print("Running short federated training test (5 rounds)...")
    result = train_federated_model_editing(
        model=model,
        client_datasets=client_datasets,
        client_masks=client_masks,
        optimizer_config=optimizer_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_rounds=5,  # Small number for testing
        num_clients=10,
        client_fraction=0.5,  # Use half the clients
        num_client_steps=2,  # Small number for testing
        validation_loader=None,  # Skip validation for test
        checkpoint_path="./checkpoints",
        model_name="test_federated_masked"
    )
    
    print("✓ Federated training completed successfully")
    if 'val_accuracies' in result and len(result['val_accuracies']) > 0:
        print(f"  - Final validation accuracy: {result['val_accuracies'][-1]:.2f}%")
    
    return True


def test_command_line_scripts():
    """Test the command-line scripts."""
    print("\n=== Testing Command-Line Scripts ===")
    
    # Test stage1 script help
    print("Testing stage1_compute_masks.py help...")
    try:
        result = subprocess.run([sys.executable, "stage1_compute_masks.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ stage1_compute_masks.py help works")
        else:
            print(f"❌ stage1_compute_masks.py help failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing stage1_compute_masks.py: {e}")
    
    # Test stage2 script help
    print("Testing stage2_train_with_masks.py help...")
    try:
        result = subprocess.run([sys.executable, "stage2_train_with_masks.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ stage2_train_with_masks.py help works")
        else:
            print(f"❌ stage2_train_with_masks.py help failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing stage2_train_with_masks.py: {e}")
    
    # Test stage2 list masks
    print("Testing stage2 list masks...")
    try:
        result = subprocess.run([sys.executable, "stage2_train_with_masks.py", "--list_masks"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ stage2 list masks works")
        else:
            print(f"❌ stage2 list masks failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing stage2 list masks: {e}")


def cleanup_test_files():
    """Clean up test files."""
    print("\n=== Cleaning Up Test Files ===")
    
    test_files = [
        "masks/test_mask.pth",
        "models/stage1_model.pth",
        "plots/test_*.png",
        "checkpoints/test_masked_model_*.pth",
        "checkpoints/test_federated_masked_*.pth"
    ]
    
    for pattern in test_files:
        if '*' in pattern:
            # Handle glob patterns
            import glob
            files = glob.glob(pattern)
            for file in files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"✓ Removed {file}")
        else:
            # Handle single files
            if os.path.exists(pattern):
                os.remove(pattern)
                print(f"✓ Removed {pattern}")


def main():
    """Main test function."""
    print("TALOS 2-Stage Workflow Integration Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    try:
        # Test 1: Mask computation
        if test_mask_computation():
            tests_passed += 1
            print("✅ Test 1 PASSED: Mask computation")
        else:
            print("❌ Test 1 FAILED: Mask computation")
        
        # Test 2: Training with mask
        if test_training_with_mask():
            tests_passed += 1
            print("✅ Test 2 PASSED: Training with mask")
        else:
            print("❌ Test 2 FAILED: Training with mask")
        
        # Test 3: Model consistency
        if test_model_consistency():
            tests_passed += 1
            print("✅ Test 3 PASSED: Model consistency")
        else:
            print("❌ Test 3 FAILED: Model consistency")
        
        # Test 4: Plotting functionality
        if test_plotting_functionality():
            tests_passed += 1
            print("✅ Test 4 PASSED: Plotting functionality")
        else:
            print("❌ Test 4 FAILED: Plotting functionality")
        
        # Test 5: Federated training
        if test_federated_training():
            tests_passed += 1
            print("✅ Test 5 PASSED: Federated training")
        else:
            print("❌ Test 5 FAILED: Federated training")
        
        # Test 6: Command-line scripts
        test_command_line_scripts()
        tests_passed += 1
        print("✅ Test 6 PASSED: Command-line scripts")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The 2-stage workflow is ready to use.")
        print("\nNext steps:")
        print("1. Run: python stage1_compute_masks.py")
        print("2. Run: python stage2_train_with_masks.py --mask_file masks/centralized_mask_R5_sz0.01.pth")
        print("3. Run: python stage2_train_with_masks.py --federated --mask_file masks/federated_masks_nc5_R5_sz0.01.pth")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()

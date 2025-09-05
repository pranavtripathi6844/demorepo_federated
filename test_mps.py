#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) functionality on Apple Silicon.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils.device_utils import get_device, print_device_info, is_mps_available


def test_mps_functionality():
    """Test MPS functionality and basic operations."""
    print("=== Apple Silicon MPS Test ===\n")
    
    # Check MPS availability
    print(f"MPS Available: {is_mps_available()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"macOS Version: {os.popen('sw_vers -productVersion').read().strip()}\n")
    
    # Test device selection
    print("Testing device selection:")
    device = get_device('auto')
    print_device_info(device)
    print()
    
    # Test basic tensor operations
    print("Testing basic tensor operations:")
    try:
        # Create tensors on device
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Perform matrix multiplication
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        z = torch.mm(x, y)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            print(f"Matrix multiplication time: {elapsed_time:.2f} ms")
        else:
            print("Matrix multiplication completed successfully")
        
        print(f"Result tensor shape: {z.shape}")
        print(f"Result tensor device: {z.device}")
        print("✅ Basic tensor operations working!\n")
        
    except Exception as e:
        print(f"❌ Error in tensor operations: {e}\n")
    
    # Test model creation
    print("Testing model creation:")
    try:
        from src.models.vision_transformer import create_vision_transformer
        
        model = create_vision_transformer(model_size='tiny', num_classes=100)
        model = model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 32, 32, device=device)
        output = model(dummy_input)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("✅ Model creation and forward pass working!\n")
        
    except Exception as e:
        print(f"❌ Error in model creation: {e}\n")
    
    # Performance comparison
    print("Performance comparison:")
    try:
        # Test on different devices
        devices_to_test = []
        if is_mps_available():
            devices_to_test.append(('mps', torch.device('mps')))
        if torch.cuda.is_available():
            devices_to_test.append(('cuda', torch.device('cuda')))
        devices_to_test.append(('cpu', torch.device('cpu')))
        
        for device_name, test_device in devices_to_test:
            print(f"\nTesting {device_name.upper()}:")
            
            # Create test tensors
            a = torch.randn(500, 500, device=test_device)
            b = torch.randn(500, 500, device=test_device)
            
            # Time matrix multiplication
            import time
            start_time = time.time()
            
            for _ in range(10):
                c = torch.mm(a, b)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
            
            print(f"  Average matrix multiplication time: {avg_time:.2f} ms")
            
    except Exception as e:
        print(f"❌ Error in performance test: {e}\n")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_mps_functionality()

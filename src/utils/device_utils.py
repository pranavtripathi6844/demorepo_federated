"""
Device utilities for handling different compute devices including Apple Silicon MPS.
"""

import torch
from typing import Optional


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    return torch.backends.mps.is_available()


def get_device(device_preference: str = 'auto') -> torch.device:
    """
    Get the best available device for training.
    
    Args:
        device_preference: Device preference ('auto', 'mps', 'cuda', 'cpu')
        
    Returns:
        torch.device: The selected device
    """
    if device_preference == 'auto':
        # Try MPS (Apple Silicon) first
        if torch.backends.mps.is_available():
            return torch.device('mps')
        # Try CUDA next
        elif torch.cuda.is_available():
            return torch.device('cuda')
        # Fall back to CPU
        else:
            return torch.device('cpu')
    elif device_preference == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS not available, falling back to CPU")
            return torch.device('cpu')
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device('cpu')
    else:
        return torch.device('cpu')


def get_optimal_batch_size(device: torch.device, model_size: str = 'small') -> int:
    """
    Get optimal batch size based on device and model size.
    
    Args:
        device: The device to use
        model_size: Model size ('tiny', 'small', 'base', 'large')
        
    Returns:
        int: Recommended batch size
    """
    if device.type == 'mps':
        # Apple Silicon GPU batch sizes
        batch_sizes = {
            'tiny': 256,
            'small': 128,
            'base': 64,
            'large': 32
        }
    elif device.type == 'cuda':
        # NVIDIA GPU batch sizes
        batch_sizes = {
            'tiny': 256,
            'small': 128,
            'base': 64,
            'large': 32
        }
    else:
        # CPU batch sizes (smaller due to memory constraints)
        batch_sizes = {
            'tiny': 64,
            'small': 32,
            'base': 16,
            'large': 8
        }
    
    return batch_sizes.get(model_size, 128)


def get_device_info(device: torch.device) -> dict:
    """
    Get information about the device.
    
    Args:
        device: The device to get info for
        
    Returns:
        dict: Device information
    """
    info = {
        'type': device.type,
        'index': device.index if device.index is not None else 0
    }
    
    if device.type == 'mps':
        info['name'] = 'Apple Silicon GPU (MPS)'
        info['memory'] = 'Unified Memory (shared with CPU)'
    elif device.type == 'cuda':
        info['name'] = f'NVIDIA GPU {device.index}'
        info['memory'] = f'{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB'
    else:
        info['name'] = 'CPU'
        info['memory'] = 'System RAM'
    
    return info


def print_device_info(device: torch.device):
    """Print device information."""
    info = get_device_info(device)
    print(f"Using device: {info['name']}")
    print(f"Device type: {info['type']}")
    print(f"Memory: {info['memory']}")
    
    if device.type == 'mps':
        print("Note: MPS (Metal Performance Shaders) provides GPU acceleration on Apple Silicon")
    elif device.type == 'cuda':
        print(f"CUDA device {device.index}: {info['name']}")
    else:
        print("Training on CPU - consider using GPU for faster training")

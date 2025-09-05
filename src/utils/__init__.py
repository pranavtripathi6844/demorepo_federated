"""
Utility functions for the federated learning project.
"""

from .device_utils import get_device, is_mps_available

__all__ = ['get_device', 'is_mps_available']

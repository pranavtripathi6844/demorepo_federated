#!/usr/bin/env python3
"""
Debug script to check checkpoint structure
"""

import torch
import os

# Check what's actually in the checkpoints
checkpoints = [
    'federated_nc1_j4_final.pth',
    'federated_nc5_j4_final.pth', 
    'federated_nc10_j4_final.pth',
    'federated_nc50_j4_final.pth',
    'federated_nc100_j4_final.pth'
]

print("Checking checkpoint structure...")
for ckpt in checkpoints:
    path = f"./checkpoints/{ckpt}"
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            print(f"\n{ckpt}:")
            print(f"  Keys: {list(checkpoint.keys())}")
            if 'model_state' in checkpoint:
                print(f"  model_state type: {type(checkpoint['model_state'])}")
                print(f"  model_state keys: {list(checkpoint['model_state'].keys())[:5]}...")
            elif 'model_state_dict' in checkpoint:
                print(f"  model_state_dict type: {type(checkpoint['model_state_dict'])}")
                print(f"  model_state_dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")
            else:
                print(f"  Direct state dict type: {type(checkpoint)}")
                print(f"  Direct state dict keys: {list(checkpoint.keys())[:5]}...")
        except Exception as e:
            print(f"{ckpt}: ERROR - {e}")
    else:
        print(f"{ckpt}: FILE NOT FOUND")


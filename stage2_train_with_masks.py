#!/usr/bin/env python3
"""
Stage 2: Train models with pre-computed masks.
This script loads pre-computed masks and trains models using SparseSGDM.
"""

import torch
import argparse
import os
import time
from typing import Dict, List, Any

# Import project modules
from src.models.vision_transformer import LinearFlexibleDino  # Changed from DINOBackboneClassifier
from src.training.sparse_optimizer import SparseSGDWithMomentum
from src.training.centralized_training import train_centralized_model
from src.data.dataset_loader import CIFAR100DataManager


def load_stage1_model(model_path: str = "models/stage1_model.pth") -> LinearFlexibleDino:
    """
    Load the model saved from Stage 1.
    
    Args:
        model_path: Path to the saved model from Stage 1
        
    Returns:
        Loaded LinearFlexibleDino model
    """
    print(f"Loading Stage 1 model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stage 1 model not found at {model_path}. Run stage1_compute_masks.py first.")
    
    # Load the saved model data
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {
        'num_classes': 100,
        'num_layers_to_freeze': 12,
        'model_type': 'LinearFlexibleDino'
    })
    
    # Create model with saved configuration
    model = LinearFlexibleDino(
        num_classes=model_config['num_classes'],
        num_layers_to_freeze=model_config.get('num_layers_to_freeze', 12)
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Stage 1 model loaded successfully")
    print(f"  - Model configuration: {model_config}")
    
    return model


def train_centralized_with_mask(mask_file: str, 
                               learning_rate: float = 0.005, 
                               weight_decay: float = 0.001, 
                               momentum: float = 0.8, 
                               num_epochs: int = 30, 
                               device: str = 'cuda',
                               scheduler_type: str = 'cosine',
                               model_path: str = 'models/stage1_model.pth') -> Dict[str, Any]:
    """
    Train centralized model with pre-computed mask using SparseSGDM.
    
    Args:
        mask_file: Path to the pre-computed mask file
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        momentum: Momentum factor for SparseSGDM
        num_epochs: Number of training epochs
        device: Device to use for training
        scheduler_type: Type of learning rate scheduler
        
    Returns:
        Training history dictionary
    """
    print(f"=== Training Centralized with Mask: {mask_file} ===")
    
    # Load model from Stage 1
    model = load_stage1_model(model_path)
    
    # Load mask
    print(f"Loading mask from {mask_file}...")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    mask = torch.load(mask_file)
    
    # Calculate mask statistics
    total_params = sum(m.numel() for m in mask.values())
    pruned_params = sum((m == 0).sum().item() for m in mask.values())
    sparsity = pruned_params / total_params
    
    print(f"Mask statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Pruned parameters: {pruned_params:,}")
    print(f"  - Sparsity: {sparsity:.4f} ({sparsity*100:.1f}%)")
    print(f"  - Active parameters: {total_params - pruned_params:,}")
    
    # Create SparseSGDM optimizer
    print("Creating SparseSGDM optimizer...")
    optimizer = SparseSGDWithMomentum(
        parameters=model.named_parameters(),
        parameter_masks=mask,
        learning_rate=learning_rate,
        momentum_factor=momentum,
        weight_decay_factor=weight_decay
    )
    
    print(f"Optimizer configuration:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Momentum: {momentum}")
    print(f"  - Scheduler: {scheduler_type}")
    
    # Train model
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    
    training_history = train_centralized_model(
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        scheduler_type=scheduler_type,
        checkpoint_dir="./checkpoints",
        model_name=f"centralized_masked_{os.path.basename(mask_file).replace('.pth', '')}"
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Centralized training completed!")
    print(f"Training time: {training_time:.2f}s")
    print(f"Final training accuracy: {training_history['train_accuracy'][-1]:.2f}%")
    print(f"Final validation accuracy: {training_history['val_accuracy'][-1]:.2f}%")
    
    return training_history


def train_federated_with_masks(mask_file: str,
                              learning_rate: float = 0.005,
                              weight_decay: float = 0.001,
                              momentum: float = 0.8,
                              num_rounds: int = 400,
                              num_clients: int = 100,
                              client_fraction: float = 0.1,
                              num_client_steps: int = 4,
                              device: str = 'cuda',
                              model_path: str = 'models/stage1_model.pth') -> Dict[str, Any]:
    """
    Train federated model with pre-computed client masks using SparseSGDM.
    
    Args:
        mask_file: Path to the pre-computed client masks file
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        momentum: Momentum factor for SparseSGDM
        num_rounds: Number of federated learning rounds
        num_clients: Total number of clients
        client_fraction: Fraction of clients participating per round
        num_client_steps: Number of local training steps per client
        device: Device to use for training
        
    Returns:
        Training result dictionary
    """
    print(f"=== Training Federated with Masks: {mask_file} ===")
    
    # Load model from Stage 1
    model = load_stage1_model(model_path)
    
    # Load client masks
    print(f"Loading client masks from {mask_file}...")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    client_masks = torch.load(mask_file)
    
    # Calculate mask statistics
    if len(client_masks) > 0:
        total_params = sum(m.numel() for m in client_masks[0].values())
        avg_pruned_params = 0
        for client_mask in client_masks:
            pruned_params = sum((m == 0).sum().item() for m in client_mask.values())
            avg_pruned_params += pruned_params
        avg_pruned_params /= len(client_masks)
        avg_sparsity = avg_pruned_params / total_params
        
        print(f"Client mask statistics:")
        print(f"  - Number of clients: {len(client_masks)}")
        print(f"  - Total parameters per client: {total_params:,}")
        print(f"  - Average pruned parameters: {avg_pruned_params:.0f}")
        print(f"  - Average sparsity: {avg_sparsity:.4f} ({avg_sparsity*100:.1f}%)")
        print(f"  - Average active parameters: {total_params - avg_pruned_params:.0f}")
    
    # Load data and create client datasets
    print("Loading CIFAR-100 dataset and creating client splits...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_data_loaders()
    
    # Extract Nc from filename (e.g., federated_masks_nc5_R5_sz0.01.pth -> nc=5)
    filename = os.path.basename(mask_file)
    if 'nc' in filename:
        nc_start = filename.find('nc') + 2
        nc_end = filename.find('_', nc_start)
        if nc_end == -1:
            nc_end = filename.find('.', nc_start)
        nc = int(filename[nc_start:nc_end])
    else:
        nc = 100  # Default to IID
    
    print(f"Creating non-IID client datasets (Nc={nc})...")
    client_datasets = data_manager.create_non_iid_splits(
        train_loader.dataset,
        num_clients=num_clients,
        classes_per_client=nc
    )
    
    print(f"Federated learning configuration:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Momentum: {momentum}")
    print(f"  - Communication rounds: {num_rounds}")
    print(f"  - Clients per round: {int(num_clients * client_fraction)}")
    print(f"  - Local steps per client: {num_client_steps}")
    print(f"  - Classes per client (Nc): {nc}")
    
    # Import federated training function
    from src.training.federated_training import train_federated_model_editing
    
    # Prepare optimizer config
    optimizer_config = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum
    }
    
    # Load validation data
    _, val_loader, _ = data_manager.get_centralized_loaders()
    
    # Train federated model
    print(f"\nStarting federated training for {num_rounds} rounds...")
    start_time = time.time()
    
    result = train_federated_model_editing(
        model=model,
        client_datasets=client_datasets,
        client_masks=client_masks,
        optimizer_config=optimizer_config,
        device=device,
        num_rounds=num_rounds,
        num_clients=num_clients,
        client_fraction=client_fraction,
        num_client_steps=num_client_steps,
        validation_loader=val_loader,
        checkpoint_path="./checkpoints",
        model_name=f"federated_masked_{os.path.basename(mask_file).replace('.pth', '')}"
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Federated training completed!")
    print(f"Training time: {training_time:.2f}s")
    if 'val_accuracies' in result and len(result['val_accuracies']) > 0:
        print(f"Final validation accuracy: {result['val_accuracies'][-1]:.2f}%")
    if 'train_losses' in result and len(result['train_losses']) > 0:
        print(f"Final training loss: {result['train_losses'][-1]:.4f}")
    
    return result


def list_available_masks():
    """List all available mask files."""
    print("=== Available Mask Files ===")
    
    if not os.path.exists("masks"):
        print("No masks directory found. Run stage1_compute_masks.py first.")
        return
    
    mask_files = [f for f in os.listdir("masks") if f.endswith('.pth')]
    
    if not mask_files:
        print("No mask files found. Run stage1_compute_masks.py first.")
        return
    
    centralized_masks = [f for f in mask_files if f.startswith('centralized_')]
    federated_masks = [f for f in mask_files if f.startswith('federated_')]
    
    print(f"Centralized masks ({len(centralized_masks)}):")
    for mask_file in sorted(centralized_masks):
        print(f"  - masks/{mask_file}")
    
    print(f"\nFederated masks ({len(federated_masks)}):")
    for mask_file in sorted(federated_masks):
        print(f"  - masks/{mask_file}")
    
    print(f"\nTotal: {len(mask_files)} mask files available")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Stage 2: Train with pre-computed masks")
    parser.add_argument('--mask_file', type=str, required=True,
                       help='Path to mask file (e.g., masks/centralized_mask_R5_sz0.01.pth)')
    parser.add_argument('--federated', action='store_true',
                       help='Use federated training instead of centralized')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='Learning rate (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.8,
                       help='Momentum factor (default: 0.8)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs for centralized training (default: 30)')
    parser.add_argument('--num_rounds', type=int, default=400,
                       help='Number of rounds for federated training (default: 400)')
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients for federated training (default: 100)')
    parser.add_argument('--client_fraction', type=float, default=0.1,
                       help='Fraction of clients per round (default: 0.1)')
    parser.add_argument('--num_client_steps', type=int, default=4,
                       help='Local steps per client (default: 4)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler for centralized training (default: cosine)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--model_path', type=str, default='models/stage1_model.pth',
                       help='Path to Stage 1 model (default: models/stage1_model.pth)')
    parser.add_argument('--list_masks', action='store_true',
                       help='List available mask files and exit')
    
    args = parser.parse_args()
    
    # List masks if requested
    if args.list_masks:
        list_available_masks()
        return
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU.")
        args.device = 'cpu'
    
    print("TALOS 2-Stage Mask Calibration - Stage 2: Training with Masks")
    print("=" * 60)
    
    try:
        if args.federated:
            result = train_federated_with_masks(
                mask_file=args.mask_file,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                num_rounds=args.num_rounds,
                num_clients=args.num_clients,
                client_fraction=args.client_fraction,
                num_client_steps=args.num_client_steps,
                device=args.device,
                model_path=args.model_path
            )
        else:
            result = train_centralized_with_mask(
                mask_file=args.mask_file,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                num_epochs=args.num_epochs,
                device=args.device,
                scheduler_type=args.scheduler,
                model_path=args.model_path
            )
        
        print("\n" + "=" * 60)
        print("✅ Stage 2 Complete: Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()

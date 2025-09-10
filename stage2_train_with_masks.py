#!/usr/bin/env python3
"""
Stage 2: Train models with pre-computed masks.
This script loads pre-computed masks and trains models using SparseSGDM with TaLoS constraints.
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
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits


def load_stage1_model(model_path: str = "models/stage1_model.pth") -> LinearFlexibleDino:
    """
    Load the model saved from Stage 1 or use a pre-trained federated baseline.
    This ensures we start with a model that has a trained federated head.
    
    Args:
        model_path: Path to the saved model from Stage 1
        
    Returns:
        Loaded LinearFlexibleDino model with trained federated head
    """
    print(f"Loading Stage 1 model from {model_path}...")
    
    # Try to load Stage 1 model first
    if os.path.exists(model_path):
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
    
    # If Stage 1 model doesn't exist, try to load a baseline federated checkpoint
    print(f"Stage 1 model not found at {model_path}")
    print("Looking for baseline federated checkpoint...")
    
    # Look for the best baseline federated checkpoint
    baseline_checkpoints = [
        "checkpoints/baseline_fedavg_round_600.pth",
        "checkpoints/baseline_fedavg_round_550.pth", 
        "checkpoints/baseline_fedavg_round_500.pth",
        "checkpoints/baseline_fedavg_round_450.pth",
        "checkpoints/baseline_fedavg_round_400.pth"
    ]
    
    for checkpoint_path in baseline_checkpoints:
        if os.path.exists(checkpoint_path):
            print(f"Loading baseline federated checkpoint: {checkpoint_path}")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Create model (assuming standard configuration)
                model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
                
                # Load the trained federated weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                print(f"✓ Baseline federated model loaded successfully")
                print(f"  - Using checkpoint: {checkpoint_path}")
                print(f"  - This model has a trained federated head!")
                
                return model
                
            except Exception as e:
                print(f"Failed to load {checkpoint_path}: {e}")
                continue
    
    # Fallback: create fresh model
    print("Warning: No pre-trained model found. Creating fresh LinearFlexibleDino.")
    print("This model will have an untrained head - not ideal for federated TaLoS.")
    
    model = LinearFlexibleDino(num_classes=100, num_layers_to_freeze=12)
    model.eval()
    
    return model


def configure_talos_training(model: LinearFlexibleDino, device: str = 'cuda') -> None:
    """
    Configure model for TaLoS training (matching working implementation):
    - Unfreeze backbone blocks globally
    - Keep classification head TRAINABLE (critical for federated learning)
    - Freeze embeddings, cls_token, backbone final norm only
    - Set requires_grad appropriately
    """
    print("Configuring model for TaLoS training...")
    
    # Move model to device first
    model = model.to(device)
    
    # Unfreeze backbone blocks AND classification head (like working implementation)
    print("  - Unfreezing backbone blocks and classification head")
    for name, param in model.named_parameters():
        if 'backbone.blocks' in name:
            param.requires_grad = True
            print(f"    {name}: requires_grad=True (backbone)")
        elif 'head' in name:
            # Keep classification head TRAINABLE (critical fix!)
            param.requires_grad = True
            print(f"    {name}: requires_grad=True (head - trainable)")
        elif any(x in name for x in ['patch_embed', 'pos_embed', 'cls_token', 'backbone.norm']):
            # Freeze embeddings, cls_token, and backbone final norm
            param.requires_grad = False
            print(f"    {name}: requires_grad=False (frozen)")
        else:
            # Other parameters (shouldn't exist in LinearFlexibleDino)
            param.requires_grad = False
            print(f"    {name}: requires_grad=False (other)")
    
    print("✓ TaLoS configuration complete - Head is now TRAINABLE")


def apply_soft_zero_masking(model: LinearFlexibleDino, 
                           mask: Dict[str, torch.Tensor], 
                           soft_zero_value: float = 0.01,
                           device: str = 'cuda') -> None:
    """
    Apply soft-zero masking to model parameters.
    This enforces the TaLoS constraint by setting masked weights to soft_zero_value.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask and 'backbone.blocks' in name:
                # Move mask to same device as parameter
                mask_tensor = mask[name].to(param.device)
                
                # Create soft-zero mask: 0 -> soft_zero_value, 1 -> 1.0
                soft_mask = mask_tensor.clone()
                soft_mask[soft_mask == 0] = soft_zero_value
                
                # Apply soft masking
                param.data.mul_(soft_mask)
                print(f"  Applied soft-zero masking to {name}: {soft_zero_value} for pruned weights")


def train_centralized_with_mask(mask_file: str, 
                               learning_rate: float = 0.005, 
                               weight_decay: float = 0.001, 
                               momentum: float = 0.8, 
                               num_epochs: int = 30, 
                               device: str = 'cuda',
                               scheduler_type: str = 'cosine',
                               model_path: str = 'models/stage1_model.pth',
                               soft_zero_value: float = 0.01) -> Dict[str, Any]:
    """
    Train centralized model with pre-computed mask using SparseSGDM with TaLoS constraints.
    """
    print(f"=== Training Centralized with TaLoS Mask: {mask_file} ===")
    
    # Load model from Stage 1
    model = load_stage1_model(model_path)
    
    # Configure for TaLoS training
    configure_talos_training(model, device)
    
    # Load mask
    print(f"Loading mask from {mask_file}...")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    mask = torch.load(mask_file)
    
    # Apply soft-zero masking to enforce TaLoS constraints
    print("Applying soft-zero masking...")
    apply_soft_zero_masking(model, mask, soft_zero_value, device)
    
    # Create a combined mask: backbone uses the computed mask, head uses all-ones mask
    print("Creating combined mask for SparseSGDM...")
    combined_mask = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'head' in name:
                # Head gets all-ones mask (no pruning) - ensure same device as param
                combined_mask[name] = torch.ones_like(param, device=param.device)
                print(f"  Head parameter {name}: all-ones mask ({param.numel()} params)")
            elif 'backbone.blocks' in name and name in mask:
                # Backbone gets the computed mask - move to same device as param
                combined_mask[name] = mask[name].to(param.device)
                active_params = (mask[name] == 1).sum().item()
                total_params = mask[name].numel()
                print(f"  Backbone parameter {name}: pruned mask ({active_params}/{total_params} active)")
            else:
                # Other trainable parameters get all-ones mask - ensure same device as param
                combined_mask[name] = torch.ones_like(param, device=param.device)
                print(f"  Other parameter {name}: all-ones mask ({param.numel()} params)")
    
    # CRITICAL FIX: Move all masks to the selected device before creating optimizer
    print(f"Moving all masks to {device}...")
    for name in combined_mask:
        combined_mask[name] = combined_mask[name].to(device)
        print(f"  Moved {name} mask to {device}")
    
    # Calculate mask statistics
    total_params = sum(m.numel() for m in combined_mask.values())
    pruned_params = sum((m == 0).sum().item() for m in combined_mask.values())
    sparsity = pruned_params / total_params if total_params > 0 else 0
    
    print(f"Combined mask statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Pruned parameters: {pruned_params:,}")
    print(f"  - Overall sparsity: {sparsity:.4f} ({sparsity*100:.1f}%)")
    print(f"  - Active parameters: {total_params - pruned_params:,}")
    
    # Create SparseSGDM optimizer with combined mask
    print("Creating SparseSGDM optimizer...")
    optimizer = SparseSGDWithMomentum(
        parameters=model.named_parameters(),
        parameter_masks=combined_mask,
        learning_rate=learning_rate,
        momentum_factor=momentum,
        weight_decay_factor=weight_decay
    )
    
    print(f"Optimizer configuration:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Momentum: {momentum}")
    print(f"  - Scheduler: {scheduler_type}")
    print(f"  - Soft zero value: {soft_zero_value}")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    data_manager = CIFAR100DataManager()
    train_loader, val_loader, test_loader = data_manager.get_centralized_loaders(val_split=0.2)
    
    # Move model to device
    model = model.to(device)
    
    # Create scheduler
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    print(f"\nStarting TaLoS training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Note: Soft-zero masking is handled by SparseSGDM optimizer, not per-step reapplication
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Store history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {learning_rates[-1]:.6f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Create training history
    training_history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'learning_rates': learning_rates,
        'epochs': list(range(1, num_epochs + 1)),
        'config': {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'scheduler': scheduler_type,
            'sparsity': sparsity,
            'soft_zero_value': soft_zero_value
        }
    }
    
    # Save training history
    model_name = f"centralized_talos_{os.path.basename(mask_file).replace('.pth', '')}"
    history_file = f"checkpoints/{model_name}_epoch_{num_epochs}.json"
    os.makedirs("checkpoints", exist_ok=True)
    
    import json
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save model
    model_file = f"checkpoints/{model_name}_epoch_{num_epochs}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history,
        'config': training_history['config']
    }, model_file)
    
    print(f"\n✅ Centralized TaLoS training completed!")
    print(f"Training time: {training_time:.2f}s")
    print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    print(f"Training history saved: {history_file}")
    print(f"Model saved: {model_file}")
    
    return training_history


def train_federated_with_masks(mask_file: str,
                              learning_rate: float = 0.005,
                              weight_decay: float = 0.001,
                              momentum: float = 0.8,
                              num_rounds: int = 300,  # Changed from 400 to 300 as per requirements
                              num_clients: int = 100,
                              client_fraction: float = 0.1,
                              num_client_steps: int = 4,
                              device: str = 'cuda',
                              model_path: str = 'models/stage1_model.pth',
                              soft_zero_value: float = 0.01) -> Dict[str, Any]:
    """
    Train federated model with pre-computed client masks using SparseSGDM with TaLoS constraints.
    
    Args:
        mask_file: Path to the pre-computed client masks file
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        momentum: Momentum factor for SparseSGDM
        num_rounds: Number of federated learning rounds (300 as per requirements)
        num_clients: Total number of clients
        client_fraction: Fraction of clients participating per round
        num_client_steps: Number of local training steps per client
        device: Device to use for training
        soft_zero_value: Soft zero value for TaLoS constraints
        
    Returns:
        Training result dictionary
    """
    print(f"=== Training Federated with TaLoS Masks: {mask_file} ===")
    
    # Load model from Stage 1
    model = load_stage1_model(model_path)
    
    # Configure for TaLoS training
    configure_talos_training(model, device)
    
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
    else:
        raise ValueError("Loaded client_masks is empty; cannot proceed.")

    # Option A: use number of masks to set num_clients
    num_clients = len(client_masks)
    print(f"Using number of masks to set num_clients = {num_clients}")
    
    # Load data and create client datasets
    print("Loading CIFAR-100 dataset and creating client splits...")
    data_manager = CIFAR100DataManager()
    train_loader, _, _ = data_manager.get_centralized_loaders(val_split=0.2)
    
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
    client_datasets = create_non_iid_splits(
        train_loader.dataset,
        num_clients=num_clients,
        classes_per_client=nc
    )
    
    print(f"Federated TaLoS learning configuration:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Momentum: {momentum}")
    print(f"  - Communication rounds: {num_rounds}")
    print(f"  - Clients per round: {int(num_clients * client_fraction)}")
    print(f"  - Local steps per client: {num_client_steps}")
    print(f"  - Classes per client (Nc): {nc}")
    print(f"  - Soft zero value: {soft_zero_value}")
    
    # Import federated training function
    from src.training.federated_training import train_federated_model_editing_talos
    
    # Prepare optimizer config
    optimizer_config = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'soft_zero_value': soft_zero_value
    }
    
    # Load validation data
    _, val_loader, _ = data_manager.get_centralized_loaders()
    
    # Train federated model
    print(f"\nStarting federated TaLoS training for {num_rounds} rounds...")
    start_time = time.time()
    
    result = train_federated_model_editing_talos(
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
        model_name=f"federated_talos_{os.path.basename(mask_file).replace('.pth', '')}"
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Federated TaLoS training completed!")
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
    parser = argparse.ArgumentParser(description="Stage 2: Train with pre-computed TaLoS masks")
    parser.add_argument('--mask_file', type=str, required=True,
                       help='Path to mask file (e.g., masks/federated_masks_nc1_R3_sz0.01.pth)')
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
    parser.add_argument('--num_rounds', type=int, default=300,
                       help='Number of rounds for federated training (default: 300)')
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
    parser.add_argument('--soft_zero_value', type=float, default=0.01,
                       help='Soft zero value for TaLoS constraints (default: 0.01)')
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
    
    print("TALOS 2-Stage Mask Calibration - Stage 2: TaLoS Training with Masks")
    print("=" * 70)
    
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
                model_path=args.model_path,
                soft_zero_value=args.soft_zero_value
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
                model_path=args.model_path,
                soft_zero_value=args.soft_zero_value
            )
        
        print("\n" + "=" * 70)
        print("✅ Stage 2 Complete: TaLoS training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()

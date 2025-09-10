"""
Federated learning training implementation with model editing capabilities.
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os

from src.training.sparse_optimizer import SparseSGDWithMomentum
from src.utils.checkpoint_manager import CheckpointManager


class FederatedTrainer:
    """
    Handles federated learning training with model editing capabilities.
    """
    
    def __init__(self, model: nn.Module, 
                 client_datasets: List[torch.utils.data.Dataset],
                 client_masks: List[Dict[str, torch.Tensor]],
                 optimizer_config: Dict[str, Any],
                 device: str = 'cuda',
                 num_clients: int = 100,
                 client_fraction: float = 0.1,
                 num_client_steps: int = 4,
                 batch_size: int = 64):
        """
        Initialize the federated trainer.
        
        Args:
            model: Global model to train
            client_datasets: List of client datasets
            client_masks: List of client masks for model editing
            optimizer_config: Optimizer configuration
            device: Device to train on
            num_clients: Total number of clients
            client_fraction: Fraction of clients to select per round
            num_client_steps: Number of local training steps per client
            batch_size: Batch size for training
        """
        # Ensure device is set first, then move model to that device
        self.device = device
        self.global_model = model.to(self.device)
        self.client_datasets = client_datasets
        self.client_masks = client_masks
        self.optimizer_config = optimizer_config
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.num_client_steps = num_client_steps
        self.batch_size = batch_size
        
        self.checkpoint_manager = CheckpointManager()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def train_federated_rounds(self, num_rounds: int, 
                              validation_loader: DataLoader,
                              checkpoint_path: str,
                              log_interval: int = 5,
                              model_name: str = "federated_model",
                              resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the federated model for multiple rounds.
        
        Args:
            num_rounds: Number of federated training rounds
            validation_loader: Data loader for validation
            checkpoint_path: Path to save checkpoints
            log_interval: Interval for logging metrics
            model_name: Name for the model
            
        Returns:
            Dictionary containing training history
        """
        training_history = {
            'round_losses': [],
            'round_accuracies': [],
            'validation_losses': [],
            'validation_accuracies': [],
            'selected_clients': []
        }
        
        # Initialize wandb if available
        try:
            wandb.init(
                project="FederatedLearning_ModelEditing",
                name=f"{model_name}_federated",
                config={
                    "model": model_name,
                    "num_rounds": num_rounds,
                    "num_clients": self.num_clients,
                    "client_fraction": self.client_fraction,
                    "num_client_steps": self.num_client_steps,
                    "batch_size": self.batch_size,
                    **self.optimizer_config
                }
            )
        except Exception as e:
            print(f"Wandb initialization failed: {e}")
            wandb.init = lambda **kwargs: None
        
        # Resume support
        os.makedirs(checkpoint_path, exist_ok=True)
        start_round = 0
        if resume_from is not None and os.path.exists(resume_from):
            try:
                payload = torch.load(resume_from, map_location='cpu')
                if isinstance(payload, dict):
                    if 'model_state' in payload:
                        self.global_model.load_state_dict(payload['model_state'])
                    if 'history' in payload and isinstance(payload['history'], dict):
                        training_history = payload['history']
                    start_round = int(payload.get('round', 0))
                    print(f"Resuming from checkpoint: {resume_from} at round {start_round}")
            except Exception as e:
                print(f"Failed to resume from {resume_from}: {e}")

        for round_num in range(start_round, num_rounds):
            print(f"\n--- Federated Round {round_num + 1}/{num_rounds} ---")
            
            # Select clients for this round
            selected_clients = self._select_clients()
            training_history['selected_clients'].append(selected_clients)
            
            # Train selected clients
            client_models, client_metrics, client_sizes = self._train_selected_clients(selected_clients)
            
            # Aggregate client updates (weighted FedAvg by client size)
            self._aggregate_client_updates(client_models, client_sizes)
            
            # Evaluate global model
            val_loss, val_accuracy = self._evaluate_global_model(validation_loader)
            
            # Log metrics
            avg_client_loss = np.mean([m['loss'] for m in client_metrics])
            avg_client_accuracy = np.mean([m['accuracy'] for m in client_metrics])
            
            training_history['round_losses'].append(avg_client_loss)
            training_history['round_accuracies'].append(avg_client_accuracy)
            training_history['validation_losses'].append(val_loss)
            training_history['validation_accuracies'].append(val_accuracy)
            
            # Log to wandb
            wandb.log({
                "round": round_num + 1,
                "client_avg_loss": avg_client_loss,
                "client_avg_accuracy": avg_client_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy
            })
            
            # Save checkpoint
            if (round_num + 1) % log_interval == 0:
                self.checkpoint_manager.save_federated_checkpoint(
                    self.global_model, round_num + 1, client_metrics, 
                    f"{model_name}_round_{round_num + 1}.pth"
                )
                # Also save a self-contained payload for robust resume
                self._save_resume_checkpoint(
                    checkpoint_dir=checkpoint_path,
                    model_name=model_name,
                    round_completed=round_num + 1,
                    history=training_history
                )
            
            print(f"Round {round_num + 1} - Client Loss: {avg_client_loss:.4f}, "
                  f"Client Acc: {avg_client_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        return training_history
    
    def _select_clients(self) -> List[int]:
        """Select clients for the current round."""
        num_selected = max(int(self.num_clients * self.client_fraction), 1)
        return np.random.choice(range(self.num_clients), num_selected, replace=False)
    
    def _train_selected_clients(self, selected_clients: List[int]) -> Tuple[List[Dict], List[Dict], List[int]]:
        """Train the selected clients and return their models, metrics, and client sizes."""
        client_models = []
        client_metrics = []
        client_sizes = []
        
        for client_idx in selected_clients:
            # Create data loader for this client
            client_loader = DataLoader(
                self.client_datasets[client_idx], 
                batch_size=self.batch_size, 
                shuffle=True
            )
            
            # Train client
            client_model, client_loss, client_accuracy = self._train_single_client(
                client_idx, client_loader
            )
            
            client_models.append(client_model)
            client_metrics.append({
                'client_id': client_idx,
                'loss': client_loss,
                'accuracy': client_accuracy
            })
            client_sizes.append(len(self.client_datasets[client_idx]))
        
        return client_models, client_metrics, client_sizes
    
    def _train_single_client(self, client_idx: int, 
                            client_loader: DataLoader) -> Tuple[Dict, float, float]:
        """Train a single client and return the updated model and metrics."""
        # Create local model copy
        local_model = copy.deepcopy(self.global_model).to(self.device)
        
        # Get client mask
        client_mask = self.client_masks[client_idx]
        
        # Create sparse optimizer
        local_optimizer = SparseSGDWithMomentum(
            local_model.named_parameters(),
            client_mask,
            learning_rate=self.optimizer_config['lr'],
            momentum_factor=self.optimizer_config.get('momentum', 0.9),
            weight_decay_factor=self.optimizer_config.get('weight_decay', 0.0001)
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        local_model.train()
        accumulated_loss = 0.0
        accumulated_correct = 0
        accumulated_total = 0
        
        # Create data iterator for multiple steps
        data_iterator = iter(client_loader)
        
        for step in range(self.num_client_steps):
            try:
                images, labels = next(data_iterator)
            except StopIteration:
                # Restart iterator if we run out of data
                data_iterator = iter(client_loader)
                images, labels = next(data_iterator)
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            local_optimizer.zero_grad()
            loss.backward()
            local_optimizer.step()
            
            # Accumulate metrics
            accumulated_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            accumulated_total += labels.size(0)
            accumulated_correct += (predicted == labels).sum().item()
        
        # Calculate final metrics
        final_loss = accumulated_loss / accumulated_total if accumulated_total > 0 else 0.0
        final_accuracy = 100 * accumulated_correct / accumulated_total if accumulated_total > 0 else 0.0
        
        return local_model.state_dict(), final_loss, final_accuracy
    
    def _aggregate_client_updates(self, client_models: List[Dict], client_sizes: List[int]):
        """Aggregate client model updates using weighted FedAvg (by client dataset size)."""
        num_clients = len(client_models)
        if num_clients == 0:
            return
        
        # Normalize weights
        weights = torch.tensor(client_sizes, dtype=torch.float32)
        weights = weights / weights.sum()
        
        aggregated_state = {}
        model_keys = client_models[0].keys()
        
        for key in model_keys:
            stacked = torch.stack([client_models[i][key] for i in range(num_clients)])
            # Broadcast weights over param dims
            view = (num_clients,) + tuple([1] * (stacked.dim() - 1))
            weighted = (stacked * weights.view(view)).sum(dim=0)
            aggregated_state[key] = weighted
        
        self.global_model.load_state_dict(aggregated_state)
    
    def _evaluate_global_model(self, validation_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the global model on validation data."""
        # Make sure the global model is on the correct device before evaluation
        self.global_model.to(self.device)
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    def _save_resume_checkpoint(self,
                                checkpoint_dir: str,
                                model_name: str,
                                round_completed: int,
                                history: Dict[str, List[float]]) -> None:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            filename = f"{model_name}_round_{round_completed}.pth"
            path = os.path.join(checkpoint_dir, filename)
            payload = {
                'round': round_completed,
                'model_state': self.global_model.state_dict(),
                'history': history
            }
            torch.save(payload, path)
            print(f"Saved resume checkpoint: {path}")
        except Exception as e:
            print(f"Failed to save resume checkpoint: {e}")


def train_federated_model_editing(model: nn.Module,
                                client_datasets: List[torch.utils.data.Dataset],
                                client_masks: List[Dict[str, torch.Tensor]],
                                optimizer_config: Dict[str, Any],
                                device: str = 'cuda',
                                num_rounds: int = 100,
                                num_clients: int = 100,
                                client_fraction: float = 0.1,
                                num_client_steps: int = 4,
                                batch_size: int = 64,
                                validation_loader: Optional[DataLoader] = None,
                                checkpoint_path: str = "./checkpoints",
                                model_name: str = "federated_model",
                                checkpoint_interval: int = 50,
                                resume_from: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Train a federated model with model editing capabilities.
    
    Args:
        model: Global model to train
        client_datasets: List of client datasets
        client_masks: List of client masks
        optimizer_config: Optimizer configuration
        device: Device to train on
        num_rounds: Number of federated rounds
        num_clients: Total number of clients
        client_fraction: Fraction of clients per round
        num_client_steps: Local training steps per client
        batch_size: Training batch size
        validation_loader: Validation data loader
        checkpoint_path: Path for checkpoints
        model_name: Name for the model
        
    Returns:
        Training history dictionary
    """
    # Create federated trainer
    trainer = FederatedTrainer(
        model=model,
        client_datasets=client_datasets,
        client_masks=client_masks,
        optimizer_config=optimizer_config,
        device=device,
        num_clients=num_clients,
        client_fraction=client_fraction,
        num_client_steps=num_client_steps,
        batch_size=batch_size
    )
    
    # Train the model
    training_history = trainer.train_federated_rounds(
        num_rounds=num_rounds,
        validation_loader=validation_loader,
        checkpoint_path=checkpoint_path,
        log_interval=checkpoint_interval,
        model_name=model_name,
        resume_from=resume_from
    )
    
    return training_history


def train_federated_model_editing_talos(model: nn.Module,
                                       client_datasets: List[torch.utils.data.Dataset],
                                       client_masks: List[Dict[str, torch.Tensor]],
                                       optimizer_config: Dict[str, Any],
                                       device: str = 'cuda',
                                       num_rounds: int = 300,
                                       num_clients: int = 100,
                                       client_fraction: float = 0.1,
                                       num_client_steps: int = 4,
                                       batch_size: int = 64,
                                       validation_loader: Optional[DataLoader] = None,
                                       checkpoint_path: str = "./checkpoints",
                                       model_name: str = "federated_talos_model",
                                       checkpoint_interval: int = 50,
                                       resume_from: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Train a federated model with TaLoS constraints and model editing capabilities.
    
    Args:
        model: Global model to train
        client_datasets: List of client datasets
        client_masks: List of client masks
        optimizer_config: Optimizer configuration (including soft_zero_value)
        device: Device to train on
        num_rounds: Number of federated rounds (300 as per requirements)
        num_clients: Total number of clients
        client_fraction: Fraction of clients per round
        num_client_steps: Local training steps per client
        batch_size: Training batch size
        validation_loader: Validation data loader
        checkpoint_path: Path for checkpoints
        model_name: Name for the model
        checkpoint_interval: Interval for saving checkpoints
        resume_from: Path to resume from checkpoint
        
    Returns:
        Training history dictionary
    """
    # Create federated trainer with TaLoS constraints
    trainer = FederatedTrainerTaLoS(
        model=model,
        client_datasets=client_datasets,
        client_masks=client_masks,
        optimizer_config=optimizer_config,
        device=device,
        num_clients=num_clients,
        client_fraction=client_fraction,
        num_client_steps=num_client_steps,
        batch_size=batch_size
    )
    
    # Train the model
    training_history = trainer.train_federated_rounds(
        num_rounds=num_rounds,
        validation_loader=validation_loader,
        checkpoint_path=checkpoint_path,
        log_interval=checkpoint_interval,
        model_name=model_name,
        resume_from=resume_from
    )
    
    return training_history


class FederatedTrainerTaLoS(FederatedTrainer):
    """
    Handles federated learning training with TaLoS constraints and model editing capabilities.
    """
    
    def __init__(self, model: nn.Module, 
                 client_datasets: List[torch.utils.data.Dataset],
                 client_masks: List[Dict[str, torch.Tensor]],
                 optimizer_config: Dict[str, Any],
                 device: str = 'cuda',
                 num_clients: int = 100,
                 client_fraction: float = 0.1,
                 num_client_steps: int = 4,
                 batch_size: int = 64):
        """
        Initialize the TaLoS federated trainer.
        """
        super().__init__(model, client_datasets, client_masks, optimizer_config, 
                        device, num_clients, client_fraction, num_client_steps, batch_size)
        
        self.soft_zero_value = optimizer_config.get('soft_zero_value', 0.01)
        print(f"TaLoS Federated Trainer initialized with soft_zero_value: {self.soft_zero_value}")
    
    def _apply_soft_zero_masking(self, model: nn.Module, 
                                mask: Dict[str, torch.Tensor]) -> None:
        """
        Apply soft-zero masking to model parameters for TaLoS constraints.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask and 'backbone.blocks' in name:
                    # Move mask to same device as parameter
                    mask_tensor = mask[name].to(param.device)
                    
                    # Create soft-zero mask: 0 -> soft_zero_value, 1 -> 1.0
                    soft_mask = mask_tensor.clone()
                    soft_mask[soft_mask == 0] = self.soft_zero_value
                    
                    # Apply soft masking
                    param.data.mul_(soft_mask)
    
    def _train_single_client(self, client_idx: int, 
                            client_loader: DataLoader) -> Tuple[Dict, float, float]:
        """Train a single client with TaLoS constraints and return the updated model and metrics."""
        # Create local model copy
        local_model = copy.deepcopy(self.global_model).to(self.device)
        
        # Get client mask
        client_mask = self.client_masks[client_idx]
        
        # Create sparse optimizer
        local_optimizer = SparseSGDWithMomentum(
            local_model.named_parameters(),
            client_mask,
            learning_rate=self.optimizer_config['lr'],
            momentum_factor=self.optimizer_config.get('momentum', 0.9),
            weight_decay_factor=self.optimizer_config.get('weight_decay', 0.0001)
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop (no per-step soft-zero re-application; grad masking only)
        local_model.train()
        accumulated_loss = 0.0
        accumulated_correct = 0
        accumulated_total = 0
        
        # Create data iterator for multiple steps
        data_iterator = iter(client_loader)
        
        for step in range(self.num_client_steps):
            try:
                images, labels = next(data_iterator)
            except StopIteration:
                # Restart iterator if we run out of data
                data_iterator = iter(client_loader)
                images, labels = next(data_iterator)
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            local_optimizer.zero_grad()
            loss.backward()
            local_optimizer.step()
            
            # Accumulate metrics
            accumulated_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            accumulated_total += labels.size(0)
            accumulated_correct += (predicted == labels).sum().item()
        
        # Calculate final metrics
        final_loss = accumulated_loss / accumulated_total if accumulated_total > 0 else 0.0
        final_accuracy = 100 * accumulated_correct / accumulated_total if accumulated_total > 0 else 0.0
        
        return local_model.state_dict(), final_loss, final_accuracy
    
    def _aggregate_client_updates(self, client_models: List[Dict], client_sizes: List[int]):
        """Aggregate client model updates using weighted FedAvg (by client dataset size)."""
        num_clients = len(client_models)
        if num_clients == 0:
            return
        
        weights = torch.tensor(client_sizes, dtype=torch.float32)
        weights = weights / weights.sum()
        
        aggregated_state = {}
        model_keys = client_models[0].keys()
        for key in model_keys:
            stacked = torch.stack([client_models[i][key] for i in range(num_clients)])
            view = (num_clients,) + tuple([1] * (stacked.dim() - 1))
            weighted = (stacked * weights.view(view)).sum(dim=0)
            aggregated_state[key] = weighted
        
        self.global_model.load_state_dict(aggregated_state)
    
    def _apply_global_masking(self):
        """No-op: global model remains unmasked after FedAvg (masking only local via optimizer)."""
        return

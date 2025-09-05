"""
Centralized training implementation for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import wandb

from src.utils.checkpoint_manager import CheckpointManager


class CentralizedTrainer:
    """
    Handles centralized training of neural network models.
    """
    
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 checkpoint_dir: str = "./checkpoints"):
        """
        Initialize the centralized trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Set random seed
        torch.manual_seed(42)
        np.random.seed(42)
    
    def train_model(self, 
                   num_epochs: int,
                   learning_rate: float = 0.01,
                   weight_decay: float = 0.0001,
                   momentum: float = 0.9,
                   scheduler_type: str = 'cosine',
                   checkpoint_interval: int = 5,
                   model_name: str = "centralized_model",
                   enable_wandb: bool = True) -> Dict[str, List[float]]:
        """
        Train the model using centralized training.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay factor
            momentum: Momentum factor
            scheduler_type: Type of learning rate scheduler
            checkpoint_interval: Interval for saving checkpoints
            model_name: Name for the model
            enable_wandb: Whether to enable wandb logging
            
        Returns:
            Training history dictionary
        """
        # Initialize optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            scheduler = None
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb if enabled
        if enable_wandb:
            try:
                wandb.init(
                    project="FederatedLearning_Centralized",
                    name=f"{model_name}_training",
                    config={
                        "model": model_name,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "momentum": momentum,
                        "scheduler": scheduler_type,
                        "batch_size": self.train_loader.batch_size
                    }
                )
            except Exception as e:
                print(f"Wandb initialization failed: {e}")
                enable_wandb = False
        
        # Training loop
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Train for one epoch
            train_loss, train_accuracy = self._train_epoch(
                self.model, self.train_loader, optimizer, criterion
            )
            
            # Validate
            val_loss, val_accuracy = self._validate_epoch(
                self.model, self.val_loader, criterion
            )
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Log to wandb
            if enable_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                is_best = val_accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = val_accuracy
                
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    metrics={
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    },
                    filename=f"{model_name}_epoch_{epoch + 1}.pth",
                    is_best=is_best
                )
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def _train_epoch(self, model: nn.Module, 
                     train_loader: DataLoader,
                     optimizer: optim.Optimizer,
                     criterion: nn.Module) -> Tuple[float, float]:
        """Train the model for one epoch."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module,
                       val_loader: DataLoader,
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model for one epoch."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def test_model(self, test_loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Test the trained model.
        
        Args:
            test_loader: Test data loader (uses self.test_loader if None)
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        if test_loader is None:
            raise ValueError("No test loader provided")
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy = self._validate_epoch(
            self.model, test_loader, criterion
        )
        
        print(f"\n--- Test Results ---")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        return test_loss, test_accuracy
    
    def load_best_checkpoint(self, model_name: str = "centralized_model"):
        """Load the best checkpoint for the model."""
        try:
            epoch, checkpoint_data = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                checkpoint_path=None  # Will load best_model.pth
            )
            print(f"Loaded best checkpoint from epoch {epoch}")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False


def train_centralized_model(model: nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           test_loader: Optional[DataLoader] = None,
                           device: str = 'cuda',
                           num_epochs: int = 100,
                           learning_rate: float = 0.01,
                           weight_decay: float = 0.0001,
                           momentum: float = 0.9,
                           scheduler_type: str = 'cosine',  # ADD THIS LINE
                           checkpoint_dir: str = "./checkpoints",
                           model_name: str = "centralized_model") -> Dict[str, List[float]]:
    """
    Train a model using centralized training.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        momentum: Momentum factor
        checkpoint_dir: Directory for checkpoints
        model_name: Name for the model
        
    Returns:
        Training history dictionary
    """
    # Create trainer
    trainer = CentralizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train the model
    training_history = trainer.train_model(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        scheduler_type=scheduler_type,  # ADD THIS LINE
        model_name=model_name
    )
    
    # Test the model
    if test_loader is not None:
        trainer.test_model(test_loader)
    
    return training_history

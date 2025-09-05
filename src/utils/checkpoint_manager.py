"""
Checkpoint manager for saving and loading model checkpoints.
"""

import torch
import os
from typing import Dict, Any, Optional, Tuple


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       metrics: Dict[str, float], filename: str, 
                       is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        
        # Save checkpoint
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, model, checkpoint_path: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint['epoch'], checkpoint

"""
Sparse optimizer implementation for model editing with parameter masking.
"""

import torch
from torch.optim.optimizer import Optimizer, required
from typing import Dict, Iterable, Optional, Tuple


class SparseSGDWithMomentum(Optimizer):
    """
    Implements Stochastic Gradient Descent with Momentum and parameter masking.
    
    This optimizer extends the standard SGD optimizer to support sparse updates
    where only a subset of model parameters are updated based on a binary mask.
    """
    
    def __init__(self, parameters: Iterable[Tuple[str, torch.Tensor]], 
                 parameter_masks: Dict[str, torch.Tensor], 
                 learning_rate: float = required, 
                 momentum_factor: float = 0.0, 
                 dampening_factor: float = 0.0,
                 weight_decay_factor: float = 0.0, 
                 nesterov_acceleration: bool = False):
        """
        Initialize the sparse SGD optimizer.
        
        Args:
            parameters: Iterable of (name, parameter) tuples
            parameter_masks: Dictionary mapping parameter names to binary masks
            learning_rate: Learning rate for parameter updates
            momentum_factor: Momentum coefficient for acceleration
            dampening_factor: Dampening factor for momentum
            weight_decay_factor: L2 regularization coefficient
            nesterov_acceleration: Whether to use Nesterov momentum
        """
        # Validate input parameters
        if learning_rate is not required and learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if momentum_factor < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum_factor}")
        if weight_decay_factor < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay_factor}")
        if nesterov_acceleration and (momentum_factor <= 0 or dampening_factor != 0):
            raise ValueError("Nesterov momentum requires positive momentum and zero dampening.")
        
        # Create parameter groups with masks
        parameter_groups = []
        for param_name, param_tensor in parameters:
            if param_name in parameter_masks:
                mask_tensor = parameter_masks[param_name]
                
                # Validate mask shape
                if param_tensor.shape != mask_tensor.shape:
                    raise ValueError(
                        f"Parameter '{param_name}' shape {param_tensor.shape} "
                        f"doesn't match mask shape {mask_tensor.shape}"
                    )
                
                # Create parameter group with mask
                parameter_groups.append({
                    'params': [param_tensor], 
                    'mask': mask_tensor, 
                    'name': param_name
                })
        
        if not parameter_groups:
            raise ValueError(
                "No parameters to optimize. Ensure parameter names match mask keys."
            )
        
        # Set default hyperparameters
        defaults = {
            'lr': learning_rate, 
            'momentum': momentum_factor, 
            'dampening': dampening_factor,
            'weight_decay': weight_decay_factor, 
            'nesterov': nesterov_acceleration
        }
        
        # Initialize parent optimizer
        super().__init__(parameter_groups, defaults)
    
    def __setstate__(self, state):
        """Restore optimizer state from checkpoint."""
        super().__setstate__(state)
    
    def step(self, closure=None):
        """
        Perform a single optimization step with parameter masking.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                state = self.state[param]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                
                state['step'] += 1
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad.add(momentum_buffer, alpha=momentum)
                    else:
                        grad = momentum_buffer
                
                # Apply learning rate
                grad = grad.mul(-group['lr'])
                
                # Apply parameter mask
                param_mask = group['mask']
                grad = grad * param_mask
                
                # Update parameter
                param.data.add_(grad)
        
        return loss
    
    def get_active_parameter_count(self) -> int:
        """
        Get the total number of active (unmasked) parameters.
        
        Returns:
            Count of active parameters
        """
        active_count = 0
        for group in self.param_groups:
            param_mask = group['mask']
            active_count += param_mask.sum().item()
        
        return active_count
    
    def get_sparsity_ratio(self) -> float:
        """
        Calculate the current sparsity ratio of the model.
        
        Returns:
            Sparsity ratio (fraction of masked parameters)
        """
        total_params = 0
        masked_params = 0
        
        for group in self.param_groups:
            param_mask = group['mask']
            total_params += param_mask.numel()
            masked_params += (param_mask == 0).sum().item()
        
        return masked_params / total_params if total_params > 0 else 0.0

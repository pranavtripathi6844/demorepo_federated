"""
Core model editing functionality using Fisher Information Matrix and iterative pruning.
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import time
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


class FisherInformationCalculator:
    """
    Computes Fisher Information Matrix diagonal for model parameter importance scoring.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize the Fisher Information calculator.
        
        Args:
            device: Device to perform computations on
        """
        self.device = device
    
    def compute_fisher_diagonal(self, model: torch.nn.Module, 
                              dataloader: DataLoader, 
                              parameter_masks: Dict[str, torch.Tensor],
                              soft_zero_value: float = 0.01,
                              max_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher diagonal scores for model parameters.
        
        Args:
            model: The neural network model
            dataloader: Data loader for computing Fisher scores
            parameter_masks: Binary masks for each parameter
            soft_zero_value: Value to use for masked parameters during computation
            max_samples: Maximum number of samples to use (None for all)
            
        Returns:
            Dictionary mapping parameter names to Fisher scores
        """
        # Store original weights for restoration
        original_weights = {
            name: param.clone().detach() 
            for name, param in model.named_parameters()
        }
        
        # Apply masks with soft-zeroing
        self._apply_soft_masks(model, parameter_masks, soft_zero_value)
        
        # Move model to device
        model.to(self.device)
        
        # Initialize Fisher diagonal accumulator
        fisher_scores = {
            name: torch.zeros_like(param, device='cpu')
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
        
        # Compute Fisher scores
        sample_count = 0
        for batch_inputs, _ in dataloader:
            batch_size = batch_inputs.size(0)
            
            for sample_idx in range(batch_size):
                if max_samples is not None and sample_count >= max_samples:
                    break
                
                # Process single sample
                input_sample = batch_inputs[sample_idx].unsqueeze(0).to(self.device)
                fisher_scores = self._compute_sample_fisher(
                    model, input_sample, fisher_scores, parameter_masks
                )
                sample_count += 1
            
            if max_samples is not None and sample_count >= max_samples:
                break
        
        # Normalize scores
        if sample_count > 0:
            for param_name in fisher_scores:
                fisher_scores[param_name] /= sample_count
        
        # Restore original weights
        self._restore_weights(model, original_weights)
        
        return fisher_scores
    
    def _apply_soft_masks(self, model: torch.nn.Module, 
                          parameter_masks: Dict[str, torch.Tensor],
                          soft_zero_value: float):
        """Apply soft masks to model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in parameter_masks:
                    current_mask = parameter_masks[name].to(self.device)
                    # Replace masked values with soft zero
                    current_mask[current_mask == 0] = soft_zero_value
                    param.mul_(current_mask)
    
    def _compute_sample_fisher(self, model: torch.nn.Module, 
                              input_sample: torch.Tensor,
                              fisher_scores: Dict[str, torch.Tensor],
                              parameter_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute Fisher scores for a single sample."""
        # Forward pass
        logits = model(input_sample)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sample from predicted distribution
        distribution = torch.distributions.Categorical(logits=logits)
        sampled_label = distribution.sample()
        
        # Compute loss
        loss = F.nll_loss(log_probs, sampled_label, reduction='sum')
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Accumulate gradient squares
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in parameter_masks:
                    fisher_scores[name] += (param.grad.detach().cpu() ** 2)
        
        return fisher_scores
    
    def _restore_weights(self, model: torch.nn.Module, 
                         original_weights: Dict[str, torch.Tensor]):
        """Restore original model weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(original_weights[name])


class IterativeMaskGenerator:
    """
    Generates pruning masks iteratively using Fisher Information scores.
    """
    
    def __init__(self, target_sparsity: float = 0.9, 
                 num_iterations: int = 5,
                 soft_zero_value: float = 0.01):
        """
        Initialize the mask generator.
        
        Args:
            target_sparsity: Target fraction of parameters to prune
            num_iterations: Number of iterative pruning rounds
            soft_zero_value: Soft zero value for Fisher computation
        """
        self.target_sparsity = target_sparsity
        self.num_iterations = num_iterations
        self.soft_zero_value = soft_zero_value
        self.fisher_calculator = FisherInformationCalculator()
    
    def generate_mask(self, model: torch.nn.Module, 
                     dataloader: DataLoader,
                     device: str = 'cuda',
                     enable_visualization: bool = False,
                     debug_mode: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate pruning mask through iterative Fisher-based pruning.
        
        Args:
            model: The model to generate mask for
            dataloader: Data loader for Fisher computation
            device: Device to use for computations
            enable_visualization: Whether to show pruning progress
            debug_mode: Enable debug output
            
        Returns:
            Binary mask dictionary for each parameter
        """
        # Create local copy of model
        local_model = copy.deepcopy(model)
        
        # Configure parameter gradients for backbone-only training
        self._configure_parameter_gradients(local_model)
        
        # Set training mode for gradient computation
        local_model.train()
        
        # Initialize mask (all parameters active)
        current_mask = self._initialize_mask(local_model)
        
        # Iterative pruning loop
        for iteration in range(1, self.num_iterations + 1):
            if debug_mode:
                print(f"--- Starting Iteration {iteration}/{self.num_iterations} ---")
            
            # Compute Fisher scores with current mask
            fisher_scores = self.fisher_calculator.compute_fisher_diagonal(
                local_model, dataloader, current_mask, 
                self.soft_zero_value
            )
            
            # Calculate target sparsity for this iteration
            current_sparsity_target = self._calculate_iteration_sparsity(iteration)
            
            if debug_mode:
                print(f"Current sparsity target: {current_sparsity_target:.4f}")
            
            # Update mask for this iteration
            current_mask = self._update_mask_for_iteration(
                fisher_scores, current_sparsity_target, current_mask, debug_mode
            )
            
            # Debug output
            if debug_mode:
                self._report_mask_statistics(current_mask, iteration)
            
            # Visualization
            if enable_visualization:
                self._visualize_mask_evolution(current_mask, iteration)
        
        return current_mask
    
    def _configure_parameter_gradients(self, model: torch.nn.Module):
        """Configure which parameters require gradients - ONLY BACKBONE."""
        for name, param in model.named_parameters():
            # For LinearFlexibleDino, ONLY train backbone parameters
            # Include: backbone blocks (transformer layers)
            # Exclude: head, patch_embed, pos_embed, cls_token, backbone.norm
            if 'backbone.blocks' in name:
                # Only backbone transformer blocks
                param.requires_grad = True
            else:
                # Exclude everything else: head, embeddings, final norm
                param.requires_grad = False
    
    def _initialize_mask(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Initialize mask with all parameters active."""
        return {
            name: torch.ones_like(param, device='cpu')
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
    
    def _calculate_iteration_sparsity(self, iteration: int) -> float:
        """Calculate target sparsity for current iteration."""
        return 1 - (1 - self.target_sparsity) ** (iteration / self.num_iterations)
    
    def _update_mask_for_iteration(self, fisher_scores: Dict[str, torch.Tensor],
                                 target_sparsity: float,
                                 current_mask: Dict[str, torch.Tensor],
                                 debug_mode: bool) -> Dict[str, torch.Tensor]:
        """Update mask based on Fisher scores and target sparsity."""
        # Collect active scores
        active_scores, total_params, pruned_params = self._collect_active_scores(
            fisher_scores, current_mask
        )
        
        if debug_mode:
            print(f"Target: {target_sparsity:.4f} ({int(total_params * target_sparsity)} params)")
            print(f"Total params: {total_params}, Already pruned: {pruned_params}")
        
        # Calculate threshold for this iteration
        threshold = self._calculate_pruning_threshold(
            active_scores, target_sparsity, total_params, pruned_params
        )
        
        # Update mask
        updated_mask = copy.deepcopy(current_mask)
        for param_name in fisher_scores:
            score_tensor = fisher_scores[param_name]
            mask_tensor = updated_mask[param_name]
            
            # Prune parameters below threshold
            pruning_indices = score_tensor < threshold
            mask_tensor[pruning_indices] = 0
        
        return updated_mask
    
    def _collect_active_scores(self, fisher_scores: Dict[str, torch.Tensor],
                             current_mask: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int, int]:
        """Collect Fisher scores for currently active parameters."""
        active_scores = []
        total_params = 0
        pruned_params = 0
        
        for param_name in fisher_scores:
            scores = fisher_scores[param_name].view(-1)
            mask = current_mask[param_name].view(-1)
            
            # Only consider active parameters
            active_scores.append(scores[mask == 1])
            
            total_params += len(mask)
            pruned_params += (mask == 0).sum().item()
        
        active_scores = torch.cat(active_scores)
        return active_scores, total_params, pruned_params
    
    def _calculate_pruning_threshold(self, active_scores: torch.Tensor,
                                   target_sparsity: float,
                                   total_params: int,
                                   pruned_params: int) -> float:
        """Calculate threshold for pruning in current iteration."""
        target_pruned = int(total_params * target_sparsity)
        additional_prune = target_pruned - pruned_params
        
        if additional_prune <= 0:
            return float('inf')  # No more pruning needed
        
        # Sort scores and find threshold
        sorted_scores = torch.sort(active_scores, descending=True).values
        threshold_idx = additional_prune
        
        if threshold_idx >= len(sorted_scores):
            threshold_idx = len(sorted_scores) - 1
        
        return sorted_scores[threshold_idx].item()
    
    def _report_mask_statistics(self, mask: Dict[str, torch.Tensor], iteration: int):
        """Report current mask statistics."""
        total_params = sum(m.numel() for m in mask.values())
        pruned_params = sum((m == 0).sum().item() for m in mask.values())
        actual_sparsity = pruned_params / total_params
        
        print(f"Iteration {iteration} - Achieved sparsity: {actual_sparsity:.4f}")
        print("-" * 30)
    
    def _visualize_mask_evolution(self, mask: Dict[str, torch.Tensor], iteration: int):
        """Visualize mask evolution across layers."""
        layer_sparsities = []
        layer_names = []
        
        for name, mask_tensor in mask.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            layer_sparsities.append(sparsity)
            layer_names.append(name.split('.')[-1])  # Extract layer name
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(layer_names)), layer_sparsities)
        plt.xlabel('Layer')
        plt.ylabel('Sparsity')
        plt.title(f'Layer-wise Sparsity - Iteration {iteration}')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.tight_layout()
        plt.show()


def create_client_masks(model: torch.nn.Module,
                       client_datasets: List[torch.utils.data.Dataset],
                       classes_per_client: int = 100,
                       batch_size: int = 128,
                       target_sparsity: float = 0.9,
                       num_iterations: int = 10,
                       soft_zero_value: float = 0.01,
                       max_samples: int = 25,
                       debug_mode: bool = False) -> List[Dict[str, torch.Tensor]]:
    """
    Create pruning masks for multiple clients with adaptive stratified sampling.
    
    Args:
        model: Base model for mask generation
        client_datasets: List of client datasets
        classes_per_client: Number of classes per client (Nc) for adaptive sampling
        batch_size: Batch size for data loading
        target_sparsity: Target sparsity for masks
        num_iterations: Number of pruning iterations
        soft_zero_value: Soft zero value for Fisher computation
        max_samples: Maximum samples for Fisher computation
        debug_mode: Enable debug output
        
    Returns:
        List of masks for each client
    """
    mask_generator = IterativeMaskGenerator(
        target_sparsity=target_sparsity,
        num_iterations=num_iterations,
        soft_zero_value=soft_zero_value
    )
    
    client_masks = []
    
    for client_idx, client_dataset in enumerate(client_datasets):
        if debug_mode:
            print(f"Generating mask for client {client_idx + 1}/{len(client_datasets)}")
        
        # Create data loader for this client
        client_loader = torch.utils.data.DataLoader(
            client_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        
        # Create adaptive stratified loader based on client's class distribution
        from src.data.dataset_loader import CIFAR100DataManager
        data_manager = CIFAR100DataManager()
        stratified_loader = data_manager.get_adaptive_stratified_loader(
            client_loader, classes_per_client
        )
        
        # Generate mask
        start_time = time.time()
        client_mask = mask_generator.generate_mask(
            model, stratified_loader, device='cuda',
            enable_visualization=False, debug_mode=debug_mode
        )
        end_time = time.time()
        
        if debug_mode:
            print(f"Mask generation time: {end_time - start_time:.2f}s")
            print(f"Target sparsity: {target_sparsity}")
            print(f"Soft zero value: {soft_zero_value}")
        
        client_masks.append(client_mask)
    
    return client_masks


def compute_mask(model: torch.nn.Module, 
                 dataloader: DataLoader,
                 sparsity_target: float = 0.9,
                 R: int = 5,
                 soft_zero: float = 0.01,
                 num_examples: Optional[int] = None,
                 device: str = 'cuda',
                 enable_plot: bool = False,
                 debug: bool = False) -> Dict[str, torch.Tensor]:
    """
    Standalone function to compute iterative mask using Fisher Information.
    
    Args:
        model: The neural network model to generate mask for
        dataloader: Data loader for Fisher computation
        sparsity_target: Target fraction of parameters to prune (0.9 = 90% pruned)
        R: Number of iterative refinement rounds
        soft_zero: Value to replace exact zeros with during Fisher computation
        num_examples: Maximum number of examples to use (None for all)
        device: Device to run computations on
        enable_plot: Whether to show pruning progress plots
        debug: Enable debug output
        
    Returns:
        Binary mask dictionary for each parameter
    """
    mask_generator = IterativeMaskGenerator(
        target_sparsity=sparsity_target,
        num_iterations=R,
        soft_zero_value=soft_zero
    )
    
    return mask_generator.generate_mask(
        model=model,
        dataloader=dataloader,
        device=device,
        enable_visualization=enable_plot,
        debug_mode=debug
    )

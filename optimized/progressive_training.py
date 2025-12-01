"""
Progressive training strategy for Gaussian Splatting.

This module provides a training strategy that:
- Starts with fewer points and grows to maximum
- Densifies (adds points) where gradients are high
- Prunes points with very low opacity
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ProgressiveTraining:
    """
    Progressive training strategy for 3D Gaussian points.
    
    Manages point cloud growth through densification and pruning.
    
    Args:
        initial_points: Starting number of points
        max_points: Maximum number of points allowed
        densify_interval: Training steps between densification
        prune_interval: Training steps between pruning
        densify_grad_threshold: Gradient threshold for cloning points
        prune_opacity_threshold: Opacity threshold below which points are pruned
        device: Device for tensors
    """
    
    def __init__(
        self,
        initial_points: int = 500,
        max_points: int = 10000,
        densify_interval: int = 100,
        prune_interval: int = 200,
        densify_grad_threshold: float = 0.0002,
        prune_opacity_threshold: float = 0.01,
        device: torch.device = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.initial_points = initial_points
        self.max_points = max_points
        self.densify_interval = densify_interval
        self.prune_interval = prune_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.prune_opacity_threshold = prune_opacity_threshold
        self.device = device
        
        # Tracking
        self.xyz_gradient_accum = None
        self.gradient_count = None
        self.step = 0
    
    def reset_gradient_accum(self, num_points: int) -> None:
        """
        Reset gradient accumulation buffers.
        
        Args:
            num_points: Current number of points
        """
        self.xyz_gradient_accum = torch.zeros(num_points, device=self.device)
        self.gradient_count = torch.zeros(num_points, device=self.device)
    
    def accumulate_gradients(self, xyz: torch.Tensor) -> None:
        """
        Accumulate position gradients for densification decisions.
        
        Args:
            xyz: Point positions with gradients attached
        """
        if xyz.grad is None:
            return
        
        grad_norm = xyz.grad.norm(dim=1)
        
        # Initialize accumulators if needed
        if self.xyz_gradient_accum is None or len(self.xyz_gradient_accum) != len(grad_norm):
            self.reset_gradient_accum(len(grad_norm))
        
        self.xyz_gradient_accum += grad_norm
        self.gradient_count += 1
    
    def should_densify(self) -> bool:
        """Check if densification should occur at current step."""
        return self.step > 0 and self.step % self.densify_interval == 0
    
    def should_prune(self) -> bool:
        """Check if pruning should occur at current step."""
        return self.step > 0 and self.step % self.prune_interval == 0
    
    def densify(
        self,
        xyz: nn.Parameter,
        rgb: nn.Parameter,
        opacity: nn.Parameter,
        scale: nn.Parameter,
        perturbation: float = 0.1
    ) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        """
        Add new points where gradients are high.
        
        Clones points that have high accumulated gradients with small perturbation.
        
        Args:
            xyz: Point positions parameter
            rgb: Point colors parameter
            opacity: Point opacity parameter
            scale: Point scale parameter
            perturbation: Random noise scale for cloned positions
        
        Returns:
            Tuple of updated (xyz, rgb, opacity, scale) parameters
        """
        if self.xyz_gradient_accum is None:
            return xyz, rgb, opacity, scale
        
        current_points = xyz.shape[0]
        
        # Don't exceed max points
        if current_points >= self.max_points:
            return xyz, rgb, opacity, scale
        
        # Compute average gradient per point
        avg_gradient = self.xyz_gradient_accum / (self.gradient_count + 1e-8)
        
        # Find points with high gradients
        high_grad_mask = avg_gradient > self.densify_grad_threshold
        
        # Limit number of new points
        max_new_points = min(
            self.max_points - current_points,
            high_grad_mask.sum().item(),
            current_points // 4  # Don't more than 25% at once
        )
        
        if max_new_points <= 0:
            self.reset_gradient_accum(current_points)
            return xyz, rgb, opacity, scale
        
        # Select top gradient points
        high_grad_indices = torch.where(high_grad_mask)[0]
        if len(high_grad_indices) > max_new_points:
            top_grads = avg_gradient[high_grad_indices]
            _, top_indices = torch.topk(top_grads, max_new_points)
            high_grad_indices = high_grad_indices[top_indices]
        
        # Clone points with perturbation
        with torch.no_grad():
            new_xyz = xyz.data[high_grad_indices].clone()
            new_rgb = rgb.data[high_grad_indices].clone()
            new_opacity = opacity.data[high_grad_indices].clone()
            new_scale = scale.data[high_grad_indices].clone()
            
            # Add small perturbation to positions
            scale_vals = torch.exp(new_scale)  # Convert from log scale
            new_xyz += torch.randn_like(new_xyz) * perturbation * scale_vals
            
            # Concatenate
            updated_xyz = nn.Parameter(torch.cat([xyz.data, new_xyz], dim=0))
            updated_rgb = nn.Parameter(torch.cat([rgb.data, new_rgb], dim=0))
            updated_opacity = nn.Parameter(torch.cat([opacity.data, new_opacity], dim=0))
            updated_scale = nn.Parameter(torch.cat([scale.data, new_scale], dim=0))
        
        # Reset accumulators with new size
        self.reset_gradient_accum(updated_xyz.shape[0])
        
        return updated_xyz, updated_rgb, updated_opacity, updated_scale
    
    def prune(
        self,
        xyz: nn.Parameter,
        rgb: nn.Parameter,
        opacity: nn.Parameter,
        scale: nn.Parameter
    ) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        """
        Remove points with very low opacity.
        
        Args:
            xyz: Point positions parameter
            rgb: Point colors parameter
            opacity: Point opacity parameter (raw values, sigmoid applied)
            scale: Point scale parameter
        
        Returns:
            Tuple of updated (xyz, rgb, opacity, scale) parameters
        """
        # Convert raw opacity to actual opacity via sigmoid
        actual_opacity = torch.sigmoid(opacity.data)
        
        # Find points to keep (above threshold)
        keep_mask = (actual_opacity.squeeze(-1) > self.prune_opacity_threshold)
        
        # Always keep at least initial_points
        num_to_keep = keep_mask.sum().item()
        if num_to_keep < self.initial_points:
            # Sort by opacity and keep top initial_points
            _, top_indices = torch.topk(
                actual_opacity.squeeze(-1),
                min(self.initial_points, xyz.shape[0])
            )
            keep_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=self.device)
            keep_mask[top_indices] = True
        
        if keep_mask.all():
            return xyz, rgb, opacity, scale
        
        # Prune points
        with torch.no_grad():
            updated_xyz = nn.Parameter(xyz.data[keep_mask])
            updated_rgb = nn.Parameter(rgb.data[keep_mask])
            updated_opacity = nn.Parameter(opacity.data[keep_mask])
            updated_scale = nn.Parameter(scale.data[keep_mask])
        
        # Reset accumulators with new size
        self.reset_gradient_accum(updated_xyz.shape[0])
        
        return updated_xyz, updated_rgb, updated_opacity, updated_scale
    
    def step_update(self) -> None:
        """Increment the step counter."""
        self.step += 1
    
    def get_status(self) -> dict:
        """Get current training status."""
        return {
            'step': self.step,
            'gradient_accum_size': len(self.xyz_gradient_accum) if self.xyz_gradient_accum is not None else 0,
            'should_densify': self.should_densify(),
            'should_prune': self.should_prune()
        }

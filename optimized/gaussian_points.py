"""
Learnable 3D Gaussian point cloud for neural head reconstruction.

This module provides a GaussianPoints class that represents a set of 3D Gaussian
primitives with learnable properties including position, color, opacity, and scale.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class GaussianPoints(nn.Module):
    """
    Learnable 3D Gaussian point cloud class with full properties.
    
    Each Gaussian primitive has:
    - XYZ positions (learnable)
    - RGB colors with sigmoid activation for [0,1] range
    - Opacity (alpha) with sigmoid activation for [0,1] range
    - Scale (size) with exp activation for positive values
    
    Args:
        num_points: Number of Gaussian points to initialize
        init_scale: Initial spatial extent for point positions
        device: Device to create tensors on (cuda/cpu)
    """
    
    def __init__(
        self,
        num_points: int = 1000,
        init_scale: float = 70.0,
        device: torch.device = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_points = num_points
        
        # Initialize XYZ positions randomly within [-init_scale, init_scale]
        self._xyz = nn.Parameter(
            (torch.rand((num_points, 3), device=device) * 2 - 1) * init_scale
        )
        
        # Initialize raw RGB values (will be passed through sigmoid)
        # Start with values that map to ~0.5 after sigmoid
        self._rgb = nn.Parameter(
            torch.zeros((num_points, 3), device=device)
        )
        
        # Initialize raw opacity values (will be passed through sigmoid)
        # Start with values that map to ~0.5 after sigmoid
        self._opacity = nn.Parameter(
            torch.zeros((num_points, 1), device=device)
        )
        
        # Initialize raw scale values (will be passed through exp)
        # Start with log(3.0) so initial scale is ~3.0
        self._scale = nn.Parameter(
            torch.ones((num_points, 1), device=device) * 1.1
        )
    
    @property
    def xyz(self) -> torch.Tensor:
        """Get XYZ positions (raw values, no activation)."""
        return self._xyz
    
    @property
    def rgb(self) -> torch.Tensor:
        """Get RGB colors with sigmoid activation for [0,1] range."""
        return torch.sigmoid(self._rgb)
    
    @property
    def opacity(self) -> torch.Tensor:
        """Get opacity with sigmoid activation for [0,1] range."""
        return torch.sigmoid(self._opacity)
    
    @property
    def scale(self) -> torch.Tensor:
        """Get scale with exp activation for positive values."""
        return torch.exp(self._scale)
    
    def parameters(self) -> List[nn.Parameter]:
        """Return all learnable parameters."""
        return [self._xyz, self._rgb, self._opacity, self._scale]
    
    def get_point_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all point properties with activations applied.
        
        Returns:
            Tuple of (xyz, rgb, opacity, scale) tensors
        """
        return self.xyz, self.rgb, self.opacity, self.scale
    
    def clone_points(self, indices: torch.Tensor, perturbation: float = 0.1) -> None:
        """
        Clone points at given indices with small perturbation.
        
        Used for densification during progressive training.
        
        Args:
            indices: Tensor of point indices to clone
            perturbation: Amount of random noise to add to cloned positions
        """
        if len(indices) == 0:
            return
            
        # Get current values
        new_xyz = self._xyz.data[indices].clone()
        new_rgb = self._rgb.data[indices].clone()
        new_opacity = self._opacity.data[indices].clone()
        new_scale = self._scale.data[indices].clone()
        
        # Add perturbation to positions
        new_xyz += torch.randn_like(new_xyz) * perturbation * self.scale[indices]
        
        # Concatenate new points
        self._xyz = nn.Parameter(torch.cat([self._xyz.data, new_xyz], dim=0))
        self._rgb = nn.Parameter(torch.cat([self._rgb.data, new_rgb], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity.data, new_opacity], dim=0))
        self._scale = nn.Parameter(torch.cat([self._scale.data, new_scale], dim=0))
        
        self.num_points = self._xyz.shape[0]
    
    def prune_points(self, indices_to_keep: torch.Tensor) -> None:
        """
        Remove points not in the given indices.
        
        Used for pruning low-opacity points during progressive training.
        
        Args:
            indices_to_keep: Tensor of point indices to retain
        """
        self._xyz = nn.Parameter(self._xyz.data[indices_to_keep])
        self._rgb = nn.Parameter(self._rgb.data[indices_to_keep])
        self._opacity = nn.Parameter(self._opacity.data[indices_to_keep])
        self._scale = nn.Parameter(self._scale.data[indices_to_keep])
        
        self.num_points = self._xyz.shape[0]
    
    def __len__(self) -> int:
        """Return the number of points."""
        return self.num_points

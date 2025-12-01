"""
Camera model with learnable extrinsics for multi-view reconstruction.

This module provides a camera class with learnable parameters:
- Per-view yaw and pitch angles
- Shared focal length across views
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class LearnableCamera(nn.Module):
    """
    Camera model with learnable extrinsics for multi-view reconstruction.
    
    Supports learnable yaw/pitch per view and shared focal length.
    
    Args:
        num_views: Number of camera views
        initial_yaw: Initial yaw angles per view (radians)
        initial_pitch: Initial pitch angles per view (radians)
        initial_focal_length: Initial focal length value
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_views: int = 4,
        initial_yaw: Optional[List[float]] = None,
        initial_pitch: Optional[List[float]] = None,
        initial_focal_length: float = 400.0,
        device: torch.device = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_views = num_views
        
        # Default initial angles (typical 4-view setup)
        if initial_yaw is None:
            # Views: top-right, top-left, bottom-right, bottom-left
            initial_yaw = [0.4, -0.4, 0.4, -0.4]
        if initial_pitch is None:
            initial_pitch = [0.25, 0.25, -0.25, -0.25]
        
        # Ensure we have enough initial values
        while len(initial_yaw) < num_views:
            initial_yaw.append(0.0)
        while len(initial_pitch) < num_views:
            initial_pitch.append(0.0)
        
        # Learnable yaw angles per view
        self._yaw = nn.Parameter(
            torch.tensor(initial_yaw[:num_views], device=device, dtype=torch.float32)
        )
        
        # Learnable pitch angles per view
        self._pitch = nn.Parameter(
            torch.tensor(initial_pitch[:num_views], device=device, dtype=torch.float32)
        )
        
        # Learnable focal length (shared across views)
        # Store as log for positive constraint
        self._log_focal_length = nn.Parameter(
            torch.tensor([initial_focal_length], device=device, dtype=torch.float32).log()
        )
    
    @property
    def yaw(self) -> torch.Tensor:
        """Get yaw angles for all views."""
        return self._yaw
    
    @property
    def pitch(self) -> torch.Tensor:
        """Get pitch angles for all views."""
        return self._pitch
    
    @property
    def focal_length(self) -> torch.Tensor:
        """Get focal length (positive via exp)."""
        return torch.exp(self._log_focal_length)
    
    def get_view(self, view_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get camera parameters for a specific view.
        
        Args:
            view_idx: Index of the view (0 to num_views-1)
        
        Returns:
            Tuple of (yaw, pitch, focal_length) tensors
        """
        if view_idx < 0 or view_idx >= self.num_views:
            raise ValueError(f"View index {view_idx} out of range [0, {self.num_views})")
        
        return (
            self._yaw[view_idx],
            self._pitch[view_idx],
            self.focal_length.squeeze()
        )
    
    def get_all_views(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get camera parameters for all views.
        
        Returns:
            List of (yaw, pitch, focal_length) tuples
        """
        focal = self.focal_length.squeeze()
        return [
            (self._yaw[i], self._pitch[i], focal)
            for i in range(self.num_views)
        ]
    
    def parameters(self):
        """Return iterator over all learnable parameters."""
        return iter([self._yaw, self._pitch, self._log_focal_length])
    
    def get_angles_as_dict(self) -> dict:
        """
        Get current camera angles as a dictionary.
        
        Returns:
            Dictionary with yaw, pitch, and focal length values
        """
        return {
            'yaw': self._yaw.detach().cpu().tolist(),
            'pitch': self._pitch.detach().cpu().tolist(),
            'focal_length': self.focal_length.item()
        }
    
    def clamp_angles(self, max_angle: float = 1.0) -> None:
        """
        Clamp angles to reasonable range.
        
        Prevents angles from becoming too extreme during training.
        
        Args:
            max_angle: Maximum absolute angle in radians
        """
        with torch.no_grad():
            self._yaw.clamp_(-max_angle, max_angle)
            self._pitch.clamp_(-max_angle, max_angle)
    
    def clamp_focal_length(self, min_focal: float = 100.0, max_focal: float = 1000.0) -> None:
        """
        Clamp focal length to reasonable range.
        
        Args:
            min_focal: Minimum focal length
            max_focal: Maximum focal length
        """
        with torch.no_grad():
            self._log_focal_length.clamp_(
                torch.tensor(min_focal).log().item(),
                torch.tensor(max_focal).log().item()
            )


class FixedCamera:
    """
    Fixed camera model (non-learnable) for comparison or testing.
    
    Args:
        yaw: List of yaw angles per view
        pitch: List of pitch angles per view
        focal_length: Focal length value
        device: Device for tensors
    """
    
    def __init__(
        self,
        yaw: List[float],
        pitch: List[float],
        focal_length: float = 400.0,
        device: torch.device = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.num_views = len(yaw)
        self._yaw = torch.tensor(yaw, device=device, dtype=torch.float32)
        self._pitch = torch.tensor(pitch, device=device, dtype=torch.float32)
        self._focal_length = torch.tensor([focal_length], device=device, dtype=torch.float32)
    
    def get_view(self, view_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get camera parameters for a specific view."""
        return (
            self._yaw[view_idx],
            self._pitch[view_idx],
            self._focal_length.squeeze()
        )
    
    def get_all_views(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get camera parameters for all views."""
        return [self.get_view(i) for i in range(self.num_views)]
    
    def parameters(self) -> List:
        """Return empty list (no learnable parameters)."""
        return []

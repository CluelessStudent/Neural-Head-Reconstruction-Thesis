"""
Depth-aware renderer with proper alpha compositing for Gaussian Splatting.

This module provides rendering functions that correctly handle:
- 3D to 2D perspective projection with yaw/pitch rotation
- Depth sorting (back-to-front) for proper alpha compositing
- Batched rendering to prevent VRAM overflow
- Alpha compositing instead of additive blending
"""

import torch
from typing import Tuple

# Constants for numerical stability
MIN_Z_CLAMP = 1.0  # Minimum Z value to prevent division by zero


def project_3d(
    points_xyz: torch.Tensor,
    canvas_size: int,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    focal_length: float = 400.0,
    z_offset: float = 500.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D with perspective projection.
    
    Applies yaw (Y-axis) and pitch (X-axis) rotation, then perspective projection.
    
    Args:
        points_xyz: Tensor of shape (N, 3) with XYZ coordinates
        canvas_size: Size of the output canvas (assumes square)
        yaw: Rotation around Y-axis (horizontal rotation)
        pitch: Rotation around X-axis (vertical rotation)
        focal_length: Camera focal length for perspective
        z_offset: Distance to push object back from camera
    
    Returns:
        Tuple of (screen_x, screen_y, perspective_scale, z_depth):
        - screen_x: X coordinates on screen (N,)
        - screen_y: Y coordinates on screen (N,)
        - perspective_scale: Scale factor for each point (N,)
        - z_depth: Z depth after rotation for sorting (N,)
    """
    H, W = canvas_size, canvas_size
    cx, cy = W / 2, H / 2
    
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    # Rotation matrices
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    
    # Apply Yaw rotation (around Y-axis)
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y
    y_rot = y
    
    # Apply Pitch rotation (around X-axis)
    y_final = y_rot * cos_p - z_rot * sin_p
    z_final = y_rot * sin_p + z_rot * cos_p
    x_final = x_rot
    
    # Perspective projection
    z_cam = z_final + z_offset
    perspective = focal_length / torch.clamp(z_cam, min=MIN_Z_CLAMP)
    
    screen_x = x_final * perspective + cx
    screen_y = y_final * perspective + cy
    
    return screen_x, screen_y, perspective, z_final


def render_depth_aware(
    points_xyz: torch.Tensor,
    points_rgb: torch.Tensor,
    points_opacity: torch.Tensor,
    points_scale: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    canvas_size: int,
    focal_length: float = 400.0,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    Render 3D Gaussian points with depth-aware alpha compositing.
    
    Features:
    - Proper depth sorting (back-to-front)
    - Alpha compositing (not additive blending)
    - Batched rendering to prevent VRAM overflow
    - On-screen filtering
    
    Args:
        points_xyz: Point positions (N, 3)
        points_rgb: Point colors (N, 3), should be in [0, 1]
        points_opacity: Point opacities (N, 1), should be in [0, 1]
        points_scale: Point scales (N, 1), should be positive
        yaw: Horizontal rotation angle
        pitch: Vertical rotation angle
        canvas_size: Output image size (square)
        focal_length: Camera focal length
        batch_size: Number of points to render per batch (prevents VRAM overflow)
    
    Returns:
        Rendered image tensor of shape (H, W, 3) with values in [0, 1]
    """
    device = points_xyz.device
    H, W = canvas_size, canvas_size
    
    # Project all points
    sx, sy, scale, z_depth = project_3d(
        points_xyz, canvas_size, yaw, pitch, focal_length
    )
    
    # Filter visible points (on-screen check)
    visible_mask = (
        (scale > 0) &
        (sx >= -20) & (sx < W + 20) &
        (sy >= -20) & (sy < H + 20)
    )
    
    if visible_mask.sum() == 0:
        return torch.zeros((H, W, 3), device=device)
    
    # Extract visible point data
    vis_sx = sx[visible_mask]
    vis_sy = sy[visible_mask]
    vis_scale = scale[visible_mask]
    vis_z = z_depth[visible_mask]
    vis_rgb = points_rgb[visible_mask]
    vis_opacity = points_opacity[visible_mask].squeeze(-1)
    vis_point_scale = points_scale[visible_mask].squeeze(-1)
    
    # Sort by depth (back-to-front for proper alpha compositing)
    # In this coordinate system, larger z_final values are further from camera
    # We sort descending to render far points first, then composite closer points on top
    sorted_indices = torch.argsort(vis_z, descending=True)
    
    vis_sx = vis_sx[sorted_indices]
    vis_sy = vis_sy[sorted_indices]
    vis_scale = vis_scale[sorted_indices]
    vis_rgb = vis_rgb[sorted_indices]
    vis_opacity = vis_opacity[sorted_indices]
    vis_point_scale = vis_point_scale[sorted_indices]
    
    # Initialize canvas
    canvas = torch.zeros((H, W, 3), device=device)
    accumulated_alpha = torch.zeros((H, W, 1), device=device)
    
    # Create coordinate grids
    y_coords = torch.arange(H, device=device).float()
    x_coords = torch.arange(W, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    num_visible = vis_sx.shape[0]
    
    # Render in batches to prevent VRAM overflow
    for batch_start in range(0, num_visible, batch_size):
        batch_end = min(batch_start + batch_size, num_visible)
        
        b_sx = vis_sx[batch_start:batch_end]
        b_sy = vis_sy[batch_start:batch_end]
        b_scale = vis_scale[batch_start:batch_end]
        b_rgb = vis_rgb[batch_start:batch_end]
        b_opacity = vis_opacity[batch_start:batch_end]
        b_point_scale = vis_point_scale[batch_start:batch_end]
        
        # Compute Gaussian weights for batch
        # Shape: (batch, H, W)
        dx = x_grid.unsqueeze(0) - b_sx.view(-1, 1, 1)
        dy = y_grid.unsqueeze(0) - b_sy.view(-1, 1, 1)
        dist_sq = dx ** 2 + dy ** 2
        
        # Radius based on scale and perspective
        radius = b_point_scale.view(-1, 1, 1) * b_scale.view(-1, 1, 1)
        radius = torch.clamp(radius, min=1.0, max=20.0)
        
        # Gaussian weights
        gaussian_weights = torch.exp(-dist_sq / (2 * radius ** 2))
        
        # Threshold small weights for efficiency
        gaussian_weights = torch.where(
            gaussian_weights > 0.01,
            gaussian_weights,
            torch.zeros_like(gaussian_weights)
        )
        
        # Alpha for this batch: gaussian_weight * point_opacity
        batch_alpha = gaussian_weights * b_opacity.view(-1, 1, 1)
        
        # Alpha compositing: C_out = C_new * alpha + C_old * (1 - alpha)
        # Process each point in the batch
        for i in range(batch_end - batch_start):
            alpha = batch_alpha[i:i+1].permute(1, 2, 0)  # (H, W, 1)
            color = b_rgb[i:i+1].view(1, 1, 3)  # (1, 1, 3)
            
            # Remaining transparency
            remaining = 1.0 - accumulated_alpha
            
            # Composite this point
            canvas = canvas + remaining * alpha * color
            accumulated_alpha = accumulated_alpha + remaining * alpha
            
            # Early termination if fully opaque
            if accumulated_alpha.min() > 0.99:
                break
    
    return torch.clamp(canvas, 0, 1)


def render_fast(
    points_xyz: torch.Tensor,
    points_rgb: torch.Tensor,
    points_opacity: torch.Tensor,
    points_scale: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    canvas_size: int,
    focal_length: float = 400.0,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    Fast approximate rendering without per-point alpha compositing.
    
    Uses weighted sum instead of proper alpha compositing for speed.
    Suitable for training iterations where exact rendering is less critical.
    
    Args:
        points_xyz: Point positions (N, 3)
        points_rgb: Point colors (N, 3)
        points_opacity: Point opacities (N, 1)
        points_scale: Point scales (N, 1)
        yaw: Horizontal rotation angle
        pitch: Vertical rotation angle
        canvas_size: Output image size
        focal_length: Camera focal length
        batch_size: Points per batch
    
    Returns:
        Rendered image tensor of shape (H, W, 3)
    """
    device = points_xyz.device
    H, W = canvas_size, canvas_size
    
    sx, sy, scale, z_depth = project_3d(
        points_xyz, canvas_size, yaw, pitch, focal_length
    )
    
    # Filter visible points
    visible_mask = (
        (scale > 0) &
        (sx >= -15) & (sx < W + 15) &
        (sy >= -15) & (sy < H + 15)
    )
    
    if visible_mask.sum() == 0:
        return torch.zeros((H, W, 3), device=device)
    
    vis_sx = sx[visible_mask]
    vis_sy = sy[visible_mask]
    vis_scale = scale[visible_mask]
    vis_rgb = points_rgb[visible_mask]
    vis_opacity = points_opacity[visible_mask].squeeze(-1)
    vis_point_scale = points_scale[visible_mask].squeeze(-1)
    
    # Sort by depth for better results
    sorted_indices = torch.argsort(z_depth[visible_mask], descending=True)
    vis_sx = vis_sx[sorted_indices]
    vis_sy = vis_sy[sorted_indices]
    vis_scale = vis_scale[sorted_indices]
    vis_rgb = vis_rgb[sorted_indices]
    vis_opacity = vis_opacity[sorted_indices]
    vis_point_scale = vis_point_scale[sorted_indices]
    
    # Initialize canvas
    canvas = torch.zeros((H, W, 3), device=device)
    
    y_grid = torch.arange(H, device=device).float().view(1, H, 1)
    x_grid = torch.arange(W, device=device).float().view(1, 1, W)
    
    num_visible = vis_sx.shape[0]
    
    for batch_start in range(0, num_visible, batch_size):
        batch_end = min(batch_start + batch_size, num_visible)
        
        b_sx = vis_sx[batch_start:batch_end].view(-1, 1, 1)
        b_sy = vis_sy[batch_start:batch_end].view(-1, 1, 1)
        b_scale = vis_scale[batch_start:batch_end].view(-1, 1, 1)
        b_rgb = vis_rgb[batch_start:batch_end].view(-1, 1, 1, 3)
        b_opacity = vis_opacity[batch_start:batch_end].view(-1, 1, 1, 1)
        b_point_scale = vis_point_scale[batch_start:batch_end].view(-1, 1, 1)
        
        dist_sq = (x_grid - b_sx) ** 2 + (y_grid - b_sy) ** 2
        radius = b_point_scale * b_scale
        radius = torch.clamp(radius, min=1.0, max=15.0)
        
        strength = torch.exp(-dist_sq / (2 * radius ** 2))
        strength = torch.where(strength > 0.05, strength, torch.zeros_like(strength))
        
        # Weighted accumulation with opacity
        batch_canvas = (strength.unsqueeze(-1) * b_rgb * b_opacity).sum(dim=0)
        canvas = canvas + batch_canvas
    
    return torch.clamp(canvas, 0, 1)

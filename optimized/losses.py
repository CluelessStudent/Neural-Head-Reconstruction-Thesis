"""
Combined loss functions with regularization for Gaussian Splatting training.

This module provides loss functions that combine:
- Photometric loss (L1 + SSIM)
- Opacity regularization
- Scale regularization
- Spatial regularization
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image tensor (H, W, C) or (B, H, W, C)
        img2: Second image tensor (H, W, C) or (B, H, W, C)
        window_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian window
    
    Returns:
        SSIM value as a scalar tensor
    """
    # Handle different input shapes
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Convert from (B, H, W, C) to (B, C, H, W) for conv operations
    if img1.shape[-1] <= 4:  # Assuming channels is the last dim if <= 4
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)
    
    device = img1.device
    channels = img1.shape[1]
    
    # Create Gaussian window
    gauss = torch.arange(window_size, device=device).float()
    gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    # Create 2D window
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    padding = window_size // 2
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def photometric_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.2
) -> torch.Tensor:
    """
    Compute photometric loss combining L1 and SSIM.
    
    L1 loss provides sharp edges, SSIM preserves structure.
    
    Args:
        rendered: Rendered image (H, W, 3)
        target: Ground truth image (H, W, 3)
        ssim_weight: Weight for SSIM loss (1 - ssim_weight for L1)
    
    Returns:
        Combined photometric loss
    """
    # L1 loss (sharper than MSE)
    l1_loss = torch.abs(rendered - target).mean()
    
    # SSIM loss (1 - SSIM since we want to maximize SSIM)
    ssim_val = compute_ssim(rendered, target)
    ssim_loss = 1.0 - ssim_val
    
    # Combine losses
    total_loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss
    
    return total_loss


def opacity_regularization(
    opacity: torch.Tensor,
    target_opacity: float = 0.8
) -> torch.Tensor:
    """
    Regularization to encourage points to be visible.
    
    Penalizes opacity values far from target (encourages visible points).
    
    Args:
        opacity: Point opacities (N, 1) in [0, 1]
        target_opacity: Target opacity value to encourage
    
    Returns:
        Opacity regularization loss
    """
    # Encourage opacity to be close to target
    # Using L2 distance from target
    loss = ((opacity - target_opacity) ** 2).mean()
    return loss


def scale_regularization(
    scale: torch.Tensor,
    min_scale: float = 1.0,
    max_scale: float = 10.0
) -> torch.Tensor:
    """
    Regularization to prevent tiny or huge points.
    
    Penalizes scales outside the desired range.
    
    Args:
        scale: Point scales (N, 1), positive values
        min_scale: Minimum allowed scale
        max_scale: Maximum allowed scale
    
    Returns:
        Scale regularization loss
    """
    # Penalize scales below min
    too_small = F.relu(min_scale - scale) ** 2
    
    # Penalize scales above max
    too_large = F.relu(scale - max_scale) ** 2
    
    loss = (too_small.mean() + too_large.mean())
    return loss


def spatial_regularization(
    xyz: torch.Tensor,
    max_extent: float = 100.0
) -> torch.Tensor:
    """
    Regularization to keep points in a reasonable volume.
    
    Penalizes points that stray too far from the origin.
    
    Args:
        xyz: Point positions (N, 3)
        max_extent: Maximum allowed distance from origin per axis
    
    Returns:
        Spatial regularization loss
    """
    # Penalize positions outside the box [-max_extent, max_extent]
    outside = F.relu(torch.abs(xyz) - max_extent)
    loss = (outside ** 2).mean()
    return loss


def combined_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    points_xyz: torch.Tensor,
    points_opacity: torch.Tensor,
    points_scale: torch.Tensor,
    ssim_weight: float = 0.2,
    opacity_weight: float = 0.01,
    scale_weight: float = 0.01,
    spatial_weight: float = 0.001,
    max_extent: float = 100.0
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss with all regularization terms.
    
    Args:
        rendered: Rendered image (H, W, 3)
        target: Ground truth image (H, W, 3)
        points_xyz: Point positions (N, 3)
        points_opacity: Point opacities (N, 1)
        points_scale: Point scales (N, 1)
        ssim_weight: Weight for SSIM in photometric loss
        opacity_weight: Weight for opacity regularization
        scale_weight: Weight for scale regularization
        spatial_weight: Weight for spatial regularization
        max_extent: Maximum spatial extent
    
    Returns:
        Tuple of (total_loss, loss_dict with individual terms)
    """
    # Photometric loss
    photo_loss = photometric_loss(rendered, target, ssim_weight)
    
    # Regularization losses
    opacity_loss = opacity_regularization(points_opacity)
    scale_loss = scale_regularization(points_scale)
    spatial_loss = spatial_regularization(points_xyz, max_extent)
    
    # Combine
    total_loss = (
        photo_loss +
        opacity_weight * opacity_loss +
        scale_weight * scale_loss +
        spatial_weight * spatial_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'photometric': photo_loss.item(),
        'opacity_reg': opacity_loss.item(),
        'scale_reg': scale_loss.item(),
        'spatial_reg': spatial_loss.item()
    }
    
    return total_loss, loss_dict


def multi_view_loss(
    rendered_views: list,
    target_views: list,
    points_xyz: torch.Tensor,
    points_opacity: torch.Tensor,
    points_scale: torch.Tensor,
    ssim_weight: float = 0.2,
    opacity_weight: float = 0.01,
    scale_weight: float = 0.01,
    spatial_weight: float = 0.001
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss across multiple views.
    
    Args:
        rendered_views: List of rendered images
        target_views: List of target images
        points_xyz: Point positions
        points_opacity: Point opacities
        points_scale: Point scales
        ssim_weight: Weight for SSIM
        opacity_weight: Weight for opacity regularization
        scale_weight: Weight for scale regularization
        spatial_weight: Weight for spatial regularization
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    num_views = len(rendered_views)
    total_photo_loss = 0
    
    for rendered, target in zip(rendered_views, target_views):
        total_photo_loss = total_photo_loss + photometric_loss(rendered, target, ssim_weight)
    
    total_photo_loss = total_photo_loss / num_views
    
    # Regularization (applied once, not per view)
    opacity_loss = opacity_regularization(points_opacity)
    scale_loss = scale_regularization(points_scale)
    spatial_loss = spatial_regularization(points_xyz)
    
    total_loss = (
        total_photo_loss +
        opacity_weight * opacity_loss +
        scale_weight * scale_loss +
        spatial_weight * spatial_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'photometric': total_photo_loss.item(),
        'opacity_reg': opacity_loss.item(),
        'scale_reg': scale_loss.item(),
        'spatial_reg': spatial_loss.item()
    }
    
    return total_loss, loss_dict

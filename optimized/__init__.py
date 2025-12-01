"""
Optimized Gaussian Splatting for Neural Head Reconstruction.

This package provides improved implementations of 3D Gaussian Splatting
for neural head reconstruction from sparse camera views.
"""

from .gaussian_points import GaussianPoints
from .learnable_camera import LearnableCamera, FixedCamera
from .depth_aware_render import project_3d, render_depth_aware, render_fast
from .losses import (
    compute_ssim,
    photometric_loss,
    opacity_regularization,
    scale_regularization,
    spatial_regularization,
    combined_loss,
    multi_view_loss
)
from .progressive_training import ProgressiveTraining
from . import train

__all__ = [
    # Point cloud
    'GaussianPoints',
    # Camera
    'LearnableCamera',
    'FixedCamera',
    # Rendering
    'project_3d',
    'render_depth_aware',
    'render_fast',
    # Losses
    'compute_ssim',
    'photometric_loss',
    'opacity_regularization',
    'scale_regularization',
    'spatial_regularization',
    'combined_loss',
    'multi_view_loss',
    # Training
    'ProgressiveTraining',
    'train',
]

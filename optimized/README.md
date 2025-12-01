# Optimized Gaussian Splatting for Neural Head Reconstruction

This folder contains improved implementations of 3D Gaussian Splatting for neural head reconstruction from sparse views.

## Overview

This optimized implementation improves upon the basic `aistudio/` implementation with:

- **Full Gaussian Properties**: Each point has learnable position, color, opacity, and scale
- **Depth-Aware Rendering**: Proper back-to-front sorting with alpha compositing
- **Advanced Loss Functions**: L1 + SSIM photometric loss with regularization
- **Progressive Training**: Dynamic point densification and pruning
- **Learnable Camera**: Jointly optimize camera extrinsics during training
- **Memory Efficient**: Batched rendering to prevent VRAM overflow

## Modules

### `gaussian_points.py`
Learnable 3D Gaussian point cloud class with:
- XYZ positions (raw learnable values)
- RGB colors (sigmoid activation for [0,1] range)
- Opacity/Alpha (sigmoid activation for [0,1] range)
- Scale (exp activation for positive values)
- Methods for cloning and pruning points

### `depth_aware_render.py`
Improved renderer with:
- `project_3d()`: 3D to 2D perspective projection with yaw/pitch rotation
- `render_depth_aware()`: Full alpha compositing with depth sorting
- `render_fast()`: Faster approximate rendering for training
- On-screen point filtering
- Batched rendering (1000 points per batch) to prevent VRAM overflow

### `losses.py`
Combined loss functions:
- `photometric_loss()`: L1 loss (sharper edges) + SSIM loss (structure preservation)
- `opacity_regularization()`: Encourage points to be visible
- `scale_regularization()`: Prevent tiny/huge points
- `spatial_regularization()`: Keep points in reasonable volume
- `compute_ssim()`: Helper for SSIM computation
- `combined_loss()`: All losses combined with configurable weights
- `multi_view_loss()`: Combined loss across multiple views

### `progressive_training.py`
Progressive training strategy:
- Start with fewer points (e.g., 500) and grow to max (e.g., 10000)
- `densify()`: Clone high-gradient points with small perturbation
- `prune()`: Remove low-opacity points
- Gradient accumulation for intelligent densification

### `learnable_camera.py`
Camera model with learnable extrinsics:
- Per-view yaw and pitch angles
- Shared focal length across views
- `get_view()`: Get parameters for specific view
- `parameters()`: For optimizer integration
- Angle and focal length clamping

### `train.py`
Main training script integrating all components:
- Dual optimizers: geometry (0.01 LR) and camera (0.001 LR)
- Cosine annealing learning rate scheduler
- Loads 4 ground truth images (1.jpg - 4.jpg)
- Real-time visualization with OpenCV
- Periodic checkpointing

## Usage

### Basic Training

```python
from optimized.train import train

# Train with default settings
train()

# Or with custom parameters
train(
    image_paths=["view1.jpg", "view2.jpg", "view3.jpg", "view4.jpg"],
    canvas_size=256,
    num_steps=2000,
    initial_points=500,
    max_points=10000,
    geometry_lr=0.01,
    camera_lr=0.001
)
```

### Using Individual Components

```python
import torch
from optimized.gaussian_points import GaussianPoints
from optimized.learnable_camera import LearnableCamera
from optimized.depth_aware_render import render_fast
from optimized.losses import combined_loss

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create point cloud
points = GaussianPoints(num_points=1000, device=device)
xyz, rgb, opacity, scale = points.get_point_data()

# Create camera
camera = LearnableCamera(num_views=4, device=device)
yaw, pitch, focal = camera.get_view(0)

# Render
rendered = render_fast(xyz, rgb, opacity, scale, yaw, pitch, canvas_size=256, focal_length=focal.item())

# Compute loss
target = torch.zeros(256, 256, 3, device=device)  # Your ground truth
loss, loss_dict = combined_loss(rendered, target, xyz, opacity, scale)
```

## Requirements

```bash
pip install torch numpy opencv-python
```

### Hardware Requirements
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4070 or better)
- **Minimum**: Any CUDA-capable GPU or CPU (training will be slower)

## Comparison with Basic Implementation

| Feature | Basic (`aistudio/`) | Optimized |
|---------|---------------------|-----------|
| **Point Properties** | XYZ, RGB only | XYZ, RGB, Opacity, Scale |
| **Rendering** | Additive blending | Alpha compositing with depth sorting |
| **Loss Function** | MSE only | L1 + SSIM + regularization |
| **Training** | Fixed point count | Progressive densification/pruning |
| **Camera** | Fixed angles | Learnable extrinsics |
| **Memory** | Can overflow VRAM | Batched rendering (memory safe) |
| **Activations** | Raw values | Proper sigmoid/exp activations |

## Output

Training produces:
- Real-time visualization window (press ESC to stop)
- Periodic checkpoint images in `output/` folder
- Final model saved as `output/final_model.pt`

## Tips

1. **Start with fewer points** (500) and let progressive training add more
2. **Adjust camera angles** if initial reconstruction is misaligned
3. **Reduce batch_size** in renderer if running out of VRAM
4. **Increase SSIM weight** for smoother results (may lose fine details)
5. **Monitor opacity regularization** to ensure points become visible

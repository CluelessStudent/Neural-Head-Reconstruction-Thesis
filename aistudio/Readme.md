# AI Studio - Neural Head Reconstruction

This folder contains experimental scripts for 3D head reconstruction using Gaussian splatting techniques. The scripts implement differentiable rendering pipelines to reconstruct 3D point clouds from multiple 2D images. 

## Overview

The reconstruction pipeline works by:
1.  Loading 4 input images (`1. jpg`, `2. jpg`, `3. jpg`, `4. jpg`) captured from different viewpoints
2. Initializing a random 3D point cloud with trainable positions (XYZ) and colors (RGB)
3.  Using differentiable rendering to project 3D points onto 2D canvases
4.  Optimizing the point cloud to minimize the difference between rendered views and ground truth images

## Scripts

### `aistudio. py`
Base implementation for 4-view 3D reconstruction. 
- **Device**: Apple MPS (Metal Performance Shaders) or CPU
- **Resolution**: 128×128
- **Points**: 1,000 Gaussian blobs
- **Features**: Basic yaw/pitch rotation, perspective projection

### `4070superrtx.py`
CUDA-optimized version for NVIDIA GPUs.
- **Device**: CUDA (NVIDIA GPU)
- **Resolution**: 256×256
- **Points**: 2,000 Gaussian blobs
- **Features**: Real-time spinning visualization of the reconstructed 3D model

### `turbo.py`
High-performance vectorized CUDA implementation.
- **Device**: CUDA with CuDNN benchmark enabled
- **Resolution**: 256×256
- **Points**: 2,000 Gaussian blobs
- **Features**: Fully vectorized rendering for maximum GPU utilization, on-screen point filtering for efficiency

### `yellow_killer (background killer).py`
Memory-efficient version with background removal.
- **Device**: CUDA with memory management
- **Resolution**: 256×256
- **Points**: 10,000 Gaussian blobs
- **Features**: 
  - HSV-based yellow background masking
  - Batched rendering (2,000 points per batch) to prevent VRAM overflow
  - Stochastic view sampling during training

## Input Images

Place 4 portrait images in the same directory as the scripts:
- `1. jpg` - View looking up/left
- `2. jpg` - View looking up/right
- `3.jpg` - View looking down/left
- `4. jpg` - View looking down/right

## Requirements

```bash
pip install torch numpy opencv-python
```

### Hardware Requirements
- **For `aistudio.py`**: macOS with Apple Silicon (MPS) or any CPU
- **For CUDA scripts**: NVIDIA GPU with 8GB+ VRAM (RTX 4070 Super recommended)

## Usage

```bash
# For Mac users
python aistudio. py

# For NVIDIA GPU users
python turbo.py

# For images with yellow backgrounds
python "yellow_killer (background killer).py"
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_blobs` | Number of 3D Gaussian points | 1000-10000 |
| `CANVAS_SIZE` | Render resolution | 128-256 |
| `yaw_val` | Horizontal rotation angle (radians) | 0.35-0.4 |
| `pitch_val` | Vertical rotation angle (radians) | 0.2-0.25 |
| `focal_length` | Camera focal length | 350-450 |
| `lr` | Learning rate | 1.5-3.0 |

## Controls

- Press **ESC** to stop the optimization early
- The visualization window shows a spinning view of the reconstructed 3D model

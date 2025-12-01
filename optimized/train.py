"""
Main training script for optimized Gaussian Splatting head reconstruction.

This script integrates all components:
- GaussianPoints: Learnable 3D point cloud
- LearnableCamera: Camera model with learnable extrinsics
- Depth-aware rendering with alpha compositing
- Combined loss functions with regularization
- Progressive training with densification and pruning
"""

import os
import math
import torch
import numpy as np
import cv2

# Import from other modules in this folder
from .gaussian_points import GaussianPoints
from .learnable_camera import LearnableCamera
from .depth_aware_render import render_fast, render_depth_aware
from .losses import combined_loss, multi_view_loss
from .progressive_training import ProgressiveTraining


def setup_device() -> torch.device:
    """Setup compute device with CUDA preference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    return device


def load_image(filepath: str, size: int, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess an image.
    
    Args:
        filepath: Path to image file
        size: Target size (square)
        device: Device to place tensor on
    
    Returns:
        Image tensor (H, W, 3) in [0, 1] range
    """
    if not os.path.exists(filepath):
        print(f"Warning: Image not found: {filepath}")
        return torch.zeros((size, size, 3), device=device)
    
    img = cv2.imread(filepath)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float().to(device) / 255.0


def save_visualization(
    rendered_views: list,
    step: int,
    output_dir: str = "output"
) -> None:
    """Save visualization of rendered views."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and create grid
    imgs = [v.detach().cpu().numpy() for v in rendered_views]
    
    # 2x2 grid for 4 views
    if len(imgs) >= 4:
        top = np.hstack([imgs[0], imgs[1]])
        bottom = np.hstack([imgs[2], imgs[3]])
        grid = np.vstack([top, bottom])
    else:
        grid = np.hstack(imgs)
    
    grid = (grid * 255).astype(np.uint8)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f"{output_dir}/step_{step:05d}.png", grid)


# Default camera parameters
DEFAULT_YAW_ANGLES = [0.4, -0.4, 0.4, -0.4]  # Views: top-right, top-left, bottom-right, bottom-left
DEFAULT_PITCH_ANGLES = [0.25, 0.25, -0.25, -0.25]
DEFAULT_FOCAL_LENGTH = 400.0
DEFAULT_INIT_SCALE = 70.0

# Animation parameters for visualization
ROTATION_SPEED = 0.02  # Radians per step
ROTATION_AMPLITUDE = 0.6  # Maximum rotation angle


def train(
    image_paths: list = None,
    canvas_size: int = 256,
    num_steps: int = 2000,
    initial_points: int = 500,
    max_points: int = 10000,
    geometry_lr: float = 0.01,
    camera_lr: float = 0.001,
    visualize_interval: int = 50,
    save_interval: int = 200,
    use_progressive: bool = True,
    output_dir: str = "output"
) -> None:
    """
    Main training function.
    
    Args:
        image_paths: List of 4 image paths (1.jpg, 2.jpg, 3.jpg, 4.jpg)
        canvas_size: Render resolution
        num_steps: Total training steps
        initial_points: Starting number of Gaussian points
        max_points: Maximum points (with progressive training)
        geometry_lr: Learning rate for geometry (positions, colors, etc.)
        camera_lr: Learning rate for camera parameters
        visualize_interval: Steps between visualizations
        save_interval: Steps between saving images
        use_progressive: Whether to use progressive densification/pruning
        output_dir: Directory for output images
    """
    # Setup
    device = setup_device()
    
    # Default image paths
    if image_paths is None:
        image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    
    # Load ground truth images
    print("Loading images...")
    gt_images = [load_image(p, canvas_size, device) for p in image_paths]
    
    # Initialize Gaussian points
    print(f"Initializing {initial_points} Gaussian points...")
    points = GaussianPoints(
        num_points=initial_points,
        init_scale=DEFAULT_INIT_SCALE,
        device=device
    )
    
    # Initialize learnable camera
    camera = LearnableCamera(
        num_views=4,
        initial_yaw=DEFAULT_YAW_ANGLES,
        initial_pitch=DEFAULT_PITCH_ANGLES,
        initial_focal_length=DEFAULT_FOCAL_LENGTH,
        device=device
    )
    
    # Setup optimizers (separate LRs)
    geometry_optimizer = torch.optim.Adam(
        points.parameters(),
        lr=geometry_lr
    )
    camera_optimizer = torch.optim.Adam(
        camera.parameters(),
        lr=camera_lr
    )
    
    # Cosine annealing schedulers
    geometry_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        geometry_optimizer,
        T_max=num_steps,
        eta_min=geometry_lr * 0.01
    )
    camera_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        camera_optimizer,
        T_max=num_steps,
        eta_min=camera_lr * 0.01
    )
    
    # Progressive training
    if use_progressive:
        progressive = ProgressiveTraining(
            initial_points=initial_points,
            max_points=max_points,
            densify_interval=100,
            prune_interval=200,
            device=device
        )
    
    print(f"Starting training for {num_steps} steps...")
    print("Press ESC in visualization window to stop early")
    
    # Training loop
    for step in range(num_steps):
        geometry_optimizer.zero_grad()
        camera_optimizer.zero_grad()
        
        # Get point data
        xyz, rgb, opacity, scale = points.get_point_data()
        
        # Render all 4 views
        rendered_views = []
        for i in range(4):
            yaw, pitch, focal = camera.get_view(i)
            rendered = render_fast(
                xyz, rgb, opacity, scale,
                yaw, pitch, canvas_size, focal.item()
            )
            rendered_views.append(rendered)
        
        # Compute combined loss
        loss, loss_dict = multi_view_loss(
            rendered_views,
            gt_images,
            xyz, opacity, scale,
            ssim_weight=0.2,
            opacity_weight=0.01,
            scale_weight=0.01,
            spatial_weight=0.001
        )
        
        # Backpropagation
        loss.backward()
        
        # Progressive training gradient accumulation
        if use_progressive:
            progressive.accumulate_gradients(points._xyz)
        
        # Optimization step
        geometry_optimizer.step()
        camera_optimizer.step()
        geometry_scheduler.step()
        camera_scheduler.step()
        
        # Clamp camera parameters
        camera.clamp_angles(max_angle=1.0)
        camera.clamp_focal_length(min_focal=200.0, max_focal=800.0)
        
        # Progressive densification and pruning
        if use_progressive:
            progressive.step_update()
            
            if progressive.should_densify() and len(points) < max_points:
                points._xyz, points._rgb, points._opacity, points._scale = \
                    progressive.densify(
                        points._xyz, points._rgb,
                        points._opacity, points._scale
                    )
                points.num_points = points._xyz.shape[0]
                
                # Recreate optimizer with new parameters
                geometry_optimizer = torch.optim.Adam(
                    points.parameters(),
                    lr=geometry_scheduler.get_last_lr()[0]
                )
            
            if progressive.should_prune():
                points._xyz, points._rgb, points._opacity, points._scale = \
                    progressive.prune(
                        points._xyz, points._rgb,
                        points._opacity, points._scale
                    )
                points.num_points = points._xyz.shape[0]
                
                # Recreate optimizer with new parameters
                geometry_optimizer = torch.optim.Adam(
                    points.parameters(),
                    lr=geometry_scheduler.get_last_lr()[0]
                )
        
        # Logging
        if step % 50 == 0:
            print(f"Step {step:5d} | Loss: {loss_dict['total']:.4f} | "
                  f"Photo: {loss_dict['photometric']:.4f} | "
                  f"Points: {len(points)}")
        
        # Visualization
        if step % visualize_interval == 0:
            with torch.no_grad():
                # Render a spinning novel view
                angle = math.sin(step * ROTATION_SPEED) * ROTATION_AMPLITUDE
                novel_view = render_fast(
                    xyz, rgb, opacity, scale,
                    torch.tensor(angle, device=device),
                    torch.tensor(0.0, device=device),
                    canvas_size,
                    camera.focal_length.item()
                )
                
                disp = novel_view.cpu().numpy()
                disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
                cv2.putText(disp, f"Step: {step}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(disp, f"Points: {len(points)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow("Optimized 3D Reconstruction", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Training stopped by user")
                    break
        
        # Save visualization
        if step % save_interval == 0:
            with torch.no_grad():
                save_visualization(rendered_views, step, output_dir)
    
    # Final save
    print("\nTraining complete!")
    print(f"Final points: {len(points)}")
    print(f"Camera angles: {camera.get_angles_as_dict()}")
    
    # Save final state
    torch.save({
        'xyz': points._xyz.data,
        'rgb': points._rgb.data,
        'opacity': points._opacity.data,
        'scale': points._scale.data,
        'camera': camera.state_dict()
    }, os.path.join(output_dir, "final_model.pt"))
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    train()

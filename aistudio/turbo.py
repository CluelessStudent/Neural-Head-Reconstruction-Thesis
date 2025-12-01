import torch
import numpy as np
import cv2
import os
import sys
import math

# --- 1. SETUP HARDWARE (RTX 4070 SUPER) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Enable CuDNN benchmark for extra speed on fixed size inputs
    torch.backends.cudnn.benchmark = True
    print(f"SUCCESS: Detected NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    print(" TURBO MODE ENABLED: Using Vectorized CUDA Operations")
else:
    print("WARNING: CUDA not found. This script requires an NVIDIA GPU to run fast.")
    device = torch.device("cpu")


# --- 2. 3D MATH (Projection) ---
def project_3d(points_xyz, canvas_size, yaw, pitch):
    H, W = canvas_size, canvas_size
    cx, cy = W / 2, H / 2
    focal_length = 400.0

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    # Rotations
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)

    # Apply Yaw (Y-axis)
    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y
    y1 = y

    # Apply Pitch (X-axis)
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p
    x2 = x1

    # Perspective (Push object 600 units back)
    z_cam = z2 + 600.0
    perspective = focal_length / torch.clamp(z_cam, min=1.0)

    screen_x = x2 * perspective + cx
    screen_y = y2 * perspective + cy

    return screen_x, screen_y, perspective


# --- 3. TURBO RENDERER (Vectorized) ---
def render_turbo(points_xyz, points_rgb, yaw, pitch, canvas_size):
    H, W = canvas_size, canvas_size

    # 1. Project ALL points
    sx, sy, scale = project_3d(points_xyz, canvas_size, yaw, pitch)

    # 2. FILTERING (Crucial for Speed)
    # Only keep points that are actually on the screen
    # This prevents the GPU from calculating pixels for invisible points
    mask = (scale > 0) & (sx >= -20) & (sx < W + 20) & (sy >= -20) & (sy < H + 20)

    if mask.sum() == 0:
        # If camera looks at nothing, return black screen
        return torch.zeros((H, W, 3), device=device)

    # Extract valid data
    valid_sx = sx[mask].view(-1, 1, 1)
    valid_sy = sy[mask].view(-1, 1, 1)
    valid_scale = scale[mask].view(-1, 1, 1)
    valid_rgb = points_rgb[mask].view(-1, 1, 1, 3)

    # 3. CREATE GRIDS (Broadcasting)
    # Y grid shape: [1, H, 1]
    # X grid shape: [1, 1, W]
    y_grid = torch.arange(H, device=device).float().view(1, H, 1)
    x_grid = torch.arange(W, device=device).float().view(1, 1, W)

    # 4. CALCULATE GAUSSIANS (The Heavy Lifting)
    # (x - cx)^2 + (y - cy)^2
    # This line calculates millions of pixel distances instantly
    dist_sq = (x_grid - valid_sx) ** 2 + (y_grid - valid_sy) ** 2

    radius = 4.0 * valid_scale

    # Gaussian Formula
    strength = torch.exp(-dist_sq / (2 * radius ** 2))

    # Optimization: Zero out very weak pixels to save accumulation precision
    strength = torch.where(strength > 0.05, strength, torch.zeros_like(strength))

    # 5. ACCUMULATE
    # Expand strength to [N, H, W, 1] and multiply by color [N, 1, 1, 3]
    # Then Sum all blobs together (dim=0)
    # * 0.3 is the Opacity/Alpha factor to prevent white-out
    final_image = (strength.unsqueeze(-1) * valid_rgb * 0.3).sum(dim=0)

    # Clip values to valid image range (0.0 to 1.0)
    return torch.clamp(final_image, 0, 1)


# --- 4. SETUP DATA ---
CANVAS_SIZE = 256


def load_img(name):
    if not os.path.exists(name):
        print(f"Error: {name} missing.")
        return torch.zeros((CANVAS_SIZE, CANVAS_SIZE, 3)).to(device)
    img = cv2.imread(name)
    img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float().to(device) / 255.0


print("Loading images...")
gt_1 = load_img("1.jpg")
gt_2 = load_img("2.jpg")
gt_3 = load_img("3.jpg")
gt_4 = load_img("4.jpg")

# Camera Angles (Yaw, Pitch)
yaw_val, pitch_val = 0.4, 0.25
angles = [
    (torch.tensor(yaw_val).to(device), torch.tensor(pitch_val).to(device)),  # View 1
    (torch.tensor(-yaw_val).to(device), torch.tensor(pitch_val).to(device)),  # View 2
    (torch.tensor(yaw_val).to(device), torch.tensor(-pitch_val).to(device)),  # View 3
    (torch.tensor(-yaw_val).to(device), torch.tensor(-pitch_val).to(device))  # View 4
]

# --- 5. INITIALIZE MODEL ---
num_blobs = 2000
print(f"Initializing {num_blobs} Gaussian points...")

# Random Cloud
trainable_xyz = (torch.rand((num_blobs, 3), device=device) * 140.0) - 70.0
trainable_xyz = trainable_xyz.detach().requires_grad_(True)

# Random Colors
trainable_rgb = torch.rand((num_blobs, 3), device=device).detach().requires_grad_(True)

optimizer = torch.optim.Adam([trainable_xyz, trainable_rgb], lr=3.0)

# --- 6. OPTIMIZATION LOOP ---
print("Starting CUDA Optimization... (Please wait 5-10s for first frame)")

rotation_angle = 0.0

for i in range(5001):
    optimizer.zero_grad()
    loss = 0

    # --- A. TRAINING (Inverse Rendering) ---
    # Render all 4 views efficiently
    r1 = render_turbo(trainable_xyz, trainable_rgb, angles[0][0], angles[0][1], CANVAS_SIZE)
    r2 = render_turbo(trainable_xyz, trainable_rgb, angles[1][0], angles[1][1], CANVAS_SIZE)
    r3 = render_turbo(trainable_xyz, trainable_rgb, angles[2][0], angles[2][1], CANVAS_SIZE)
    r4 = render_turbo(trainable_xyz, trainable_rgb, angles[3][0], angles[3][1], CANVAS_SIZE)

    # Calculate difference from Ground Truth
    loss += torch.mean((r1 - gt_1) ** 2)
    loss += torch.mean((r2 - gt_2) ** 2)
    loss += torch.mean((r3 - gt_3) ** 2)
    loss += torch.mean((r4 - gt_4) ** 2)

    loss.backward()
    optimizer.step()

    # --- B. VISUALIZATION (The Spinning Head) ---
    if i % 10 == 0:
        with torch.no_grad():
            # Calculate Spinning Angle (Sin wave for left-right sway)
            rotation_angle += 0.05
            rot_tensor = torch.tensor(math.sin(rotation_angle) * 0.6).to(device)

            # Render Novel View
            preview = render_turbo(trainable_xyz, trainable_rgb, rot_tensor, torch.tensor(0.0).to(device), 300)

            # Prepare for OpenCV
            disp = preview.cpu().numpy()
            disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

            # UI Text
            cv2.putText(disp, f"Step: {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disp, "Single 3D Object (Live)", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Thesis: Turbo 3D Reconstruction", disp)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to stop
                break

    if i % 100 == 0:
        print(f"Step {i} | Loss: {loss.item():.4f}")

print("Done.")
cv2.destroyAllWindows()
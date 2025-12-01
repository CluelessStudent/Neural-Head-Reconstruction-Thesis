import torch
import numpy as np
import cv2
import os
import sys
import math

# --- 1. SETUP HARDWARE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"SUCCESS: Detected NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not found. This will be slow.")
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

    # Perspective
    z_cam = z2 + 600.0
    perspective = focal_length / torch.clamp(z_cam, min=1.0)

    screen_x = x2 * perspective + cx
    screen_y = y2 * perspective + cy

    return screen_x, screen_y, perspective


# --- 3. RENDERER ---
def render(points_xyz, points_rgb, yaw, pitch, canvas_size):
    H, W = canvas_size, canvas_size
    # Start with a Dark Grey background so we can see the head better
    canvas = torch.ones((H, W, 3), device=device) * 0.1

    sx, sy, scale = project_3d(points_xyz, canvas_size, yaw, pitch)

    # Optimization: Vectorized rendering logic
    # This is a simplified "Splat"
    for i in range(len(points_xyz)):
        if scale[i] < 0: continue
        px, py = sx[i], sy[i]

        if px < 0 or px >= W or py < 0 or py >= H: continue

        # Size of the blob based on distance
        radius = 4.0 * scale[i]

        y_grid = torch.arange(H, device=device).float().unsqueeze(1)
        x_grid = torch.arange(W, device=device).float().unsqueeze(0)

        dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2

        # Sharp Gaussian
        strength = torch.exp(-dist_sq / (2 * radius ** 2))

        # Only draw if visible
        if strength.max() > 0.1:
            # Additive blending with opacity control (0.3 to prevent whiteout)
            blob = strength.unsqueeze(-1) * points_rgb[i].unsqueeze(0).unsqueeze(0) * 0.3
            canvas += blob

    return torch.clamp(canvas, 0, 1)


# --- 4. SETUP DATA ---
CANVAS_SIZE = 256


def load_img(name):
    if not os.path.exists(name): return torch.zeros((CANVAS_SIZE, CANVAS_SIZE, 3)).to(device)
    img = cv2.imread(name)
    img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float().to(device) / 255.0


print("Loading images...")
gt_1 = load_img("1.jpg")
gt_2 = load_img("2.jpg")
gt_3 = load_img("3.jpg")
gt_4 = load_img("4.jpg")

# Training Angles (Fixed)
yaw_val, pitch_val = 0.4, 0.25
angles = [
    (torch.tensor(yaw_val).to(device), torch.tensor(pitch_val).to(device)),
    (torch.tensor(-yaw_val).to(device), torch.tensor(pitch_val).to(device)),
    (torch.tensor(yaw_val).to(device), torch.tensor(-pitch_val).to(device)),
    (torch.tensor(-yaw_val).to(device), torch.tensor(-pitch_val).to(device))
]

# --- 5. INITIALIZE MODEL ---
num_blobs = 2000
trainable_xyz = (torch.rand((num_blobs, 3), device=device) * 140.0) - 70.0
trainable_xyz = trainable_xyz.detach().requires_grad_(True)
trainable_rgb = torch.rand((num_blobs, 3), device=device).detach().requires_grad_(True)

optimizer = torch.optim.Adam([trainable_xyz, trainable_rgb], lr=3.0)

# --- 6. THE LOOP ---
print("Starting 3D Reconstruction... (Window will appear shortly)")

rotation_angle = 0.0  # For visualization only

for i in range(2001):
    optimizer.zero_grad()
    loss = 0

    # 1. MATH: Calculate errors for all 4 views (The Learning part)
    # We do this invisibly in the background
    loss += torch.mean((render(trainable_xyz, trainable_rgb, angles[0][0], angles[0][1], CANVAS_SIZE) - gt_1) ** 2)
    loss += torch.mean((render(trainable_xyz, trainable_rgb, angles[1][0], angles[1][1], CANVAS_SIZE) - gt_2) ** 2)
    loss += torch.mean((render(trainable_xyz, trainable_rgb, angles[2][0], angles[2][1], CANVAS_SIZE) - gt_3) ** 2)
    loss += torch.mean((render(trainable_xyz, trainable_rgb, angles[3][0], angles[3][1], CANVAS_SIZE) - gt_4) ** 2)

    loss.backward()
    optimizer.step()

    # 2. VISUALIZATION: Show ONLY ONE ROTATING VIEW (The "Fun" part)
    # Update screen only every 20 steps to prevent "Not Responding"
    if i % 20 == 0:
        with torch.no_grad():  # Don't calculate gradients for the display
            # Rotate the camera slightly
            rotation_angle += 0.1
            rot_tensor = torch.tensor(math.sin(rotation_angle) * 0.5).to(device)  # Swing left/right

            # Render the "Novel View" (A view that doesn't exist in photos)
            preview = render(trainable_xyz, trainable_rgb, rot_tensor, torch.tensor(0.0).to(device), 350)

            # Convert to display format
            disp = preview.cpu().numpy()
            disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

            cv2.putText(disp, f"Step: {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disp, "ONE 3D MODEL (Spinning)", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Thesis: 3D Result", disp)
            cv2.waitKey(1)

    if i % 100 == 0:
        print(f"Step {i} | Loss: {loss.item():.4f}")

cv2.destroyAllWindows()
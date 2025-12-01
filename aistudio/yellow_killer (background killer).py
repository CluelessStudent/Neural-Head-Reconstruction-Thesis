import torch
import numpy as np
import cv2
import os
import math
import gc

# --- 1. SETUP HARDWARE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Clear memory from previous crashes
    torch.cuda.empty_cache()
    gc.collect()
    print(f"SUCCESS: RTX GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not found.")
    device = torch.device("cpu")


# --- 2. 3D MATH ---
def project_3d(points_xyz, canvas_size, yaw, pitch):
    H, W = canvas_size, canvas_size
    cx, cy = W / 2, H / 2
    focal_length = 450.0

    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)

    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y
    y1 = y

    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p
    x2 = x1

    z_cam = z2 + 500.0
    perspective = focal_length / torch.clamp(z_cam, min=1.0)

    screen_x = x2 * perspective + cx
    screen_y = y2 * perspective + cy

    return screen_x, screen_y, perspective


# --- 3. MEMORY SAFE RENDERER (Batched) ---
def render_batched(points_xyz, points_rgb, yaw, pitch, canvas_size):
    H, W = canvas_size, canvas_size

    sx, sy, scale = project_3d(points_xyz, canvas_size, yaw, pitch)

    # Filter valid points
    mask = (scale > 0) & (sx >= -15) & (sx < W + 15) & (sy >= -15) & (sy < H + 15)

    if mask.sum() == 0:
        return torch.zeros((H, W, 3), device=device)

    # Extract valid data
    valid_sx = sx[mask]
    valid_sy = sy[mask]
    valid_scale = scale[mask]
    valid_rgb = points_rgb[mask]

    # --- BATCHING LOGIC ---
    # We process points in chunks of 2000 to prevent crashing the 12GB VRAM
    BATCH_SIZE = 2000
    num_valid = valid_sx.shape[0]

    # Initialize blank canvas
    final_canvas = torch.zeros((H, W, 3), device=device)

    y_grid = torch.arange(H, device=device).float().view(1, H, 1)
    x_grid = torch.arange(W, device=device).float().view(1, 1, W)

    for i in range(0, num_valid, BATCH_SIZE):
        end = min(i + BATCH_SIZE, num_valid)

        # Slice the batch
        b_sx = valid_sx[i:end].view(-1, 1, 1)
        b_sy = valid_sy[i:end].view(-1, 1, 1)
        b_scale = valid_scale[i:end].view(-1, 1, 1)
        b_rgb = valid_rgb[i:end].view(-1, 1, 1, 3)

        # Render just this batch
        dist_sq = (x_grid - b_sx) ** 2 + (y_grid - b_sy) ** 2
        radius = 2.5 * b_scale

        strength = torch.exp(-dist_sq / (2 * radius ** 2))
        # Optimization threshold
        strength = torch.where(strength > 0.05, strength, torch.zeros_like(strength))

        # Accumulate this batch to the final image
        batch_image = (strength.unsqueeze(-1) * b_rgb * 0.5).sum(dim=0)
        final_canvas += batch_image

    return torch.clamp(final_canvas, 0, 1)


# --- 4. SETUP DATA ---
CANVAS_SIZE = 256


def load_and_mask_yellow(name):
    if not os.path.exists(name):
        return torch.zeros((CANVAS_SIZE, CANVAS_SIZE, 3)).to(device)
    img = cv2.imread(name)
    img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow Masking
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_inv = cv2.bitwise_not(mask)
    kernel = np.ones((2, 2), np.uint8)
    mask_inv = cv2.erode(mask_inv, kernel, iterations=1)
    img_masked = cv2.bitwise_and(img, img, mask=mask_inv)

    return torch.from_numpy(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)).float().to(device) / 255.0


print("Loading images...")
gt_1 = load_and_mask_yellow("1.jpg")
gt_2 = load_and_mask_yellow("2.jpg")
gt_3 = load_and_mask_yellow("3.jpg")
gt_4 = load_and_mask_yellow("4.jpg")

yaw_val, pitch_val = 0.4, 0.25
angles = [
    (torch.tensor(yaw_val).to(device), torch.tensor(pitch_val).to(device)),
    (torch.tensor(-yaw_val).to(device), torch.tensor(pitch_val).to(device)),
    (torch.tensor(yaw_val).to(device), torch.tensor(-pitch_val).to(device)),
    (torch.tensor(-yaw_val).to(device), torch.tensor(-pitch_val).to(device))
]

# --- 5. INITIALIZE ---
# We can keep 10,000 points now because we render them in batches!
num_blobs = 10000
print(f"Initializing {num_blobs} points...")

trainable_xyz = (torch.rand((num_blobs, 3), device=device) * 100.0) - 50.0
trainable_xyz = trainable_xyz.detach().requires_grad_(True)
trainable_rgb = torch.rand((num_blobs, 3), device=device).detach().requires_grad_(True)

optimizer = torch.optim.Adam([trainable_xyz, trainable_rgb], lr=1.5)

# --- 6. LOOP ---
print("Starting Optimization...")
rotation_angle = 0.0

for i in range(5001):
    optimizer.zero_grad()

    # Random view
    view_idx = np.random.randint(0, 4)
    if view_idx == 0:
        gt, ang = gt_1, angles[0]
    elif view_idx == 1:
        gt, ang = gt_2, angles[1]
    elif view_idx == 2:
        gt, ang = gt_3, angles[2]
    else:
        gt, ang = gt_4, angles[3]

    # Render (Using the new Safe Batched Renderer)
    render_img = render_batched(trainable_xyz, trainable_rgb, ang[0], ang[1], CANVAS_SIZE)

    loss = torch.mean((render_img - gt) ** 2)
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        with torch.no_grad():
            rotation_angle += 0.05
            rot_tensor = torch.tensor(math.sin(rotation_angle) * 0.5).to(device)

            # Preview uses the same safe renderer
            preview = render_batched(trainable_xyz, trainable_rgb, rot_tensor, torch.tensor(0.0).to(device), 300)

            disp = preview.cpu().numpy()
            disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
            cv2.putText(disp, f"Step: {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Thesis: Fixed Memory", disp)
            if cv2.waitKey(1) & 0xFF == 27: break

    if i % 200 == 0:
        # Clear cache occasionally to be safe
        torch.cuda.empty_cache()
        print(f"Step {i} | Loss: {loss.item():.4f}")

cv2.destroyAllWindows()
import torch
import numpy as np
import cv2
import os

# --- 1. SETUP ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Running 4-View Reconstruction on: {device}")


# --- 2. ADVANCED 3D PROJECTION (Yaw + Pitch) ---
def project_3d_to_2d_advanced(points_xyz, canvas_size, yaw, pitch):
    """
    Projects 3D points considering both Horizontal (Yaw) and Vertical (Pitch) angles.
    """
    H, W = canvas_size, canvas_size
    cx, cy = W / 2, H / 2
    focal_length = 350.0  # Adjusted for portrait photos

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    # --- A. YAW ROTATION (Around Y-axis) ---
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y
    y_rot = y

    # --- B. PITCH ROTATION (Around X-axis) ---
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    y_final = y_rot * cos_p - z_rot * sin_p
    z_final = y_rot * sin_p + z_rot * cos_p
    x_final = x_rot

    # --- C. PERSPECTIVE ---
    z_cam = z_final + 500.0  # Push object back
    perspective = focal_length / torch.clamp(z_cam, min=1.0)

    screen_x = x_final * perspective + cx
    screen_y = y_final * perspective + cy

    return screen_x, screen_y, perspective


# --- 3. RENDERER ---
def renderer_3d(points_xyz, points_rgb, yaw, pitch, canvas_size):
    H, W = canvas_size, canvas_size
    canvas = torch.zeros((H, W, 3), device=device)

    sx, sy, scale = project_3d_to_2d_advanced(points_xyz, canvas_size, yaw, pitch)

    # Render loop
    # We use a simplified loop for the demo. In production, this is vectorized.
    for i in range(len(points_xyz)):
        if scale[i] < 0: continue  # Behind camera

        px, py = sx[i], sy[i]

        # Check if point is on screen
        if px < 0 or px >= W or py < 0 or py >= H: continue

        radius = 3.5 * scale[i]

        # Create grid around the point to save compute
        y_grid = torch.arange(H, device=device).float().unsqueeze(1)
        x_grid = torch.arange(W, device=device).float().unsqueeze(0)

        dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
        strength = torch.exp(-dist_sq / (2 * radius ** 2))

        # Only render if strength is significant
        if strength.max() > 0.1:
            blob = strength.unsqueeze(-1) * points_rgb[i].unsqueeze(0).unsqueeze(0)
            canvas += blob

    return torch.clamp(canvas, 0, 1)


# --- 4. LOAD AND PROCESS YOUR PHOTOS ---
def load_and_process_image(filename, size=128):
    if not os.path.exists(filename):
        print(f"ERROR: Could not find {filename}. Make sure it is in the folder!")
        # Return a dummy black image so script doesn't crash
        return torch.zeros((size, size, 3), device=device)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to square
    img = cv2.resize(img, (size, size))

    # Convert to Tensor
    return torch.from_numpy(img).float().to(device) / 255.0


# --- 5. MAIN SETUP ---
CANVAS_SIZE = 128

# Load Images (Ground Truths)
print("Loading images 1.jpg, 2.jpg, 3.jpg, 4.jpg...")
gt_1 = load_and_process_image("1.jpg", CANVAS_SIZE)  # Look Up/Left
gt_2 = load_and_process_image("2.jpg", CANVAS_SIZE)  # Look Up/Right
gt_3 = load_and_process_image("3.jpg", CANVAS_SIZE)  # Look Down/Left
gt_4 = load_and_process_image("4.jpg", CANVAS_SIZE)  # Look Down/Right

# Define Angles (Radians) - THESE ARE GUESSTIMATES
# You might need to tweak these numbers if the face rotates the wrong way!
# Yaw: + is Right, - is Left
# Pitch: + is Down, - is Up
angle_1 = (torch.tensor(0.35), torch.tensor(0.2))  # Yaw Right, Pitch Down (Camera perspective)
angle_2 = (torch.tensor(-0.35), torch.tensor(0.2))  # Yaw Left, Pitch Down
angle_3 = (torch.tensor(0.35), torch.tensor(-0.2))  # Yaw Right, Pitch Up
angle_4 = (torch.tensor(-0.35), torch.tensor(-0.2))  # Yaw Left, Pitch Up

# Initialize 3D Cloud
num_blobs = 1000
trainable_xyz = (torch.rand((num_blobs, 3), device=device) * 150.0) - 75.0
trainable_xyz = trainable_xyz.detach().requires_grad_(True)
trainable_rgb = torch.rand((num_blobs, 3), device=device).detach().requires_grad_(True)

optimizer = torch.optim.Adam([trainable_xyz, trainable_rgb], lr=3.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

print("Starting 4-Way Optimization...")

# --- 6. OPTIMIZATION LOOP ---
for i in range(501):
    optimizer.zero_grad()
    loss = 0

    # Render View 1
    r1 = renderer_3d(trainable_xyz, trainable_rgb, angle_1[0], angle_1[1], CANVAS_SIZE)
    loss += torch.mean((r1 - gt_1) ** 2)

    # Render View 2
    r2 = renderer_3d(trainable_xyz, trainable_rgb, angle_2[0], angle_2[1], CANVAS_SIZE)
    loss += torch.mean((r2 - gt_2) ** 2)

    # Render View 3
    r3 = renderer_3d(trainable_xyz, trainable_rgb, angle_3[0], angle_3[1], CANVAS_SIZE)
    loss += torch.mean((r3 - gt_3) ** 2)

    # Render View 4
    r4 = renderer_3d(trainable_xyz, trainable_rgb, angle_4[0], angle_4[1], CANVAS_SIZE)
    loss += torch.mean((r4 - gt_4) ** 2)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % 20 == 0:
        # Convert to numpy for display
        img1 = r1.detach().cpu().numpy()
        img2 = r2.detach().cpu().numpy()
        img3 = r3.detach().cpu().numpy()
        img4 = r4.detach().cpu().numpy()

        # Stitch them into a 2x2 grid
        top_row = np.hstack((img1, img2))
        bot_row = np.hstack((img3, img4))
        grid = np.vstack((top_row, bot_row))

        display = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        display = cv2.resize(display, (600, 600), interpolation=cv2.INTER_NEAREST)

        cv2.putText(display, f"Step {i} | Loss: {loss.item():.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        cv2.imshow("Thesis 4-View 3D Reconstruction", display)
        cv2.waitKey(1)

print("Optimization Finished.")
cv2.waitKey(0)
cv2.destroyAllWindows()
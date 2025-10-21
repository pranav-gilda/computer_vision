import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- Configuration ---
BASE_PATH = './2011_09_26/2011_09_26_drive_0005_sync/'
IMAGE_PATH = os.path.join(BASE_PATH, 'image_02/data/')
LIDAR_PATH = os.path.join(BASE_PATH, 'velodyne_points/data/')
CALIB_PATH = './2011_09_26/'

def load_calibration_matrices(calib_path):
    """Loads and parses the KITTI calibration files with robust parsing."""
    calib = {}
    
    # --- Camera-to-Camera Calibration ---
    with open(os.path.join(calib_path, 'calib_cam_to_cam.txt'), 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            # We only care about these specific keys. This ignores headers/dates.
            if key in ['P_rect_02', 'R_rect_00']:
                calib[key] = np.array([float(x) for x in value.split()])

    # --- Velodyne-to-Camera Calibration ---
    # This file has a different format, with R and T separate
    with open(os.path.join(calib_path, 'calib_velo_to_cam.txt'), 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            # We now read R and T separately
            if key in ['R', 'T']:
                 calib[key] = np.array([float(x) for x in value.split()])

    # Extract the specific matrices we need
    P_rect_02 = calib['P_rect_02'].reshape(3, 4)
    
    R_rect_00 = np.eye(4)
    R_rect_00[:3, :3] = calib['R_rect_00'].reshape(3, 3)
    
    # Combine R and T into a single 4x4 homogenous transformation matrix
    T_velo_to_cam = np.eye(4)
    T_velo_to_cam[:3, :3] = calib['R'].reshape(3, 3)
    T_velo_to_cam[:3, 3] = calib['T']
    
    print("Calibration matrices loaded successfully.")
    return P_rect_02, R_rect_00, T_velo_to_cam


def project_lidar_to_image(lidar_points, P_rect, R_rect, T_velo_cam):
    """Projects 3D LiDAR points to the 2D image plane."""
    front_points_mask = lidar_points[:, 0] > 0
    lidar_points = lidar_points[front_points_mask]

    num_points = lidar_points.shape[0]
    points_homo = np.hstack((lidar_points[:, :3], np.ones((num_points, 1))))

    points_cam = T_velo_cam @ points_homo.T
    points_cam = R_rect @ points_cam
    points_img = P_rect @ points_cam
    
    points_img_2d = points_img[:2, :] / points_img[2, :]

    return points_img_2d.T, lidar_points[:, 0]


def main():
    print("Starting Project 3: 3D Data Visualization")
    
    # --- 1. Load Calibration ---
    try:
        P_rect_02, R_rect_00, T_velo_to_cam = load_calibration_matrices(CALIB_PATH)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading calibration files: {e}")
        print("Please ensure your directory structure and file contents are correct.")
        return

    # --- 2. Load a Sample Frame ---
    sample_idx = 59
    image_file = os.path.join(IMAGE_PATH, f'{sample_idx:010d}.png')
    lidar_file = os.path.join(LIDAR_PATH, f'{sample_idx:010d}.bin')

    if not os.path.exists(image_file) or not os.path.exists(lidar_file):
        print("Error: Sample image or LiDAR file not found.")
        print("Please ensure you have downloaded the '2011_09_26_drive_0005_sync' dataset.")
        return

    image = cv2.imread(image_file)
    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    # --- 3. Project LiDAR points onto the image ---
    projected_points_2d, depths = project_lidar_to_image(lidar_points, P_rect_02, R_rect_00, T_velo_to_cam)

    # --- 4. Visualize the Result ---
    h, w, _ = image.shape
    mask = (projected_points_2d[:, 0] >= 0) & (projected_points_2d[:, 0] < w) & \
           (projected_points_2d[:, 1] >= 0) & (projected_points_2d[:, 1] < h)
    
    filtered_points = projected_points_2d[mask]
    filtered_depths = depths[mask]

    cmap = plt.get_cmap('viridis')
    colors = cmap(filtered_depths / np.max(filtered_depths))[:, :3] * 255
    
    vis_image = image.copy()
    for i, point in enumerate(filtered_points):
        x, y = int(point[0]), int(point[1])
        color = colors[i]
        cv2.circle(vis_image, (x, y), radius=2, color=tuple(color), thickness=-1)

    print("Visualization complete. Displaying the result.")
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title("LiDAR Points Projected onto Camera Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()


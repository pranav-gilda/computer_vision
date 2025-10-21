import cv2
import numpy as np
import os

# --- Configuration ---
# The script now assumes the image file is in the same directory.
IMAGE_PATH = "test_image.png"
BEV_WIDTH = 400
BEV_HEIGHT = 600

# Global variables to store points from mouse clicks
source_points = []

def mouse_callback(event, x, y, flags, param):
    """
    OpenCV mouse callback function. Appends the (x,y) coordinates of a left-click.
    """
    global source_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(source_points) < 4:
            source_points.append((x, y))
            print(f"Point {len(source_points)} selected: ({x}, {y})")
        else:
            print("Already have 4 points. Press 'w' to warp or 'r' to reset.")

def main():
    print("Starting Project 4: Camera to Bird's-Eye View Transformation")

    # --- 1. Load the Local Image ---
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'.")
        print("Please download the image and save it in the same folder as this script.")
        return
        
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Failed to read image from '{IMAGE_PATH}'. The file might be corrupted.")
        return
    print(f"Successfully loaded image '{IMAGE_PATH}'.")

    # --- 2. Interactive Point Selection ---
    window_name = "Original Image - Select 4 Points"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\nPlease select 4 points on the 'Original Image' window by clicking on the corners of the road lane.")
    print("Select in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
    print("Press 'w' to warp | 'r' to reset points | 'q' to quit.")

    while True:
        display_frame = frame.copy()
        
        for point in source_points:
            cv2.circle(display_frame, point, 7, (0, 0, 255), -1) # Made circles bigger

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        if key == ord('r'):
            source_points.clear()
            print("Points reset.")
        
        if key == ord('w') and len(source_points) == 4:
            print("Warping image...")
            destination_points = np.float32([
                [0, 0],
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT],
                [0, BEV_HEIGHT]
            ])

            src_np = np.float32(source_points)
            matrix = cv2.getPerspectiveTransform(src_np, destination_points)
            warped_image = cv2.warpPerspective(frame, matrix, (BEV_WIDTH, BEV_HEIGHT))
            
            cv2.imshow("Bird's-Eye View", warped_image)
            print("BEV image displayed. Press 'q' on the original window to exit all.")
        
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()


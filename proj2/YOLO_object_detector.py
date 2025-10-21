from ultralytics import YOLO
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---
MODEL_NAME = 'yolov8n.pt'
VIDEO_PATH = "test_video (1).mp4" # Assumes you have a video file named this in the same folder
OUTPUT_VIDEO_PATH = "tracked_output_new.mp4" # The name of the saved output file

def main():
    """
    This is the core object tracking pipeline.
    It processes a video and saves the output to a new file.
    """
    print("--- Starting Modern Object Tracking with YOLOv8 and DeepSORT ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load the Model & Tracker ---
    try:
        model = YOLO(MODEL_NAME)
        print(f"Successfully loaded '{MODEL_NAME}'.")
        # Initialize the DeepSORT tracker
        tracker = DeepSort(max_age=30)
        print("Successfully initialized DeepSORT tracker.")
    except Exception as e:
        print(f"Error loading model or tracker: {e}")
        return

    # --- 2. Open Video File and Prepare Output ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{VIDEO_PATH}'")
        return
    
    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # A common codec for .mp4 files
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print("Successfully opened video file. Starting tracking and saving output...")

    # --- 3. Process Video Frame by Frame ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Run Inference with YOLOv8 ---
        results = model(frame)
        
        # --- Format Detections for DeepSORT ---
        detections_for_tracker = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            w, h = x2 - x1, y2 - y1
            
            if class_name in ['person', 'car', 'truck', 'bicycle', 'motorcycle']:
                 detections_for_tracker.append(([x1, y1, w, h], confidence, class_name))

        # --- Update the Tracker ---
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # --- Visualize the Results ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(i) for i in ltrb]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output file
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # --- Cleanup ---
    cap.release()
    out.release() # Release the output file writer
    cv2.destroyAllWindows()
    print(f"\n--- Tracking finished successfully! Output saved to {OUTPUT_VIDEO_PATH} ---")

if __name__ == "__main__":
    main()


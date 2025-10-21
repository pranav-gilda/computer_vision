import cv2
import numpy as np
import torch
import os
import json
import time
from PIL import Image
from torchvision import models, transforms 
import traceback

# --- Configuration ---
VIDEO_PATH = 'Test_Drive.mp4'
CONFIG_FILE = 'calibration.json'
YOLO_MODEL_NAME = 'yolov5s'
CONF_THRESHOLD = 0.4
SEGMENTATION_MODEL_PATH = 'fcn_road_segmentation_v2.pth'

# --- Global variable for calibration ---
calibration_points = []
def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 4:
        calibration_points.append([x, y])
        print(f"  [CALIBRATION] Point {len(calibration_points)} selected: ({x}, {y})")

def run_calibration(video_path):
    print("\n--- LAUNCHING INTERACTIVE CALIBRATION ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("  [ERROR] Cannot open video for calibration."); return None
    ret, frame = cap.read()
    cap.release()
    if not ret: print("  [ERROR] Cannot read frame for calibration."); return None

    window_name = "CALIBRATION: Click 4 Lane Corners (TL, TR, BR, BL), then press 's'"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_frame = frame.copy()
        for point in calibration_points:
            cv2.circle(display_frame, tuple(point), 7, (0, 0, 255), -1)
        if len(calibration_points) == 4:
            cv2.putText(display_frame, "Press 's' to save and exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(calibration_points) == 4:
            with open(CONFIG_FILE, 'w') as f: json.dump({"source_points": calibration_points}, f, indent=4)
            print("  [INFO] Calibration saved successfully!")
            cv2.destroyAllWindows()
            return calibration_points
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None

# --- Road Segmentation using OpenCV DNN (Working approach from lane_hybrid.py) ---
class RoadSegmenter:
    def __init__(self, model_path):
        self.model_input_size = (256, 128)  # Width, Height
        print(f"Loading segmentation model from: {model_path}")
        self.net = cv2.dnn.readNet(model_path)
        print("Segmentation model loaded successfully.")

    def segment(self, frame):
        # Pre-process the frame for the model
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=self.model_input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Perform inference
        output = self.net.forward()
        
        # Post-process the output mask to create a binary image of the road
        mask = output[0, 1, :, :]  # Assumes class 1 corresponds to "road"
        original_h, original_w, _ = frame.shape
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        
        return binary_mask.astype(np.uint8)

# --- Lane Line Smoothing Class ---
class LaneLine:
    def __init__(self):
        self.last_fit = None
        self.recent_fits = []
    
    def average_fit(self):
        if not self.recent_fits:
            return None
        return np.mean(self.recent_fits, axis=0)
    
    def add_fit(self, new_fit):
        if new_fit is not None:
            self.recent_fits.append(new_fit)
            if len(self.recent_fits) > 10:
                self.recent_fits.pop(0)
        self.last_fit = self.average_fit()

# --- Working Lane Detection System (from lane_hybrid.py) ---
class LaneDetector:
    def __init__(self, img_size, src, dst):
        self.img_size = img_size
        self.src_points = np.float32(src)
        self.dst_points = np.float32(dst)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)  # type: ignore
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)  # type: ignore
        self.left_line = LaneLine()
        self.right_line = LaneLine()
        self.detection_failed = True

    def thresholding(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sx_thresh = (20, 100)
        sx_binary = cv2.inRange(scaled_sobel, sx_thresh[0], sx_thresh[1])  # type: ignore
        return sx_binary

    def warp(self, img):
        h, w = self.img_size
        return cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def find_lane_pixels_sliding_window(self, warped_img):
        histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((warped_img, warped_img, warped_img))
        midpoint = np.int32(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows, margin, minpix = 9, 100, 50
        window_height = np.int32(warped_img.shape[0] // nwindows)
        nonzero = warped_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            cv2.rectangle(out_img, (int(win_xleft_low), int(win_y_low)), (int(win_xleft_high), int(win_y_high)), (0, 255, 0), 2)  # type: ignore
            cv2.rectangle(out_img, (int(win_xright_low), int(win_y_low)), (int(win_xright_high), int(win_y_high)), (0, 255, 0), 2)  # type: ignore
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            if len(good_left) > minpix: leftx_current = np.int32(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix: rightx_current = np.int32(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else []
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else []
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        return leftx, lefty, rightx, righty, out_img

    def search_around_poly(self, warped_img):
        margin = 100
        nonzero = warped_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        left_fit = self.left_line.last_fit
        right_fit = self.right_line.last_fit
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        out_img = np.dstack((warped_img, warped_img, warped_img))
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, leftx, lefty, rightx, righty):
        left_fit, right_fit = None, None
        if len(leftx) > 150:
            try: left_fit = np.polyfit(lefty, leftx, 2)
            except: print("Warning: Polyfit may be poorly conditioned for left line.")
        if len(rightx) > 150:
            try: right_fit = np.polyfit(righty, rightx, 2)
            except: print("Warning: Polyfit may be poorly conditioned for right line.")
        return left_fit, right_fit

    def draw_final_result(self, original_img, warped_img, left_fit, right_fit):
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        color_warp = np.zeros_like(original_img)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, [pts.astype(np.int32)], (0, 255, 0))  # type: ignore
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (original_img.shape[1], original_img.shape[0]))
        return cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

    def process_frame(self, frame, road_mask):
        gradient_thresholds = self.thresholding(frame)
        warped_thresholds = self.warp(gradient_thresholds)
        warped_road_mask = self.warp(road_mask)
        
        # KEY INTEGRATION: Use the road mask to eliminate noise
        cleaned_input = cv2.bitwise_and(warped_thresholds, warped_thresholds, mask=warped_road_mask)
        
        if self.detection_failed:
            leftx, lefty, rightx, righty, search_img = self.find_lane_pixels_sliding_window(cleaned_input)
        else:
            # Use previous lane positions for faster detection
            leftx, lefty, rightx, righty, search_img = self.search_around_poly(cleaned_input)
        
        left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)

        if left_fit is not None and right_fit is not None:
            h = frame.shape[0]
            left_bottom = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
            right_bottom = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
            lane_width = right_bottom - left_bottom
            if 300 < lane_width < 1200:  # Reasonable lane width
                self.left_line.add_fit(left_fit)
                self.right_line.add_fit(right_fit)
                self.detection_failed = False
            else:
                self.detection_failed = True
        else:
            self.detection_failed = True

        stable_left_fit = self.left_line.last_fit
        stable_right_fit = self.right_line.last_fit
        
        debug_panel = self.create_debug_panel(frame, road_mask, cleaned_input, search_img)
        
        if stable_left_fit is None or stable_right_fit is None:
            return frame, debug_panel

        final_image = self.draw_final_result(frame, cleaned_input, stable_left_fit, stable_right_fit)
        return final_image, debug_panel

    def create_debug_panel(self, original, road_mask, cleaned_warped, search_img):
        diag_w, diag_h = 426, 240
        original_thumb = cv2.resize(original, (diag_w, diag_h))
        
        # Visualize the segmentation mask for debugging
        road_mask_thumb = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        road_mask_thumb = cv2.resize(road_mask_thumb, (diag_w, diag_h))
        
        # Visualize the final cleaned input
        cleaned_thumb = cv2.cvtColor(cleaned_warped, cv2.COLOR_GRAY2BGR)
        cleaned_thumb = cv2.resize(cleaned_thumb, (diag_w, diag_h))
        search_img_thumb = cv2.resize(search_img, (diag_w, diag_h))

        top_row = np.concatenate((original_thumb, road_mask_thumb), axis=1)
        bottom_row = np.concatenate((cleaned_thumb, search_img_thumb), axis=1)
        
        debug_panel_full = np.concatenate((top_row, bottom_row), axis=0)
        return cv2.resize(debug_panel_full, (0,0), fx=0.75, fy=0.75)


def main():
    # Check for required files
    ONNX_MODEL_PATH = 'proj1/road_segmentation.onnx'
    if not all(os.path.exists(p) for p in [VIDEO_PATH, ONNX_MODEL_PATH]):
        print(f"FATAL: A required file is missing. Check VIDEO_PATH and ONNX_MODEL_PATH.")
        return

    src_points = None
    if not os.path.exists(CONFIG_FILE): 
        src_points = run_calibration(VIDEO_PATH)
    else:
        with open(CONFIG_FILE, 'r') as f: 
            src_points = json.load(f)['source_points']
        print(f"  [INFO] Loaded calibration from '{CONFIG_FILE}'.")
    if src_points is None: 
        return

    print("--- Loading models... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load YOLO model
    yolo_model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True, _verbose=False)
    yolo_model.to(device).eval()  # type: ignore
    
    # Load DeepSORT tracker
    from deep_sort_realtime.deepsort_tracker import DeepSort
    tracker = DeepSort(max_age=30)
    
    # Load road segmentation model (ONNX)
    segmenter = RoadSegmenter(ONNX_MODEL_PATH)
    
    print(f"--- Models loaded successfully on {device} ---")

    cap = cv2.VideoCapture(VIDEO_PATH)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_size = (h, w)
    dst_points = [[w*0.2, 0], [w*0.8, 0], [w*0.8, h], [w*0.2, h]]
    
    # Initialize lane detector
    lane_detector = LaneDetector(img_size=img_size, src=src_points, dst=dst_points)
    
    debug_mode = True # Start with debug on by default
    frame_count = 0
    performance_stats = {
        'lane_detection_success': 0,
        'total_frames': 0,
        'lane_times': [],
        'object_times': [],
        'tracking_times': []
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        frame_start_time = time.time()
        print(f"\n[FRAME {frame_count}] Processing...")
        
        output_frame = frame.copy()

        # Step 1: Lane Detection with timing
        print("  [STEP 1] Running Hybrid Lane Detection...")
        lane_start = time.time()
        
        # Get road mask from segmentation model
        road_mask = segmenter.segment(frame)
        
        # Process frame with lane detection
        final_image, debug_panel = lane_detector.process_frame(frame, road_mask)
        
        lane_time = time.time() - lane_start
        performance_stats['lane_times'].append(lane_time)
        
        if final_image is not None and not np.array_equal(final_image, frame):
            output_frame = final_image
            performance_stats['lane_detection_success'] += 1
            print(f"    -> Lane overlay generated successfully. ({lane_time:.3f}s)")
        else:
            print(f"    -> Lane detection failed for this frame. ({lane_time:.3f}s)")

        # Step 2: Object Detection with timing
        print("  [STEP 2] Running Object Detection (YOLOv5)...")
        obj_start = time.time()
        results = yolo_model(frame)  # type: ignore
        detections = []
        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            for detection in results.xyxy[0]:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    if conf > CONF_THRESHOLD:
                        class_name = getattr(yolo_model, 'names', {}).get(int(cls), 'unknown')
                        if class_name in ['person','bicycle','car','motorcycle','bus','truck']:
                            detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(conf), class_name))
        obj_time = time.time() - obj_start
        performance_stats['object_times'].append(obj_time)
        print(f"    -> Found {len(detections)} raw objects. ({obj_time:.3f}s)")

        # Step 3: Object Tracking with timing
        print("  [STEP 3] Running Object Tracking (DeepSORT)...")
        track_start = time.time()
        tracks = tracker.update_tracks(detections, frame=frame)
        track_time = time.time() - track_start
        performance_stats['tracking_times'].append(track_time)
        print(f"    -> Tracking {len(tracks)} objects. ({track_time:.3f}s)")
        
        # Step 4: Visualization and BEV mapping
        bev_map = np.zeros((h, w, 3), dtype=np.uint8)
        print("  [STEP 4] Fusing and Visualizing...")
        for track in tracks:
            if not track.is_confirmed(): continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(i) for i in ltrb]
            bottom_center = np.array([[[ (x1+x2)/2, y2 ]]], dtype=np.float32)
            bev_point = cv2.perspectiveTransform(bottom_center, lane_detector.M)[0][0]
            if 0 <= bev_point[0] < w and 0 <= bev_point[1] < h:
                 cv2.circle(bev_map, (int(bev_point[0]), int(bev_point[1])), 8, (0,0,255), -1)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(output_frame, f"{track.get_det_class()}:{track.track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        # Calculate performance metrics
        total_time = time.time() - frame_start_time
        performance_stats['total_frames'] += 1
        
        # Add performance overlay to main frame
        fps = 1.0 / total_time if total_time > 0 else 0
        lane_success_rate = (performance_stats['lane_detection_success'] / performance_stats['total_frames']) * 100 if performance_stats['total_frames'] > 0 else 0
        
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Lane Success: {lane_success_rate:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Objects: {len(tracks)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Final Perception Dashboard', output_frame)
        
        if debug_mode:
            # Use the debug panel from lane detector
            if debug_panel is not None:
                cv2.imshow('Debug Panel', debug_panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('d'):
            debug_mode = not debug_mode
            if not debug_mode:
                try: cv2.destroyWindow('Debug Panel')
                except cv2.error: pass

    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance summary
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    
    if performance_stats['total_frames'] > 0:
        lane_success_rate = (performance_stats['lane_detection_success'] / performance_stats['total_frames']) * 100
        avg_lane_time = np.mean(performance_stats['lane_times']) if performance_stats['lane_times'] else 0
        avg_obj_time = np.mean(performance_stats['object_times']) if performance_stats['object_times'] else 0
        avg_track_time = np.mean(performance_stats['tracking_times']) if performance_stats['tracking_times'] else 0
        
        print(f"Total Frames Processed: {performance_stats['total_frames']}")
        print(f"Lane Detection Success Rate: {lane_success_rate:.1f}%")
        print(f"Average Lane Detection Time: {avg_lane_time:.3f}s")
        print(f"Average Object Detection Time: {avg_obj_time:.3f}s")
        print(f"Average Tracking Time: {avg_track_time:.3f}s")
        print(f"Average Total Processing Time: {avg_lane_time + avg_obj_time + avg_track_time:.3f}s")
        print(f"Estimated FPS: {1.0 / (avg_lane_time + avg_obj_time + avg_track_time):.1f}")
    
    print("="*60)
    print("Processing finished.")

if __name__ == "__main__":
    main()

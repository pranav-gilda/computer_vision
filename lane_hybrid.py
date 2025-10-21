import cv2
import numpy as np
import os
# import onnxruntime as ort

# --- STEP 1: DEEP LEARNING ROAD SEGMENTATION ---
class RoadSegmenter:
    """
    This class handles loading and running the ONNX road segmentation model.
    """
    def __init__(self, model_path):
        self.model_input_size = (256, 128) # The model was trained on this resolution (Width, Height)
        print(f"Loading segmentation model from: {model_path}")
        self.net = cv2.dnn.readNet(model_path)
        print("Segmentation model loaded successfully.")

    def segment(self, frame):
        # 1. Pre-process the frame for the model
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=self.model_input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 2. Perform inference
        output = self.net.forward()
        
        # 3. Post-process the output mask to create a binary image of the road
        mask = output[0, 1, :, :] # Assumes class 1 in the model's output corresponds to "road"
        original_h, original_w, _ = frame.shape
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        
        return binary_mask.astype(np.uint8)

# --- HELPER CLASS FOR SMOOTHING (Unchanged) ---
class LaneLine:
    def __init__(self):
        self.last_fit = None
        self.recent_fits = []
    def average_fit(self):
        if not self.recent_fits: return None
        return np.mean(self.recent_fits, axis=0)
    def add_fit(self, new_fit):
        if new_fit is not None:
            self.recent_fits.append(new_fit)
            if len(self.recent_fits) > 10: self.recent_fits.pop(0)
        self.last_fit = self.average_fit()

# --- STEP 2: CLASSICAL COMPUTER VISION LANE DETECTION (Upgraded) ---
class LaneDetector:
    def __init__(self, img_size, src, dst):
        self.img_size = img_size
        self.src_points = np.float32(src)
        self.dst_points = np.float32(dst)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
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
        sx_binary = cv2.inRange(scaled_sobel, sx_thresh[0], sx_thresh[1])
        return sx_binary

    def warp(self, img):
        h, w = self.img_size
        return cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)

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
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
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

    def fit_polynomial(self, leftx, lefty, rightx, righty):
        left_fit, right_fit = None, None
        if len(leftx) > 150:
            try: left_fit = np.polyfit(lefty, leftx, 2)
            except np.linalg.RankWarning: print("Warning: Polyfit may be poorly conditioned for left line.")
        if len(rightx) > 150:
            try: right_fit = np.polyfit(righty, rightx, 2)
            except np.linalg.RankWarning: print("Warning: Polyfit may be poorly conditioned for right line.")
        return left_fit, right_fit

    def draw_final_result(self, original_img, warped_img, left_fit, right_fit):
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        color_warp = np.zeros_like(original_img)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (original_img.shape[1], original_img.shape[0]))
        return cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

    def process_frame(self, frame, road_mask):
        gradient_thresholds = self.thresholding(frame)
        warped_thresholds = self.warp(gradient_thresholds)
        warped_road_mask = self.warp(road_mask)
        
        # --- KEY INTEGRATION STEP: Use the road mask to eliminate all noise ---
        cleaned_input = cv2.bitwise_and(warped_thresholds, warped_thresholds, mask=warped_road_mask)
        
        if self.detection_failed:
            leftx, lefty, rightx, righty, search_img = self.find_lane_pixels_sliding_window(cleaned_input)
        else:
            leftx, lefty, rightx, righty, search_img = self.search_around_poly(cleaned_input)
        
        left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)

        if left_fit is not None and right_fit is not None:
            h = frame.shape[0]
            left_bottom = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
            right_bottom = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
            lane_width = right_bottom - left_bottom
            if 300 < lane_width < 1200: # Widened the range for more flexibility
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

# --- CALIBRATION FUNCTIONS (Unchanged) ---
points = []
def click_event(event, x, y, flags, params):
    img = params['img']
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration - Click 4 Points", img)

def calibrate_perspective(image_path):
    print("--- CALIBRATION MODE ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}' for calibration.")
        return
    cv2.imshow("Calibration - Click 4 Points", img)
    cv2.setMouseCallback("Calibration - Click 4 Points", click_event, {'img': img})
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) == 4:
        print(f"\nsrc_points = {points}")
    else:
        print("\nCalibration failed.")

# --- MAIN EXECUTION ---
def main():
    MODE = "VIDEO"
    CALIBRATE = False
    VIDEO_PATH = 'Test_Drive.mp4'
    MODEL_PATH = 'proj1/road_segmentation.onnx'
    
    # Initialize the deep learning segmenter
    segmenter = RoadSegmenter(MODEL_PATH)
    
    if CALIBRATE:
        # Calibration logic...
        return

    # Paste your calibrated points for the video here
    src_points = [[955, 515], [1001, 519], [1238, 753], [391, 762]] 
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_size = (h, w)
    
    dst_points = [[w*0.2, 0], [w*0.8, 0], [w*0.8, h], [w*0.2, h]]
    
    detector = LaneDetector(img_size=img_size, src=src_points, dst=dst_points)

    debug_mode = True
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret: 
            print("End of video.")
            break
        
        # Get the road mask from the neural network
        road_mask = segmenter.segment(frame)
        
        # Pass the frame AND the mask to the detector
        final_image, debug_panel = detector.process_frame(frame, road_mask)
        
        cv2.imshow('Final Result (Hybrid Model)', final_image)
        if debug_mode and debug_panel is not None:
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

if __name__ == "__main__":
    main()
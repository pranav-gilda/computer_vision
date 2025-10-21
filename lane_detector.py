import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- A Class to Manage Lane Line State ---
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


# --- The Main LaneDetector Class (UPGRADED) ---
class LaneDetector:
    def __init__(self, img_size, src, dst):
        self.img_size = img_size
        self.src_points = np.float32(src)
        self.dst_points = np.float32(dst)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        self.left_line = LaneLine()
        self.right_line = LaneLine()
        
        # Track if the last detection was successful
        self.detection_failed = True 

    # --- IMPROVEMENT #3: More Robust Thresholding ---
    def thresholding(self, img):
        # Apply a Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # HLS S-channel for color
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        s_thresh = (170, 255)
        s_binary = cv2.inRange(s_channel, s_thresh[0], s_thresh[1])

        # Sobel X gradient
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sx_thresh = (20, 100)
        sx_binary = cv2.inRange(scaled_sobel, sx_thresh[0], sx_thresh[1])
        
        # Combine color and gradient thresholds
        combined_binary = np.zeros_like(sx_binary)
        combined_binary[(s_binary == 255) | (sx_binary == 255)] = 255
        return combined_binary

    def warp(self, img):
        h, w = self.img_size
        return cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)

    # --- IMPROVEMENT #1A: Targeted Search Around Previous Fit ---
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

        # Create a debug image
        out_img = np.dstack((warped_img, warped_img, warped_img))
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return leftx, lefty, rightx, righty, out_img


    def find_lane_pixels_sliding_window(self, warped_img):
        # This function is now the fallback when targeted search fails
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
        if len(leftx) > 150: # Increased threshold for a more confident fit
            try: left_fit = np.polyfit(lefty, leftx, 2)
            except np.linalg.LinAlgError: print("Warning: Polyfit failed for left line.")
        if len(rightx) > 150:
            try: right_fit = np.polyfit(righty, rightx, 2)
            except np.linalg.LinAlgError: print("Warning: Polyfit failed for right line.")
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

    # --- IMPROVEMENT #1B & #2: Main Processing Pipeline ---
    def process_frame(self, frame):
        thresholded = self.thresholding(frame)
        warped = self.warp(thresholded)

        # Decide whether to do a full search or search around the previous fit
        if self.detection_failed:
            # Perform a full sliding window search
            leftx, lefty, rightx, righty, search_img = self.find_lane_pixels_sliding_window(warped)
        else:
            # Perform a faster search around the previously found polynomial
            leftx, lefty, rightx, righty, search_img = self.search_around_poly(warped)
        
        left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)

        # --- Sanity Check ---
        if left_fit is not None and right_fit is not None:
            # Check if lanes are roughly parallel and at a reasonable distance
            h = frame.shape[0]
            left_bottom = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
            right_bottom = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
            lane_width = right_bottom - left_bottom
            # A reasonable lane width in pixels for this perspective
            if 300 < lane_width < 1000: 
                self.left_line.add_fit(left_fit)
                self.right_line.add_fit(right_fit)
                self.detection_failed = False
            else:
                # Failed sanity check
                self.detection_failed = True
        else:
            # Polynomial fit failed
            self.detection_failed = True

        stable_left_fit = self.left_line.last_fit
        stable_right_fit = self.right_line.last_fit
        
        if stable_left_fit is None or stable_right_fit is None:
            return frame, self.create_debug_panel(frame, thresholded, warped, search_img)

        final_image = self.draw_final_result(frame, warped, stable_left_fit, stable_right_fit)
        debug_panel = self.create_debug_panel(frame, thresholded, warped, search_img)
        return final_image, debug_panel

    def create_debug_panel(self, original, thresholded, warped, search_img):
        diag_w, diag_h = 426, 240
        
        original_thumb = cv2.resize(original, (diag_w, diag_h))
        
        thresh_thumb = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        thresh_thumb = cv2.resize(thresh_thumb, (diag_w, diag_h))
        
        warped_thumb = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        warped_thumb = cv2.resize(warped_thumb, (diag_w, diag_h))

        search_img_thumb = cv2.resize(search_img, (diag_w, diag_h))

        top_row = np.concatenate((original_thumb, thresh_thumb), axis=1)
        bottom_row = np.concatenate((warped_thumb, search_img_thumb), axis=1)
        
        debug_panel_full = np.concatenate((top_row, bottom_row), axis=0)
        return cv2.resize(debug_panel_full, (0,0), fx=0.75, fy=0.75)
    
# --- Perspective Calibration Tool ---
points = []
def click_event(event, x, y, flags, params):
    img = params['img']
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration - Click 4 Points", img)

def calibrate_perspective(image_path):
    print("--- CALIBRATION MODE ---")
    print("Click on the 4 corners of the lane on the road.")
    print("Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}' for calibration.")
        return

    cv2.imshow("Calibration - Click 4 Points", img)
    cv2.setMouseCallback("Calibration - Click 4 Points", click_event, {'img': img})
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        print("\nCalibration successful! Here are your source points:")
        print(f"src_points = {points}")
        print("\nCopy this list and replace the 'src_points' variable in the main() function.")
        print("Then, set CALIBRATE = False and run the script again.")
    else:
        print("\nCalibration failed. You must click exactly 4 points.")

def main():
    # --- Configuration ---
    MODE = "VIDEO"  # "IMAGE" or "VIDEO"
    CALIBRATE = False  # Set to True to run calibration
    
    IMAGE_PATH = 'test_image.png'
    VIDEO_PATH = 'Test_Drive.mp4'

    print("--- Script Starting ---")
    print(f"Mode: {MODE}, Calibrate: {CALIBRATE}")

    if CALIBRATE:
        # Build an absolute path for the temporary file to avoid ambiguity
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_frame_path = os.path.join(script_dir, "temp_calibration_frame.jpg")

        calibration_source_path = IMAGE_PATH if MODE == "IMAGE" else VIDEO_PATH
        
        # --- DEBUG: Verify that the source file for calibration exists ---
        abs_source_path = os.path.join(script_dir, calibration_source_path)
        print(f"DEBUG: Checking for calibration source file at: {abs_source_path}")

        if not os.path.exists(abs_source_path):
            print(f"--- FATAL ERROR ---")
            print(f"The source file '{calibration_source_path}' was not found in the same directory as the script.")
            print("Please make sure your image/video file is in the correct location and the name is correct.")
            return # Exit the script

        print("DEBUG: Source file found. Proceeding with calibration.")

        if MODE == "VIDEO":
            print("DEBUG: Video mode selected. Attempting to extract the first frame.")
            cap = cv2.VideoCapture(abs_source_path)

            if not cap.isOpened():
                print("--- FATAL ERROR ---")
                print(f"OpenCV could not open the video file: '{calibration_source_path}'.")
                print("This could be due to a missing video codec or an invalid file path.")
                return # Exit the script

            print("DEBUG: Video file opened successfully.")
            ret, frame = cap.read()
            cap.release()

            if ret:
                print("DEBUG: First frame extracted successfully.")
                cv2.imwrite(temp_frame_path, frame)
                calibrate_perspective(temp_frame_path)
            else:
                print("--- FATAL ERROR ---")
                print("Failed to read the first frame from the video, even though it was opened.")
        else: # Image Mode
            print("DEBUG: Image mode selected.")
            calibrate_perspective(abs_source_path)
        return

    # --- The rest of the function remains the same for detection mode ---
    # (Ensure you've pasted your calibrated points here)
    src_points = [[955, 515], [1001, 519], [1238, 753], [391, 762]]
    
    img_size = None
    if MODE == "IMAGE":
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print(f"Error: Could not read image at '{IMAGE_PATH}'.")
            return
        h, w, _ = frame.shape
        img_size = (h, w)
    elif MODE == "VIDEO":
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{VIDEO_PATH}'")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img_size = (h, w)
    
    h, w = img_size
    dst_points = [[w*0.15, 0], [w*0.85, 0], [w*0.85, h], [w*0.15, h]]
    
    detector = LaneDetector(img_size=img_size, src=src_points, dst=dst_points)

    if MODE == "IMAGE":
        final_image, debug_panel = detector.process_frame(frame)
        if final_image is not None: cv2.imshow('Final Result', final_image)
        if debug_panel is not None: cv2.imshow('Debug Panel', debug_panel)
        cv2.waitKey(0)
        
    elif MODE == "VIDEO":
        debug_mode = True
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret: 
                print("End of video.")
                break
            
            final_image, debug_panel = detector.process_frame(frame)
            cv2.imshow('Final Result', final_image)

            if debug_mode and debug_panel is not None:
                cv2.imshow('Debug Panel', debug_panel)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): break
            if key == ord('d'): 
                debug_mode = not debug_mode
                if not debug_mode:
                    try: cv2.destroyWindow('Debug Panel')
                    except cv2.error: pass
        cap.release()

    cv2.destroyAllWindows()
    print("Processing finished.")

if __name__ == "__main__":
    main()

import numpy as np
import cv2 as cv
import time
from collections import deque
import sys
import traceback

# Import math_func from package `src`
from src import math_func as mf

class TwoDTrack:
    def __init__(self, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0])):
        self.prev_poss = deque(maxlen=5)
        self.prev_time = None
        self.pos = position
        self.vel = velocity

    def predict_next_position(self, dt):            
        if len(self.prev_poss) < 3:
            return self.pos
        p1 = np.array(self.prev_poss[-3])
        p2 = np.array(self.prev_poss[-2])
        p3 = np.array(self.prev_poss[-1])
        v1 = (p2 - p1) / dt
        v2 = (p3 - p2) / dt
        a = (v2 - v1) / dt
        predicted_pos = p3 + v2 * dt + 0.5 * a * dt * dt
        return predicted_pos, v2, a
    
    def update_position(self, new_position):
        self.pos = new_position

    def get_position(self):
        return self.pos


class MonoMotionTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_length, camera_index):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.prev_position = None
        self.prev_time = None
        self.camera = cv.VideoCapture(camera_index)
        # store last time as a timestamp
        self.last_t = time.time()
        self.tracks = []
        self.dt = 0
        self.prev_gray = None
        # optional debug flag
        self.debug = False
        self.prev_diff_map = None

    def detect_markers(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def estimate_pose(self, corners):
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, self.marker_length,
                                                             self.camera_matrix, self.dist_coeffs)
        return rvecs, tvecs

    def detect_posible_tracks(self, frame, old_tracks):
        new_tracks = []
        # blob detection
        blobs = cv.SimpleBlobDetector().detect(frame)
        for blob in blobs:
            # get blob position and calculate SNR
            pos = np.array([blob.pt[0], blob.pt[1]])
            # Check if the blob is close to any old track
            

    def process_frame(self):
        if self.camera is None or not self.camera.isOpened():
            raise ValueError("Camera is not initialized or cannot be opened.")
        # compute elapsed time
        self.dt = time.time() - self.last_t
        self.last_t = time.time()
        ret, frame = self.camera.read()
        if not ret:
            raise ValueError("Failed to read frame from camera.")
        if frame is None:
            raise ValueError("Captured frame is None.")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(mask, (int(pred_pos[0]), int(pred_pos[1])), 50, 255, -1)
        mask = cv.GaussianBlur(mask, (9, 9), 0, dst=mask)
        # initialize local vars defensively
        diff_map = None
        overlay = None
        # use a safe prev_diff_map for computations
        if self.prev_diff_map is None:
            prev_diff_map = np.ones_like(gray, dtype=np.uint8)
        else:
            prev_diff_map = self.prev_diff_map

        if self.prev_gray is not None:
            diff_map = mf.picture_diference(gray, self.prev_gray)
            colored_diff = cv.applyColorMap(diff_map, cv.COLORMAP_JET)
            overlay = cv.addWeighted(frame, 0.2, colored_diff, 0.8, 0)

        # compute contrast; if no diff_map assume high contrast to continue
        if diff_map is None:
            contrast = 255
        else:
            contrast = mf.calculate_pixel_contrast(diff_map, mask)

        if diff_map is None:
            w_diff = np.zeros_like(prev_diff_map, dtype=np.uint8)
        else:
            w_diff = cv.normalize((diff_map.astype(float) * mask), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        combined = cv.addWeighted(w_diff, 0.6, prev_diff_map, 0.4, 0)
        
        if contrast < 1.2:
            # Development fallback: low contrast detected — use previous diff map
            if self.debug:
                print(contrast)
            pass
        else:   
            
            self.pos = (mf.center_of_change(combined))

            if self.debug:

                colored_combined = cv.applyColorMap(combined, cv.COLORMAP_JET)
                cv.circle(colored_combined, (int(self.pos[0]), int(self.pos[1])), 10, (0, 255, 255), -1)
                # show only valid images to avoid OpenCV assertion errors
                try:
                    if colored_combined is not None and getattr(colored_combined, 'size', 0) > 0:
                        cv.imshow("Debug - Combined", colored_combined)
                except Exception:
                    pass

                if overlay is not None and getattr(overlay, 'size', 0) > 0:
                    try:
                        cv.imshow("Debug - Overlay", overlay)
                    except Exception:
                        pass

                if mask is not None and getattr(mask, 'size', 0) > 0:
                    cv.imshow("Debug - Mask", mask)

                if diff_map is not None and getattr(diff_map, 'size', 0) > 0:
                    try:
                        cv.imshow("Debug - Diff", diff_map)
                    except Exception:
                        pass

                if self.prev_diff_map is not None and getattr(self.prev_diff_map, 'size', 0) > 0: #prev_gray
                    cv.imshow("Debug - Gray", self.prev_diff_map)

                if frame is not None and getattr(frame, 'size', 0) > 0:
                    cv.imshow("Debug - Frame", frame)

                cv.waitKey(1)
        
            
        # store combined for next frame
        self.prev_diff_map = combined.copy()
        # store prev_gray and history
        self.prev_gray = gray.copy()
        # append current position if available
        try:
            self.pos_hst.append((int(self.pos[0]), int(self.pos[1])))
        except Exception:
            pass
        

    def release(self):
        if self.camera is not None:
            self.camera.release()
        cv.destroyAllWindows()
    
    def set_debug(self, debug):
        self.debug = debug

if __name__ == "__main__":
    # Example usage
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    marker_length = 0.05  # in meters

    # create tracker and do basic startup diagnostics
    try:
        tracker = MonoMotionTracker(camera_matrix, dist_coeffs, marker_length, camera_index=0)
    except Exception:
        print("ERROR: failed to create MonoMotionTracker:")
        traceback.print_exc()
        sys.exit(1)

    tracker.set_debug(False)
    print("MonoMotionTracker created. Debug mode ON.")

    # camera open check
    try:
        if tracker.camera is None or not tracker.camera.isOpened():
            print("ERROR: Camera could not be opened. Check the camera index (0) and that no other app uses it.")
            # show available devices hint (best-effort)
            print("Tip: try changing camera_index in the script or run a small test script to list devices.")
            sys.exit(1)
        else:
            print("Camera opened successfully.")
    except Exception:
        print("ERROR: exception while checking camera:")
        traceback.print_exc()
        sys.exit(1)

    # run loop with top-level exception handling so failures are visible
    try:
        while True:
            tracker.process_frame()
    except KeyboardInterrupt:
        print("Interrupted by user — releasing resources.")
        tracker.release()
    except Exception:
        print("Unhandled exception in tracker loop:")
        traceback.print_exc()
        try:
            tracker.release()
        except Exception:
            pass
        sys.exit(1)


            



        
        
"""
Example: Using the calibration module

This script demonstrates how to use the calibration utilities
in your tracking code.
"""

import numpy as np
import cv2 as cv
from src.calibration import load_calibration, RealtimeUndistorter, apply_calibration

def example_1_basic_usage():
    """Example 1: Load calibration and undistort a single image."""
    print("Example 1: Basic usage")
    
    # Load calibration
    calib = load_calibration('src/calibration/note_manu_cam_calibration.json')
    print(f"Loaded calibration: {calib}")
    
    # Load test image
    img = cv.imread('test_image.jpg')
    if img is not None:
        # Apply undistortion
        from src.calibration import undistort_image
        corrected = undistort_image(img, calib, alpha=1.0)
        
        # Save result
        cv.imwrite('corrected_image.jpg', corrected)
        print("Saved corrected image")


def example_2_realtime_video():
    """Example 2: Real-time undistortion in video loop."""
    print("\nExample 2: Real-time video undistortion")
    
    # Initialize undistorter (pre-computes maps for efficiency)
    undistorter = RealtimeUndistorter(
        'src/calibration/note_manu_cam_calibration.json',
        alpha=1.0  # 0=crop to valid, 1=keep all pixels
    )
    
    # Open camera
    cap = cv.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save a frame")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply undistortion (very fast with pre-computed maps)
        corrected = undistorter(frame)
        
        # Display both
        combined = np.hstack([frame, corrected])
        cv.imshow('Original (left) vs Corrected (right)', combined)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.imwrite(f'frame_{frame_count}_corrected.jpg', corrected)
            print(f"Saved frame {frame_count}")
            frame_count += 1
    
    cap.release()
    cv.destroyAllWindows()


def example_3_one_liner():
    """Example 3: Quick one-liner undistortion."""
    print("\nExample 3: One-liner usage")
    
    img = cv.imread('test_image.jpg')
    if img is not None:
        # One function call: load calibration + apply
        corrected = apply_calibration(img, 'src/calibration/note_manu_cam_calibration.json')
        cv.imwrite('quick_corrected.jpg', corrected)
        print("Quick correction done")


def example_4_integrate_with_tracker():
    """Example 4: Integration with MonoMotionTracker."""
    print("\nExample 4: Tracker integration example")
    
    # In your MonoMotionTracker.__init__:
    from src.calibration import RealtimeUndistorter
    
    # Option A: Pass calibration JSON path
    calibration_json = 'src/calibration/note_manu_cam_calibration.json'
    undistorter = RealtimeUndistorter(calibration_json, alpha=1.0)
    
    # Option B: Or construct CalibrationData from existing camera_matrix, dist_coeffs
    from src.calibration import CalibrationData
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    calib = CalibrationData(camera_matrix, dist_coeffs, (640, 480))
    undistorter = RealtimeUndistorter(calib, alpha=1.0)
    
    # Then in process_frame():
    # ret, frame = self.camera.read()
    # frame = self.undistorter(frame)  # Apply correction
    # ... rest of processing
    
    print("See code comments for integration pattern")


if __name__ == "__main__":
    print("Camera Calibration Module - Usage Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # example_1_basic_usage()
    # example_2_realtime_video()
    # example_3_one_liner()
    example_4_integrate_with_tracker()
    
    print("\nFor more details, see src/calibration/calib.py docstrings")

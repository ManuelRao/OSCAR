"""
Integration guide: Adding calibration to MonoMotionTracker

This shows the changes needed in motion_tracking.py to use the calibration module.
"""

# At the top of motion_tracking.py, add this import:
from src.calibration import RealtimeUndistorter

# In MonoMotionTracker.__init__, add these parameters and initialization:

class MonoMotionTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_length, camera_index,
                 calibration_json=None, apply_undistortion=True):
        """
        Args:
            ...existing args...
            calibration_json: Path to calibration JSON file (optional)
                             If None, uses camera_matrix and dist_coeffs directly
            apply_undistortion: Whether to apply undistortion correction
        """
        # ... existing initialization ...
        
        # NEW: Initialize undistorter if requested
        self.apply_undistortion = apply_undistortion
        self.undistorter = None
        
        if apply_undistortion:
            if calibration_json:
                # Load from JSON file
                self.undistorter = RealtimeUndistorter(calibration_json, alpha=1.0)
            else:
                # Use provided camera_matrix and dist_coeffs
                from src.calibration import CalibrationData
                # Assume 640x480 or get from first frame
                calib = CalibrationData(camera_matrix, dist_coeffs, (640, 480))
                self.undistorter = RealtimeUndistorter(calib, alpha=1.0)


# In process_frame(), add undistortion right after reading the frame:

    def process_frame(self):
        # ... existing checks ...
        ret, frame = self.camera.read()
        if not ret:
            raise ValueError("Failed to read frame from camera.")
        if frame is None:
            raise ValueError("Captured frame is None.")
        
        # NEW: Apply undistortion if enabled
        if self.undistorter is not None:
            frame = self.undistorter(frame)
        
        # Continue with existing processing...
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # ... rest of your code ...


# Usage example in __main__:

if __name__ == "__main__":
    # Option 1: Load calibration from JSON
    tracker = MonoMotionTracker(
        camera_matrix=None,  # Will be loaded from JSON
        dist_coeffs=None,
        marker_length=0.05,
        camera_index=0,
        calibration_json='src/calibration/note_manu_cam_calibration.json',
        apply_undistortion=True
    )
    
    # Option 2: Use manual calibration with undistortion
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    tracker = MonoMotionTracker(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length=0.05,
        camera_index=0,
        calibration_json=None,
        apply_undistortion=True
    )
    
    # Option 3: No undistortion (original behavior)
    tracker = MonoMotionTracker(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length=0.05,
        camera_index=0,
        apply_undistortion=False
    )

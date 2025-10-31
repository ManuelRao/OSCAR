"""
Calibration package - Camera distortion correction utilities

Provides easy-to-use functions for loading and applying camera calibration data.

Usage:
    from src.calibration import load_calibration, undistort_image, RealtimeUndistorter
    
    # Option 1: Load and apply
    calib = load_calibration('camera_calibration.json')
    corrected_frame = undistort_image(frame, calib)
    
    # Option 2: Real-time efficient undistortion
    undistorter = RealtimeUndistorter('camera_calibration.json')
    corrected_frame = undistorter(frame)
    
    # Option 3: One-liner
    corrected_frame = apply_calibration(frame, 'camera_calibration.json')
"""

from .calib import (
    CalibrationData,
    load_calibration,
    undistort_image,
    undistort_points,
    get_undistort_maps,
    RealtimeUndistorter,
    save_calibration,
    apply_calibration,
)

__all__ = [
    'CalibrationData',
    'load_calibration',
    'undistort_image',
    'undistort_points',
    'get_undistort_maps',
    'RealtimeUndistorter',
    'save_calibration',
    'apply_calibration',
]

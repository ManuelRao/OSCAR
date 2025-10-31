"""
calib.py - Camera calibration utilities

Provides functions to:
- Load calibration data from JSON files
- Apply distortion corrections to images and video frames
- Get optimal camera matrices for undistortion
- Support real-time calibration workflows

Usage example:
    from src.calibration import load_calibration, undistort_image
    
    calib = load_calibration('camera_calibration.json')
    corrected_frame = undistort_image(frame, calib)
"""

import json
import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


class CalibrationData:
    """Container for camera calibration parameters."""
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                 image_size: Tuple[int, int], metadata: Optional[Dict] = None):
        """
        Initialize calibration data.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3, ...)
            image_size: (width, height) of calibrated images
            metadata: Optional dict with camera name, reprojection error, etc.
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).flatten()
        self.image_size = tuple(image_size)  # (width, height)
        self.metadata = metadata or {}
        
        # Cache for optimal new camera matrix (computed on-demand)
        self._optimal_camera_matrix = None
        self._roi = None
    
    def get_optimal_camera_matrix(self, alpha: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Get optimal new camera matrix for undistortion.
        
        Args:
            alpha: Free scaling parameter (0=no invalid pixels, 1=all pixels retained)
        
        Returns:
            (new_camera_matrix, roi) where roi is (x, y, w, h) for valid pixels
        """
        if self._optimal_camera_matrix is None or self._roi is None:
            self._optimal_camera_matrix, self._roi = cv.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, self.image_size, alpha, self.image_size
            )
        return self._optimal_camera_matrix, self._roi
    
    def __repr__(self):
        return (f"CalibrationData(image_size={self.image_size}, "
                f"camera={self.metadata.get('camera_name', 'unknown')}, "
                f"error={self.metadata.get('reprojection_error', 'N/A')})")


def load_calibration(json_path: Union[str, Path]) -> CalibrationData:
    """
    Load calibration data from a JSON file.
    
    Args:
        json_path: Path to calibration JSON file
    
    Returns:
        CalibrationData object
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract calibration section
    if "calibration" not in data:
        raise ValueError(f"JSON file missing 'calibration' key: {json_path}")
    
    calib = data["calibration"]
    
    # Parse required fields
    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32).flatten()
    
    # Image size from metadata or infer from camera matrix
    if "image_size" in data:
        image_size = tuple(data["image_size"])
    else:
        # Fallback: use principal point * 2 as rough estimate
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        image_size = (int(cx * 2), int(cy * 2))
    
    # Collect metadata
    metadata = {
        "camera_name": data.get("camera_name", "unknown"),
        "reprojection_error": calib.get("reprojection_error"),
        "pattern_size": data.get("pattern_size"),
        "square_size": data.get("square_size"),
        "timestamp": data.get("timestamp"),
        "used_image_count": data.get("used_image_count"),
    }
    
    return CalibrationData(camera_matrix, dist_coeffs, image_size, metadata)


def undistort_image(image: np.ndarray, calib: CalibrationData, 
                   alpha: float = 1.0, crop_to_roi: bool = False) -> np.ndarray:
    """
    Apply distortion correction to an image.
    
    Args:
        image: Input image (BGR or grayscale)
        calib: CalibrationData object
        alpha: Free scaling parameter for getOptimalNewCameraMatrix
               0 = no invalid pixels (image cropped)
               1 = all pixels retained (black borders may appear)
        crop_to_roi: If True, crop result to ROI with valid pixels
    
    Returns:
        Undistorted image
    """
    new_camera_matrix, roi = calib.get_optimal_camera_matrix(alpha)
    
    undistorted = cv.undistort(image, calib.camera_matrix, calib.dist_coeffs, 
                               None, new_camera_matrix)
    
    if crop_to_roi and roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted


def undistort_points(points: np.ndarray, calib: CalibrationData) -> np.ndarray:
    """
    Undistort 2D image points.
    
    Args:
        points: Array of shape (N, 1, 2) or (N, 2) with image coordinates
        calib: CalibrationData object
    
    Returns:
        Undistorted points with same shape as input
    """
    points = np.array(points, dtype=np.float32)
    original_shape = points.shape
    
    # cv.undistortPoints expects (N, 1, 2)
    if points.ndim == 2:
        points = points.reshape(-1, 1, 2)
    
    undistorted = cv.undistortPoints(points, calib.camera_matrix, calib.dist_coeffs, 
                                     None, calib.camera_matrix)
    
    return undistorted.reshape(original_shape)


def get_undistort_maps(calib: CalibrationData, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute undistortion maps for fast real-time correction.
    
    Use with cv.remap() for efficient per-frame undistortion.
    
    Args:
        calib: CalibrationData object
        alpha: Free scaling parameter
    
    Returns:
        (map1, map2) - Use with cv.remap(image, map1, map2, cv.INTER_LINEAR)
    """
    new_camera_matrix, _ = calib.get_optimal_camera_matrix(alpha)
    
    map1, map2 = cv.initUndistortRectifyMap(
        calib.camera_matrix, calib.dist_coeffs, None, 
        new_camera_matrix, calib.image_size, cv.CV_32FC1
    )
    
    return map1, map2


class RealtimeUndistorter:
    """
    Efficient real-time undistortion using pre-computed maps.
    
    Usage:
        undistorter = RealtimeUndistorter('camera_calibration.json')
        
        # In video loop:
        corrected_frame = undistorter.undistort(frame)
    """
    
    def __init__(self, calibration_path: Union[str, Path, CalibrationData], alpha: float = 1.0):
        """
        Initialize with calibration data.
        
        Args:
            calibration_path: Path to JSON file or CalibrationData object
            alpha: Free scaling parameter for undistortion
        """
        if isinstance(calibration_path, CalibrationData):
            self.calib = calibration_path
        else:
            self.calib = load_calibration(calibration_path)
        
        self.alpha = alpha
        self.map1, self.map2 = get_undistort_maps(self.calib, alpha)
    
    def undistort(self, image: np.ndarray, interpolation: int = cv.INTER_LINEAR) -> np.ndarray:
        """
        Undistort an image using pre-computed maps.
        
        Args:
            image: Input image
            interpolation: Interpolation method (cv.INTER_LINEAR, cv.INTER_CUBIC, etc.)
        
        Returns:
            Undistorted image
        """
        return cv.remap(image, self.map1, self.map2, interpolation)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow calling undistorter as a function."""
        return self.undistort(image)


def save_calibration(calib: CalibrationData, json_path: Union[str, Path], 
                     rvecs: Optional[list] = None, tvecs: Optional[list] = None,
                     used_image_names: Optional[list] = None):
    """
    Save calibration data to JSON file.
    
    Args:
        calib: CalibrationData object
        json_path: Output path for JSON file
        rvecs: Optional list of rotation vectors from calibration
        tvecs: Optional list of translation vectors from calibration
        used_image_names: Optional list of image filenames used
    """
    def arr_to_list(a):
        return np.array(a).tolist()
    
    data = {
        "camera_name": calib.metadata.get("camera_name", "camera"),
        "image_size": list(calib.image_size),
        "timestamp": calib.metadata.get("timestamp"),
        "pattern_size": calib.metadata.get("pattern_size"),
        "square_size": calib.metadata.get("square_size"),
        "used_image_count": calib.metadata.get("used_image_count"),
        "calibration": {
            "camera_matrix": arr_to_list(calib.camera_matrix),
            "dist_coeffs": arr_to_list(calib.dist_coeffs),
            "reprojection_error": calib.metadata.get("reprojection_error"),
        }
    }
    
    if rvecs is not None:
        data["calibration"]["rvecs"] = [arr_to_list(r) for r in rvecs]
    if tvecs is not None:
        data["calibration"]["tvecs"] = [arr_to_list(t) for t in tvecs]
    if used_image_names is not None:
        data["used_image_names"] = used_image_names
    
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved calibration to: {json_path}")


# Convenience function for quick one-liner usage
def apply_calibration(image: np.ndarray, json_path: Union[str, Path], 
                     alpha: float = 1.0) -> np.ndarray:
    """
    One-shot undistortion: load calibration and apply to image.
    
    Args:
        image: Input image
        json_path: Path to calibration JSON
        alpha: Scaling parameter
    
    Returns:
        Undistorted image
    """
    calib = load_calibration(json_path)
    return undistort_image(image, calib, alpha)


if __name__ == "__main__":
    # Example usage / quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python calib.py <calibration.json> [test_image.jpg]")
        print("\nThis module provides camera calibration utilities.")
        print("Import it in your code:")
        print("  from src.calibration import load_calibration, undistort_image")
        sys.exit(0)
    
    calib_path = sys.argv[1]
    
    # Load and display calibration info
    calib = load_calibration(calib_path)
    print(f"\nLoaded: {calib}")
    print(f"Camera matrix:\n{calib.camera_matrix}")
    print(f"Distortion coefficients: {calib.dist_coeffs}")
    print(f"Image size: {calib.image_size}")
    
    # If test image provided, undistort it
    if len(sys.argv) >= 3:
        test_image_path = sys.argv[2]
        img = cv.imread(test_image_path)
        if img is None:
            print(f"Could not load test image: {test_image_path}")
            sys.exit(1)
        
        print(f"\nUndistorting test image: {test_image_path}")
        undistorted = undistort_image(img, calib, alpha=1.0)
        
        output_path = "undistorted_output.jpg"
        cv.imwrite(output_path, undistorted)
        print(f"Saved undistorted image to: {output_path}")

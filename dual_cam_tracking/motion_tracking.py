import numpy as np
import cv2 as cv

class MonoMotionTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_length, track, camera_index):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.track = track
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        self.parameters = cv.aruco.DetectorParameters()
        self.prev_position = None
        self.prev_time = None
        self.camera = cv.VideoCapture(camera_index)

    def detect_markers(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def estimate_pose(self, corners):
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, self.marker_length,
                                                             self.camera_matrix, self.dist_coeffs)
        return rvecs, tvecs

    def compute_velocity(self, current_position, current_time):
        if self.prev_position is None or self.prev_time is None:
            self.prev_position = current_position
            self.prev_time = current_time
            return np.array([0.0, 0.0, 0.0])

        dt = current_time - self.prev_time
        if dt <= 0:
            return np.array([0.0, 0.0, 0.0])

        velocity = (current_position - self.prev_position) / dt
        self.prev_position = current_position
        self.prev_time = current_time
        return velocity

    def process_frame(self, current_time):
        ret, frame = self.camera.read()

        
        
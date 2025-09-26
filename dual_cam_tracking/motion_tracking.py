import numpy as np
import cv2 as cv
import time
import math_func as mf
from collections import deque

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
        self.last_t = time.time
        self.pos = []
        self.vel = []
        self.pos_hst = deque(maxlen = 5)

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

    def process_frame(self):
        if self.camera is None or not self.camera.isOpened():
            raise ValueError("Camera is not initialized or cannot be opened.")
        self.dt = time.time - self.last_t
        self.last_t = time.time
        ret, frame = self.camera.read()
        if not ret:
            raise ValueError("Failed to read frame from camera.")
        if frame is None:
            raise ValueError("Captured frame is None.")
        pred_pos, theta, speed, acc = mf.predict_next_position(self.pos_hst[4], self.pos_hst[3], self.pos_hst[2], self.dt)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(mask, (int(pred_pos[0]), int(pred_pos[1])), 50, 255, -1)
        contrast = mf.calculate_pixel_contrast(frame, mask)
        if contrast < 30:
            raise NotImplementedError("Low contrast area, implement alternative tracking method.")
        else:



            



        
        
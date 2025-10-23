import cv2 as cv
import cv2.aruco as aruco
import numpy as np
from src import math_func as mf
import time
import serial
import pygame
from .track_lib import Track
import math
import json

# ----- CONFIG -----
SERIAL_PORT = 'COM3'     # Change this to your ESP32 COM port
BAUD_RATE = 115200
SEND_INTERVAL = 0.02      # Seconds between sends
WRAP_POINTS_FILE = "wrap.json"  # points file for wrap mode

# ----- INIT SERIAL -----
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

# ----- INIT JOYSTICK -----
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick found!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using joystick: {joystick.get_name()}")

# -- init PID controllers --
steering_pid = mf.pid(kp=1.0, ki=0.20, kd=0.1)
throttle_pid = mf.pid(kp=1.0, ki=0.0, kd=0.1)
head_lpf = mf.low_pass_filter(alpha=0.3)

# ---------------INIT TRACK ----------------
track = Track()
print(f"Track loaded with {len(track.points)} points, total length {track.total_length:.2f}")

# ---------------- CAMERA CALIBRATION PARAMETERS ----------------
# Replace with your own camera intrinsics (from calibration)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=float)

dist_coeffs = np.zeros((5, 1))  # assume no lens distortion

# ---------------- ARUCO SETUP ----------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Marker size in meters (must match your printed marker!)
marker_length = 0.05  

# ---------------- CAMERA STREAM ----------------
cap = cv.VideoCapture(1)  # change index if you have multiple cameras

pos = (0, 0)
pos_history = []
radius = 20
wrap_points = json.load(open(WRAP_POINTS_FILE, "r"))
wrap_points = [tuple(p) for p in wrap_points]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    MARGIN = 100
    frame = cv.copyMakeBorder(
        frame,
        top=MARGIN, bottom=MARGIN,
        left=MARGIN, right=MARGIN,
        borderType=cv.BORDER_CONSTANT,
        value=(255, 255, 255)  # white
    )
    frame = mf.warp_to_rectangle(frame, wrap_points)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose for each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            # Draw axis for orientation (NEW FUNCTION NAME)
            cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Convert rotation vector to rotation matrix
            R, _ = cv.Rodrigues(rvec)

            # Extract Euler angles
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x_angle = np.arctan2(R[2, 1], R[2, 2])
                y_angle = np.arctan2(-R[2, 0], sy)
                z_angle = np.arctan2(R[1, 0], R[0, 0])
            else:
                x_angle = np.arctan2(-R[1, 2], R[1, 1])
                y_angle = np.arctan2(-R[2, 0], sy)
                z_angle = 0
            heading = head_lpf.apply(z_angle)
            # Print position + orientation
            print("Marker Position (tvec):", tvec.ravel())
            print("Marker Orientation (Euler angles deg):", np.degrees([x_angle, y_angle, z_angle]))
        pos = mf.quadrilateral_center(corners[0].reshape(4, 2))
        radius = mf.aruco_apparent_radius(corners[0].reshape(4, 2))[0]
        cv.putText(frame, f"Center: {pos}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.circle(frame, (int(pos[0]), int(pos[1])), 50, (255, 0, 0), -1)
    else:
        # get difference heatmap
        if 'prev_gray' in locals():
            diff_map = mf.picture_diference(gray, prev_gray)
            colored_diff = cv.applyColorMap(diff_map, cv.COLORMAP_JET)
            overlay = cv.addWeighted(frame, 0.2, colored_diff, 0.8, 0)

        # predict next position if we have enough history
        if 'pos_history' in locals() and len(pos_history) >= 3:
            pred_pos, pred_rot, spd_p, acc = mf.predict_next_position(pos_history[-3], pos_history[-2], pos_history[-1], dt)
            
        # generate weighted diff map
        if 'diff_map' in locals() and 'pred_pos' in locals() and 'prev_diff_map' in locals():
            weight_map = np.zeros_like(diff_map, dtype=float)
            cv.circle(weight_map, (int(pred_pos[0]), int(pred_pos[1])), int(radius*2), (1.0,), -1)
            weight_map = cv.GaussianBlur(weight_map, (9, 9), 6)
            w_diff = cv.normalize(diff_map.astype(float) * weight_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            combined = cv.addWeighted(w_diff, 0.6, prev_diff_map, 0.4, 0)
            colored_combined = cv.applyColorMap(combined, cv.COLORMAP_JET)  
            
        # calculate change clump center
        if 'combined' in locals():
            pos = mf.center_of_change(combined)
            cv.circle(frame, (int(pos[0]), int(pos[1])), 10, (0, 255, 255), -1)

    if False:  # debug: draw entire track
        n = 400
        for i in range(n):
            cv.circle(frame, (int(track.get_position(i/n)[0]), int(track.get_position(i/n)[1])), 5, (0, 255, 0), -1)

    # calculate nearest point on track
    if 'pos' in locals():
        closest_t = None
        closest_dist = float('inf')
        for i in range(len(track.segments)):
            p1, p2 = track.segments[i]
            seg_vec = (p2[0]-p1[0], p2[1]-p1[1])
            pt_vec = (pos[0]-p1[0], pos[1]-p1[1])
            seg_len = math.hypot(seg_vec[0], seg_vec[1])
            if seg_len == 0:
                continue
            seg_unit = (seg_vec[0]/seg_len, seg_vec[1]/seg_len)
            proj_len = pt_vec[0]*seg_unit[0] + pt_vec[1]*seg_unit[1]
            proj_len = max(0, min(seg_len, proj_len))
            closest_point = (p1[0] + seg_unit[0]*proj_len, p1[1] + seg_unit[1]*proj_len)
            dist = math.hypot(pos[0]-closest_point[0], pos[1]-closest_point[1])
            if dist < closest_dist:
                closest_dist = dist
                closest_t = track.cumulative[i] + (proj_len / track.lengths[i]) * (track.cumulative[i+1] - track.cumulative[i])
            heading = head_lpf.apply(mf.angle_beetween_points(pos, pos_history[-1]) if len(pos_history) > 0 else 0.0)
        if closest_t is not None:
            cv.putText(frame, f"Track t: {closest_t:.3f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            track_pos = track.get_position(closest_t+0.05)
            cv.circle(frame, (int(track_pos[0]), int(track_pos[1])), 10, (255, 255, 0), -1)
            cv.line(frame, (int(pos[0]), int(pos[1])), (int(track_pos[0]), int(track_pos[1])), (255, 255, 0), 2)

    
    ideal_t = closest_t+0.05 if 'closest_t' in locals() else 0.0
    ideal_pos = track.get_position(ideal_t) if 'ideal_t' in locals() else track.get_position(0.05)
    ideal_heading = mf.angle_beetween_points(pos, ideal_pos) if 'ideal_pos' in locals() else 0.0
    error = (ideal_heading - heading) if 'ideal_heading' in locals() else 0.0
    cv.putText(frame, f"Steering error (rad): {error:.2f}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steering = steering_pid.compute(error, dt) if 'error' in locals() and 'dt' in locals() else 0.0
    steering = -steering + 68
    
    # show images
    if 'overlay' in locals():
        cv.imshow("Overlay", overlay)
    if 'colored_combined' in locals():
        cv.imshow("Weighted Difference Map", colored_combined) 
    if 'weight_map' in locals():      
        cv.imshow("Weight Map", (weight_map * 255).astype(np.uint8))
    cv.imshow("ARUCO Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prev_diff_map = combined.copy() if 'combined' in locals() else np.zeros_like(gray)
    prev_gray = gray.copy()
    pos_history.append(pos)
    if len(pos_history) > 10:
        pos_history.pop(0)

    dt = time.time() - lt if 'lt' in locals() else 0.1
    lt = time.time() 

    pygame.event.pump()

    # Read axes
    x_axis = -joystick.get_axis(0)  # Left-right
    thr = -joystick.get_axis(2) # Forward-backward (inverted)  
    brk = -joystick.get_axis(1)

    # Map values
    # steering = int((x_axis + 1) * 68)
    throttle = int(((1 - thr) / 2 * 256)-((1 - brk) / 2
                                            * 256))

    # Send over serial
    cmd = f"{throttle} {steering}\n"
    ser.write(cmd.encode())                                          


cap.release()
cv.destroyAllWindows()

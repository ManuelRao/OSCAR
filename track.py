import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math_func as mf
import time

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
cap = cv.VideoCapture(0)  # change index if you have multiple cameras

pos = (0, 0)
pos_history = []
radius = 20
while True:
    ret, frame = cap.read()
    if not ret:
        break

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
            pred_pos, th_p, spd_p, acc = mf.predict_next_position(pos_history[-3], pos_history[-2], pos_history[-1], dt)
            cv.circle(frame, (int(pred_pos[0]), int(pred_pos[1])), 10, (0, 0, 255), -1)
            cv.putText(frame, f"Predicted: {pred_pos}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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

cap.release()
cv.destroyAllWindows()

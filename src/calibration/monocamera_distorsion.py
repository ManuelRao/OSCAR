import cv2
import numpy as np
import json
import glob
import os
import time
from datetime import datetime

# monocamera_distorsion.py
# Requires: opencv-python, numpy
# Run: python monocamera_distorsion.py


def ask(prompt, default=None):
    s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    return s if s else (default if default is not None else "")

def load_images_from_folder(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            imgs.append((f, img))
    return imgs

def capture_from_camera(cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Unable to open camera index", cam_index)
        return []
    cv2.namedWindow("Calibration Capture (press C to stop)", cv2.WINDOW_NORMAL)
    imgs = []
    idx = 0
    print("Starting camera capture. Frames will be grabbed once per second.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, stopping.")
            break
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"camera_frame_{idx}_{timestamp}.png"
        imgs.append((name, frame.copy()))
        idx += 1
        cv2.imshow("Calibration Capture (press C to stop)", frame)
        key = cv2.waitKey(1000) & 0xFF  # wait 1 second between captures
        if key in (ord('c'), ord('C')):
            print("Capture stopped by user (C pressed).")
            break
    cap.release()
    cv2.destroyAllWindows()
    return imgs

def find_corners(images, pattern_size, criteria):
    objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    used_image_names = []
    image_size = None
    for name, img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            object_points.append(objp.copy())
            image_points.append(corners_refined)
            used_image_names.append(name)
            # draw and show for feedback
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
            cv2.imshow("Detected corners (press any key)", vis)
            cv2.waitKey(500)
        else:
            print(f"Chessboard not found in image: {name}")
    cv2.destroyAllWindows()
    return object_points, image_points, used_image_names, image_size

def calibrate(object_points, image_points, image_size):
    if not object_points or not image_points:
        raise ValueError("No object points or image points for calibration.")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)
    # compute reprojection error
    total_error = 0
    total_points = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)
        total_error += error**2
        total_points += len(imgpoints2)
    mean_error = np.sqrt(total_error / total_points) if total_points > 0 else float('inf')
    return {
        "ret": float(ret),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reprojection_error": float(mean_error)
    }

def serialize_calibration(calib_dict, metadata):
    def arr_to_list(a):
        return np.array(a).tolist()
    data = {}
    data.update(metadata)
    data["calibration"] = {
        "ret": calib_dict["ret"],
        "camera_matrix": arr_to_list(calib_dict["camera_matrix"]),
        "dist_coeffs": arr_to_list(calib_dict["dist_coeffs"]),
        "rvecs": [arr_to_list(r) for r in calib_dict["rvecs"]],
        "tvecs": [arr_to_list(t) for t in calib_dict["tvecs"]],
        "reprojection_error": calib_dict["reprojection_error"]
    }
    return data

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved calibration to:", filepath)

def main():
    print("Monocular camera distortion calibration")
    camera_name = ask("Enter camera name (will be saved in json)", "camera")
    mode = ask("Choose input mode: 'folder' or 'camera' (type folder or camera)", "folder").lower()
    images = []
    if mode.startswith("f"):
        folder = ask("Enter folder path with calibration images", ".")
        if not os.path.isdir(folder):
            print("Folder does not exist:", folder)
            return
        images = load_images_from_folder(folder)
        if not images:
            print("No images found in folder:", folder)
            return
    else:
        idx_str = ask("Enter camera index (integer)", "0")
        try:
            idx = int(idx_str)
        except:
            print("Invalid camera index")
            return
        images = capture_from_camera(idx)
        if not images:
            print("No images captured from camera.")
            return

    # get chessboard pattern size and square size
    defaults = "9 6"
    pattern_str = ask("Chessboard inner corners (cols rows) e.g. '9 6'", defaults)
    try:
        cols, rows = [int(x) for x in pattern_str.split()]
        pattern_size = (cols, rows)
    except Exception:
        print("Invalid pattern size.")
        return
    sq_default = "1.0"
    square_size_str = ask("Square size (any unit, default 1.0)", sq_default)
    try:
        square_size = float(square_size_str)
    except:
        square_size = 1.0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_pts, img_pts, used_names, img_size = find_corners(images, pattern_size, criteria)
    if not used_names:
        print("No valid images with detectable chessboard corners. Exiting.")
        return

    # scale object points by square_size
    for obj in obj_pts:
        obj[:, :2] *= square_size

    calib = calibrate(obj_pts, img_pts, img_size)

    metadata = {
        "camera_name": camera_name,
        "image_size": img_size,
        "used_image_count": len(used_names),
        "used_image_names": used_names,
        "pattern_size": {"cols": pattern_size[0], "rows": pattern_size[1]},
        "square_size": square_size,
        "timestamp": datetime.now().isoformat()
    }

    data = serialize_calibration(calib, metadata)
    filename = f"{camera_name}_calibration.json"
    outpath = ask("Enter output json path (file) to save", filename)
    outdir = os.path.dirname(outpath)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    save_json(data, outpath)
    print("Calibration complete. Reprojection error:", calib["reprojection_error"])

if __name__ == "__main__":
    main()
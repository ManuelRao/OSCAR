import cv2
import json
import os
import math_func as mf

# JSON file for storing points
POINTS_FILE = "points.json"
MARGIN = 100  # margin size
RADIUS = 8    # marker radius in pixels
WRAP_POINTS_FILE = "wrap.json"  # points file for wrap mode

wrap_points = json.load(open(WRAP_POINTS_FILE, "r"))
wrap_points = [tuple(p) for p in wrap_points]

# Load existing points if available
if os.path.exists(POINTS_FILE):
    with open(POINTS_FILE, "r") as f:
        points = json.load(f)
else:
    points = []

# Convert to list of tuples
points = [tuple(p) for p in points]

# For dragging points
dragging_index = None


def save_points():
    with open(POINTS_FILE, "w") as f:
        json.dump(points, f)


def find_point(x, y):
    """Return index of point if (x, y) is close to one, else None."""
    for i, (px, py) in enumerate(points):
        if (x - px) ** 2 + (y - py) ** 2 <= RADIUS**2:
            return i
    return None


def mouse_callback(event, x, y, flags, param):
    global points, dragging_index

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = find_point(x, y)
        if idx is None:
            # Add new point
            points.append((x, y))
            save_points()
        else:
            # Start dragging
            dragging_index = idx

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_index is not None:
            # Update dragging point
            points[dragging_index] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragging_index is not None:
            # Finish dragging
            save_points()
            dragging_index = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = find_point(x, y)
        if idx is not None:
            # Delete point
            points.pop(idx)
            save_points()


# Open camera
cap = cv2.VideoCapture(1)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Add white border (margin) around the frame
    bordered = cv2.copyMakeBorder(
        frame,
        top=MARGIN, bottom=MARGIN,
        left=MARGIN, right=MARGIN,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)  # white
    )

    bordered = mf.warp_to_rectangle(bordered, wrap_points)
    
    # Draw points
    for i, (x, y) in enumerate(points):
        cv2.circle(bordered, (x, y), RADIUS, (0, 0, 255), -1)
        cv2.putText(bordered, str(i + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Camera", bordered)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

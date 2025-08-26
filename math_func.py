import numpy as np
import cv2 as cv



def dist_between_points(p1, p2):
    return np.sqrt(abs(p1[0]-p2[0])**2 + abs(p1[1]-p2[1])**2)



def middle_point(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]



def picture_diference(im1, im2):
    diff = cv.absdiff(im1, im2)
    change_map = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)
    return diff



def quadrilateral_center(corners):
    """
    Compute the center of a quadrilateral as the intersection of its diagonals.

    Parameters:
        corners (list or ndarray): List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                                   Should be ordered consistently (e.g., clockwise).

    Returns:
        (cx, cy): Coordinates of the intersection point (center).
    """
    if len(corners) != 4:
        raise ValueError("Must provide exactly 4 corners")

    # unpack points
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners

    # Diagonal 1: (x1,y1) -> (x3,y3)
    # Diagonal 2: (x2,y2) -> (x4,y4)
    A = np.array([[x3 - x1, x2 - x4],
                  [y3 - y1, y2 - y4]], dtype=float)
    b = np.array([x2 - x1, y2 - y1], dtype=float)

    # Solve for intersection (parametric form)
    try:
        t, s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("Diagonals are parallel or invalid input")

    cx = x1 + t * (x3 - x1)
    cy = y1 + t * (y3 - y1)

    return (cx, cy)



def predict_next_position(p0, p1, p2, dt=10):
    """
    Predict the next position of a moving object using last 3 positions.
    
    Args:
        p0, p1, p2 : tuple (x,y) - last 3 positions (oldest to newest)
        dt         : time step between positions (default = 1)
        
    Returns:
        (x_pred, y_pred): predicted next position
    """
    p0, p1, p2 = np.array(p0, dtype=float), np.array(p1, dtype=float), np.array(p2, dtype=float)

    # Velocities
    v1 = (p1 - p0) / dt
    v2 = (p2 - p1) / dt

    # Speeds
    s1 = np.linalg.norm(v1)
    s2 = np.linalg.norm(v2)

    # Acceleration (scalar, along direction of motion)
    acceleration = (s2 - s1) / dt

    # Directions (angles)
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])

    # Change in angle (rotation rate)
    dtheta = theta2 - theta1
    # Normalize angle difference to [-pi, pi]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    # Predict next angle and speed
    theta_pred = theta2 + dtheta
    speed_pred = max(0, s2 + acceleration * dt)  # avoid negative speed

    # Predict displacement
    displacement = np.array([np.cos(theta_pred), np.sin(theta_pred)]) * speed_pred * dt
    p_pred = p2 + displacement

    return tuple(p_pred), theta_pred, speed_pred, acceleration




def aruco_apparent_radius(corners):
    """
    Calculate the apparent size of an ArUco marker as the average radius
    from the geometric center to its corners.

    Args:
        corners (list or ndarray): List of 4 points [(x1,y1),..., (x4,y4)],
                                   ordered clockwise or counterclockwise.

    Returns:
        radius (float): Average distance from center to corners.
        center (tuple): Geometric center (x, y).
    """
    if len(corners) != 4:
        raise ValueError("Must provide exactly 4 corners")

    corners = np.array(corners, dtype=float)
    
    p0, p1, p2, p3 = corners
    center_x = (p0[0] + p2[0]) / 2
    center_y = (p0[1] + p2[1]) / 2
    center = np.array([center_x, center_y])

    # Distances from center to each corner
    distances = np.linalg.norm(corners - center, axis=1)

    # Average distance as radius
    radius = np.mean(distances)

    return radius, tuple(center)




def weighted_moving_average(prev_avg, new_img, weight_map):
    """
    Compute a weighted moving average of heatmap images.

    Parameters:
        prev_avg (ndarray): Previous averaged image (HxW or HxWxC).
        new_img (ndarray): New input image (same shape as prev_avg).
        weight_map (ndarray): Weight map (HxW) with values in [0,1].
                              1 = trust new image fully, 0 = keep old average fully.

    Returns:
        updated_avg (ndarray): Updated moving average image.
    """
    if prev_avg.shape != new_img.shape:
        raise ValueError("prev_avg and new_img must have the same shape")

    if weight_map.shape != prev_avg.shape[:2]:
        raise ValueError("weight_map must have same HxW as images")

    # Expand weight_map for color images (HxWxC)
    if len(prev_avg.shape) == 3:
        weight_map = np.expand_dims(weight_map, axis=-1)

    # Weighted update
    updated_avg = (1 - weight_map) * prev_avg + weight_map * new_img
    return updated_avg





def center_of_change(change_map):
    """
    Compute the center of the most change in a change map.
    
    Args:
        change_map (ndarray): 2D array with intensity values representing change.
    
    Returns:
        (cx, cy): Center coordinates (x, y) of change.
    """
    # Ensure it's float for weighted computation
    change_map = change_map.astype(float)

    # Compute total weight
    total_weight = np.sum(change_map)
    if total_weight == 0:
        # No change detected, return center of image
        h, w = change_map.shape
        return w/2, h/2

    # Create coordinate grids
    h, w = change_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Weighted average of coordinates
    cx = np.sum(x_coords * change_map) / total_weight
    cy = np.sum(y_coords * change_map) / total_weight

    return cx, cy

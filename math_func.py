import numpy as np

def dist_between_points(p1, p2):
    return np.sqrt(abs(p1[0]-p2[0])**2 + abs(p1[1]-p2[1])**2)

def middle_point(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
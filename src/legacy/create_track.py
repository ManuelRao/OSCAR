import numpy as np
import matplotlib as plt
import json
import tkinter
from src import math_func as mf
import cv2 as cv
import math

class bezier:
    def __init__(self, _p0, _p1, _p2, _p3):
        self.p0 = _p0
        self.p1 = _p1
        self.p2 = _p2
        self.p3 = _p3
        self.len = self.get_len(20)
        self.hand1ang = 0
        self.hand2ang = 0
        self.get_hand_angs()

    def get_p(self, t: float):
        if 1 < t or 0 > t:
            raise ValueError
            return None
        
        x = ((1-t)**3)*self.p0[0] + 3*((1-t)**2)*t*self.p1[0] + 3*(1-t)*(t**2)*self.p2[0] + (t**3)*self.p3[0]
        y = ((1-t)**3)*self.p0[1] + 3*((1-t)**2)*t*self.p1[1] + 3*(1-t)*(t**2)*self.p2[1] + (t**3)*self.p3[1]

        return [x, y]

    def get_hand_angs(self):
        self.hand1ang = math.atan2(self.p1[1] - self.p0[1], self.p1[0] - self.p0[0])
        self.hand2ang = math.atan2(self.p3[1] - self.p2[1], self.p3[0] - self.p2[0])

    def get_len(self, iter: int):
        points = []
        dst = 0
        for n in range(iter):
            points.append(self.get_p(1/iter*n))

        print("Points:", points)
        for p in range(iter-1):
            dst = dst + mf.dist_between_points(points[p], points[p+1])
            
        return dst
    
    def draw_bezier(self, img: np.ndarray, iter: int = 100, control_points: bool = True):
        if img is None:
            raise ValueError("Image cannot be None")
        if control_points:
            cv.circle(img, (int(self.p0[0]), int(self.p0[1])), 5, (0, 255, 0), -1)
            cv.circle(img, (int(self.p1[0]), int(self.p1[1])), 5, (0, 0, 255), -1)
            cv.circle(img, (int(self.p2[0]), int(self.p2[1])), 5, (0, 0, 255), -1)
            cv.circle(img, (int(self.p3[0]), int(self.p3[1])), 5, (0, 255, 0), -1)
            cv.putText(img, "P0", (int(self.p0[0]), int(self.p0[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(img, "P1", (int(self.p1[0]), int(self.p1[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(img, "P2", (int(self.p2[0]), int(self.p2[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(img, "P3", (int(self.p3[0]), int(self.p3[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.line(img, (int(self.p0[0]), int(self.p0[1])), (int(self.p1[0]), int(self.p1[1])), (0, 0, 255), 1)
            cv.line(img, (int(self.p2[0]), int(self.p2[1])), (int(self.p3[0]), int(self.p3[1])), (0, 0, 255), 1)
        for n in range(iter):
            p1 = self.get_p(1/iter*n)
            p2 = self.get_p(1/iter*(n+1))
            cv.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)

    def set_hand_angle(self, angle: float, which_point: int):
        if which_point == 1:
            self.hand1ang = angle
            d = int(mf.dist_between_points(self.p0, self.p1))
            self.p1[0] = self.p0[0] + d * math.cos(self.hand1ang)
            self.p1[1] = self.p0[1] + d * math.sin(self.hand1ang)
        elif which_point == 2:
            self.hand2ang = angle
            d = int(mf.dist_between_points(self.p2, self.p3))
            self.p2[0] = self.p3[0] - d * math.cos(self.hand2ang)
            self.p2[1] = self.p3[1] - d * math.sin(self.hand2ang)

def create_track(points: list, img_size: tuple = (1000, 1000)):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    bezier_curves = []
    
    for i in range(0, len(points)-1):
        mid = mf.middle_point(points[i], points[i+1])
        b = bezier(points[i], mf.middle_point(mid, points[i]), mf.middle_point(mid, points[i+1]), points[i+1])
        bezier_curves.append(b)
    mid = mf.middle_point(points[0], points[1])
    bezier_curves.append(bezier(points[0], mf.middle_point(mid, points[0]), mf.middle_point(mid, points[len(points)-1]), points[len(points)-1]))

    bezier_curves[0].set_hand_angle(bezier_curves[len(bezier_curves)-1].hand1ang, 1)
    for n, b in enumerate(bezier_curves):
        if n < len(bezier_curves) - 1:
            b.set_hand_angle(bezier_curves[n+1].hand2ang, 2)
    
    for b in bezier_curves:
        b.draw_bezier(img, 10, True)
    

    return img, bezier_curves
if __name__ == "__main__":
    points = [[100, 100], [120, 250], [600, 300], [550, 100]]
    img, b = create_track(points, img_size=(1000, 1000))

    cv.imshow("Bezier Curve", img)
    cv.waitKey(0)
    cv.destroyAllWindows()



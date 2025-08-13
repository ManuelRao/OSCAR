import numpy as np
import matplotlib as plt
import json
import tkinter
import math_func as mf

class bezier:
    def __init__(self, _p0, _p1, _p2, _p3):
        self.p0 = _p0
        self.p1 = _p1
        self.p2 = _p2
        self.p3 = _p3
        self.len = self.get_len()

    def get_p(self, t: float):
        if 1 < t or 0 > t:
            raise ValueError
            return None
        
        x = ((1-t)**3)*p0[0] + 3*((1-t)**2)*t*p1[0] + 3*(1-t)*(t**2)*p2[0] + (t**3)*p3[0]
        y = ((1-t)**3)*p0[1] + 3*((1-t)**2)*t*p1[1] + 3*(1-t)*(t**2)*p2[1] + (t**3)*p3[1]

        return [x, y]

    def get_len(self, iter: int):
        points = []
        dst = 0
        for n in range(iter):
            points.append(self.get_p(1/iter*n))
        for p in range(iter-1):
            dst = dst + mf.dist_between_points(points[p], points[p+1])
        return dst
        
        
        


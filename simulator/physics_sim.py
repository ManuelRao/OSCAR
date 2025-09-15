import numpy as np

class car:
    def __init__(self, dimensions, thrust_map, wheel_grip, center_of_mass, mass):
        # dimensions: (length, width) in meters
        # thrust_map: function mapping throttle input and wheel speed to force
        # wheel_grip: coefficient of friction for the wheels
        # center_of_mass: (x, y, z) offset from geometric center in meters
        # mass: in kilograms
        self.dim = dimensions
        self.th_map = thrust_map
        self.wh_grip = wheel_grip
        self.cm = center_of_mass
        self.mass = mass
        self.position = np.zeros(3)  # x, y, z in meters
        self.velocity = np.zeros(3)  # vx, vy, vz in m/s
        self.orientation = 0.0  # heading in radians
        self.angular_velocity = 0.0  # radians per second
        self.wheel_speeds = np.zeros(4)  # front-left, front-right, rear-left, rear-right in rad/s

        def get_state(self):
            return {
                'position': self.position,
                'velocity': self.velocity,
                'orientation': self.orientation,
                'angular_velocity': self.angular_velocity,
                'wheel_speeds': self.wheel_speeds
            }
        
        def get_wheel_forces(self, throttle_inputs):
            
            
        
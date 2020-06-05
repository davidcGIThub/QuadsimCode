# system imports
import numpy as np


class Trajectory:
    def __init__(self, pos = np.zeros(3), vel=np.zeros(3), accel = np.zeros(3),
            jerk = np.zeros(3), yaw = 0.0, yaw_rate = 0.0):
        self.pos = pos
        self.vel = vel
        self.accel = accel
        self.jerk = jerk
        self.yaw = yaw
        self.yaw_rate = yaw_rate
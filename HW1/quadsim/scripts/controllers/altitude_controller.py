#altitude controller

import numpy as np


class AltitudeController:
    """Controller to stabalize the desired altitude of a quadrotor UAV.
    kp is proportional gain(s), kd is the derivative gain, and ki is the integral gain.
    max_altitude is the absolute value of max angular velocity.

    kp and max_ang_vel can be scalars or 1D numpy arrays of length 3
    """
    def __init__(self, kp=1.0, kd=1.0, ki = 1.0, min_altitude=1.0, throttle_eq = 0.54447, max_throttle = 0.85):
        self.kp = kp
        self.kd = kd
        self.min_altitude = min_altitude
        self.throttle_eq = throttle_eq
        self.max_throttle = max_throttle
        self.altitude = 0

    def saturate(self, throttle):
        """use self.max_cmd to saturate control"""
        throttle_clipped = np.clip(throttle, 0, self.max_throttle)
        return throttle_clipped

    def calc_throttle(self, alt_des, alt, dt):
        """Compute the throttle command from current altitude
        and desired altitude. Returns floating point value
        """
        #proportional gain variables
        max_diff = 30 # max alt error before goes full throttle
        diff_normalizer = max_diff / (1 - self.throttle_eq) #If 30 meters away makes go full throttle
        kp = self.kp/diff_normalizer
        if alt_des < self.min_altitude:
            alt_des = self.min_altitude
        alt_error = alt_des - alt

        #derivative gain variables
        max_vel = 15 #max velocity before throttle diff is nulled
        vel_normalizer = max_vel / (1 - self.throttle_eq)
        kd = self.kd / vel_normalizer
        velocity = (alt-self.altitude)/dt

        #calculate commanded throttle
        throttle_cmd = kp*alt_error - kd*velocity + self.throttle_eq
        throttle_cmd_sat = self.saturate(throttle_cmd)
        self.altitude = alt
        return throttle_cmd_sat
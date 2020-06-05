import numpy as np

import utils.quaternion as quat
import utils.math_tools as mt


class AttitudeController:
    """Controller to stabilize the attitude of a quadrotor UAV.
    kp is proportional gain(s) multiplied by attitude error.
    max_ang_vel is the absolute value of max angular velocity.

    kp and max_ang_vel can be scalars or 1D numpy arrays of length 3
    """
    def __init__(self, kp=1.0, max_ang_vel=np.pi):
        self.kp = kp
        self.max_cmd = max_ang_vel

    def saturate(self, cmd):
        """use self.max_cmd to saturate control"""
        cmd = np.clip(cmd, -self.max_cmd, self.max_cmd)
        return cmd

    def calc_command(self, att_des: quat.Quaternion, ang_vel_des,
            att: quat.Quaternion):
        """Compute the angular velocity command from current attitude
        and desired attitude. Returns 1D numpy array of length 3. att_des and 
        att are quaternions equivalent to rotations from inertial to body
        """
        q_des2body = att_des.inverse() @ att # return quaternion rotation from desired to body
        feedforward = q_des2body.rotp(ang_vel_des) #rotate ang_vel_des from des frame to body frame
        
        error = att_des.box_minus(att) #returns Axis-Angle representation of how much you should rotate about each of the body frame axes
        feedback = self.kp * error

        ang_vel_cmd = self.saturate(feedforward + feedback)
        return ang_vel_cmd 

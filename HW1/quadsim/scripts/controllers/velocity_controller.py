# system imports
import numpy as np

# local imports
import utils.quaternion as quat
import utils.math_tools as mt


class VelocityController:
    """Velocity controller class. Trajectory received contains desired
    position, velocity, acceleration, and yaw in order to generate 
    throttle, attitude, and desired angular velocity commands.
    
    Kd are 1D numpy arrays of length 3. All
    other arguments are scalars."""
    def __init__(self, Kd, throttle_eq, max_throttle, min_altitude):
        self.Kd = Kd
        self.throttle_eq = throttle_eq
        self.max_throttle = max_throttle
        self.min_altitude = min_altitude
        # Assume linear throttle model: throttle = k_thrust*thrust
        # This means that "thrust" is being used as Force/mass (acceleration)
        self.g = 9.81
        self.k_thrust = throttle_eq / self.g
        self.e3 = mt.ei(2)
        self.att_prev = quat.identity()
        self.R_di_prev = np.eye(3)
        self.pos_prev = np.array((0,0,0))

    def saturate(self, throttle):
        return np.clip(throttle, 0, self.max_throttle)

    def calc_command(self, vel, vel_des, R_bi, yaw, dt):
        """Calculate the throttle and attitude commands from desired
        position and heading. Returns both scalar throttle command as
        well as a unit quaternion attitude command.

        Positions, velocities, and accel_des are 1D numpy arrays of 
        length 3. Headings and dt are scalars.
        """

        vel_err = vel_des - vel
        # yaw_des = -np.pi/2.0
        yaw_des = 0.0

        thrust_vec = self.g*self.e3 - self.Kd@(vel_err)
        thrust_cmd = mt.norm2(thrust_vec)

        throttle_cmd = self.calc_throttle_cmd(thrust_cmd)
        att_cmd = self.calc_att_cmd(thrust_vec, thrust_cmd, yaw_des)
        ang_vel_des = self.calc_ang_vel_des(att_cmd, dt)

        return throttle_cmd, att_cmd, ang_vel_des

    def calc_throttle_cmd(self, thrust_cmd):
        throttle_cmd = self.k_thrust*thrust_cmd
        return self.saturate(throttle_cmd)
    
    def calc_att_cmd(self, Tc_vec, Tc, yaw_des):
        s_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])

        R = np.empty([3,3])
        R[:,2] = Tc_vec / Tc
        R[:,1] = mt.skew(R[:,2]) @ s_yaw
        R[:,1] /= mt.norm2(R[:,1])
        R[:,0] = mt.skew(R[:,1]) @ R[:,2] 
        return quat.from_R(R)

    def calc_ang_vel_des(self, att_cmd, dt):
        R = att_cmd.R
        omega_x = 1/dt*mt.logm(self.R_di_prev.T @ R)
        R_dot = R @ omega_x
        self.R_di_prev = R
        ang_vel_des = mt.vee(R.T @ R_dot)
        # print(att_cmd.euler)
        return ang_vel_des

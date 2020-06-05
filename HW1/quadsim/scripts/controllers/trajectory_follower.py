# system imports
import numpy as np

# local imports
import utils.quaternion as quat
import utils.math_tools as mt


class TrajectoryFollower:
    """Trajectory following class. Trajectory received contains desired
    position, velocity, acceleration, and yaw in order to generate 
    throttle, attitude, and desired angular velocity commands.
    
    kp and kd can be scalars or 1D numpy arrays of length 3. All
    other arguments are scalars."""
    def __init__(self, kp, kd, throttle_eq, max_throttle, min_altitude):
        self.kp = kp
        self.kd = kd
        self.throttle_eq = throttle_eq
        self.max_throttle = max_throttle
        self.min_altitude = min_altitude
        # Assume linear throttle model: throttle = k_thrust*thrust
        # This means that "thrust" is being used as Force/mass (acceleration)
        self.g = 9.81
        self.k_thrust = throttle_eq / self.g
        self.e2 = mt.ei(2) #np.array([0,0,1])
        self.att_prev = quat.identity()

    def saturate(self, throttle):
        return np.clip(throttle, 0, self.max_throttle)

    def calc_command(self, traj, pos, vel, att, dt):
        """Calculate the throttle and attitude commands from desired
        position and heading. Returns both scalar throttle command as
        well as a unit quaternion attitude command.

        Positions, velocities, and accel_des are 1D numpy arrays of 
        length 3. Headings and dt are scalars.
        """
        pos_des = traj.pos #desired position in intertial frame
        vel_des = traj.vel #desired velocity in the inertial frame
        accel_des = traj.accel #desired acceleration in the intertial frame
        yaw_des = traj.yaw #desired yaw in the inertial frame

        pos_err = pos - pos_des #position error in the inertial frame
        vel_err = vel - vel_des #velocity error in the inertial frame

        yaw = att.euler[2] #yaw in the inertial frame, transformed from quaternion

        thrust_feedforward = -accel_des + self.g*self.e2 #gravity minus acceleration desired
        thrust_feedback = self.kp*pos_err + self.kd*vel_err #pid control using position error*kp velocity error*kd in the inertial frame
        thrust_vec = thrust_feedforward + thrust_feedback #cmd thrust or cmd acceleration in 3 directions

        throttle_cmd = self.calc_throttle_cmd(thrust_vec, pos, att)
        att_cmd = self.calc_att_cmd(thrust_vec, yaw_des)
        ang_vel_des = self.calc_ang_vel_des(att_cmd, pos, vel, yaw, traj)

        return throttle_cmd, att_cmd, ang_vel_des

    def calc_throttle_cmd(self, thrust_vec, pos, att):
        if -pos[2] < self.min_altitude:
            # if altitude is too low, apply high throttle
            throttle_cmd = 0.9 * self.max_throttle
        else:
            #att.R.T @ self.e2 = passive rotation unit vector e2 from body to inertial
            throttle_cmd = self.k_thrust \
                * thrust_vec.dot(att.R.T @ self.e2) 
        
        return self.saturate(throttle_cmd)
    
    def calc_att_cmd(self, thrust_vec, yaw_des):
        thrust_mag = mt.norm2(thrust_vec)
        s_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])

        R = np.empty([3,3])
        R[:,2] = thrust_vec / thrust_mag
        R[:,1] = mt.skew(R[:,2]) @ s_yaw
        R[:,1] /= mt.norm2(R[:,1])
        R[:,0] = mt.skew(R[:,1]) @ R[:,2] 

        return quat.from_R(R)

    # def calc_ang_vel_des(self, att_cmd, dt):
    #     ang_vel_des = att_cmd.box_minus(self.att_prev) / dt
    #     self.att_prev = att_cmd
    #
    #     return ang_vel_des

    def calc_ang_vel_des(self, att_cmd, pos, vel, yaw, traj):
        pos_des = traj.pos
        vel_des = traj.vel
        accel_des = traj.accel
        jerk_des = traj.jerk
        yaw_des = traj.yaw
        yaw_rate_des = traj.yaw_rate

        pos_err = pos - pos_des
        vel_err = vel - vel_des

        R = att_cmd.R
        r1 = R[:,0]
        r2 = R[:,1]
        r3 = R[:,2]

        s_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        s_yaw_dot = np.array([-yaw_rate_des*np.sin(yaw_des),
                            yaw_rate_des*np.cos(yaw_des), 0])

        a = accel_des - self.g*np.array([0.,0.,1.]) - self.kp*pos_err - self.kd*vel_err
        accel_err = -self.kp*pos_err - self.kd*vel_err
        a_dot = jerk_des - self.kp*vel_err - self.kd*accel_err
        r3_dot = -a_dot/np.linalg.norm(a) + a * (a_dot @ a) / np.linalg.norm(a)**3

        num = mt.skew(r3) @ s_yaw
        num_dot = mt.skew(r3_dot) @ s_yaw + mt.skew(r3) @ s_yaw_dot
        r2_dot = num_dot/np.linalg.norm(num) - num * (num_dot @ num) / np.linalg.norm(num)**3

        r1_dot = mt.skew(r2_dot) @ r3 + mt.skew(r2) @ r3_dot

        R_dot = np.zeros((3,3))
        R_dot[:,0] = r1_dot
        R_dot[:,1] = r2_dot
        R_dot[:,2] = r3_dot

        ang_vel_des = mt.vee(R.T @ R_dot)
        return ang_vel_des

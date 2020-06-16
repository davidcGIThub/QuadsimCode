#!/usr/bin/env python3

# system imports
import numpy as np
import scipy.linalg

# ROS imports
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from quadsim.msg import ImageData 
import utils.quaternion as quat
from rosflight_msgs.msg import Status

# local imports
import utils.math_tools as mt


class ImuIntegratorNode:
    """ROS node that integrates IMU and published the resulting pose"""
    def __init__(self):
        # Initialize ROS
        self.pose_msg = PoseStamped()
        self.use_imu = rospy.get_param('~use_imu', False)
        if self.use_imu:
            rospy.Subscriber('imu/data', Imu, self.imuCallback, queue_size=1)
        else:
            rospy.Subscriber('truth/NED', Odometry, self.truthCallback, queue_size=1)
        rospy.Subscriber("status", Status, self.statusCallback, queue_size=1)
        self.pose_pub = rospy.Publisher('pose', PoseStamped, queue_size=1)
        self.a_lm1 = np.zeros(3)
        self.a_lm1[2] = -9.80665
        self.w_lm1 = np.zeros(3)
        self.p_l_lm1 = np.zeros(3)
        self.v_l_lm1 = np.zeros(3)
        self.q_l_lm1 = quat.identity()
        self.q_l_i = quat.identity()
        self.v_l_i = np.zeros(3)
        self.p_l_i = np.zeros(3)
        self.t_prev = rospy.Time.now().to_sec()
        self.GYRO_X_BIAS = rospy.get_param("GYRO_X_BIAS", 0.0)
        self.GYRO_Y_BIAS = rospy.get_param("GYRO_Y_BIAS", -0.00537)
        self.GYRO_Z_BIAS = rospy.get_param("GYRO_Z_BIAS", 0.0)
        self.ACC_X_BIAS = rospy.get_param("ACC_X_BIAS", 0.2144661992788315)
        self.ACC_Y_BIAS = rospy.get_param("ACC_Y_BIAS", 0.5216714143753052)
        self.ACC_Z_BIAS = rospy.get_param("ACC_Z_BIAS", 0.02329552546143532)
        self.init = True
        self.armed = False
        
    def run(self):
        """This function runs until ROS node is killed"""
        rospy.spin()
    
    def statusCallback(self, msg):
        self.armed = msg.armed

    def imuCallback(self, msg):
        t = msg.header.stamp.to_sec()
        dt = t - self.t_prev 
        self.t_prev = t

        if self.armed:
            accel = msg.linear_acceleration
            accel.x += self.ACC_X_BIAS
            accel.y += self.ACC_Y_BIAS
            accel.z += self.ACC_Z_BIAS
            ang_vel = msg.angular_velocity
            ang_vel.x += self.GYRO_X_BIAS
            ang_vel.y += self.GYRO_Y_BIAS
            ang_vel.z += self.GYRO_Z_BIAS
            a_l = np.array([accel.x, accel.y, accel.z])
            w_l = np.array([ang_vel.x, ang_vel.y, ang_vel.z])

            ## ------------------------ Relative to inertial approach ------------------------
            #Eqn 8.4
            self.q_l_i = quat.Exp(-w_l * dt) @ self.q_l_i
            # Eqn 8.5
            self.v_l_i = self.v_l_i + dt * 9.80665 * mt.ei(2) + dt * self.q_l_i.rotp(a_l)
            # Eqn 8.6
            self.p_l_i = self.p_l_i + dt * self.v_l_i + dt**2/2.0 * (9.80665*mt.ei(2) + self.q_l_i.rotp(a_l))

            ## ----------------- IMU relative integration approach (TODO: doesn't work yet) -----------------
            # if self.init:
            #     R_1_km1, v_1_km1, p_1_km1 = self.imu_init(msg, dt)
            #     self.v_l_lm1 = v_1_km1
            #     self.q_l_lm1 = quat.from_R(R_1_km1)
            #     self.p_l_lm1 = p_1_km1
            #     self.init = False

            # # Eqn 8.7
            # q_lp1_l = quat.Exp(w_l * dt) #Not using - sign on w_l because using Quaternion Exp
            # # Eqn 8.8
            # # v = self.v_l_lm1 + dt * (self.q_l_lm1.rotp(self.a_lm1) - a_l)
            # v = self.v_l_lm1 + dt * (self.q_l_lm1.R @ (self.a_lm1) - a_l)
            # # v_lp1_l = q_lp1_l.rotp(v)
            # v_lp1_l = q_lp1_l.R @ (v)
            # # Eqn 8.9
            # # p = self.p_l_lm1 + dt * self.v_l_lm1 + dt**2/2 * (self.q_l_lm1.rotp(self.a_lm1) - a_l)
            # p = self.p_l_lm1 + dt * self.v_l_lm1 + dt**2/2 * (self.q_l_lm1.R @ (self.a_lm1) - a_l)
            # # p_lp1_l = q_lp1_l.rotp(p)
            # p_lp1_l = q_lp1_l.R @ (p)

            # self.q_l_lm1 = q_lp1_l
            # self.v_l_lm1 = v_lp1_l
            # self.p_l_lm1 = p_lp1_l
            # self.a_lm1 = a_l
            # self.w_lm1 = w_l

            # # pose iterations
            # #   We aren't resetting the pose between frames as outlined in the book, so instead 
            # #   of k-1 (km1) we just use inertial
            # R_lp1_i = q_lp1_l.R @ self.q_l_i.R
            # # v_lp1_i = q_lp1_l.R @ v_lp1_l
            # p_lp1_i = q_lp1_l.R @ self.p_l_i  + p_lp1_l

            # # store latest estimate
            # self.p_l_i = p_lp1_i
            # self.q_l_i = quat.from_R(R_lp1_i)
            ## --------------------------------------------------------------------------------

            # pack into message
            self.pose_msg.pose.position.x = self.p_l_i.item(0)
            self.pose_msg.pose.position.y = self.p_l_i.item(1)
            self.pose_msg.pose.position.z = self.p_l_i.item(2)
            self.pose_msg.pose.orientation.w = self.q_l_i.q0
            self.pose_msg.pose.orientation.x = self.q_l_i.qx
            self.pose_msg.pose.orientation.y = self.q_l_i.qy
            self.pose_msg.pose.orientation.z = self.q_l_i.qz
            self.pose_msg.header = msg.header
            self.pose_pub.publish(self.pose_msg)
    
    ## ----------------- IMU relative integration approach (TODO: doesn't work yet) -----------------
    # def imu_init(self, msg, dt):
    #     omega = np.array((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))
    #     a = np.array((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
    #     R_1_km1 = scipy.linalg.expm(-mt.skew(omega)*dt)
    #     v_1_km1 = np.zeros(3)
    #     p_1_km1 = np.zeros(3)
    #     return R_1_km1, v_1_km1, p_1_km1
    ## --------------------------------------------------------------------------------

    def truthCallback(self, msg):
        self.pose_msg.header = msg.header
        self.pose_msg.pose = msg.pose.pose
        self.pose_pub.publish(self.pose_msg)       

if __name__ == '__main__':
    rospy.init_node('imu_integrator', anonymous=True)
    try:
        ros_node = ImuIntegratorNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass
    
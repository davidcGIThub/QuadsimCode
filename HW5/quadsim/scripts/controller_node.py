#!/usr/bin/env python3

# system imports
import numpy as np

# ROS imports
import rospy
from nav_msgs.msg import Odometry
from rosflight_msgs.msg import Command, Status
from std_msgs.msg import Float32MultiArray

# local imports
import utils.state as state
import utils.trajectory as trajectory
import utils.quaternion as quat
import controllers.attitude_controller as ac
import controllers.trajectory_follower as tf


class ControllerNode:
    """ROS node that runs the quadsim controller"""
    def __init__(self):
        self.traj_follower = self.setup_trajectory_follower()
        self.att_controller = self.setup_attitude_stabilizer()
        self.state = state.State()
        self.traj = trajectory.Trajectory()
        self.armed = False
        self.t0 = 0.0
        self.t_prev = 0.0

        # Initialize ROS
        rospy.Subscriber("truth/NED", Odometry, self.stateCallback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("status", Status, self.statusCallback, queue_size=1)
        self.cmd_pub = rospy.Publisher("command", Command, queue_size=1)
        rospy.Subscriber("trajectory", Float32MultiArray, 
            self.trajectoryCallback)
        self.cmd_msg = Command()
        self.cmd_msg.mode = Command.MODE_ROLLRATE_PITCHRATE_YAWRATE_THROTTLE

    def run(self):
        """This loop runs until ROS node is killed"""
        rospy.spin()

    def setup_trajectory_follower(self):
        ns = 'trajectory_follower/'
        kp_raw = rospy.get_param(ns+'kp')
        kd_raw = rospy.get_param(ns+'kd')
        kp = np.array(kp_raw)
        kd = np.array(kd_raw)

        throttle_eq = rospy.get_param(ns+"equilibrium_throttle", 0.5)
        max_throttle = rospy.get_param(ns+"max_throttle", 1.0)
        min_altitude = rospy.get_param(ns+"min_altitude", 1.0)

        return tf.TrajectoryFollower(kp, kd, throttle_eq, max_throttle,
            min_altitude)

    def setup_attitude_stabilizer(self):
        ns = 'attitude_controller/'
        kp_raw = rospy.get_param(ns+'kp')
        kp = np.array(kp_raw)

        max_roll_rate = rospy.get_param(ns+"max_roll_rate", 1.57)
        max_pitch_rate = rospy.get_param(ns+"max_pitch_rate", 1.57)
        max_yaw_rate = rospy.get_param(ns+"max_yaw_rate", 0.785)
        max_ang_vel = np.array([max_roll_rate, max_pitch_rate, max_yaw_rate])

        return ac.AttitudeController(kp, max_ang_vel)

    def statusCallback(self, msg):
        self.armed = msg.armed

    def trajectoryCallback(self, msg):
        traj = np.array(msg.data).reshape(4,-1)
        self.traj.pos = traj[:3, 0]
        self.traj.vel = traj[:3, 1]
        self.traj.accel = traj[:3, 2]
        self.traj.jerk = traj[:3, 3]
        self.traj.yaw = traj[3, 0]
        self.traj.yaw_rate = traj[3, 1]

    def stateCallback(self, msg: Odometry):
        if self.t_prev == 0.0:
            self.t_prev = msg.header.stamp.to_sec()
            return
        now = msg.header.stamp.to_sec()
        dt = now - self.t_prev
        self.t_prev = now

        position = msg.pose.pose.position   # NED position in i frame
        orient = msg.pose.pose.orientation  # orientation in quaternion
        linear = msg.twist.twist.linear     # linear velocity in i? frame
        angular = msg.twist.twist.angular   # angular velocity in b frame

        self.state.pos = np.array([position.x, position.y, position.z])
        self.state.att = np.array([orient.w, orient.x, orient.y, orient.z])
        self.state.vel = np.array([linear.x, linear.y, linear.z])
        self.state.ang_vel = np.array([angular.x, angular.y, angular.z])

        if self.armed:
            if self.t0 == 0.0:
                self.t0 = now
            self.compute_control(dt)
            self.publish_command()
        
    def compute_control(self, dt):
        # Trajectory follower
        throttle_cmd, att_cmd, ang_vel_des = self.traj_follower.calc_command(
            self.traj, self.state.pos, self.state.att.rota(self.state.vel),
            self.state.att, dt)
        
        # Attitude stabilizer
        ang_vel_cmd = self.att_controller.calc_command(att_cmd, ang_vel_des,
            self.state.att)

        # Formulate ROS msg
        self.cmd_msg.F = throttle_cmd
        self.cmd_msg.x = ang_vel_cmd[0]
        self.cmd_msg.y = ang_vel_cmd[1]
        self.cmd_msg.z = ang_vel_cmd[2]

    def publish_command(self):
        self.cmd_msg.header.stamp = rospy.Time.now()
        self.cmd_pub.publish(self.cmd_msg)

if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True)
    try:
        ros_node = ControllerNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass

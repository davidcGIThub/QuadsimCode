#!/usr/bin/env python3

# system imports
import numpy as np

# ROS imports
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from quadsim.msg import ImageData

# local imports
# import the imu integrator


class ImuIntegratorNode:
    """ROS node that integrates IMU and published the resulting pose"""
    def __init__(self):
        # Initialize ROS
        self.pose_msg = PoseStamped()
        self.use_imu = rospy.get_param('~use_imu', True)
        if self.use_imu:
            rospy.Subscriber('imu/data', Imu, self.imuCallback, queue_size=1)
        else:
            rospy.Subscriber('truth/NED', Odometry, self.truthCallback,
                             queue_size=1)
        self.pose_pub = rospy.Publisher('pose', PoseStamped, queue_size=1)

    def run(self):
        """This function runs until ROS node is killed"""
        rospy.spin()

    def imuCallback(self, msg):
        # IMU integration

        self.pose_msg.header = msg.header
        self.pose_pub(self.pose_msg)

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

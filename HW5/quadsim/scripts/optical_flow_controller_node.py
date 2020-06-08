#!/usr/bin/env python3

# system imports
import numpy as np

# ROS imports
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from quadsim.msg import ImageData

# local imports
import controllers.optical_flow_controller as ofc

class OpticalFlowControllerNode:
    def __init__(self):
        self.opt_flow = self.setup_optical_flow_controller()
        self.velocity_command_msg = Vector3()
        self.Kc = np.zeros((3,3))
        self.calc_camera_matrix()

        # Initialize ROS
        self.cv_bridge = CvBridge() # converts ROS img to OpenCV img
        self.setup_ros_subscribers()
        self.setup_ros_publishers()

    def run(self):
        """This function runs until ROS node is killed"""
        rospy.spin()

    def setup_optical_flow_controller(self):
        ns = 'optical_flow_controller/'
        follow_option = rospy.get_param(ns+"follow_option", 0)
        wall_option = rospy.get_param(ns+"wall_option", 1)
        k_terrain = rospy.get_param(ns+"k_terrain", 0.01)
        Dv_terrain = rospy.get_param(ns+"Dv_terrain", 30.0)
        k_wall = rospy.get_param(ns+"k_wall", 0.01)
        Dv_wall = rospy.get_param(ns+"Dv_wall", 30.0)
        k_canyon = rospy.get_param(ns+"k_canyon", 2.0)
        return ofc.OpticalFlowController(follow_option, wall_option, k_terrain, Dv_terrain, k_wall, Dv_wall, k_canyon)

    def setup_ros_subscribers(self):
        rospy.Subscriber("image_data", ImageData, self.featuresCallback,
                         queue_size=1)

    def setup_ros_publishers(self):
        self.velocity_command_pub = rospy.Publisher("optical_flow/velocity_cmd", Vector3, queue_size=1)

    def calc_camera_matrix(self):
        ns = 'optical_flow_controller/'
        h_fov = rospy.get_param(ns+"h_fov", 90)
        size = rospy.get_param(ns+"img_size", [640, 480])
        f = 0.5*size[0]/np.tan(np.radians(h_fov)/2.0)
        self.Kc = np.array([[f, 0, size[0]/2.0],
                            [0, f, size[1]/2.0],
                            [0, 0, 1]])

    def featuresCallback(self, msg):
        self.process_features(msg)
        self.publish_velocity_commands()

    def process_features(self, img_data):
        self.opt_flow.process_optical_flow(img_data, self.Kc)

    def publish_velocity_commands(self):
        self.velocity_command_msg.x =  self.opt_flow.velocity_command.item(0)
        self.velocity_command_msg.y =  self.opt_flow.velocity_command.item(1)
        self.velocity_command_msg.z =  self.opt_flow.velocity_command.item(2)
        self.velocity_command_pub.publish(self.velocity_command_msg)

if __name__ == '__main__':
    rospy.init_node('optical_flow_controller', anonymous=True)
    try:
        ros_node = OpticalFlowControllerNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass

#! /usr/bin/env python3


import numpy as np

import rospy
from sensor_msgs.msg import PointCloud2, PointCloud
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

from quadsim.msg import ImageData
from utils.quaternion import Quaternion
import utils.vision_tools as vt



class PointTriangulationNode:
    """
    ROS wrapper for feature point triangulation.
    Input is a set of tracked feature pairs,
    Output is a point cloud.
    """

    def __init__(self):
        ns = 'triangulation/'
        self._use_pointcloud2 = rospy.get_param(ns + 'use_pointcloud2', False)
        self.setup_ros_publishers()
        self.setup_ros_subscribers()
        self.setup_camera()
        self.setup_triangulation()

    def setup_ros_subscribers(self):
        rospy.Subscriber("image_data", ImageData, callback=self.imageDataCallback)
        rospy.Subscriber("truth/NED", Odometry, callback=self.stateCallback)

    def setup_ros_publishers(self):
        pointcloud_topic_name = "triang_points"
        if self._use_pointcloud2:
            self.pointcloud_pub = rospy.Publisher(pointcloud_topic_name, PointCloud2,queue_size=1)
        else:
            self.pointcloud_pub = rospy.Publisher(pointcloud_topic_name, PointCloud,queue_size=1)

    def setup_camera(self):
        ns = 'camera/'
        h_fov = rospy.get_param(ns + "horizontal_fov", 90)
        img_size = rospy.get_param(ns + "img_size", [640, 480])
        self.Kc = vt.calculate_calibration_matrix(h_fov, img_size)
        self.Kc_inv = np.linalg.inv(self.Kc)

        # get rotation from body frame to camera frame
        self.R_bc = np.array([rospy.get_param(ns + "R_bc")]).reshape(3,3)

    def setup_triangulation(self):
        ns = 'triangulation/'
        self.triang_method = rospy.get_param(ns + "method", 0)

    def imageDataCallback(self, msg):

        trans_msg = msg.relative_pose.position  # NED translation
        rot_msg = msg.relative_pose.orientation  # orientation in quaternion

        # these come in as body to body, need camera to camera
        translation = np.array([trans_msg.x, trans_msg.y, trans_msg.z])
        rotation = Quaternion(
            np.array([rot_msg.w, rot_msg.x, rot_msg.y, rot_msg.z]))

        # convert from body to camera frame
        translation = self.R_bc @ translation

        features_next = np.array(msg.features_next).reshape(-1, 2).T
        features_prev = np.array(msg.features_prev).reshape(-1, 2).T

        #dt = msg.dt
        if np.linalg.norm(translation) > 0.05:
            # points in the next camera frame
            p_next = vt.triangulate_points(features_prev, features_next,
                                       translation, rotation, self.triang_method, self.Kc_inv)

            p_inertial = self.current_att.R.T @ self.R_bc.T @ p_next + self.current_pos.reshape((3,1))
            self.publish_pointCloud(p_inertial.T)


    def stateCallback(self, msg):
        position = msg.pose.pose.position   # NED position in i frame
        orient = msg.pose.pose.orientation  # orientation in quaternion

        self.current_pos = np.array([position.x, position.y, position.z])
        self.current_att = Quaternion(np.array([orient.w, orient.x, orient.y, orient.z]))


    def publish_pointCloud(self, points: np.ndarray):
        assert (points.ndim == 2 and points.shape[1] == 3)
        if self._use_pointcloud2:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'ned'
            cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
            self.pointcloud_pub.publish(cloud_msg)
        else:
            cloud_msg = PointCloud()
            def make_point(x: float, y: float, z: float) -> Point32:
                p = Point32()
                p.x = x
                p.y = y
                p.z = z
                return p
            cloud_msg.points = [make_point(row[0], row[1], row[2]) for row in points]
            self.pointcloud_pub.publish(cloud_msg)


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('point_triangulation', anonymous=True)
    try:
        ros_node = PointTriangulationNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass

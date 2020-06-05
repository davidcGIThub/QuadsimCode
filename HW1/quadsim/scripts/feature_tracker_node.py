#!/usr/bin/env python3

# system imports
import numpy as np

# ROS imports
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
# from quadsim.msg import ImageData

# local imports
import controllers.feature_tracker as ft
import utils.quaternion as quat

class FeatureTrackerNode:
    def __init__(self):
        self.feature_tracker = self.setup_feature_tracker()
        self.show = rospy.get_param('~display_img', False)
        # self.pos_prev = np.zeros(3)
        # self.att_prev = quat.identity()
        # self.pos = np.zeros(3)
        # self.att = quat.identity()

        # Initialize ROS
        self.cv_bridge = CvBridge() # converts ROS img to OpenCV img
        rospy.Subscriber("holodeck/image", Image, self.imageCallback)
        # rospy.Subscriber("pose", PoseStamped, self.poseCallback)
        # self.img_data_pub = rospy.Publisher("image_data", ImageData,
        #                                     queue_size=1)
        # self.t_prev = rospy.Time.now()
        # self.img_data_msg = ImageData()
        # self.img_width = 0
        # self.img_height = 0
        # self.init = True

    def run(self):
        """This function runs until ROS node is killed"""
        rospy.spin()

    def setup_feature_tracker(self):
        max_features = rospy.get_param('~max_features', 500)
        quality = rospy.get_param('~quality', 0.1)
        min_dist = rospy.get_param('~min_dist', 10.0)
        return ft.FeatureTracker(max_features, quality, min_dist)

    def imageCallback(self, msg):
        # rel_pos, rel_att = self.calc_relative_pose()

        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # if self.init:
        #     self.img_width = msg.width
        #     self.img_height = msg.height
        #     self.init = False
        self.process_img(img)
        # self.publish_processed_img_data(rel_pos, rel_att)

        # self.pos_prev = self.pos
        # self.att_prev = self.att

    # def poseCallback(self, msg):
    #     position = msg.pose.position
    #     orient = msg.pose.orientation
    #     self.pos = np.array([position.x, position.y, position.z])
    #     self.att = quat.Quaternion([orient.w, orient.x, orient.y, orient.z])

    # def calc_relative_pose(self):
    #     rel_att = self.att_prev.inverse() @ self.att
    #     rel_pos = rel_att.rotp(self.pos_prev) - self.pos
    #     return rel_pos, rel_att

    def process_img(self, img):
        # t = rospy.Time.now()
        # self.img_data_msg.dt = t - self.t_prev
        # self.t_prev = rospy.Time.now()
        self.feature_tracker.save_trackable_features(img, show=self.show)

    # def publish_processed_img_data(self, rel_pos, rel_att):
    #     self.img_data_msg.relative_pose.position.x = rel_pos[0]
    #     self.img_data_msg.relative_pose.position.y = rel_pos[1]
    #     self.img_data_msg.relative_pose.position.z = rel_pos[2]
    #     self.img_data_msg.relative_pose.orientation.w = rel_att.q0
    #     self.img_data_msg.relative_pose.orientation.x = rel_att.qx
    #     self.img_data_msg.relative_pose.orientation.y = rel_att.qy
    #     self.img_data_msg.relative_pose.orientation.z = rel_att.qz

    #     self.img_data_msg.img_width = self.img_width
    #     self.img_data_msg.img_height = self.img_height
    #     self.img_data_msg.feature_count = len(self.feature_tracker.features_paired)
    #     if self.img_data_msg.feature_count > 0:
    #         self.img_data_msg.features_prev = \
    #             self.feature_tracker.features_paired[:,0,:].flatten().astype('int32').tolist()
    #         self.img_data_msg.features_next = \
    #             self.feature_tracker.features_paired[:,1,:].flatten().astype('int32').tolist()

    #         self.img_data_pub.publish(self.img_data_msg)

if __name__ == '__main__':
    rospy.init_node('feature_tracker', anonymous=True)
    try:
        ros_node = FeatureTrackerNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass
    
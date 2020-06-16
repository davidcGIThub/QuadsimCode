#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros


class QuadsimTF2:
    def __init__(self):
        rospy.init_node('quadsim_tf2')
        world_frame_name = rospy.get_param('~world_frame_name', 'world')
        ned_frame_name = rospy.get_param('~ned_frame_name', 'ned')
        vehicle_frame_name = rospy.get_param('~vehicle_frame_name', 'vehicle')
        body_frame_name = rospy.get_param('~body_frame_name', 'body')
        self._static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._transform_broadcaster = tf2_ros.TransformBroadcaster()
        ned_transform = TransformStamped()
        ned_transform.header.frame_id = world_frame_name
        ned_transform.header.stamp = rospy.Time.now()
        ned_transform.child_frame_id = ned_frame_name
        ned_transform.transform.rotation.w = 0.
        ned_transform.transform.rotation.x = 1.
        ned_transform.transform.rotation.y = 0.
        ned_transform.transform.rotation.z = 0.
        self._static_transform_broadcaster.sendTransform(ned_transform)
        self._vehicle_transform = TransformStamped()
        self._vehicle_transform.header.frame_id = ned_frame_name
        self._vehicle_transform.child_frame_id = vehicle_frame_name
        self._vehicle_transform.transform.rotation.w = 1.
        self._body_transform = TransformStamped()
        self._body_transform.header.frame_id = vehicle_frame_name
        self._body_transform.child_frame_id = body_frame_name
        self._truth_sub = rospy.Subscriber('truth/NED', Odometry, self._truth_cb)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

    def _truth_cb(self, truth: Odometry):
        time = rospy.Time.now()
        self._vehicle_transform.header.stamp = time
        self._vehicle_transform.transform.translation.x = truth.pose.pose.position.x
        self._vehicle_transform.transform.translation.y = truth.pose.pose.position.y
        self._vehicle_transform.transform.translation.z = truth.pose.pose.position.z
        self._body_transform.header.stamp = time
        self._body_transform.transform.rotation = truth.pose.pose.orientation
        self._transform_broadcaster.sendTransform(self._vehicle_transform)
        self._transform_broadcaster.sendTransform(self._body_transform)


if __name__ == '__main__':
    node = QuadsimTF2()
    node.run()

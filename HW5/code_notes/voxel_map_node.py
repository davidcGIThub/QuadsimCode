#!/usr/bin/env python3

# system imports
import numpy as np
import copy
import matplotlib.pyplot as plt

# ROS imports
import rospy
from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry

from utils.state import State
from voxel_map import VoxelMap

class VoxelMapNode:
    def __init__(self):
        ns = 'voxel_map/'
        self._visual_threshold = rospy.get_param(ns + 'visual_threshold')
        self.map = VoxelMap()
        rospy.Subscriber("truth/NED", Odometry, callback=self.state_cb, queue_size=1)
        rospy.Subscriber("triang_points", PointCloud, callback=self.measurement_cb)
        self.map_pub = rospy.Publisher("voxel_map", PointCloud, queue_size=1)
        self.vis_pub = rospy.Publisher("voxel_map_visual" , PointCloud, queue_size=1)
        self.state = State()
        self.blank_cloud = PointCloud()
        zero_point = Point32(0,0,0)
        self.blank_cloud.points = [zero_point]*self.map.N**3

    def run(self):
        rospy.spin()

    def state_cb(self, msg):
        position = msg.pose.pose.position
        shift = np.array([position.x - self.state.pos[0],
                          position.y - self.state.pos[1],
                          position.z - self.state.pos[2]])
        self.map.shift_map(shift)
        self.state.pos = np.array([position.x, position.y, position.z])

    def measurement_cb(self, msg):
        num_points = len(msg.points)
        if num_points > 0:
            measurements_in = np.zeros((num_points,3))
            # There may be a slicker way to fill up this array.
            for i in range(num_points):
                measurements_in[i,0] = msg.points[i].x - self.state.pos[0]
                measurements_in[i,1] = msg.points[i].y - self.state.pos[1]
                measurements_in[i,2] = msg.points[i].z - self.state.pos[2]
            self.map.update_map(measurements_in)
        else:
            self.map.update_map()
        # We may want to change this to where we only publsh at the voxel map at a set frequency 
        # instead of as soon as we get any updates to our measurements
        self.publish_map()

    def publish_map(self):
        out_cloud = copy.deepcopy(self.blank_cloud)
        out_cloud.header.stamp = rospy.Time.now()
        out_cloud.header.frame_id = "ned"
        N = self.map.N
        offset = self.map.offset
        voxel_width = self.map.voxel_width
        point_it = 0
        # TODO Get these map probabilities into a pointcloud without a loop
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    out_cloud.points[point_it] = (Point32(voxel_width*(i-offset)+self.state.pos[0],
                                                          voxel_width*(j-offset)+self.state.pos[1],
                                                          voxel_width*(k-offset)+self.state.pos[2]))
                    out_cloud.channels.append(ChannelFloat32())
                    out_cloud.channels[point_it].name = "occupancy probability"
                    out_cloud.channels[point_it].values.append(self.map.prob_map[i,j,k])
                    point_it += 1
        self.publish_visual(out_cloud)
        self.map_pub.publish(out_cloud)

    def publish_visual(self, probability_cloud):
        vis_cloud = PointCloud()
        vis_cloud.header.stamp = rospy.Time.now()
        vis_cloud.header.frame_id = "ned"
        num_points = len(probability_cloud.points)
        for i in range(num_points):
            probability = probability_cloud.channels[i].values[0]
            #print("probability: " , probability)
            if(probability > self._visual_threshold):
                vis_cloud.points.append(probability_cloud.points[i])
        self.vis_pub.publish(vis_cloud)

if __name__ == '__main__':
    rospy.init_node('voxel_map', anonymous=True)
    try:
        ros_node = VoxelMapNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass

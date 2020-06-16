#!/usr/bin/env python3
"""
Takes in voxel maps and current voxel position, and produces a smooth path.
See TODO's for what needs to be finished.
"""
import numpy as np

import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseArray, Pose

from quad_a_star.a_star import AStarSearch
from quad_a_star.path_smoother import PathSmoother

class AStarNode:
    def __init__(self):
        ns = "path_planner/"
        self.goal = np.array(rospy.get_param(ns + 'goal', np.array([0., 0., np.pi/2]))) # What params file will the goal come from?
        self.current_cell = np.array([0.0, 0.0, 0.0])
        self.step = 4 # TODO: From voxel map params?
        self.fov = 12 # TODO: From voxel map params?

        self.map_width = self.fov*2 + 1
        self.prob_map = np.ones((self.map_width, self.map_width, self.map_width))*0.01

        self.planner = AStarSearch(start=self.current_cell, goal=self.goal,
            prob_map=self.prob_map, fov=self.fov)

        rospy.Subscriber("status", Status, self.statusCallback, queue_size=1)
        self.counter = 0
        self.armed = False

        self.cell_sub = rospy.Subscriber("---------<TOPIC HERE>-----------", <---MESSAGE_TYPE--->, self.cellCallback) #TODO: Subscribe to the current cell position from voxel_map_node
        self.cloud_sub = rospy.Subscriber("voxel_map", PointCloud, self.mapCallback)
        self.wpt_pub = rospy.Publisher("waypoints", PoseArray, queue_size=1)
        self.wpt_msg = PoseArray()
        self.num_waypoints = 10 # TODO: From waypoints?

    def run(self):
        rospy.spin()

    def cellCallback(self, cell): #TODO: Implement from whatever topic voxel_map_node publishes
        self.current_cell = np.array([cell.x, cell.y, cell.z])

    def mapCallback(self, map):
        # self.step = int(np.abs(map.points[0].x - map.points[1].x)) #TODO: See if necessary or if voxel map node can provide
        # self.fov = (len(map.points) - 1) / 2 #TODO: See if necessary or if voxel map node can provide
        self.pointcloud_to_prob_map(map)
        self.plan()

    def plan(self):
        path = self.planner.planPath(self.current_cell, self.prob_map).tolist() # TODO: Do we need .tolist?
        while len(path) < self.num_waypoints:
            path.append(self.goal)
        smooth_path = self.path_smoother.update(current, path)
        self.wpt_msg.poses = self.path_to_poses(smooth_path)
        self.wpt_pub.publish(self.wpt_msg)

    def pointcloud_to_prob_map(self, map):
        # TODO: Check that the pointcloud points will be in meters away from origin
        for i in range(len(map.points)): # len(map.points) should equal self.map_width
            x = int(round(map.points[i].x / self.step)) + self.fov
            y = int(round(map.points[i].y / self.step)) + self.fov
            z = int(round(map.points[i].z / self.step)) + self.fov
            self.prob_map[x][y][z] = map.channels[i].values[0] # TODO: Verify that this works with voxel_map_node

    def path_to_poses(self, path):
        data = []
        for i in range(self.num_waypoints):
            pose_msg = Pose()
            pose_msg.position.x = path[i][0]
            pose_msg.position.y = path[i][1]
            pose_msg.position.z = path[i][2] # TODO: Make sure this doesn't need to be negated
            data.append(pose_msg)
        return data

    if __name__=="__main__":
        rospy.init_node('path_planner', anonymous=True)
        try:
            ros_node = AStarNode()
            ros_node.run()
        except rospy.ROSInterruptException:
            pass

"""
Test for 3d functionality.
"""
from a_star import *
import numpy as np

# Empty point_cloud
point_cloud = np.ones((9,9,9))*0.01

# Add obstacles
for x in range(len(point_cloud)):
    for y in range(len(point_cloud[x])):
        for z in range(len(point_cloud[x][y])):
            if y == 5 and z < 5:
                point_cloud[x][y][z] = 0.9

goal = np.array([1., 4., 1.])
start = np.array([0., 0., 0.])
step = 1
fov = 4
current_cell = np.array([0., 0., 0.])

a_star = AStarSearch(start=start,goal=goal,step=step,prob_map=point_cloud,fov=fov)

path = a_star.planPath(current_cell, point_cloud)

print(path)

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a_star import *
from path_smoother import PathSmoother
import numpy as np

from nav_msgs.msg import Odometry, OccupancyGrid

def updateMap(position, global_map, fov):
    """
    Updates the limited fov probability map from the global map.
    """
    map = np.zeros((2*fov + 1, 2*fov + 1))
    i = 0
    for i in range(fov*2 + 1):
        j = 0
        for j in range(fov*2 + 1):
            map[i][j] = global_map[i + int(position[0]/step) - fov][j + int(position[1]/step) - fov]
            j += 1
        i += 1
    map[fov][fov] = 0
    return map

class animation:

    def __init__(self, obstacles, fov, start, goal, N):
        self.flagInit = True
        self.fig, self.ax = plt.subplots()
        self.fov = fov
        self.handle = []
        self.obs_handle = []
        self.drawObstacles(obstacles)
        plt.axis([-N/2,N/2,-N/2,N/2])
        plt.ion()

    def drawObstacles(self,obstacles):

        for i in range(len(obstacles)):
            if self.flagInit:
                self.obs_handle.append(mpatches.Rectangle(xy=(obstacles[i][0],obstacles[i][1]),width=obstacles[i][2],height=obstacles[i][3],fill=True,fc='orange'))
                self.ax.add_patch(self.obs_handle[i])
            else:
                self.obs_handle[i].set_xy((obstacles[i][0],obstacles[i][1]))

    def drawEverything(self,point,path):
        self.drawObject(point)
        self.drawFrame(point)
        self.drawPath(path)
        plt.pause(0.001)

        if self.flagInit:
            self.flagInit = False

    def drawFrame(self,point):
        x = point[0] - self.fov
        y = point[1] - self.fov
        xy = (x,y)

        if self.flagInit:
            self.handle.append(mpatches.Rectangle(xy,2*self.fov,2*self.fov,fill=False,ec='black',lw = 1))
            self.ax.add_patch(self.handle[1])
        else:
            self.handle[1].set_xy(xy)

    def drawPath(self,path):
        x_list = []
        y_list = []

        for i in path:
            x_list.append(i[0])
            y_list.append(i[1])

        plt.scatter(x_list,y_list,s=10)

    def drawObject(self,point):
        xy = (point[0]-0.5,point[1]-0.5)

        if self.flagInit:
            self.handle.append(mpatches.Rectangle(xy,height = 1,width=1,fill=True,ec='blue'))
            self.ax.add_patch(self.handle[0])
        else:
            self.handle[0].set_xy(xy)

#start and goal will come from the current state which is a 1d np array of length 3
#a_star algorithm uses only the first two elements and converts to a list
#a_star converts back to np arrays of length 3 when it passes values back out
start = np.array([0,0,20])
goal = np.array([100,-120,20]) #altitude is left alone in the a_star class
step = 1 #how many meters per voxel
fov = 25 # In voxels
current_cell = np.array([0.0, 0.0]) # Use floats
diff = start[:2] - current_cell
while (np.abs(diff[0]) > (step)/2 or np.abs(diff[1]) > (step)/2):
    #Update current_cell with position of the center of the current voxel
    idx = np.argwhere(np.abs(diff) > (step)/2).T
    current_cell[idx] = current_cell[idx] + np.sign(diff[idx]) * (step)
    diff = start[:2] - current_cell

if (goal[0]-current_cell[0])%step != 0 or (goal[1]-current_cell[1])%step != 0:
    print("GOAL WILL NOT BE REACHED") #TODO: Put an error here
    #TODO: Adjust the goal like the start is adjusted, to be in a cell

print(current_cell)
N = 250
M = int(N/step)

prob_map = np.zeros((2*fov+1, 2*fov+1)) # The limited fov map, with spacing based on the step
global_prob_map = np.zeros((M,M)) # The global prob map, with spacing based on the step

# obstacles = [[3, 3, 2, 2]]

obstacles = [[20,20,30,30], #structure [obs1:[x,y,dx,dy],obs2:[x,y,dx,dy]]
             [60,60,10,10],
             [110,100,50,25],
             [180,180,25,25],
             [-60, -60, 100, 50],
             [50, -120, 40, 90],
             [30, -88, 20, 9]]

#fill out obstacles in prob_map
a_obstacles = []
for obs in obstacles:
    for i in range(obs[0],obs[0]+obs[2]+1):
        for j in range(obs[1],obs[1]+obs[3]+1):
            a_obstacles.append([i,j])
            if i%step == 0 and j%step == 0:
                global_prob_map[int(i/step)][int(j/step)] = 1

prob_map = updateMap(start, global_prob_map, fov)

pic = animation(obstacles=obstacles,fov=fov*step,start=start,goal=goal,N=N) #fov is in voxels

start2D = start[0:2]
goal2D = goal[0:2]
a_star = AStarSearch(start=start2D,goal=goal2D,step=step,prob_map=prob_map,fov=fov)
path_smoother = PathSmoother(max_pts = 2)

current = start

plt.scatter(goal[0], goal[1], marker='*', color='y')

while np.linalg.norm(goal-current) != 0:
    path = a_star.planPath(current_cell, prob_map) # The planner plans from the current voxel to the goal.
    # path = path_smoother.update(current,path)
    pic.drawEverything(current_cell,path)
    plt.pause(.1)
    current = np.array([path[1][0], path[1][1], start[2]]) # Move to the next position on the path, assumes constant altitude
    diff = current[:2] - current_cell # Distance between current position and center of current voxel
    while (np.abs(diff[0]) > (step)/2 or np.abs(diff[1]) > (step)/2):
        # Update current_cell with position of the center of the voxel the quad is currently in
        idx = np.argwhere(np.abs(diff) > (step)/2).T # Identify in which dimensions the quad has moved to be able to shift the current voxel
        current_cell[idx] = current_cell[idx] + np.sign(diff[idx]) * (step) # Shift the current voxel
        diff = current[:2] - current_cell # Update the difference and repeat until the voxel is correct
        print(current_cell)
    prob_map = updateMap(current_cell, global_prob_map, fov)

plt.show()

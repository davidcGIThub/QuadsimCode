#!/usr/bin/env python3
"""
A* implementation
Based in code by Amit Patel, found at:
    https://www.redblobgames.com/pathfinding/a-star/implementation.html
Adapted to 3D by Nathan Toombs
"""

import heapq
import numpy as np
from math import sqrt

class PriorityQueue:
    """
    Includes a list of items sorted by priority, lowest to highest.
    Wrapper for the heapq class.
    """
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class Graph:
    """
    The graph as known by the quadrotor. All obstacle locations can either be added
    all at once or as they come into the quadrotors field of view without degrading
    performance.
    """
    def __init__(self, current, prob_map=[], field_view=2, step=1):
        self.edges = {} # Edges of the field of view
        self.step = step
        self.prob_map = prob_map
        self.current = current # Current position of the quadrotor
        self.fov = field_view # Size of the field of view, in each direction from the quadrotor
        self.update_field_of_view()

    def neighbors(self,point):
        """
        Defines the neighbors adjacent to the current block, and doesn't add
        them if they are in the obstacles list.
        """
        #TODO: See if the aesthetic code works
        cutoff = 1 #TODO Implement this if necessary
        neighbors_list = []
        step = self.step
        fov = self.fov
        for dim in range(len(point)):
            temp = point.copy()
            temp[dim] += step
            p = self.g2p(temp)
            if len(point) == 2:
                if not self.prob_map[p[0], p[1]] >= cutoff:
                    neighbors_list.append(temp)
            elif len(point) == 3:
                if not self.prob_map[p[0], p[1], p[2]] >= cutoff:
                    neighbors_list.append(temp)
            temp = point.copy()
            temp[dim] -= step
            p = self.g2p(temp)
            if len(point) == 2:
                if not self.prob_map[p[0], p[1]] >= cutoff:
                    neighbors_list.append(temp)
            elif len(point) == 3:
                if not self.prob_map[p[0], p[1], p[2]] >= cutoff:
                    neighbors_list.append(temp)
        return neighbors_list

    def g2p(self, point):
        """
        Converts points in the global frame (meters) to the prob map (voxels/steps)
        """
        p = []
        for dim in range(len(point)):
            gc = self.current[dim]
            fov = self.fov
            g = point[dim]
            step = self.step
            p.append(int(g + -gc + fov + (gc-g)/step*(step-1)))
        return p

    def borders(self,point):
        """
        Determines if the current point is along one of the edges of the
        field of view.
        """
        diff0 = np.zeros(len(point))
        diff1 = np.zeros(len(point))
        for coord in range(len(point)):
            diff0[coord] = np.abs(point[coord]-self.edges[coord][0])
            diff1[coord] = np.abs(point[coord]-self.edges[coord][1])
            if diff0[coord] < self.step or diff1[coord] < self.step:
                return True

        return False

        # bool = []
        # diff0 = np.zeros(len(point))
        # diff1 = np.zeros(len(point))
        # for coord in range(len(point)):
        #     diff0[coord] = np.abs(point[coord]-self.edges[coord][0])
        #     diff1[coord] = np.abs(point[coord]-self.edges[coord][1])
        #     if diff0[coord] >= self.step and diff1[coord] >= self.step:
        #         bool.append(False)
        #     else:
        #         bool.append(True)
        # if any(bool) == True:
        #     return True
        #
        # return False

    def prob_cost(self, point):
        """
        Returns the probability of an obstacle existing at a point,
        used as a cost added to the priority of a point.
        """
        if len(point) == 2:
            p = self.g2p(point)
            return self.prob_map[p[0], p[1]]
        elif len(point) == 3:
            p = self.g2p(point)
            return self.prob_map[p[0], p[1], p[2]]

    def update_prob_map(self,prob_map):
        """
        Update self.prob_map with the current map.
        """
        self.prob_map = prob_map

    def update_field_of_view(self):
        """
        Puts the field of view around the current quadrotor position.
        The edges are placed one step in from the maximum view so the frontier
        is never expanded beyond the field of view.
        Field of view is in voxels, not meters.
        """
        for dim in range(len(self.current)):
            self.edges[dim] = [self.current[dim]-(self.fov*self.step-self.step),self.current[dim]+(self.fov*self.step-self.step)]

    def reset(self, pos, map): # Unused so far
        self.current = pos
        self.prob_map = map
        self.update_field_or_view()

class AStarSearch:
    """
    Conducts the search. The positions are in the global map, while the
    probability map is local centered on the current voxel.
    """
    def __init__(self, start, goal, prob_map, step=1, fov=2):
        self.cost_so_far = {} # A dictionary of the minimum cost to each point explored
        self.came_from = {} # A dictionary of the point directly before each point explored
        self.goal = goal.tolist() # Global
        self.start = start.tolist() # Global
        self.step = step # Distance between points in voxels
        self.current = start.tolist() # Self.current is always in global
        self.graph = Graph(current = self.current,prob_map=prob_map,field_view=fov,step=step)
        self.came_from[str(self.current)] = None # As lists cannot be keys in dictionaries,
        # we convert them to strings. Can be replaced with a better method.
        self.cost_so_far[str(self.current)] = 0.

    def planPath(self, current, new_prob_map):
        """
        Follows A* Implementation, using borders to stop searching. Each time it
        is called, it gets the current cell location and the probability map
        that accompanies it.
        """
        self.current = current.tolist()
        self.graph.current = self.current
        self.graph.update_field_of_view() # Get the field of view from the new position
        self.graph.update_prob_map(new_prob_map) # Get the current probability map for the new field of view
        frontier = PriorityQueue()
        frontier.put(self.current, 0)
        came_from = self.came_from # Remember the total path (not really important)
        cost_so_far = {} # Reset the cost_so_far
        # came_from[str(self.current)] = None  # Don't need to clear the current point's came_from
        cost_so_far[str(self.current)] = 0 # Current cost_so_far = 0
        # came_from[str(self.current)] = self.came_from[str(self.current)] # We don't need to remember the came_from
        # cost_so_far[str(self.current)] = self.cost_so_far[str(self.current)] # We don't need to remember the cost_so_far

        while not frontier.empty():
            current = frontier.get() # Get next node with the lowest priority score

            if current == self.goal or self.graph.borders(point=current):
                break

            # Step through the available neighbors of the node
            for next_point in self.graph.neighbors(point=current):
                # Calculate the cost of moving to the next point.
                # For quadrotors the cost of moving to each node is the same
                new_cost = cost_so_far[str(current)] + self.step*self.graph.prob_cost(current)*10 #TODO: Scale the probability appropriately
                if str(next_point) not in cost_so_far or new_cost < cost_so_far[str(next_point)]:
                    cost_so_far[str(next_point)] = new_cost
                    priority = new_cost + self.heuristic(next_point)
                    frontier.put(next_point, priority)
                    came_from[str(next_point)] = current
        path = self.reconstruct_path(came_from,self.current,current)

        # self.cost_so_far[str(self.current)] = cost_so_far[str(self.current)]
        # self.came_from[str(self.current)] = came_from[str(self.current)]
        self.cost_so_far = cost_so_far
        self.came_from = came_from
        self.graph.update_field_of_view()

        #pass a np array of Mx2 to path smoother
        return np.array(path)

    def heuristic(self,point):
        """
        Finds the cost of getting from a point to the goal.
        """
        distance = 0.
        added = 0.
        for i in range(len(point)):
            for i in range(len(point)):
                distance += abs(self.goal[i]-point[i])
        # Use actual distance, with a modifier.
        for i in range(len(point)):
            added += (self.goal[i]-point[i])**2
        distance += added/self.graph.fov
        return distance #TODO: Get a better scaling (added) value

    def reconstruct_path(self,came_from,start,goal):
        """
        Rebuilds the path from the goal to the start.
        """
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[str(current)]
        path.append(start)
        path.reverse()
        return path

    def get_results(self):
        """
        Used to reconstruct the final path after the goal has been reached
        """
        path = self.reconstruct_path(self.came_from,self.start,self.goal)
        return path

    def reset(self, pos, goal, map):
        self.start = pos
        self.goal = goal
        self.graph.reset(pos, map)

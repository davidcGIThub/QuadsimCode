#!/usr/bin/env python3
import numpy as np

class PathSmoother:
    def __init__(self, max_pts=2):
        #this is just a place holder
        self.max_pts = max_pts

    def update(self, current, sharp_path): #Should we pass in the obstacle map to not over smooth
        smooth_path = [current] #sharp_path #just a place holder

        for i in range(len(sharp_path[:-1])):
            pt_i = sharp_path[i]
            pt_ip1 = sharp_path[i+1]
            avg = ((np.array(pt_i) + np.array(pt_ip1))/2).tolist()
            smooth_path.append(avg)

        #the smooth path returned will be used by min snap trajectory
        #numpy array Mx2
        return(smooth_path)

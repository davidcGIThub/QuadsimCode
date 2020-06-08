# system imports
import numpy as np


def skew(vec3):
    '''Create skew-symmetric matrix of vec3 (1d numpy array of length 3)'''
    return np.array([[0., -vec3[2], vec3[1]],
                     [vec3[2], 0., -vec3[0]],
                     [-vec3[1], vec3[0], 0.]])

def vee(mat3):
    """Create vec3 from skew-symmetric matrix mat3"""
    return np.array([mat3[2,1], mat3[0,2], mat3[1,0]])

def norm2(vec):
    '''Takes the 2-norm of a vector (1d numpy.array)'''
    return np.sqrt(vec @ vec)

def ei(i, n=3):
    """Return elementary unit vector of length n along axis i"""
    return np.array([i==j for j in range(n)], dtype=float)

def logm(R):
    temp = (np.arccos((np.trace(R)-1)/2.0)/np.sqrt((3-np.trace(R))*(1+np.trace(R))))*(R-R.T)
    if np.any(np.isnan(temp)):
        temp = np.zeros((3,3))
    return temp
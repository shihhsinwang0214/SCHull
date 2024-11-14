import math

import numpy as np


# Coordinate Transforms
# -----------------------

def cartesian_to_xspherical(x, y, z):
  " Return spherical coords from x-axis"
  r = np.sqrt(x**2 + y**2 + z**2)
  theta = np.arccos(x/r)
  phi = np.arctan2(z, y)
  return r, theta, phi

def cartesian_to_yspherical(x, y, z):
  " Return spherical coords from y-axis"
  r = np.sqrt(x**2 + y**2 + z**2)
  theta = np.arccos(y/r)
  phi = np.arctan2(z, x)
  return r, theta, phi

def cartesian_to_zspherical(x, y, z):
  " Return spherical coords from z-axis"
  r = np.sqrt(x**2 + y**2 + z**2)
  theta = math.atan2(math.sqrt(x * x + y * y), z)
  phi = math.atan2(y, x)
  return r, theta, phi



def planar_normal(v0, v1):
    return np.cross(v0, v1)/np.linalg.norm(np.cross(v0, v1))

def project_onto_plane(vectors, plane_normal):
    return vectors - np.dot(vectors, plane_normal)[:,np.newaxis]*plane_normal

def angle_between_vectors(v0, v1):
    temp = np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1))
    if temp < -1:
       temp = -1
    elif temp > 1:
       temp = 1
    val = np.arccos(temp)
    # if np.isnan(val):
    #    print(temp)
    #    print(val)
    return val

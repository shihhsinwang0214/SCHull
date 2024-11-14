# Alignment Utilities
# =====================

import math

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import cartesian_to_zspherical, cartesian_to_xspherical

# Vector Alignment
# -----------------


def xz_planar_alignment(positions, align_vec):
  " Align vector into xy-plane"
  r,theta,phi = cartesian_to_xspherical(*align_vec)
  Q = Rotation.from_euler('z',[-theta],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def xy_planar_alignment(positions, align_vec):
  " Align vector into xy-plane"
  r,theta,phi = cartesian_to_xspherical(*align_vec)
  Q = Rotation.from_euler('x',[phi],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def z_axis_alignment(positions, align_vec):
  " Align vector with z-axis"
  r,theta,phi = cartesian_to_zspherical(*align_vec)
  Qz = Rotation.from_euler('z',[-phi],degrees=False).as_matrix().squeeze()
  Qy = Rotation.from_euler('y',[-theta],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Qy@(Qz@pos)
  return positions, Qy@Qz

def zy_planar_alignment(positions, align_vec):
  " Align vector into zy-plane"
  r,theta,phi = cartesian_to_zspherical(*align_vec)
  Q = Rotation.from_euler('z',[-phi],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def sign_alignment(positions, align_vec):
  " Align vector to positive x-direction"
  val = -1 if align_vec[0]<0 else 1
  positions[:,1] = -positions[:,1]
  return positions, val
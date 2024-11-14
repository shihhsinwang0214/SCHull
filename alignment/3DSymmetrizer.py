from typing import Dict
from torch import Tensor

from abc import ABCMeta

import torch

class Symmetrizer3D(metaclass=ABCMeta):
  def __call__(self,pointcloud):
    self.reduce_symmetry(pointcloud)
    pass

  def reduce_rank0(self,
     pos : Tensor,
      symmetries : list = ['E','Ci','Cs','Cn','Sn','Cnv','Cnh','Dn','Dnh','C-v','D-h','Td','Th','Oh','Ih','Kh'],
      radial_axes : list = [],
      planar_normals : list = [],
      tol : float = 1e-2
    ):
    keep_axes, keep_normals = [], []
    # Radial
    # ~~~~~~
    for axes in radial_axes:
      inner = torch.inner(axes,pos[0]).abs()
      print(f'<A, A>: {inner}')
      if (1-tol<inner):
        print(f'Kept axes {axes}')
        keep_axes = keep_axes + [axes]
      # else:
      #   return [], []
    # Plane-H/V
    # ~~~~~~~~~
    for normal in planar_normals:
      inner = torch.inner(normal,pos[0]).abs()
      print(f'<A,N>: {inner}')
      if (inner<tol):
        print(f'Kept normal {normal}')
        keep_normals = keep_normals + [normal]
    # Return
    # ~~~~~~
    if len(keep_axes)==0:
        return [pos[0]], keep_normals
    else:
      return keep_axes, keep_normals

  def reduce_spherical_rank1(self,
      pos : Tensor,
      symmetries : list = ['E','Ci','Cs','Cn','Sn','Cnv','Cnh','Dn','Dnh','C-v','D-h','Td','Th','Oh','Ih','Kh'],
      radial_axes : list = [],
      planar_normals : list = [],
      tol : float = 1e-2
    ):
    keep_axes, keep_normals = [], []
    # Plane-i
    # ~~~~~~~
    if len(planar_normals)==0:
      keep_normals = keep_normals + [(pos[0]-pos[1])/(pos[0]-pos[1]).norm()]
    for normal in planar_normals:
      inner = torch.inner(normal,pos[0]).abs()
      if (1-tol<inner):
        print(f'<N,N>: {inner}')
        keep_normals = keep_normals + [normal]
    # Return
    # ~~~~~~
    if len(keep_axes)==0:
        return [pos[0]], keep_normals
    else:
      return keep_axes, keep_normals

  def reduce_origin_rank1(self,
      pos : Tensor,
      symmetries : list = ['E','Ci','Cs','Cn','Sn','Cnv','Cnh','Dn','Dnh','C-v','D-h','Td','Th','Oh','Ih','Kh'],
      radial_axes : list = [],
      planar_normals : list = [],
      tol : float = 1e-2
    ):
    # Rotations / Plane-H / V
    # ~~~~~~~~~~~~~~~~~~~~~~~
    keep_axes, keep_normals = self.reduce_rank0(pos, symmetries, radial_axes, planar_normals)
    print('\t Rank 1 -> Rank 0', keep_axes, keep_normals)
    # Plane-i
    # ~~~~~~~
    if len(planar_normals)==0:
      keep_normals = keep_normals + [pos[0]]
    for normal in planar_normals:
      inner = torch.inner(normal,pos[0]).abs()
      if (1-tol<inner):
        print(f'<N,N>: {inner}')
        keep_normals = keep_normals + [normal]
    # Return
    # ~~~~~~
    if len(keep_axes)==0:
        return [pos[0]], keep_normals
    else:
      return keep_axes, keep_normals

  def reduce_rank2(self,
      pos : Tensor,
      symmetries : list = ['E','Ci','Cs','Cn','Sn','Cnv','Cnh','Dn','Dnh','C-v','D-h','Td','Th','Oh','Ih','Kh'],
      radial_axes : list = [],
      planar_normals : list = [],
      tol : float = 1e-2
    ):
    # Rotations / Plane-H / V
    # ~~~~~~~~~~~~~~~~~~~~~~~
    keep_axes, keep_normals = [], []
    for i,p1 in enumerate(pos):
      # keep_axes_, keep_normals_ = reduce_spherical_rank1(p1.unsqueeze(0), symmetries, radial_axes, planar_normals)
      keep_axes_, keep_normals_ = self.reduce_rank0(p1.unsqueeze(0), symmetries, radial_axes, planar_normals)
      keep_axes = keep_axes + keep_axes_
      keep_normals = keep_normals + keep_normals_
      print('\t Rank 2 -> Rank 0', keep_axes_, keep_normals_, p1)
      for j,p2 in enumerate(pos):
        print(i,j)
        if j<i:
          p = torch.concat((p1.unsqueeze(0),p2.unsqueeze(0)),dim=0)
          keep_axes_, keep_normals_ = self.reduce_spherical_rank1(p, symmetries, radial_axes, planar_normals)
          print('\t Rank 2 -> Rank 1', keep_axes_, keep_normals_)
          keep_axes = keep_axes + keep_axes_
          keep_normals = keep_normals + keep_normals_

    # Plane-i
    # ~~~~~~~
    # if len(planar_normals)==0:
    #   keep_normals = keep_normals + [pos[0]]
    # for normal in planar_normals:
    #   inner = torch.inner(normal,pos[0]).abs()
    #   if (1-tol<inner):
    #     print(f'<N,N>: {inner}')
    #     keep_normals = keep_normals + [normal]
    # Return
    # ~~~~~~
    if len(keep_axes)==0:
        return [pos[0]], keep_normals
    else:
      return keep_axes, keep_normals


  def reduce_symmetry(self,
    pos : Tensor,
    symmetries : list = ['E','Ci','Cs','Cn','Sn','Cnv','Cnh','Dn','Dnh','C-v','D-h','Td','Th','Oh','Ih','Kh'],
    radial_axes : list = [],
    planar_normals : list = [],
    ):
    n = pos.shape[0]
    pos = torch.concat((pos,torch.zeros(1,pos.shape[1])))
    rank = torch.linalg.matrix_rank(pos,atol=1e-1,rtol=1e-1)
    pos = (pos[:-1].T/pos[:-1].norm(dim=1)).T

    if n==1 and rank!=0: # -- no need to reduce symmetries
      print(pos)
      radial_axes, planar_normals =  self.reduce_rank0(pos=pos, symmetries=symmetries, radial_axes=radial_axes, planar_normals=planar_normals)
    elif rank==1: # A line through the origin
      radial_axes, planar_normals =  self.reduce_origin_rank1(pos=pos, symmetries=symmetries, radial_axes=radial_axes, planar_normals=planar_normals)
    elif rank==2:# Two planes for all pairs, an axis and plane for all individuals AND an axis normal to the plane in which they all lie.
      radial_axes, planar_normals =  self.reduce_rank2(pos=pos, symmetries=symmetries, radial_axes=radial_axes, planar_normals=planar_normals)
      # normal1 = get_plane(pos[0],pos[1])
      # midpoint = (pos[0]+pos[1])/2
      # normal2 = get_plane(midpoint/midpoint.norm(),normal1)
      # planar_normals = [normal1, normal2]
    print(f'Rank ({rank}) | N {n} | Axes {radial_axes} | Normals {planar_normals}')
    return rank, symmetries, radial_axes, planar_normals
    

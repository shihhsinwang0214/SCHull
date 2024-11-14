from abc import ABCMeta

import torch
from torch import Tensor

from .visualizer import Visualizer, plot_axes, plot_3d_pointcloud, plot_shell, plot_point, plot_plane
from ..align.PointCloudAlignment import PointCloudAlignment


class PointCloudVisualizer(metaclass=ABCMeta):

  def __init__(self):
    self.align = PointCloudAlignment()

  def __call__(self, pointcloud: Tensor):
    self.pointcloud = pointcloud

    self.center_of_mass()
    self.shells()
    self.symmetries()


  def center_of_mass(self):
    vis = Visualizer(1,2,self.pointcloud.max(),figsize=(6,3))
    vis.fig.suptitle('Center of Mass')
    ax = vis.axes[0]
    ax = plot_3d_pointcloud(ax,self.pointcloud)
    ax = plot_axes(ax)

    self.pointcloud = self.align.align_center(self.pointcloud)

    ax = vis.axes[1]
    ax = plot_3d_pointcloud(ax,self.pointcloud)
    ax = plot_axes(ax)
    vis()

  def shells(self):
    self.shell_dict = self.align.separate_shells(self.pointcloud)
    vis = Visualizer(height=1,width=1,limits=self.pointcloud.norm(dim=1).mean(), figsize=(6,3))
    vis.fig.suptitle('Shells')
    ax = vis.axes[0]
    colors = {0:'snow',
              1: 'gainsboro',
              2:'lightgrey',
              3:'silver',
              4:'grey',
              5:'darkgrey',
              6:'black',
              }
    plot_axes(ax)
    plot_3d_pointcloud(ax, self.pointcloud)
    for i,(k,v) in enumerate(self.shell_dict.items()):
      if i>len(colors):
          break
      l = [j for j in range(len(v)) if v[i]]
      plot_shell(ax, self.pointcloud, l, k, color=colors[i])
    vis()
    return self.shell_dict

  def symmetries(self):
    vis = Visualizer(1,2,self.pointcloud.max(),figsize=(6,3))
    vis.fig.suptitle('Symmetries')
    ax = vis.axes[0]
    ax = plot_3d_pointcloud(ax,self.pointcloud)
    ax = plot_axes(ax)
    rank, symmetries, radial_axes, planar_normals = self.align.symm(self.pointcloud)
    print(radial_axes, planar_normals)
    for radial in radial_axes:
      plot_point(ax,radial,color='cyan',marker='o')
    for plane in planar_normals:
      plot_plane(ax,plane,alpha=0.5,color='cyan')
    vis()


from abc import ABCMeta

import torch
from torch import Tensor

from .visualizer import Visualizer, plot_axes, plot_3d_pointcloud, plot_point
from ..align.SVDAlignment import SVDAlignment


class SVDVisualizer(metaclass=ABCMeta):

  def __init__(self):
    self.align = SVDAlignment()

  def __call__(self, pointcloud: Tensor):
    self.pointcloud = pointcloud

    self.center_of_mass()
    self.svd_rotate()


  def center_of_mass(self):
    vis = Visualizer(1,2,self.pointcloud.max(),figsize=(6,3))
    vis.fig.suptitle('Center of Mass')
    ax = vis.axes[0]
    plot_3d_pointcloud(ax,self.pointcloud)
    plot_axes(ax)

    self.pointcloud = self.align.align_center(self.pointcloud)

    ax = vis.axes[1]
    plot_3d_pointcloud(ax,self.pointcloud)
    plot_axes(ax)
    vis()

  def svd_rotate(self):
    vis = Visualizer(1,2,self.pointcloud.max(),figsize=(6,3))
    vis.fig.suptitle('SVD Rotate')
    ax = vis.axes[0]
    plot_3d_pointcloud(ax,self.pointcloud)
    plot_axes(ax)
    e,v = self.align.get_eigs(self.pointcloud)
    for v_ in v:
      plot_point(ax,v_/v_.norm())

    self.pointcloud = self.align.svd_rotate(self.pointcloud)

    ax = vis.axes[1]
    plot_3d_pointcloud(ax,self.pointcloud)
    plot_axes(ax)
    e,v = self.align.get_eigs(self.pointcloud)
    for v_ in v:
      plot_point(ax,v_/v_.norm())
    vis()

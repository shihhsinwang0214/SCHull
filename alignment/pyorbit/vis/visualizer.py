from abc import ABCMeta

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

def plot_axes(ax, arrow_length=0.5, color='k'):
  ax.quiver(0, 0, 0, arrow_length, 0, 0, color=color)
  ax.quiver(0, 0, 0, 0, arrow_length, 0, color=color)
  ax.quiver(0, 0, 0, 0, 0, arrow_length, color=color)
  return ax

def plot_point(ax, v, color='r', marker='o'):
  return ax.scatter(v[0], v[1], v[2], color=color, marker=marker)

def plot_plane(ax, normal, alpha=0.5, color='g', extent=1):
  if normal[2]>1e-2:
    xx, yy = np.meshgrid(np.linspace(-extent, extent, 100), np.linspace(-extent, extent, 100))
    zz = (-normal[0] * xx - normal[1] * yy)/normal[2]
  else:
    if normal[0]>normal[1]:
      zz, yy = np.meshgrid(np.linspace(-extent, extent, 100), np.linspace(-extent, extent, 100))
      xx = (-normal[2] * zz - normal[1] * yy)/normal[0]
    else:
      zz, xx = np.meshgrid(np.linspace(-extent, extent, 100), np.linspace(-extent, extent, 100))
      yy = (-normal[2] * zz - normal[0] * xx)/normal[1]

  return ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

def plot_sphere(ax, r=1, color='b', alpha=0.1):
  u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
  x = r * np.sin(v) * np.cos(u) ; y = r * np.sin(v) * np.sin(u) ; z = r * np.cos(v)
  ax.plot_surface(x, y, z, color=color, alpha=alpha)
  return ax

def plot_3d_pointcloud(ax, pointcloud, alpha=1.0, color='k'):
  for p in pointcloud:
    ax.scatter(p[0], p[1], p[2], s=100, alpha=alpha, color=color)  # Adjust s for size of atoms
  return ax

def plot_3d_polyhedron(ax, pointcloud, edge_index, alpha=1.0, color='b'):
    if len(edge_index)>0:
        lines = [(pointcloud[start,:], pointcloud[end,:]) for start, end in edge_index]
        lc = Line3DCollection(lines, colors=color, linewidths=2, alpha=alpha)
        ax.add_collection3d(lc)
    return ax

def plot_mol(ax, mol, alpha=1.0):
  COLORS = {
      1: 'k',        # Black
      2: 'r',        # Red
      3: 'b',        # Blue
      4: 'y',        # Yellow
      5: 'g',        # Green
      6: 'darkred',  # Dark Red
      7: 'darkblue', # Dark Blue
      8: 'darkgreen',# Dark Green
      9: 'purple',   # Purple
      10: 'orange'   # Orange
  }
  for z,p in zip(mol.z,mol.pos):
      ax.scatter(p[0], p[1], p[2], s=100, alpha=alpha, color=COLORS[z.item()])  # Adjust s for size of atoms
  return ax

def plot_shell(ax, pointcloud, idx, radius, color='r', marker='X'):
  for p in pointcloud[idx]:
    plot_point(ax, p,color=color,marker=marker)
  return plot_sphere(ax, r=radius,color=color)

def plot_symmetries(ax, radial_axes, planar_normals, alpha=0.5, color='r', marker='X'):
  for radial in radial_axes:
    plot_point(ax,radial,color=color,marker=marker)
  for plane in planar_normals:
    plot_plane(ax,plane,alpha=alpha,color=color)
  return ax

class Visualizer(metaclass=ABCMeta):
  def __init__(self, height, width, limits, figsize=(4,3), elev=90, azim=30, name='visualizer'):
    self.name = name
    self.fig = plt.figure(figsize=figsize, tight_layout=True)
    self.axes = []
    for i in range(height):
      for j in range(width):
          ax = self.fig.add_subplot(height, width, i*width+j+1, projection='3d')
          ax.set_xlim(-limits,limits); ax.set_ylim(-limits,limits); ax.set_zlim(-limits,limits)
          ax.view_init(elev=elev, azim=azim)
          ax.axis('off')
          self.axes.append(ax)
  def __call__(self):
    plt.savefig(f'/root/workspace/out/{self.name}.pdf',format='pdf',bbox_inches='tight')
    plt.show()


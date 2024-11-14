from abc import ABCMeta

import numpy as np
import torch
from torch_geometric.data import Data

from .visualizer import Visualizer, plot_axes, plot_mol, plot_shell, plot_3d_pointcloud, plot_3d_polyhedron, plot_point, plot_plane
from pyorbit.alignment.MolecularAlignment import MolecularAlignment, NewAlignment


class MoleculeVisualizer(metaclass=ABCMeta):

  def __init__(self):
    self.align = MolecularAlignment()
    self.atom_shells = None

  def __call__(self, molecule: Data):
    self.mol = molecule

    self.center_of_mass()
    self.atomic_ordering()
    self.shells()
    self.symmetries()


  def center_of_mass(self):
    vis = Visualizer(1,2,self.mol.pos.max(),figsize=(6,3))
    vis.fig.suptitle('Center of Mass')
    ax = vis.axes[0]
    ax.set_title(self.mol.smiles)
    ax = plot_mol(ax,self.mol)
    ax = plot_axes(ax)

    self.mol.pos = self.align.align_center(self.mol.pos)

    ax = vis.axes[1]
    ax.set_title(self.mol.smiles)
    ax = plot_mol(ax,self.mol)
    ax = plot_axes(ax)
    vis()


  def atomic_ordering(self):
    K = torch.unique(self.mol.z).shape[0]
    vis = Visualizer(height=1,width=K,limits=self.mol.pos.norm(dim=1).max(), figsize=(K+2,3))
    vis.fig.suptitle('Atomic Ordering')


    self.atom_mols = self.align.separate_atoms(self.mol)

    for i,mol in enumerate(self.atom_mols):
      ax = vis.axes[i]
      ax.set_title(mol.smiles)
      ax = plot_mol(ax,mol)
      ax = plot_axes(ax)
    vis()

  def shells(self):
    self.atom_shells = self.align.separate_shells(self.atom_mols)
    K = len(self.atom_mols)
    vis = Visualizer(height=1,width=K,limits=self.mol.pos.norm(dim=1).mean(), figsize=(K+2,3))
    vis.fig.suptitle('Shells')
    colors = {0:'snow',
              1: 'gainsboro',
              2:'lightgrey',
              3:'silver',
              4:'grey',
              5:'darkgrey',
              6:'black',
              }
    for i,(k,v) in enumerate(self.atom_shells.items()):
      ax = vis.axes[i]
      ax.set_title(f'{k}')
      plot_mol(ax, self.atom_mols[i])
      plot_axes(ax)
      for j,(radius,blist) in enumerate(v.items()):
        l = [i for i in range(len(blist)) if blist[i]]
        plot_shell(ax, self.atom_mols[i].pos, l, radius, color=colors[j])
    vis()
    return self.atom_shells

  def shells(self):
    self.atom_shells = self.align.separate_shells(self.atom_mols)
    K = len(self.atom_mols)
    vis = Visualizer(height=1,width=K,limits=self.mol.pos.norm(dim=1).mean(), figsize=(K+2,3))
    vis.fig.suptitle('Shells')
    colors = {0:'snow',
              1: 'gainsboro',
              2:'lightgrey',
              3:'silver',
              4:'grey',
              5:'darkgrey',
              6:'black',
              }
    for i,(k,v) in enumerate(self.atom_shells.items()):
      ax = vis.axes[i]
      ax.set_title(f'{k}')
      plot_mol(ax, self.atom_mols[i])
      plot_axes(ax)
      for j,(radius,blist) in enumerate(v.items()):
        l = [i for i in range(len(blist)) if blist[i]]
        plot_shell(ax, self.atom_mols[i].pos, l, radius, color=colors[j])
    vis()
    return self.atom_shells

  def symmetries(self):
    if self.atom_shells is None:
      self.atom_shells = self.align.separate_shells(self.atom_mols)
    K = len(self.atom_mols)
    vis = Visualizer(height=1,width=K,limits=self.mol.pos.norm(dim=1).mean(), figsize=(K+2,3))
    vis.fig.suptitle('Symmetries')
    for i,(k,v) in enumerate(self.atom_shells.items()):
      ax = vis.axes[i]
      plot_mol(ax, self.atom_mols[i])
      plot_axes(ax)
      rank, symmetries, radial_axes, planar_normals = self.align.pca.symm(self.atom_mols[i].pos)
      for radial in radial_axes:
        plot_point(ax,radial,color='cyan',marker='o')
      for plane in planar_normals:
        plot_plane(ax,plane,alpha=0.5,color='cyan')
    vis()
    return self.atom_shells


class NewVisualizer(metaclass=ABCMeta):

  def __init__(self):
    self.align = NewAlignment()
    self.atom_shells = None

  def __call__(self, molecule: Data):
    self.mol = molecule

    self.center_of_mass()
    self.atomic_ordering()
    self.shells()
    self.polyhedra()
    # self.symmetries()


  def center_of_mass(self):
    vis = Visualizer(1,2,self.mol.pos.max(),figsize=(6,3))
    vis.fig.suptitle('Center of Mass')
    ax = vis.axes[0]
    ax.set_title(self.mol.smiles)
    ax = plot_mol(ax,self.mol)
    ax = plot_axes(ax)
    self.mol.pos = self.mol.pos.numpy()

    self.mol.pos = self.align.align_center(self.mol.pos)

    ax = vis.axes[1]
    ax.set_title(self.mol.smiles)
    ax = plot_mol(ax,self.mol)
    ax = plot_axes(ax)
    vis()


  def atomic_ordering(self):
    K = torch.unique(self.mol.z).shape[0]
    vis = Visualizer(height=1,width=K,limits=np.linalg.norm(self.mol.pos, axis=1).max(), figsize=(K+2,3))
    vis.fig.suptitle('Atomic Ordering')


    self.atom_mols = self.align.separate_atoms(self.mol)

    for i,mol in enumerate(self.atom_mols):
      ax = vis.axes[i]
      ax.set_title(mol.smiles)
      ax = plot_mol(ax,mol)
      ax = plot_axes(ax)
    vis()

  def shells(self):
    self.atom_shells = self.align.separate_shells(self.atom_mols)
    K = len(self.atom_mols)
    vis = Visualizer(height=1,width=K,limits=np.linalg.norm(self.mol.pos,axis=1).mean(), figsize=(K+2,3))
    vis.fig.suptitle('Shells')
    colors = {0:'snow',
              1: 'gainsboro',
              2:'lightgrey',
              3:'silver',
              4:'grey',
              5:'darkgrey',
              6:'black',
              }
    for i,(k,v) in enumerate(self.atom_shells.items()):
      ax = vis.axes[i]
      ax.set_title(f'{k}')
      plot_mol(ax, self.atom_mols[i])
      plot_axes(ax)
      for j,(radius,blist) in enumerate(v.items()):
        l = [i for i in range(len(blist)) if blist[i]]
        plot_shell(ax, self.atom_mols[i].pos, l, radius, color=colors[j])
    vis()
    return self.atom_shells

  def polyhedra(self):
    K = len(self.atom_mols)
    vis = Visualizer(height=1,width=K,limits=np.linalg.norm(self.mol.pos,axis=1).mean(), elev=45, figsize=(K+2,3))
    vis.fig.suptitle('Polyhedra')
    shell_colors = {0:'snow',
              1: 'gainsboro',
              2:'lightgrey',
              3:'silver',
              4:'grey',
              5:'darkgrey',
              6:'black',
              }
    poly_colors = {0:'lightblue',
              1: 'blue',
              2:'darkblue',
              3:'blue',
              4:'blue',
              5:'blue',
              6:'blue',
              }
    for i,(k,v) in enumerate(self.atom_shells.items()):
      ax = vis.axes[i]
      ax.set_title(f'{k}')
      plot_mol(ax, self.atom_mols[i])
      plot_axes(ax)
      for j,(radius,blist) in enumerate(v.items()):
        l = [i for i in range(len(blist)) if blist[i]]
        shell_data = self.atom_mols[i].pos[l]
        shell_zero_data = np.concatenate([shell_data, np.zeros((1,3))], axis=0)
        n = len(shell_data)
        rank = self.align.frame.get_ranks(shell_data)
        edge_index = self.align.frame.conv.get_chull_graph(shell_zero_data, rank, n)
        plot_3d_polyhedron(ax, shell_zero_data, edge_index, color=poly_colors[j])
        plot_shell(ax, self.atom_mols[i].pos, l, radius, color=shell_colors[j])
    vis()
    return self.atom_shells

  def symmetries(self):
    if self.atom_shells is None:
      self.atom_shells = self.align.separate_shells(self.atom_mols)
    K = len(self.atom_mols)
    vis = Visualizer(height=1,width=K,limits=np.linalg.norm(self.mol.pos,axis=1).mean(), figsize=(K+2,3))
    vis.fig.suptitle('Symmetries')
    for i,(k,v) in enumerate(self.atom_shells.items()):
      # REDO THIS
      rank, symmetries, radial_axes, planar_normals = self.align.frame.symm(self.atom_mols[i].pos)
      for radial in radial_axes:
        plot_point(ax,radial,color='cyan',marker='o')
      for plane in planar_normals:
        plot_plane(ax,plane,alpha=0.5,color='cyan')
    vis()
    return self.atom_shells

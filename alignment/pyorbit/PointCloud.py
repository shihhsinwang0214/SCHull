from abc import ABCMeta
import ast

import torch
import numpy as np

from Alignment import *
from Geometry import angle_between_vectors, planar_normal, project_onto_plane
from Hopcroft import PartitionRefinement
from Qhull import Qhull

class Frame(metaclass=ABCMeta):
    def __init__(self, tol=1e-2, *args, **kwargs):
        super().__init__()
        self.tol = tol
        self.chull = Qhull()

    def project_sphere(self, data, *args, **kwargs):
        distances = np.linalg.norm(data, axis=1, keepdims=False)
        temp =  data/np.linalg.norm(data, axis=1, keepdims=True)
        arr, key = np.unique(temp, axis=0, return_inverse=True)
        encoding = {}
        dists_hash = {}
        for val in set(key):
            dists = [custom_round(v,self.tol) for v in distances[key==val]]
            dists = tuple(sorted(dists))
            if dists not in dists_hash:
                dists_hash[dists] = id(dists)

            encoding[val] = dists_hash[dists]
        return dists_hash, encoding, arr


    def get_frame(self, data, *args, **kwargs):

        data = self.check_type(data) # Assert Type
        data = self.align_center(data) # Assert Centered
        data = data[np.linalg.norm(data, axis=1) > self.tol]

        # PROJECT ONTO SPHERE
        dist_hash, r_encoding, shell_data = self.project_sphere(data, *args, **kwargs)

        # GET CONVEX HULL
        shell_rank = np.linalg.matrix_rank(shell_data, tol=self.tol)
        shell_n = shell_data.shape[0]
        shell_graph = self.chull.get_chull_graph(shell_data, shell_rank, shell_n)
        
        # CHECK POLY HULL IS CORRECT
        bool_lst = [i in shell_graph for i in range(shell_n)]
        if not all(bool_lst):
            false_values = [i for i, x in enumerate(bool_lst) if not x]
            shell_data = np.delete(shell_data, false_values, axis=0)
            # PROJECT ONTO SPHERE
            dist_hash, r_encoding, shell_data = self.project_sphere(shell_data, *args, **kwargs)

            # GET CONVEX HULL
            shell_rank = np.linalg.matrix_rank(shell_data, tol=self.tol)
            shell_n = shell_data.shape[0]
            shell_graph = self.chull.get_chull_graph(shell_data, shell_rank, shell_n)
        
        bool_lst = [i in shell_graph for i in range(shell_n)]
        assert all(bool_lst), 'Convex Hull is not correct'

        # GET GEOMETRIC ENCODING
        adj_list = build_adjacency_list(shell_graph)
        dg = direct_graph(shell_graph)
        g_hash, g_encoding = self.geometric_encoding(shell_data, adj_list, shell_rank)

        # COMBINE ENCODINGS
        n_encoding = {}
        # for each node combine ENCODINGS
        for i in range(shell_n):
            n_encoding[i] = (r_encoding[i], g_encoding[i])

        # CONSTRUCT DFA
        dfa, edge_encoding = self.construct_dfa(n_encoding, dg)
        self.hopcroft = PartitionRefinement(dfa)
        out = self.hopcroft.refine(dfa)
        k = list(self.hopcroft._partition.keys())[0]
        for value in self.hopcroft._partition[k]:
            e = [edge for edge, enc in edge_encoding.items() if enc == value]

        # print(self.hopcroft._partition)
        sorted_edges, sorted_graph = self.convert_partition(dist_hash, g_hash, r_encoding, g_encoding)
        pth = self.traverse(sorted_graph, shell_data, shell_rank)
        return self.align(data, shell_data, pth)

    def align(self, data, shell_data, pth):
        funcs = {0: z_axis_alignment, 1: zy_planar_alignment, 2: sign_alignment}
        rot_stor = []
        frame = np.eye(3)
        for idx,val in enumerate(pth):
            data, rot = funcs[idx](data, shell_data[val])
            if rot.__class__ == np.ndarray:
                shell_data = shell_data @ rot
                frame = frame @ rot
            else:
                shell_data = shell_data*rot
                frame = frame*rot
            rot_stor.append(rot)
        return data, frame


    def traverse(self, sorted_graph, shell_data, shell_rank):
        edge = 0
        v0 = sorted_graph[edge][0][0]
        if shell_rank == 1:
            return [v0]
        s0 = shell_data[v0]

        v1 = None
        while v1 is None and edge < len(sorted_graph):
            possible_indices = sorted_graph[edge][1]
            possible_indices = [i for i in possible_indices if i != v0]
            for idx in possible_indices:
                if np.abs(np.dot(s0, shell_data[idx])) > self.tol:
                    v1 = idx
                    break
            if v1 is None:
                edge += 1

        if shell_rank == 2:
            return [v0, v1]

        v2 = self.v2_subroutine(v0, v1, edge, sorted_graph, shell_data, shell_rank)
        if v2 is None:
            edge = 0
            v2 = self.v2_subroutine(v1, v0, edge, sorted_graph, shell_data, shell_rank)

        if v2 is None:
            return [v0, v1]

        return [v0, v1, v2]

    def v2_subroutine(self, v0, v1, edge, sorted_graph, shell_data, shell_rank):
        s0 = shell_data[v0]
        s1 = shell_data[v1]
        v2 = None
        while v2 is None and edge < len(sorted_graph):
            if v1 in sorted_graph[edge][0]:
                possible_indices = sorted_graph[edge][1]
                possible_indices = [i for i in possible_indices if i != v0]
                possible_indices = [i for i in possible_indices if i != v1]
                for idx in possible_indices:
                    cond1 = np.abs(np.dot(s0, shell_data[idx])) > self.tol
                    cond2 = np.abs(np.dot(s1, shell_data[idx])) > self.tol
                    if cond1 and cond2:
                        v2 = idx
                        break
            if v2 is None:
                edge += 1
        return v2

    def convert_partition(self, dist_hash, g_hash, r_encoding, g_encoding):
        edges = list(tuple(ast.literal_eval(k)) for k in self.hopcroft._partition.keys())
        ret_edges = []
        ret_graph = []
        for edge in edges:
            a,b = edge
            r0 = get_key(dist_hash, a[0])
            g0 = get_key(g_hash, a[1])
            r1 = get_key(dist_hash, b[0])
            g1 = get_key(g_hash, b[1])
            ret_edges.append([(r0,g0),(r1,g1)])
            r0 = get_key(r_encoding, a[0])
            r1 = get_key(r_encoding, b[0])
            ret_graph.append([r0,r1])

        indexed_edges = sorted(enumerate(ret_edges), key=lambda x: x[1])
        sorted_inidces = [i for i,_ in indexed_edges]
        ret_edges = [element for index, element in indexed_edges]
        ret_graph = [ret_graph[i] for i in sorted_inidces]
        return sorted(ret_edges), ret_graph


    def construct_dfa(self, encoding, graph):
        dfa_encoding = {}
        dfa_set = list()
        for edge in graph:
            value = str([encoding[edge[0]], encoding[edge[1]]])
            dfa_encoding[(edge[0], edge[1])] = value
            dfa_set.append(value)
        return dfa_set, dfa_encoding

    def align_center(self, pointcloud):
        return pointcloud - np.mean(pointcloud,axis=0)


    def geometric_encoding(self, shell_data, adj_list, shell_rank):
        # Project edges onto relative plane
        encoding = {}
        g_hash = {}
        for point in adj_list.keys():
            r_ij = shell_data[adj_list[point]]-shell_data[point]
            if shell_rank == 1:
                d_ij = np.zeros_like(np.linalg.norm(r_ij, axis=1))
            else:
                d_ij = np.linalg.norm(r_ij, axis=1)
            projection = project_onto_plane(r_ij, shell_data[point])
            angle = []
            for i in range(len(projection)):
                if shell_rank == 3:
                    angle += [angle_between_vectors(projection[i], projection[i-1])]
                else:
                    angle += [0]

            # lexicographical shift
            lst = [(custom_round(a,self.tol),custom_round(d,self.tol)) for a,d in zip(angle, d_ij)]
            lst = tuple(list_rotate(lst))
            if lst not in g_hash:
                g_hash[lst] = id(lst)
            encoding[point] = g_hash[lst]
        return g_hash, encoding


    def check_type(self, data, *args, **kwargs):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Data type not supported {type(data)}")

if __name__ == "__main__":
    from torch_geometric.datasets import QM9
    qm9 = QM9(root='/root/workspace/data/qm9-2.4.0/')
    frame = Frame()
    # p = np.array([[1,0],[0,1]])
    # frame.get_frame(p)
    # print('====')
    # p = np.array([[1,0],[-1,0],[-2,0]])
    # frame.get_frame(p)
    # print('====')
    # p = np.array([[1,0],[0,1], [1,1]])
    # frame.get_frame(p)
    for k in range(3,6):
        print(f'Regular Polygon of {k} sides')
        theta = 2*np.pi/k
        p = np.array([[np.cos(i*theta),np.sin(i*theta), 0] for i in range(k)])
        frame.get_frame(p)
    for i,data in enumerate(qm9):
        if i>100:
            break
        else:
            print('='*10)
            print(data.smiles)
            print(data.z)
            frame.get_frame(data.pos)
    exit()
    for k in range(3,6):
        print('Regular Polyhedron of {k} sides')
        theta = 2*np.pi/k
    pass

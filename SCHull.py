from abc import ABCMeta
import ast
import torch
import numpy as np

import sys
sys.path.append('/root/workspace/UnitSphere/alignment/pyorbit/utils/')

from alignment3D import *
from qhull import Qhull


def build_adjacency_list(edges):
    adj_list = {}
    for edge in edges:
        a, b = edge
        if a not in adj_list:
            adj_list[a] = []
        if b not in adj_list:
            adj_list[b] = []
        adj_list[a].append(b)
        adj_list[b].append(a)
    for key in adj_list:
        adj_list[key].sort()
    adj_list = dict(sorted(adj_list.items()))
    return adj_list

def get_key(dct, value):
    keys = []
    for key, val in dct.items():
        if val == value:
            keys.append(key)
    return keys

def direct_graph(edges):
    dg = []
    for edge in edges:
        dg.append(list(edge))
        dg.append(list(edge[::-1]))
    return dg

def custom_round(number, tolerance):
    k = int(-np.log10(tolerance))
    return round(number, k)

def list_rotate(lst):
    idx = lst.index(min(lst))
    return lst[idx:] + lst[:idx]

class Molecule:
    def __init__(self, data=None, cat_data=None):
        self.pos = data
        self.z = cat_data

class SCHull(metaclass=ABCMeta):
    def __init__(self, tol=1e-2, *args, **kwargs):
        super().__init__()
        self.tol = tol
        self.chull = Qhull()

    def align_center(self, pointcloud):
        return pointcloud - np.mean(pointcloud,axis=0)

    def get_hull_geometric_info(self, shell_data, 
                                adj_list,
                                shell_rank):
        # Project edges onto relative plane
        s_feature = {}

        for point in adj_list.keys():
            r_ij = shell_data[adj_list[point]]-shell_data[point]
            if shell_rank == 1:
                d_ij = np.zeros_like(np.linalg.norm(r_ij, axis=1))
            else:
                d_ij = np.linalg.norm(r_ij, axis=1)
            lst = {}
            for ct in range(len(r_ij)):
                lst[adj_list[point][ct]] = (
                                            d_ij[ct],
                                            (r_ij[ct][0],
                                             r_ij[ct][1],
                                             r_ij[ct][2],
                                             )
                                            )

            s_feature[point] = lst
        return s_feature


    def check_type(self, data, *args, **kwargs):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Data type not supported {type(data)}")

    def project_sphere(self, data, cat_data, *args, **kwargs):

        distances = np.linalg.norm(data, axis=1, keepdims=False)
        temp =  data/np.linalg.norm(data, axis=1, keepdims=True)
        arr, key = np.unique(temp, axis=0, return_inverse=True)

        # record which node projected
        proj_index_record = {}
        for k in range(len(key)):
            proj_index_record[key[k]] = []
        for k in range(len(key)):
            proj_index_record[key[k]].append(k)
        ### modified by hyh ###
        
            
        encoding = {}
        dists_hash = {}
        for val in set(key):
            dists = [(custom_round(d,self.tol), custom_round(c,self.tol))  for d,c in zip(distances[key==val],cat_data[key==val])]
            dists = tuple(sorted(dists))
            if dists not in dists_hash:
                dists_hash[dists] = id(dists)

            encoding[val] = dists_hash[dists]
        
        proj_index_record_reverse = {}
        for key in proj_index_record:
            for i in range(len(proj_index_record[key])):
                proj_index_record_reverse[proj_index_record[key][i]] = key

        return dists_hash, encoding, arr, proj_index_record, proj_index_record_reverse

    def get_recover_adj(self,
                        adj_list,
                        shell_data_proj_id_rcrd):
        # step one
        recover_adj_list_1 = {}
        for key in adj_list:
            recover_key = shell_data_proj_id_rcrd[key]
            for k in range(len(recover_key)):
                recover_adj_list_1[recover_key[k]] = adj_list[key]
        
        recover_adj_list_2 = {}
        for key in recover_adj_list_1:
            lst = recover_adj_list_1[key]
            temp = []
            for k in range(len(lst)):
                temp += shell_data_proj_id_rcrd[lst[k]]
            temp.sort()
            recover_adj_list_2[key] = temp
        
        recover_adj_list_2 = dict(sorted(recover_adj_list_2.items()))
        # for key in recover_adj_list_2:
        #     recover_adj_list_2[key].sort()
        return recover_adj_list_2
    
    ### modified by hyh ###
    def get_merged_edge_index(self,
                              adj_list,
                              shell_data_proj_id_rcrd,
                              data_edge_index):
        # step one
        recover_adj_list_1 = {}
        for key in adj_list:
            recover_key = shell_data_proj_id_rcrd[key]
            for k in range(len(recover_key)):
                recover_adj_list_1[recover_key[k]] = adj_list[key]
        
        recover_adj_list_2 = {}
        for key in recover_adj_list_1:
            lst = recover_adj_list_1[key]
            temp = []
            for k in range(len(lst)):
                temp += shell_data_proj_id_rcrd[lst[k]]
            recover_adj_list_2[key] = temp

        edge_node = np.unique(data_edge_index[0])
        data_edge_index_list = {}
        for k in range(len(edge_node)):
            data_edge_index_list[edge_node[k]] = []
        for k in range(len(data_edge_index[0])):
            data_edge_index_list[int(data_edge_index[0][k])].append(int(data_edge_index[1][k]))

        for key in recover_adj_list_2:
            lst = data_edge_index_list[key]
            for ik in range(len(lst)):
                if lst[ik] not in recover_adj_list_2[key]:
                    recover_adj_list_2[key].append(lst[ik])
        
        return recover_adj_list_2
    
    def merge_coord_info(self, 
                         data, s_feature, 
                         shell_data_proj_id_rcrd):
        new_coord_fea = {}
        for key in shell_data_proj_id_rcrd:
            for k in range(len(shell_data_proj_id_rcrd[key])):
                key_ = shell_data_proj_id_rcrd[key][k]
                new_coord_fea[key_] = {'R': np.linalg.norm(data[key_])}

        return new_coord_fea

    def get_radial_arr(self, data):
        radial_arr = []
        for i in range(len(data)):
            radial_arr.append(np.linalg.norm(data[i]))
        return radial_arr

    def adj_arr(self, adj_list):
        arr = [[], []]
        for key in adj_list:
            temp = adj_list[key].copy()
            for k in range(len(temp)):
                arr[0].append(int(key))
                arr[1].append(int(temp[k]))
        return arr
    
    def edge_attr_arr(self, s_feature, 
                      proj_id_rcrd_rvrs,
                      edge_index_hull):

        attr_arr = []
        for i in range(len(edge_index_hull[0])):
            key1 = proj_id_rcrd_rvrs[edge_index_hull[0][i]]
            key2 = proj_id_rcrd_rvrs[edge_index_hull[1][i]]
            temp = s_feature[key1][key2]
            # attr_arr.append(
            #         [temp[0], 
            #          temp[1][0], 
            #          temp[1][1]]
            #     )
            attr_arr.append(
                    [temp[0], 
                    temp[1][0], 
                    temp[1][1], 
                    temp[1][2]]
                )
        return attr_arr

    def get_schull(self, data, cat_data, data_edge_index=None, *args, **kwargs):

        data = self.check_type(data) # Assert Type
        data = self.align_center(data) # Assert Centered
        indices = np.linalg.norm(data, axis=1) > self.tol
        original_data = data.copy()
        original_cat = cat_data.copy()
        data = data[indices]
        cat_data = cat_data[indices]

        ### In order to debug, intentionally make two points proj into one 
        # data[1] = data[0].copy() * 2
        ### modified by hyh ###
        
        # PROJECT ONTO SPHERE
        ### modified by hyh ###
        dist_hash, r_encoding, shell_data, shell_data_proj_id_rcrd,  shell_data_proj_id_rcrd_rvrs= self.project_sphere(data, 
                                                                                                                        cat_data, 
                                                                                                                        *args, 
                                                                                                                        **kwargs)


        
        # GET CONVEX HULL
        shell_rank = np.linalg.matrix_rank(shell_data, tol=self.tol)
        shell_n = shell_data.shape[0]
        shell_graph = self.chull.get_chull_graph(shell_data, shell_rank, shell_n)


        # bool_lst = [i in shell_graph for i in range(shell_n)]
        # if not all(bool_lst):
        #     false_values = [i for i, x in enumerate(bool_lst) if not x]
        #     shell_data = np.delete(shell_data, false_values, axis=0)
        #     # PROJECT ONTO SPHERE
        #     ### modified by hyh ###
        #     dist_hash, r_encoding, shell_data, _ = self.project_sphere(shell_data, cat_data, 
        #                                                                *args, **kwargs)
        #     cat_hash, cat_encoding = self.categorical_encoding(data, cat_data)
            
        #     # GET CONVEX HULL
        #     shell_rank = np.linalg.matrix_rank(shell_data, tol=self.tol)
        #     shell_n = shell_data.shape[0]
        #     shell_graph = self.chull.get_chull_graph(shell_data, shell_rank, shell_n)

        # bool_lst = [i in shell_graph for i in range(shell_n)]
        # assert all(bool_lst), 'Convex Hull is not correct'

        # GET GEOMETRIC ENCODING
        adj_list = build_adjacency_list(shell_graph)

        s_feature = self.get_hull_geometric_info(shell_data, 
                                                 adj_list, 
                                                 shell_rank, 
                                                 )
        
        rcvr_adj_list = self.get_recover_adj(adj_list, shell_data_proj_id_rcrd)

        edge_index_hull = self.adj_arr(rcvr_adj_list)

        edge_attr_hull = self.edge_attr_arr(s_feature,
                                            shell_data_proj_id_rcrd_rvrs,
                                            edge_index_hull)
        radial_arr = self.get_radial_arr(data)
    
        return data, cat_data, edge_index_hull, edge_attr_hull, radial_arr



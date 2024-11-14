from abc import ABCMeta
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import sys
sys.path.append('/root/workspace/UnitSphere/alignment/pyorbit/utils/')
from alignment3D import *
from geometry import angle_between_vectors, planar_normal, project_onto_plane
from hopcroft import PartitionRefinement
from qhull import Qhull

sys.path.append('/root/workspace/UnitSphere/alignment/pyorbit/vis/')
from visualizer import Visualizer, plot_axes, plot_mol, plot_shell, plot_3d_pointcloud, plot_3d_polyhedron, plot_point, plot_plane

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

class Frame(metaclass=ABCMeta):
    def __init__(self, tol=1e-2, *args, **kwargs):
        super().__init__()
        self.tol = tol
        self.chull = Qhull()


    def align(self, data, shell_data, cat_data, pth):      
        funcs = {0: z_axis_alignment, 1: zy_planar_alignment, 2: sign_alignment}
        for idx,val in enumerate(pth):
            # print('func index {}'.format(idx))
            # print('input {}'.format(val))
            # print(shell_data[val])
            data = funcs[idx](data, shell_data[val])
            shell_data = funcs[idx](shell_data, shell_data[val])
        return data, shell_data

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
            v2 = self.v2_subroutine(v1, v0, edge, sorted_graph, shell_data, shell_rank)
        
        assert v2 is not None, 'v2 is None'

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
            # print(edge)
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
        for i,edge in enumerate(graph):
            value = str([encoding[edge[0]], encoding[edge[1]]])
            dfa_encoding[(edge[0], edge[1])] = (value, i)
            dfa_set.append(value)
        return dfa_set, dfa_encoding

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
   
    def geometric_encoding(self, shell_data, 
                           adj_list, 
                           shell_rank, 
                           angle_sorted=False):
        # Project edges onto relative plane
        encoding = {}
        g_hash = {}
        s_feature = {}

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
                    # angle += [angle_between_vectors(projection[i], projection[i-1])]
                    # To do: optimize
                    if i < len(projection) - 1:
                        if angle_sorted:
                            angle.append(tuple(sorted([angle_between_vectors(projection[i], projection[i+1]), 
                                            angle_between_vectors(projection[i], projection[i-1])])))
                        else:
                            angle.append(tuple([angle_between_vectors(projection[i], projection[i-1]), 
                                            angle_between_vectors(projection[i], projection[i+1])]))
                            # if np.isnan(angle_between_vectors(projection[i], projection[i-1])):
                            #     print(projection[i])
                            #     print(projection[i-1])
                    else:
                        if angle_sorted:
                            angle.append(tuple(sorted([angle_between_vectors(projection[i], projection[0]), 
                                            angle_between_vectors(projection[i], projection[i-1])])))
                        else:
                            angle.append(tuple([angle_between_vectors(projection[i], projection[i-1]), 
                                            angle_between_vectors(projection[i], projection[0])]))
                            # if np.isnan(angle_between_vectors(projection[i], projection[i-1])):
                            #     print(projection[i])
                            #     print(projection[i-1])
                    ### modified by hyh: save two angles ###
                else:
                    angle += [(0, 0)]


            # lexicographical shift
            ### modified by hyh ###
            # lst = [(custom_round(a,self.tol), custom_round(d, self.tol)) for a,d in zip(angle, d_ij)]
            lst = {}
            ct = 0
            for angles, d in zip(angle, d_ij):
                # lst.append(
                #         (   
                #             d,
                #             (
                #                 custom_round(angles[0], self.tol), 
                #                 custom_round(angles[1], self.tol)
                #             ),
                #             (point, adj_list[point][ct]) 
                #         )
                #     )
                lst[adj_list[point][ct]] = (
                                            d,
                                            (
                                                custom_round(angles[0], self.tol), 
                                                custom_round(angles[1], self.tol)
                                            )
                                            )
                ct += 1
            s_feature[point] = lst

            # lst = tuple(list_rotate(lst))
            # if lst not in g_hash:
            #     g_hash[lst] = id(lst)
            # encoding[point] = g_hash[lst]   
            g_hash = None
            encoding = None

        return g_hash, encoding, s_feature


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

    def get_frame(self, data, cat_data, data_edge_index=None, *args, **kwargs):

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

np.random.seed(1)

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["font.size"] = 50
plt.rcParams["font.family"] = 'serif'
plt.rcParams['mathtext.default'] = 'default'
# plt.rcParams["font.weight"] = 'bold'
plt.rcParams["xtick.color"] = 'black'
plt.rcParams["ytick.color"] = 'black'
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["axes.linewidth"] = 1

from scipy.spatial.transform import Rotation as R

AZIM=110
ELEV=20
L_THC = 16
L_OP = .3
P_THC = 2000
P_OP = .4
V_THC = 10
V_OP = .1
AR_LEN=0.2
AX_LEN=0.
AX_WTH=10
AX_STY='_x'

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Init
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def plot_projection(ax, data, cat_data, shell_data, edges=[], cycle=[]):

    LIM = 0.7
    ax.set_axis_off()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim([-LIM,LIM])
    ax.set_ylim([-LIM,LIM])
    ax.set_zlim([-LIM+.2*LIM,LIM-.2*LIM])

    origin = [0,0,0]
    x = [-1,0,0]
    y = [0,1,0]
    z = [0,0,1]
    ax.quiver(origin[0], origin[1], origin[2], x[0], x[1], x[2], color='k', linewidth=AX_WTH, arrow_length_ratio=AX_LEN)
    ax.quiver(origin[0], origin[1], origin[2], y[0], y[1], y[2], color='k', linewidth=AX_WTH, arrow_length_ratio=AX_LEN)
    ax.quiver(origin[0], origin[1], origin[2], z[0], z[1], z[2], color='k', linewidth=AX_WTH, arrow_length_ratio=AX_LEN)
    
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.shade = True
    
    # Surface
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    surf = ax.plot_surface(x, y, z, cmap=cm.Greys_r, alpha=.03, linewidth=.1, edgecolor='k')
    
    # Plane
    r = 1
    center = (0, 0, 0)
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = r * np.outer(np.cos(theta), np.sin(phi)) + center[0]
    y = r * np.outer(np.sin(theta), np.sin(phi)) + center[1]
    z = r * np.outer(np.ones(np.size(theta)), np.cos(phi)) + center[2]
    plt.contour(x, y, z, [0], colors='grey')


    for edge in edges:
        x0,y0,z0=shell_data[edge[0]]
        x1,y1,z1=shell_data[edge[1]]
        ax.plot([x0, x1], [y0, y1], [z0, z1], color='grey', alpha=1, linewidth=L_THC)  # You can choose any color
        
    for edge in cycle:
        x0,y0,z0=shell_data[edge[0]]
        x1,y1,z1=shell_data[edge[1]]
        ax.quiver(x0, y0, z0, x1-x0, y1-y0, z1-z0, color='b', alpha=1.0, arrow_length_ratio=AR_LEN, linewidth=L_THC)    # x0,y0,z0=center, center, center
        # ax.plot([x0, x1], [y0, y1], [z0, z1], color='blue', alpha=1, linewidth=L_THC)  # You can choose any color  
   
    colors = {1:'k', 6:'b', 7:'g'}
    # Data
    # for i,point in enumerate(data):
    #     ax.scatter(point[0], point[1], point[2], color=colors[cat_data[i]], alpha=1.0, s=P_THC/2)
    cat_data = [1, 7, 1, 1]
    for i,point in enumerate(shell_data):
        ax.scatter(point[0], point[1], point[2], color=colors[cat_data[i]], alpha=1.0, s=P_THC)
        
    if AX_STY=='_x':
        ax.text(-1., 0.0, -.15, "$z$", color='k')
        ax.text(-.02, 1.0, -.15, "$y$", color='k')
        ax.text(-.1, 0, .98, "$x$", color='k')

if __name__ == "__main__":
    from torch_geometric.datasets import QM9
    from scipy.spatial.transform import Rotation as R
   
    qm9 = QM9(root='/root/workspace/A_data/data/qm9-2.4.0/')
    frame = Frame()
    for i,data in enumerate(qm9):
        k=37
        if i>k:
            break
        elif i<k:
            continue
        else:
            # print(data.smiles)
            cat_data = data.z.numpy()
            aligned_data = frame.get_frame(data.pos.numpy(), 
                                           cat_data,
                                           data.edge_index)
            
            print(f'\nROTATION {i}')

    pass
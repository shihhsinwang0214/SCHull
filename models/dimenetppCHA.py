import torch
from torch import nn
from torch.nn import Linear, Embedding 
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt
from torch_sparse import SparseTensor
from dimenetpp_features import dist_emb, angle_emb
from SCHull_features import angle_emb_hull, torsion_emb_hull
from leftnetCHA import get_angle_torsion

try:
    import sympy as sym
except ImportError:
    sym = None

def xyz_to_dat(pos, edge_index, num_nodes, use_torsion = False):
    """
    Compute the diatance, angle, and torsion from geometric information.

    Args:
        pos: Geometric information for every node in the graph.
        edge_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)


    if use_torsion:
        # Prepare torsion idxes.
        idx_batch = torch.arange(len(idx_i),device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n   
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

        # Calculate torsions.
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji 
        torsion1 = torch.atan2(b, a) # -pi to pi
        torsion1[torsion1<=0]+=2*PI # 0 to 2pi
        torsion = scatter(torsion1,idx_batch_t,reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji
    
    else:
        return dist, angle, i, j, idx_kj, idx_ji
    
def swish(x):
    return x * torch.sigmoid(x)

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        return dist_emb, angle_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf,_ = emb
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, 
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2 


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, 
                 num_output_layers, act, output_init,
                 cha_rate,
                 cha_scale, 
                 isangle_emb_hull,
                 num_filters=128):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init
        self.isangle_emb_hull = isangle_emb_hull
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.lin_hull = Linear(hidden_channels, num_filters, bias=False)
        if isangle_emb_hull:
            self.mlp_hull = torch.nn.Sequential(
                Linear(16, num_filters),
                Linear(num_filters, num_filters),
            )
        else:
            self.mlp_hull = torch.nn.Sequential(
                Linear(7, num_filters),
                Linear(num_filters, num_filters),
            )
        self.lin1_hull = Linear(num_filters, hidden_channels)
        self.lin2_hull = Linear(hidden_channels, int(cha_scale*out_emb_channels*(1-cha_rate)))

        self.lin_cat = Linear(int(cha_scale*out_emb_channels*(1-cha_rate)) + hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i,
                fea_hull, edge_index_hull):
        _, e2 = e
        v = scatter(e2, i, dim=0)

        # hull feature emb
        j_, i_ = edge_index_hull
        v_hull = self.lin_hull(v)
        W_hull = self.act(self.mlp_hull[0](fea_hull))
        W_hull = self.mlp_hull[1](W_hull)
        e_hull = v_hull[j_] * W_hull
        out_hull = scatter(e_hull, i_, dim=0)
        out_hull = self.lin1_hull(out_hull)
        out_hull = self.act(out_hull)
        out_hull = self.lin2_hull(out_hull)

        v = self.act(self.lin_cat(torch.cat([v, out_hull], dim=1)))

        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u

class embHull(torch.nn.Module):
    def __init__(
            self,
            hull_cos = True,
    ):
        super(embHull, self).__init__()
        self.hull_cos = hull_cos
    
    def forward(self, r, h, edge_index):
        row, col = edge_index
        fea2 = torch.cat([r[row].unsqueeze(1), r[col].unsqueeze(1)], dim=1)
        if self.hull_cos:
            h[:,1:] = torch.cos(h[:,1:])
        fea1 = h
        return torch.cat([fea1, fea2], dim=1)
    
class DimeNetPPCHA(torch.nn.Module):
    r"""
        The re-implementation for DimeNet++ from the `"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" <https://arxiv.org/abs/2011.14115>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size (int, optional): Embedding size used in the basis transformation. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`) 
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)       
    """
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=32, 
        basis_emb_size=4, out_emb_channels=32, 
        num_spherical=3, num_radial=2, 
        cha_rate = 0.5,
        cha_scale = 1,
        hull_cos = True,
        envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, 
        act=swish, 
        isangle_emb_hull=False,
        output_init='GlorotOrthogonal'):
        super(DimeNetPPCHA, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.isangle_emb_hull = isangle_emb_hull
        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(hidden_channels, out_emb_channels, 
                               out_channels, num_output_layers, act, output_init,
                               cha_rate, cha_scale, isangle_emb_hull)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.feature_emb_hull = torsion_emb_hull(num_radial=1, 
                                                 num_spherical=2)
        self.angle_emb_hull = angle_emb_hull(num_radial=1, 
                                                num_spherical=2)
        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, 
                     out_channels, num_output_layers, act, output_init,
                     cha_rate, cha_scale, isangle_emb_hull) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])
        self.embhull = embHull(hull_cos)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False)

        emb = self.emb(dist, angle, idx_kj)
        
        # convex hull feature embedding
        edge_index_hull, edge_attr_hull, r = batch_data.edge_index_hull, batch_data.edge_attr_hull, batch_data.posr
        # fea1_hull, fea2_hull = self.embhull(r, edge_attr_hull, edge_index_hull)
        dist_hull = edge_attr_hull[:, 0]
        vecs_hull = edge_attr_hull[:, 1:]
        i_hull, j_hull = edge_index_hull
        theta_hull, phi_hull, tau_hull = get_angle_torsion(edge_index = edge_index_hull,
                                                            vecs = vecs_hull, 
                                                            dist = dist_hull,
                                                            num_nodes = z.size(0))

        if self.isangle_emb_hull:
            fea1_hull = torch.cat([self.feature_emb_hull(dist_hull, theta_hull, phi_hull), 
                                   self.angle_emb_hull(dist_hull, tau_hull[0]),
                                   self.angle_emb_hull(dist_hull, tau_hull[1])], dim=1)
            
            fea2_hull = torch.cat([self.feature_emb_hull(r[i_hull].unsqueeze(1), theta_hull, phi_hull), 
                                   self.angle_emb_hull(r[j_hull].unsqueeze(1), tau_hull[0]),
                                   self.angle_emb_hull(r[j_hull].unsqueeze(1), tau_hull[1]),]
                                   , dim=1)
        else:
            fea1_hull = torch.cat([dist_hull.unsqueeze(1),
                                   theta_hull.unsqueeze(1),
                                   phi_hull.unsqueeze(1),
                                   tau_hull[0].unsqueeze(1),
                                   tau_hull[1].unsqueeze(1)], dim=1)
            fea2_hull = torch.cat([r[i_hull].unsqueeze(1), 
                                   r[j_hull].unsqueeze(1)], dim=1)  

        fea_hull = torch.cat([fea1_hull, fea2_hull], dim=1)
        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, fea_hull, edge_index_hull)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i, fea_hull, 
                         edge_index_hull)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)

        return u
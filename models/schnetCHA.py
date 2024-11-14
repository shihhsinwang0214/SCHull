from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from SCHull_features import angle_emb_hull, torsion_emb_hull
from leftnetCHA import get_angle_torsion

class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff, isangle_emb_hull):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.isangle_emb_hull = isangle_emb_hull
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.lin_hull = Linear(hidden_channels, num_filters, bias=False)
        if isangle_emb_hull:
            self.mlp_hull = Sequential(
                Linear(16, num_filters),
                ShiftedSoftplus(),
                Linear(num_filters, num_filters),
            )
        else:
            self.mlp_hull = Sequential(
                Linear(7, num_filters),
                ShiftedSoftplus(),
                Linear(num_filters, num_filters),
            )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin_hull.weight)
        torch.nn.init.xavier_uniform_(self.mlp_hull[0].weight)
        self.mlp_hull[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_hull[2].weight)
        self.mlp_hull[0].bias.data.fill_(0)

    def forward(self, v, 
                dist, dist_emb, edge_index,
                fea_hull, edge_index_hull):

        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        v_ = self.lin(v)
        e = v_[j] * W

        j_, _ = edge_index_hull
        v_hull = self.lin_hull(v)
        W_hull = self.mlp_hull(fea_hull)
        e_hull = v_hull[j_] * W_hull

        return e, e_hull


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 num_filters,
                 cha_rate,
                 cha_scale,):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, int(cha_scale*hidden_channels*cha_rate))

        self.lin1_hull = Linear(num_filters, hidden_channels)
        self.lin2_hull = Linear(hidden_channels, int(cha_scale*hidden_channels)-int(cha_scale*hidden_channels*cha_rate))

        self.lin_cat = Linear(int(cha_scale*hidden_channels), hidden_channels)
                              
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

        torch.nn.init.xavier_uniform_(self.lin1_hull.weight)
        self.lin1_hull.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2_hull.weight)
        self.lin2_hull.bias.data.fill_(0)

        torch.nn.init.xavier_uniform_(self.lin_cat.weight)
        self.lin_cat.bias.data.fill_(0)

    def forward(self, v, 
                e, edge_index,
                e_hull, edge_index_hull):
        
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)

        _, i_ = edge_index_hull
        out_hull = scatter(e_hull, i_, dim=0)
        out_hull = self.lin1_hull(out_hull)
        out_hull = self.act(out_hull)
        out_hull = self.lin2_hull(out_hull)

        out =  self.act(self.lin_cat(torch.cat([out, out_hull], 1)))
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        u = scatter(v, batch, dim=0)
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

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

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class SchNetCHA(torch.nn.Module):
    r"""
        The re-implementation for SchNet from the `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Output embedding size. (default: :obj:`1`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """
    def __init__(self, energy_and_force=False, 
                 cutoff=5.0, num_layers=6, 
                 hidden_channels=128, out_channels=1, 
                 num_filters=128, 
                 num_gaussians=25,
                 cha_rate = 0.5,
                 cha_scale = 1,
                 hull_cos = True,
                 isangle_emb_hull=False,):
        super(SchNetCHA, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.isangle_emb_hull = isangle_emb_hull
        self.feature_emb_hull = torsion_emb_hull(num_radial=1, 
                                                 num_spherical=2)
        self.angle_emb_hull = angle_emb_hull(num_radial=1, 
                                                num_spherical=2)
        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters, cha_rate, cha_scale) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff, isangle_emb_hull) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels, out_channels)

        self.embhull = embHull(hull_cos)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)
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

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e, e_hull = update_e(v, 
                                dist, dist_emb, edge_index,
                                fea_hull, edge_index_hull)
            v = update_v(v,
                         e, edge_index,
                         e_hull, edge_index_hull)
        u = self.update_u(v, batch)

        return u


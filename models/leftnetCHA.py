import math
from math import pi
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import Embedding
from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_min
from SCHull_features import angle_emb_hull, torsion_emb_hull

def nan_to_num(vec, num=0.0):
    idx = torch.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, dim=-1):
    return nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))

def swish(x):
    return x * torch.sigmoid(x)

def get_angle_torsion(edge_index,
                      vecs, dist,
                      num_nodes,
                      cutoff=9999):
    j, i = edge_index

    # Calculate distances.
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add

    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]
    # --------------------------------------------------------

    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]

    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j

    # i[argmin] = range(0, num_nodes)
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    # ----------------------------------------------------------

    # n0, n1 for i
    n0 = n0[i]
    n1 = n1[i]

    # n0, n1 for j
    n0_j = n0_j[j]
    n1_j = n1_j[j]

    mask_iref = n0 == j
    iref = torch.clone(n0)
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
        vecs,
        vecs[argmin0][i],
        vecs[argmin1][i],
        vecs[idx_iref],
        vecs[idx_jref]
    )

    # Calculate angles.
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi

    # Calculate torsions.
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0)
    plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    # Calculate right torsions.
    plane1 = torch.cross(pos_ji, pos_jref_j)
    plane2 = torch.cross(pos_ji, pos_iref)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji

    a[a>=1] = 1 - 1e-12


    a[a<=-1] = -1 + 1e-12
    b[b>=1] = 1 - 1e-12
    b[b<=-1] = -1 + 1e-12

    tau1 = torch.arccos(a)
    tau2 = math.pi/2 - torch.arcsin(b)

    return theta, phi, [tau1, tau2]

## radial basis function to embed distances
class rbf_emb(nn.Module):
    def __init__(self, num_rbf, soft_cutoff_upper, rbf_trainable=False):
        super().__init__()
        self.soft_cutoff_upper = soft_cutoff_upper
        self.soft_cutoff_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist=dist.unsqueeze(-1)
        soft_cutoff = 0.5 * \
                  (torch.cos(dist * pi / self.soft_cutoff_upper) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.soft_cutoff_upper).float()
        return soft_cutoff*torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    def __init__(self, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.embedding(z)
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j

class S_vector(MessagePassing):
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU())

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            num_radial,
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.inv_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3), nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3), )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)

        self.hull_proj = nn.Linear(num_radial, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)

    def forward(self, x, vec, 
                edge_index, edge_rbf, weight, edge_vector
                ):

        xh = self.x_proj(x)
        rbfh = self.rbf_proj(edge_rbf)
        weight = self.inv_proj(weight)
        rbfh = rbfh * weight
        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

class EquiMessagePassingHull(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            fea_dim1=3,
            fea_dim2=2,
            isangle_emb_hull=False,
            pos_require_grad=False
    ):
        super(EquiMessagePassingHull, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.pos_require_grad = pos_require_grad

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels*3),
        )

        self.fea_proj = nn.Sequential(
                            nn.Linear(fea_dim1+fea_dim2, hidden_channels),
                            nn.SiLU(),
                            nn.Linear(hidden_channels, 3 * hidden_channels),
                            )
        
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)

        nn.init.xavier_uniform_(self.fea_proj[0].weight)
        self.fea_proj[0].bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.fea_proj[2].weight)
        # self.fea_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, 
                edge_index_hull, fea1_hull, fea2_hull, edge_vector_hull
                ):

        xh = self.x_proj(x)
        fea_hull = self.fea_proj(torch.cat([fea1_hull, fea2_hull], dim=1))

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index_hull,
            xh=xh,
            vec=vec,
            rbfh_ij=fea_hull,
            r_ij = edge_vector_hull,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):

        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        # if self.pos_require_grad:
        #     x = xh_j[:,:self.hidden_channels] * rbfh_ij
        #     xh2 = xh_j[:,self.hidden_channels:self.hidden_channels*2] * rbfh_ij
        #     xh3 = xh_j[:,self.hidden_channels*2:] * rbfh_ij
        #     vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        #     vec = vec * self.inv_sqrt_h
        # else:
        #     x = xh_j * rbfh_ij
        #     vec = torch.zeros(size=[x.shape[0], 3, x.shape[1]], 
        #                     device=x.device)
        return x, vec

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
    
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
        return fea1, fea2

class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.equi_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xequi_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.equi_proj.weight)
        nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, node_frame):

        vec = self.equi_proj(vec)
        vec1,vec2 = torch.split(
                 vec, self.hidden_channels, dim=-1
             )

        scalrization = torch.sum(vec1.unsqueeze(2) * node_frame.unsqueeze(-1), dim=1)
        scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())
        scalar = torch.norm(vec1, dim=-2) # torch.sqrt(torch.sum(vec1 ** 2, dim=-2))
        
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xequi_proj(
            torch.cat(
                [x, scalar], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class aggregate_pos(MessagePassing):

    def __init__(self, aggr='mean'):  
        super(aggregate_pos, self).__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


class EquiOutput(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                # GatedEquivariantBlock(
                #     hidden_channels,
                #     hidden_channels // 2,
                # ),
                GatedEquivariantBlock(hidden_channels, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


class LEFTNetCHA(torch.nn.Module):
    r"""
        LEFTNet

        Args:
            pos_require_grad (bool, optional): If set to :obj:`True`, will require to take derivative of model output with respect to the atomic positions. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`32`)
            y_mean (float, optional): Mean value of the labels of training data. (default: :obj:`0`)
            y_std (float, optional): Standard deviation of the labels of training data. (default: :obj:`1`)

    """

    def __init__(
            self, pos_require_grad=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, 
            num_radial=32,
            y_mean=0, y_std=1, 
            cha_rate = 0.5,
            cha_scale = 1,
            hull_cos=False,
            isangle_emb_hull = False,
            **kwargs):
        super(LEFTNetCHA, self).__init__()
        self.y_std = y_std
        self.y_mean = y_mean
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.pos_require_grad = pos_require_grad
        self.cha_rate = cha_rate
        self.cha_scale = cha_scale
        self.z_emb = Embedding(95, hidden_channels)
        self.radial_emb = rbf_emb(num_radial, self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        
        self.cha_rate_lin = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            nn.Sigmoid())
        
        self.neighbor_emb = NeighborEmb(hidden_channels)
        self.feature_emb_hull = torsion_emb_hull(num_radial=1, 
                                                 num_spherical=2)
        self.angle_emb_hull = angle_emb_hull(num_radial=1, 
                                                num_spherical=2)
        self.S_vector = S_vector(hidden_channels)
        self.isangle_emb_hull = isangle_emb_hull
        self.lin = nn.Sequential(
            nn.Linear(3, hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, 1))
        
        self.embhull = embHull(hull_cos=hull_cos)
        
        self.message_layers = nn.ModuleList()
        self.message_hull_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()
        
        for _ in range(num_layers):
            self.message_layers.append(
                EquiMessagePassing(hidden_channels, num_radial).jittable()
            )
            if isangle_emb_hull:
                self.message_hull_layers.append(
                    EquiMessagePassingHull(hidden_channels,
                                        fea_dim1=8,
                                        fea_dim2=8,
                                        isangle_emb_hull=True,
                                        pos_require_grad=pos_require_grad).jittable()
                )
            else:
                self.message_hull_layers.append(
                    EquiMessagePassingHull(hidden_channels,
                                        fea_dim1=5,
                                        fea_dim2=2,
                                        pos_require_grad=pos_require_grad).jittable()
                )
            self.FTEs.append(FTE(hidden_channels))

        self.last_layer = nn.Linear(hidden_channels, out_channels)
        if self.pos_require_grad:
            self.out_forces = EquiOutput(hidden_channels)
        
        # for node-wise frame
        self.mean_neighbor_pos = aggregate_pos(aggr='mean')

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.posc, batch_data.batch
        if self.pos_require_grad:
            pos.requires_grad_()
        
        # embed z
        z_emb = self.z_emb(z)
        
        # construct edges based on the cutoff value
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        i, j = edge_index
        
        # embed pair-wise distance
        dist = torch.norm(pos[i]-pos[j], dim=-1)
        # radial_emb shape: (num_edges, num_radial), radial_hidden shape: (num_edges, hidden_channels)
        radial_emb = self.radial_emb(dist)	
        radial_hidden = self.radial_lin(radial_emb)	
        soft_cutoff = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        radial_hidden = soft_cutoff.unsqueeze(-1) * radial_hidden

        # init invariant node features
        # shape: (num_nodes, hidden_channels)
        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)

        # init equivariant node features
        # shape: (num_nodes, 3, hidden_channels)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        # bulid edge-wise frame
        edge_diff = pos[i] - pos[j]
        edge_diff = _normalize(edge_diff)
        edge_cross = torch.cross(pos[i], pos[j])
        edge_cross = _normalize(edge_cross)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # edge_frame shape: (num_edges, 3, 3)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)
        
        # build node-wise frame
        mean_neighbor_pos = self.mean_neighbor_pos(pos, edge_index)
        node_diff = pos - mean_neighbor_pos
        node_diff = _normalize(node_diff)
        node_cross = torch.cross(pos, mean_neighbor_pos)
        node_cross = _normalize(node_cross)
        node_vertical = torch.cross(node_diff, node_cross)
        # node_frame shape: (num_nodes, 3, 3)
        node_frame = torch.cat((node_diff.unsqueeze(-1), node_cross.unsqueeze(-1), node_vertical.unsqueeze(-1)), dim=-1)

        # LSE: local 3D substructure encoding
        # S_i_j shape: (num_nodes, 3, hidden_channels)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + torch.permute(scalrization1, (0, 2, 1))[:, :,
                                                                        0].unsqueeze(2)).squeeze(-1)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + torch.permute(scalrization2, (0, 2, 1))[:, :,
                                                                        0].unsqueeze(2)).squeeze(-1)
        
        A_i_j = torch.cat((scalar3, scalar4), dim=-1) * soft_cutoff.unsqueeze(-1)
        A_i_j = torch.cat((A_i_j, radial_hidden, radial_emb), dim=-1)
        
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

        # r_inp = torch.tensor([self.cha_rate], device=s.device)
        # r_ate = self.cha_rate_lin(r_inp).squeeze()
        r_ate = self.cha_rate
    
        for i in range(self.num_layers):
            # equivariant message passing
            ds, dvec = self.message_layers[i](
                s, vec, 
                edge_index, radial_emb, A_i_j, edge_diff
            )

            ds_hull, dvec_hull = self.message_hull_layers[i](
                s, vec, 
                edge_index_hull, fea1_hull, fea2_hull, vecs_hull
            )

            # s = s + self.cha_scale * (self.cha_rate * ds + (1-self.cha_rate)*ds_hull)
            # vec = vec + self.cha_scale * (self.cha_rate * dvec + (1-self.cha_rate)*dvec_hull)
            s = s + self.cha_scale * (r_ate * ds + (1-r_ate)*ds_hull)
            vec = vec + self.cha_scale * (r_ate * dvec + (1-r_ate)*dvec_hull)

            # FTE: frame transition encoding
            ds, dvec = self.FTEs[i](s, vec, node_frame)
            s = s + ds
            vec = vec + dvec

        if self.pos_require_grad:
            forces = self.out_forces(s, vec)
        s = self.last_layer(s)
        s = scatter(s, batch, dim=0)
        s = s * self.y_std + self.y_mean
        if self.pos_require_grad:
            return s, forces
        return s

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


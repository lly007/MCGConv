import torch
from torch.nn import Conv2d, Conv3d, Module, Linear, Sequential, GELU, init, Parameter
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm3d
from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.ops import ball_query, knn_points, knn_gather
from pytorch3d.ops.utils import masked_gather


class MCGConv(Module):
    def __init__(self, npoint, cin, cout, radius, nsample, m1, m2, v, ball_query=True, use_normal=True, group_all = False):
        super(MCGConv, self).__init__()
        self.use_normal = use_normal
        self.npoint = npoint
        self.cin = cin
        self.cout = cout
        self.radius = radius
        self.nsample = nsample
        self.m1 = m1
        self.m2 = m2
        self.ball_query = ball_query
        self.v = v
        self.group_all = group_all

        self.m1s = Sequential()
        self.m2s = Sequential()

        self.m1_num_layers = len(self.m1) - 1
        self.m2_num_layers = len(self.m2) - 1

        # Modify the input and output channels of M1 and M2
        self.m1[0] = 13
        self.m1[-1] = self.cin
        self.m2[0] = self.cin
        self.m2[-1] = self.cout

        # The MLP of M1
        for i in range(self.m1_num_layers - 1):
            self.m1s.add_module("M1_Conv3D_" + str(i), Conv3d(self.m1[i], self.m1[i+1], (1, 1, 1)))
            self.m1s.add_module("M1_BatchNorm3D_" + str(i), BatchNorm3d(self.m1[i+1]))
            self.m1s.add_module("M1_GELU" + str(i), GELU())
        self.m1s.add_module("M1_Conv3D_Last", Conv3d(self.m1[-2], self.m1[-1], (1, 1, 1)))

        # The MLP of M2
        for i in range(self.m2_num_layers - 1):
            self.m2s.add_module("M2_Conv2D_" + str(i), Conv2d(self.m2[i], self.m2[i+1], (1, 1)))
            self.m2s.add_module("M2_BatchNorm2D_" + str(i), BatchNorm2d(self.m2[i+1]))
        self.m2s.add_module("M2_Conv2D_Last", Conv2d(self.m2[-2], self.m2[-1], (1, 1)))

        # self.multi_centroids_idx = torch.randperm(nsample, device="cuda:0")[:v] # [B, S, v]

        self.res = Linear(self.cin, cout)

        # self.mix_max_sum = Linear(self.cout * 2, self.cout)

        self.data_structuring = Grouping(self.npoint, self.nsample, radius, cin, cout, group_all, ball_query, use_normal)

    def forward(self, xyz, f):
        if f is None:
            f = xyz
        B, N, C = f.shape
        S = self.npoint
        n = self.nsample
        V = self.v
        if self.group_all:
            S = 1
            self.npoint = 1

        # random select centroids index
        multi_centroids_idx = torch.randperm(n, device=f.device)[:V]  # [B, S, v]    

        # # data structuring
        # if S is not None:
        #     new_xyz, new_xyz_idx = fps(xyz, None, S, True)  # [B, S, 3], [B, S]
        #     if self.ball_query:
        #         _, grouped_idx, grouped_xyz = ball_query(new_xyz, xyz, None, None, n, self.radius, True)
        #     else:
        #         _, grouped_idx, grouped_xyz = knn_points(new_xyz, xyz, None, None, K=n, return_nn=True)
        # else:
        #     new_xyz = xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]  S=1
        #     if self.ball_query:
        #         _, grouped_idx, grouped_xyz = ball_query(new_xyz, xyz, None, None, n, self.radius, True)
        #     else:
        #         _, grouped_idx, grouped_xyz = knn_points(new_xyz, xyz, None, None, K=n, return_nn=True)
        #     S = 1
        #     self.npoint = 1
        new_xyz, grouped_xyz, grouped_xyz_local, grouped_f, mask = self.data_structuring(xyz, f)
        multi_centroids_idx = multi_centroids_idx.view(1, 1, V).repeat(B, S, 1)  # [B, S, v]
        multi_centroids = knn_gather(xyz, multi_centroids_idx)  # [B, S, v, 3]

        # gather the features
        # if self.ball_query:
        #     grouped_feature = masked_gather(f, grouped_idx)  # [B, S, n, cin]
        # else:
        #     grouped_feature = knn_gather(f, grouped_idx)  # [B, S, n, cin]

        # grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, 3)  # [B, S, n, 3] Local pos

        # cat new features [B, S, n, 3+3+cin]
        # grouped_f = torch.cat([grouped_xyz, grouped_xyz_local, grouped_feature], -1)
        # for residual connection
        group_skip = grouped_f
        # build rel_jk
        grouped_xyz = grouped_xyz.view(B, S, n, 1, 3)  # [B, S, n, 1, 3]
        multi_centroids = multi_centroids.view(B, S, 1, V, 3)  # [B, S, 1, V, 3]
        arrow = grouped_xyz - multi_centroids  # [B, S, n, V, 3]
        euclidean = torch.sqrt((arrow ** 2).sum(dim=-1, keepdim=True))
        rel_jk = torch.cat([-arrow, arrow, euclidean, grouped_xyz.repeat(1, 1, 1, V, 1), multi_centroids.repeat(1, 1, n, 1, 1)], -1)  # [B, S, n, V, 13]

        # [B, 6, S, n, V] to chaanel first
        rel_jk = rel_jk.permute(0, 4, 1, 2, 3)

        # m1 in the paper
        w_jk = self.m1s(rel_jk)

        grouped_f = grouped_f.permute(0, 3, 1, 2).view(B, self.cin, S, n, 1)  # [B, in, S, n, 1]

        grouped_f = grouped_f * w_jk  # [B, in, S, n, V]

        # softpooling
        grouped_f = grouped_f.permute(0, 2, 3, 1, 4)  # [B, S, n, in, V]
        soft_w = F.gumbel_softmax(grouped_f, 1, False, dim=-1)  # [B, S, n, in, V]
        grouped_f = soft_w * grouped_f  # [B, S, n, in, V]
        grouped_f = torch.sum(grouped_f, -1)  # [B, S, n, in]
        # softpooling end

        grouped_f = grouped_f + group_skip  # [B, S, n, in]

        if self.ball_query:
            grouped_f[mask] = 0.0
        # channel first  [B, in, S, n]
        grouped_f = grouped_f.permute(0, 3, 1, 2)  # [B, in, S, n]

        w = grouped_f

        # m2 in the paper
        w = self.m2s(w)  # [B, out, S, n]

        new_f = w.permute(0, 2, 3, 1)  # [B, S, n, out]

        new_f = F.gelu(new_f + self.res(group_skip))  # [B, S, n, out]

        # new_f = torch.cat([torch.max(new_f, dim=2)[0], torch.sum(new_f, dim=2)], dim=-1) # [B, S, out]

        # new_f = self.mix_max_sum(new_f)

        new_f = torch.max(new_f, dim=2)[0]

        return new_f, new_xyz
    
class Grouping(Module):
    def __init__(self, s, n, radius, cin, cout, groupall=False, ball_query=True, use_normals=True):
        super(Grouping, self).__init__()
        self.n = n
        self.radius = radius
        self.groupall = groupall
        self.ball_query = ball_query
        self.use_normals = use_normals
        self.S = s        

        self.affine_weight = Parameter(torch.ones(cin + 6))
        self.affine_bias = Parameter(torch.zeros(cin + 6))

        self.mix = Linear(2 * cin + 6, cin)


    def forward(self, xyz, f):
        B, N, cin = f.shape
        n = self.n
        radius = self.radius
        groupall = self.groupall
        ballquery = self.ball_query
        normals = self.use_normals

        if f is None:
            f = xyz
        if self.S is None:
            S = 1
        else:
            S = self.S

        if not groupall:
            new_xyz, new_xyz_idx = fps(xyz, None, S, True)  # [B, S, 3], [B, S]
            if ballquery:
                _, grouped_idx, grouped_xyz = ball_query(new_xyz, xyz, None, None, n, radius, True)
            else:
                _, grouped_idx, grouped_xyz = knn_points(new_xyz, xyz, None, None, K=n, return_nn=True)
        else:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]  S=1
            if self.ball_query:
                _, grouped_idx, grouped_xyz = ball_query(new_xyz, xyz, None, None, n, self.radius, True)
            else:
                _, grouped_idx, grouped_xyz = knn_points(new_xyz, xyz, None, None, K=n, return_nn=True)
            S = 1
                
        if self.ball_query:
            grouped_feature = masked_gather(f, grouped_idx)  # [B, S, n, cin]
        else:
            grouped_feature = knn_gather(f, grouped_idx)  # [B, S, n, cin]

        mask = grouped_idx == -1
        if not groupall:
            sampled_features = masked_gather(f, new_xyz_idx).view(B, S, 1, cin)
        else:
            sampled_features = torch.mean(grouped_feature, dim=-2, keepdim=True)

        grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, 3)  # [B, S, n, 3] Local pos
        grouped_feature_local = grouped_feature - sampled_features  # [B, S, n, in] Local feature

        grouped_f = torch.cat([grouped_xyz ,grouped_xyz_local, grouped_feature_local], -1)  # [B, S, n, in+6]

        std = torch.std(grouped_f.view(B, -1), dim=-1, keepdim=True).view(B, 1, 1, 1)

        grouped_f = grouped_f / (std + 1e-5)
        grouped_f = self.affine_weight * grouped_f + self.affine_bias

        grouped_f = torch.cat([grouped_f, sampled_features.repeat(1, 1, n, 1)], -1)  # [B, S, n, 2*in+6]

        grouped_f = self.mix(grouped_f) # [B, S, n, in+6]

        return new_xyz, grouped_xyz, grouped_xyz_local, grouped_f, mask


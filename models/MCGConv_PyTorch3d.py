from tokenize import group
import torch
from torch._C import device
from torch.nn import Conv1d, Conv2d, Conv3d, ModuleList, Module, Linear
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.ops import ball_query, knn_points, knn_gather
from pytorch3d.ops.utils import masked_gather
from pytorch3d.transforms import RotateAxisAngle


class MCGConv(Module):
    def __init__(self, npoint, cin, cout, radius, nsample, m1, m2, v, ball_query=True):
        super(MCGConv, self).__init__()        
        self.npoint = npoint
        self.cin = cin + 3 + 3
        self.cout = cout
        self.radius = radius
        self.nsample = nsample
        self.m1 = m1
        self.m2 = m2
        self.ball_query = ball_query
        self.v = v

        self.m1_filter = ModuleList()
        self.m1_bn = ModuleList()
        self.m2_filter = ModuleList()
        self.m2_bn = ModuleList()
        self.mr_filter = ModuleList()
        self.mr_bn = ModuleList()
        self.m1_num = len(self.m1) - 1
        self.m2_num = len(self.m2) - 1

        self.m1[0] = 13
        self.m1[-1] = self.cin
        self.m2[0] = self.cin
        self.m2[-1] = self.cout

        for i in range(self.m1_num - 1):
            self.m1_filter.append(Conv3d(self.m1[i], self.m1[i+1], (1, 1, 1)))
            self.m1_bn.append(BatchNorm3d(self.m1[i+1]))
        self.m1_filter.append(Conv3d(self.m1[-2], self.m1[-1], (1, 1, 1)))

        for i in range(self.m2_num - 1):
            self.m2_filter.append(Conv2d(self.m2[i], self.m2[i+1], (1, 1)))
            self.m2_bn.append(BatchNorm2d(self.m2[i+1]))
        self.m2_filter.append(Conv2d(self.m2[-2], self.m2[-1], (1, 1)))

        self.multi_centroids_idx = torch.randperm(nsample, device="cuda:0")[:v] # [B, S, v]

        self.res = Linear(self.cin, cout)

    def forward(self, xyz, f):
        B, N, C = f.shape
        S = self.npoint
        n = self.nsample
        V = self.v

        if S is not None:
            new_xyz, new_xyz_idx = fps(xyz, None, S, True)  # [B, S, 3], [B, S]
            if self.ball_query:
                _, grouped_idx, grouped_xyz = ball_query(new_xyz, xyz, None, None, n, self.radius, True)
            else:
                _, grouped_idx, grouped_xyz = knn_points(new_xyz, xyz, None, None, K=n, return_nn=True)
        else:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]  S=1
            if self.ball_query:
                _, grouped_idx, grouped_xyz = ball_query(new_xyz, xyz, None, None, n, self.radius, True)
            else:
                _, grouped_idx, grouped_xyz = knn_points(new_xyz, xyz, None, None, K=n, return_nn=True)
            S = 1
            self.npoint = 1

        multi_centroids_idx =  self.multi_centroids_idx.view(1, 1, V).repeat(B, S, 1) # [B, S, v]

        multi_centroids = knn_gather(xyz, multi_centroids_idx) # [B, S, v, 3]

        if self.ball_query:
            mask = grouped_idx == -1
            grouped_feature = masked_gather(f, grouped_idx)  # [B, S, n, cin]
        else:
            grouped_feature = knn_gather(f, grouped_idx)  # [B, S, n, cin]

        grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, 3).repeat(1, 1, n, 1)  # [B, S, n, 3] Local pos

        # [B, S, n, 3+3+cin]
        grouped_f = torch.cat([grouped_xyz, grouped_xyz_local, grouped_feature], -1)

        group_skip = grouped_f

                
        grouped_xyz = grouped_xyz.view(B, S, n, 1, 3).repeat(1, 1, 1, V, 1)  # [B, S, n, V, 3]
        multi_centroids_repeat = multi_centroids.view(B, S, 1, V, 3).repeat(1, 1, n, 1, 1)  # [B, S, n, V, 3]

        arrow = grouped_xyz - multi_centroids_repeat  # [B, S, n, V, 3]
        euclidean = torch.sqrt((arrow ** 2).sum(dim=-1, keepdim=True))

        rel_jk = torch.cat([-arrow, arrow, euclidean, grouped_xyz, multi_centroids_repeat], -1)  # [B, S, n, V, 13]
        # cocated_group = torch.cat([euclidean, grouped_xyz, framepoints_repeat], -1)  # [B, S, n, V, 7]

        # [B, 6, S, n, V] to chaanel first
        a = rel_jk.permute(0, 4, 1, 2, 3)
        # m1 in the paper
        for i in range(self.m1_num - 1):
            bn = self.m1_bn[i]
            a = F.gelu(bn(self.m1_filter[i](a)))

        a = self.m1_filter[-1](a)  # [B, in, S, n, V]

        grouped_f = grouped_f.permute(0, 3, 1, 2).view(B, self.cin, S, n, 1)  # [B, in, S, n, 1]

        grouped_f = grouped_f * a  # [B, in, S, n, V]

        # softpooling
        grouped_f = grouped_f.permute(0, 2, 3, 1, 4)  # [B, S, n, in, V]
        soft_w = F.gumbel_softmax(grouped_f, 1, False, dim= -1)  # [B, S, n, in, V]
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
        for i in range(self.m2_num - 1):
            bn = self.m2_bn[i]
            w = F.gelu(bn(self.m2_filter[i](w)))

        w = self.m2_filter[-1](w)  # [B, out, S, n]

        new_f = w.permute(0, 2, 3, 1)  # [B, S, n, out]

        new_f = F.gelu(new_f + self.res(group_skip))  # [B, S, n, out]

        new_f = torch.max(new_f, dim=2)[0] + torch.sum(new_f, dim=2) # [B, S, out]

        return new_f, new_xyz


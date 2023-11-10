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


class FMRConv(Module):
    def __init__(self, npoint, cin, cout, radius, nsample, m1, m2, framepoints, ball_query=True):
        super(FMRConv, self).__init__()
        self.device = framepoints.device
        self.npoint = npoint
        self.cin = cin + 3 + 3
        self.cout = cout
        self.radius = radius
        self.nsample = nsample
        self.m1 = m1
        self.m2 = m2
        self.ball_query = ball_query

        self.framepoints = framepoints * radius * 0.8
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

        self.rotated_framepoints = self.rotate_frame_points(
            self.framepoints)  # [V, 3]

        for i in range(self.m1_num - 1):
            self.m1_filter.append(Conv3d(self.m1[i], self.m1[i+1], (1, 1, 1)))
            self.m1_bn.append(BatchNorm3d(self.m1[i+1]))
        self.m1_filter.append(Conv3d(self.m1[-2], self.m1[-1], (1, 1, 1)))

        for i in range(self.m2_num - 1):
            self.m2_filter.append(Conv2d(self.m2[i], self.m2[i+1], (1, 1)))
            self.m2_bn.append(BatchNorm2d(self.m2[i+1]))
        self.m2_filter.append(Conv2d(self.m2[-2], self.m2[-1], (1, 1)))

        self.res = Linear(cin, cout)

    def forward(self, xyz, f):
        B, N, C = f.shape
        V = self.framepoints.shape[0]
        S = self.npoint
        n = self.nsample

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

        if self.ball_query:
            mask = grouped_idx == -1
            grouped_feature = masked_gather(f, grouped_idx)  # [B, S, n, cin]
        else:
            grouped_feature = knn_gather(f, grouped_idx)  # [B, S, n, cin]

        grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, 3).repeat(1, 1, n, 1)  # [B, S, n, 3] Local pos

        # [B, S, n, 3+3+cin]
        grouped_f = torch.cat([grouped_xyz, grouped_xyz_local, grouped_feature], -1)

        framepoints = self.rotated_framepoints  # [V, 3]
        
        grouped_xyz = grouped_xyz.view(B, S, n, 1, 3).repeat(1, 1, 1, V, 1)  # [B, S, n, V, 3]
        framepoints_repeat = framepoints.view(1, 1, 1, V, 3).repeat(B, S, n, 1, 1)

        arrow = grouped_xyz - framepoints_repeat  # [B, S, n, V, 3]
        euclidean = torch.sqrt((arrow ** 2).sum(dim=-1, keepdim=True))

        cocated_group = torch.cat([-arrow, arrow, euclidean, grouped_xyz, framepoints_repeat], -1)  # [B, S, n, V, 13]
        # cocated_group = torch.cat([euclidean, grouped_xyz, framepoints_repeat], -1)  # [B, S, n, V, 7]

        # [B, 6, S, n, V] to chaanel first
        a = cocated_group.permute(0, 4, 1, 2, 3)
        # m1 in the paper
        for i in range(self.m1_num - 1):
            bn = self.m1_bn[i]
            a = F.relu(bn(self.m1_filter[i](a)))

        a = self.m1_filter[-1](a)  # [B, in, S, n, V]

        grouped_f = grouped_f.permute(0, 3, 1, 2).view(B, self.cin, S, n, 1)  # [B, in, S, n, 1]

        grouped_f = grouped_f * a  # [B, in, S, n, V]

        # softpooling
        grouped_f = grouped_f.permute(0, 2, 3, 1, 4)  # [B, S, n, in, V]
        soft_w = F.softmax(grouped_f, -1)  # [B, S, n, in, V]
        grouped_f = soft_w * grouped_f  # [B, S, n, in, V]
        grouped_f = torch.sum(grouped_f, -1)  # [B, S, n, in]
        # softpooling end

        if self.ball_query:
            grouped_f[mask] = 0.0
        # channel first  [B, in, S, n]
        grouped_f = grouped_f.permute(0, 3, 1, 2)

        w = grouped_f
        # m2 in the paper
        for i in range(self.m2_num - 1):
            bn = self.m2_bn[i]
            w = F.relu(bn(self.m2_filter[i](w)))

        w = self.m2_filter[-1](w)  # [B, out, S, n]

        new_f = w.permute(0, 2, 3, 1)  # [B, S, n, out]

        new_f = F.gelu(new_f + self.res(grouped_feature))  # [B, S, n, out]

        new_f = torch.max(new_f, dim=2)[0]  # [B, S, out]

        return new_f, new_xyz

    # Random rotate the frame points
    def rotate_frame_points(self, framepoints):
        rotation_angle = torch.rand(1, device=self.device) * 360
        rotater = RotateAxisAngle(rotation_angle, "Y")
        return rotater.transform_points(framepoints)

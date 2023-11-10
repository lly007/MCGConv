import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from FMRConv_PyTorch3d import FMRConv


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        framepoint = torch.tensor([[1.0,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,0]]).cuda()
        self.fmrconv1 = FMRConv(npoint=1024, cin=6, cout=64, radius=0.15, nsample=32, m1=[7,16,1], m2=[1,16,64], framepoints=framepoint, ball_query=False)
        self.fmrconv2 = FMRConv(npoint=512, cin=64, cout=128, radius=0.15, nsample=32, m1=[7,32,1], m2=[1,64,128], framepoints=framepoint, ball_query=False)
        self.fmrconv3 = FMRConv(npoint=256, cin=128, cout=256, radius=0.25, nsample=16, m1=[7,32,1], m2=[1,64,256], framepoints=framepoint, ball_query=False)
        self.fmrconv4 = FMRConv(npoint=128, cin=256, cout=256, radius=0.4, nsample=16, m1=[7,64,1], m2=[1,128,256], framepoints=framepoint, ball_query=False)
        self.fmrconv5 = FMRConv(npoint=32, cin=256, cout=512, radius=0.6, nsample=8, m1=[7,64,1], m2=[1,128,512], framepoints=framepoint, ball_query=False)
        self.fmrconv6 = FMRConv(npoint=None, cin=512, cout=1024, radius=10, nsample=32, m1=[7,128,1], m2=[1,256,1024], framepoints=framepoint, ball_query=False)
        
        self.fp6 = PointNetFeaturePropagation(in_channel=1536, mlp=[512, 512])
        self.fp5 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        self.fp4 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=192, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l0_points = l0_points.transpose(1, 2) # channel last
        l0_xyz = l0_xyz.transpose(1, 2)
        l1_points, l1_xyz  = self.fmrconv1(l0_xyz, l0_points)
        l2_points, l2_xyz  = self.fmrconv2(l1_xyz, l1_points)
        l3_points, l3_xyz  = self.fmrconv3(l2_xyz, l2_points)
        l4_points, l4_xyz  = self.fmrconv4(l3_xyz, l3_points)
        l5_points, l5_xyz  = self.fmrconv5(l4_xyz, l4_points)
        l6_points, l6_xyz  = self.fmrconv6(l5_xyz, l5_points)
        l1_points = torch.transpose(l1_points,1,2)
        l2_points = torch.transpose(l2_points,1,2)
        l3_points = torch.transpose(l3_points,1,2)
        l4_points = torch.transpose(l4_points,1,2)
        l5_points = torch.transpose(l5_points,1,2)
        l6_points = torch.transpose(l6_points,1,2)
        l1_xyz = torch.transpose(l1_xyz,1,2)
        l2_xyz = torch.transpose(l2_xyz,1,2)
        l3_xyz = torch.transpose(l3_xyz,1,2)
        l4_xyz = torch.transpose(l4_xyz,1,2)
        l5_xyz = torch.transpose(l5_xyz,1,2)
        l6_xyz = torch.transpose(l6_xyz,1,2)
        # Feature Propagation layers
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_xyz = l0_xyz.transpose(1, 2) # channel first
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points.transpose(1, 2)],1), l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.cel = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred, target, trans_feat):
        total_loss = self.cel(pred, target)
        return total_loss
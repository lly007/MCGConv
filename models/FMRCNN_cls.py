import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from FMRConv_PyTorch3d import FMRConv


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        framepoint = torch.tensor([[1.0,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,0]]).cuda()
        self.fmrconv1 = FMRConv(npoint=512, cin=3, cout=128, radius=0.15, nsample=32, m1=[7,32,1], m2=[1,64,128], framepoints=framepoint, ball_query=False)
        self.fmrconv2 = FMRConv(npoint=256, cin=128, cout=256, radius=0.25, nsample=32, m1=[7,32,1], m2=[1,64,256], framepoints=framepoint, ball_query=False)
        self.fmrconv3 = FMRConv(npoint=128, cin=256, cout=256, radius=0.4, nsample=32, m1=[7,64,1], m2=[1,128,256], framepoints=framepoint, ball_query=False)
        self.fmrconv4 = FMRConv(npoint=32, cin=256, cout=512, radius=0.6, nsample=32, m1=[7,64,1], m2=[1,128,512], framepoints=framepoint, ball_query=False)
        self.fmrconv5 = FMRConv(npoint=None, cin=512, cout=1024, radius=10, nsample=32, m1=[7,128,1], m2=[1,256,1024], framepoints=framepoint, ball_query=False)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        xyz = xyz.transpose(1, 2)
        norm = norm.transpose(1, 2)
        f1, s1 = self.fmrconv1(xyz, norm)
        f2, s2 = self.fmrconv2(s1, f1)
        f3, s3 = self.fmrconv3(s2, f2)
        f4, s4 = self.fmrconv4(s3, f3)
        f5, s5 = self.fmrconv5(s4, f4)
        x = f5.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, s3



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.cel = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred, target, trans_feat):
        total_loss = self.cel(pred, target)
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from MCGConv_PyTorch3d import MCGConv


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # framepoint = torch.tensor([[1.0,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,0]]).cuda()
        self.mcgconv1 = MCGConv(npoint=512, cin=3, cout=128, radius=0.15, nsample=32, m1=[7,32,1], m2=[1,64,128], v = 4, ball_query=False, use_normal=normal_channel)
        self.mcgconv2 = MCGConv(npoint=256, cin=128, cout=256, radius=0.25, nsample=32, m1=[7,32,1], m2=[1,64,256], v = 4, ball_query=False, use_normal=normal_channel)
        self.mcgconv3 = MCGConv(npoint=128, cin=256, cout=256, radius=0.4, nsample=32, m1=[7,64,1], m2=[1,128,256], v = 4, ball_query=False, use_normal=normal_channel)
        self.mcgconv4 = MCGConv(npoint=32, cin=256, cout=512, radius=0.6, nsample=32, m1=[7,64,1], m2=[1,128,512], v = 4, ball_query=False, use_normal=normal_channel)
        self.mcgconv5 = MCGConv(npoint=None, cin=512, cout=1024, radius=10, nsample=32, m1=[7,128,1], m2=[1,256,1024], v = 4, ball_query=False, use_normal=normal_channel)
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
            xyz = xyz.transpose(1, 2)
            norm = norm.transpose(1, 2)
        else:
            norm = None
            xyz = xyz.transpose(1, 2)
        f1, s1 = self.mcgconv1(xyz, norm)
        f2, s2 = self.mcgconv2(s1, f1)
        f3, s3 = self.mcgconv3(s2, f2)
        f4, s4 = self.mcgconv4(s3, f3)
        f5, s5 = self.mcgconv5(s4, f4)
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

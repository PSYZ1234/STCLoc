import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_util import sample_and_group, sample_and_group_all, similarity


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, normalize_radius=False, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        """
        npoint: keyponts number to sample
        radius: sphere radius in a group
        nsample: how many points to group for a sphere
        in_channel: input dimension
        mlp: a list for dimension changes
        normalize_radius: scale normalization
        group_all: wheather use group_all or not
        """
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.normalize_radius = normalize_radius
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel     

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points) 
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.normalize_radius)
        
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]  
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, D']

        return new_xyz, new_points


class FeatureCorrelation(nn.Module):
    def __init__(self, steps, feat_size):
        super(FeatureCorrelation, self).__init__()
        self.steps = steps       
        self.feat_size = feat_size
        self.pos_embedding = nn.Parameter(torch.randn(1, steps, feat_size))   

    def forward(self, feat_in):
        """
        TAFA
        Input:
            feat_in: tensor([B*T, D])
        Return:
            feat_out: tensor([B*T, D])
        """
        B                   = feat_in.size(0) // self.steps  # B
        feat_in             = feat_in.view(B, self.steps, self.feat_size)  # [B, T, D]
        feat_in             = feat_in + self.pos_embedding   # [B, T, D]
        feat1, feat2, feat3 = torch.split(feat_in, 1, dim=1)  # [B, 1, D]*2
        feat1_new           = similarity(feat1, feat2, feat3)  # [B, 1, D]
        feat2_new           = similarity(feat2, feat1, feat3)  # [B, 1, D]
        feat3_new           = similarity(feat3, feat1, feat2)  # [B, 1, D]         
        feat_out            = torch.cat((feat1_new, feat2_new, feat3_new), dim=1)  # (B, T, D)
        feat_out = feat_out.view(B*self.steps, self.feat_size)   # [B*T, D]

        return feat_out


class PCLocEncoder(nn.Module):
    def __init__(self, steps=2, feature_correlation=False):
        super(PCLocEncoder, self).__init__()
        # oxford
        self.sa1 = PointNetSetAbstraction(512,  4,    32,   3,       [32, 32, 64],     False, False)
        self.sa2 = PointNetSetAbstraction(128,  8,    16,   64 + 3,  [64, 128, 256],   False, False)

        # vreloc
        # self.sa1 = PointNetSetAbstraction(512,  0.2,    32,   3,       [32, 32, 64],     False, False)
        # self.sa2 = PointNetSetAbstraction(128,  0.4,    16,   64 + 3,  [64, 128, 256],   False, False)

        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], False, True)
        self.correlation = FeatureCorrelation(steps, 1024)
        self.feature_correlation = feature_correlation

    def forward(self, xyz):
        B                 = xyz.size(0)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points         = l3_points.view(B, -1)  # [B*T, D]

        if self.feature_correlation:
            l3_points     = self.correlation(l3_points)  # [B*T, D]  TAFA

        return l3_points


class PCLocDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PCLocDecoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(fc(x)))  # [B, D]
        
        return x
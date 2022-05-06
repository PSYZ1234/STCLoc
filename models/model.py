import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.module_util import PCLocEncoder, PCLocDecoder


class STCLoc(nn.Module):
    def __init__(self, steps, num_class_loc, num_class_ori):
        super(STCLoc, self).__init__()
        self.encoder         = PCLocEncoder(steps, True)
        self.regressor       = PCLocDecoder(1024, [1024, 1024, 1024])
        self.classifier_t    = PCLocDecoder(1024, [1024, 1024])  # SRML
        self.classifier_q    = PCLocDecoder(1024, [1024, 1024])  # SRML
        self.fc_position     = nn.Linear(1024, 3)
        self.fc_orientation  = nn.Linear(1024, 3)
        self.fc_cls_loc      = nn.Linear(1024, num_class_loc)
        self.fc_cls_ori      = nn.Linear(1024, num_class_ori)
        self.fc_finall_pose  = nn.Linear(1024, 1024)
        self.bn_finall_pose  = nn.BatchNorm1d(1024)

    def forward(self, pc):
        x        = self.encoder(pc)  # [B*T, D]
        y        = self.regressor(x)  # [B*T, D]
        loc      = self.classifier_t(x)  # [B*T, D]
        ori      = self.classifier_q(x)  # [B*T, D]
        loc_norm = F.normalize(loc, dim=1)  # [B*T, D]
        ori_norm = F.normalize(ori, dim=1)  # [B*T, D]
        z        = y * loc_norm * ori_norm  # [B*T, D]
        z        = F.relu(self.bn_finall_pose(self.fc_finall_pose(z)))  # [B*T, D]
        t        = self.fc_position(z)  # [B*T, 3]
        q        = self.fc_orientation(z)  # [B*T, 3]
        loc_cls  = self.fc_cls_loc(loc)  # [B*T, D]
        ori_cls  = self.fc_cls_ori(ori)  # [B*T, D]
        loc_cls  = F.log_softmax(loc_cls, dim=1)  # [B*T, D]
        ori_cls  = F.log_softmax(ori_cls, dim=1)  # [B*T, D]
        
        return t, q, loc_cls, ori_cls


if __name__ == '__main__':
    model = STCLoc(8, 4096, 3)
import torch
from torch import nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.t_loss_fn = nn.L1Loss()
        self.q_loss_fn = nn.L1Loss()
        self.loc_loss  = nn.NLLLoss()
        self.ori_loss  = nn.NLLLoss()

    def forward(self, pred_t, pred_q, pred_loc, pred_ori, gt_t, gt_q, gt_loc, gt_ori):
        loss_pose = 1 * self.t_loss_fn(pred_t, gt_t) + 10 * self.q_loss_fn(pred_q, gt_q) 
        loss_cls  = 1 * self.loc_loss(pred_loc, gt_loc.long()) + 1 * self.ori_loss(pred_ori, gt_ori.long())
        loss      = 1.5 * loss_pose + 1 * loss_cls

        return loss
import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logpt = self.ce(input, target)
        pt = torch.exp(-logpt)
        loss = (1 - pt) ** self.gamma * logpt
        return loss.mean()

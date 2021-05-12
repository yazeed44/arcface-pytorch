# References:
# https://github.com/deepinsight/insightface
import math

import torch
from torch.nn import Parameter, Module



class Arcface(Module):
    def __init__(self, embedding_size, num_of_classes, scalar=64., margin=0.5):
        super(Arcface, self).__init__()
        self.num_of_classes = num_of_classes
        self.margin = margin
        self.scalar = scalar
        self.margin_cos = math.cos(margin)
        self.margin_sin = math.sin(margin)
        self.mm = self.margin_sin * margin
        self.threshold = math.cos(math.pi - margin)
        self.kernel = Parameter(torch.Tensor(embedding_size, num_of_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, embbedings, label):
        normalized_kernel = torch.norm(self.kernel, 2, 1, True)
        normalized_kernel = torch.div(self.kernel, normalized_kernel)
        cos_theta = torch.mm(embbedings, normalized_kernel).clamp(-1, 1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        cos_theta_m = (cos_theta * self.margin_cos - sin_theta * self.margin_sin)
        condition_mask = (cos_theta - self.threshold) <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[condition_mask] = keep_val[condition_mask]
        output = cos_theta * 1.0
        indices = torch.arange(0, len(embbedings), dtype=torch.long)
        output[indices, label] = cos_theta_m[indices, label]
        output *= self.scalar
        return output

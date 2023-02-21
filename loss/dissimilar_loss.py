import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x = torch.nn.functional.normalize(x, dim=2, p=2)
    y = torch.nn.functional.normalize(y, dim=2, p=2)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(B, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(B, n, m).transpose(-2, -1)
    dist = xx + yy
    dist = dist - 2 * (x @ y.transpose(-2, -1))
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # return 1. / dist
    return dist
    # return -torch.log(dist)


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection/(x_norm * y_norm)
    return torch.abs(dist)

class Dissimilar(object):
    def __init__(self, dynamic_balancer=False):
        self.dynamic_balancer = dynamic_balancer
    
    def __call__(self, features):
        B, N, C = features.shape
        dist_mat = cosine_dist(features, features)  # B*N*N
        # dist_mat = euclidean_dist(features, features)
        # 上三角index
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]

        # 1.用softmax替换平均，使得相似度更高的权重更大
        if self.dynamic_balancer:
          weight = F.softmax(_dist, dim=-1)
          dist = torch.mean(torch.sum(weight*_dist, dim=1))
        # 2.直接平均
        else:
          dist = torch.mean(_dist, dim=(0, 1))
        return dist
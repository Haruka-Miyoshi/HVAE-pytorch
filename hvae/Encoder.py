import torch
from torch import nn
from torch.nn import functional as F

"""Encoder"""
class Encoder(nn.Module):
    """__init__"""
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(Encoder, self).__init__()
        self.x_dim = x_dim # 入力変数次元数
        self.h_dim = h_dim # 隠れ変数次元数
        self.z_dim = z_dim # 潜在変数次元数

        # Sequential Function
        self.nn = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU()
        )

        # mu Function
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        # logvar Function
        self.logvar = nn.Linear(self.h_dim, self.z_dim)

    """forward"""
    def forward(self, x):
        y = self.nn(x) # 入力 -> 隠れ
        mu = self.mu(y) # 隠れ -> 平均:潜在空間
        logvar = self.logvar(y) # 隠れ -> 対数分散:潜在空間
        return mu, logvar
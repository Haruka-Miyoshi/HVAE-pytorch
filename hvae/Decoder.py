import torch
from torch import nn
from torch.nn import functional as F

"""Decoder"""
class Decoder(nn.Module):
    """__init__"""
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(Decoder, self).__init__()
        self.x_dim=x_dim # 入力変数次元数
        self.h_dim = h_dim # 隠れ変数次元数
        self.z_dim=z_dim # 潜在変数次元数

        # Sequential Fucntion
        self.nn=nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.x_dim),
            nn.Sigmoid()
        )
    
    """forward"""
    def forward(self, z):
        x_hat = self.nn(z)
        return x_hat
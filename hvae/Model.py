import torch
from torch import nn
from torch.nn import functional as F
from .Decoder import Decoder
from .Encoder import Encoder

"""Model"""
class Model(nn.Module):
    """__init__"""
    def __init__(self, x_dim:int, h1_dim:int, z_dim:int, h2_dim:int, z_h_dim:int, mode=True):
        super(Model, self).__init__()
        self.x_dim = x_dim # 入力変数次元数
        self.h1_dim = h1_dim # 隠れ変数次元数1
        self.z_dim = z_dim # 潜在変数次元数
        self.h2_dim = h2_dim # 隠れ変数次元数2
        self.z_h_dim = z_h_dim # 階層的潜在変数次元数

        self.mode = mode # 学習モード

        self.encoder = Encoder(self.x_dim, self.h1_dim, self.z_dim) # Encoder Module
        self.encoder_h = Encoder(self.z_dim, self.h2_dim, self.z_h_dim) # Hierarchical Encoder Module
        self.decoder_h = Decoder(self.z_dim, self.h2_dim, self.z_h_dim) # Hierarchical Decoder Module
        self.decoder = Decoder(self.x_dim, self.h1_dim, self.z_dim) # Decoder Module

    """reparameterize"""
    def reparameterize(self, mu, logvar, mode):
        if mode:
            s = torch.exp(0.5 * logvar) # 標準偏差
            e = torch.rand_like(s) # 誤差e
            return e.mul(s).add_(mu) # e * std + mu
        else:
            return mu # mu
        
    """forward"""
    def forward(self, x, mode=True):
        mu, logvar = self.encoder(x) # mu, logvarを計算
        z = self.reparameterize(mu, logvar, mode) # reparameterize
        mu_h, logvar_h = self.encoder_h(z) # mu_h, logvar_hを計算
        z_h = self.reparameterize(mu_h, logvar_h, mode) # reparameterize
        zh = self.decoder_h(z_h) # zhを計算
        xh = self.decoder(zh) # xhを計算
        return mu, logvar, z, mu_h, logvar_h, z_h, zh, xh
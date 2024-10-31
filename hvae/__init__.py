import os
import sys
sys.path.append('../')

import torch
from torch import nn
from torch.nn import functional as F
from .Model import Model

"""HVAE"""
class HVAE(object):
    """__init__"""
    def __init__(self, x_dim:int, h1_dim:int, z_dim:int, h2_dim:int, zh_dim:int, lr:float=1e-3, train_mode:bool=True, save_path="param", model_path="hvae_parameter.path"):
        self.x_dim = x_dim # 入力変数次元数
        self.h1_dim = h1_dim # 隠れ変数次元数
        self.z_dim = z_dim # 潜在変数次元数
        self.h2_dim = h2_dim
        self.zh_dim = zh_dim

        self.save_path = save_path # パラメータ保存先
        self.model_path = model_path # パラメータファイル名
        self.path = os.path.join(self.save_path, self.model_path) # パス生成
        
        self.mode = train_mode # 学習モード
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ディバイス設定
        self.model = Model(self.x_dim, self.h1_dim, self.z_dim, self.h2_dim, self.zh_dim, self.mode).to(device=self.device) # Model

        if not os.path.exists(save_path):
            os.mkdir(self.save_path)

        if not self.mode:
            # パラメータファイルがない場合における処理を追加
            try:
                self.model.load_state_dict(torch.load(self.path))
            except:
                raise("Not Found model paramter file!!")
            

        self.lr = lr # 学習係数
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) # Optimizer
        self.losses = [] # 損失関数

    """tensor"""
    def tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    """numpy"""
    def numpy(self, x):
        if self.device == "cpu":
            return x.detach().numpy()
        else:
            return x.detach().cpu().numpy()

    """BCELoss"""
    def BCE(self, theta, x):
        loss = F.binary_cross_entropy(theta, x, size_average=False)
        return loss
    
    """KLDLoss"""
    def KLD(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss

    """learn"""
    def learn(self, train_loader, epoch:int=30):
        self.model.train() # 学習モード
        for e in range(1, epoch+1):
            total_loss = 0.0
            for i, ( batch_data, batch_label ) in enumerate(train_loader):
                self.optim.zero_grad()
                x = batch_data.view(-1, self.x_dim).to(device=self.device)
                mu, logvar, z, mu_h, logvar_h, z_h, zh, xh = self.model(x)
                loss = self.BCE(xh, x) + self.KLD(mu_h, logvar_h) + self.KLD((mu-zh), logvar)
                loss.backward()
                total_loss += loss.item()
                self.optim.step()
            total_loss /= i
            self.losses.append(total_loss)
            print(f"epoch:{e}, loss:{total_loss}")

        if self.mode:
            torch.save(self.model.state_dict(), self.path)

    """x_to_z_to_xh"""
    def x_to_z_to_xh(self, x):
        self.model.eval() # 推論モード
        x = x.view(-1, self.x_dim).to(device=self.device)
        with torch.no_grad():
            mu, logvar, z, mu_h, logvar_h, z_h, zh, xh = self.model(x)
        return mu, logvar, z, mu_h, logvar_h, z_h, zh, xh
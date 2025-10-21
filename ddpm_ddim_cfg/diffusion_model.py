import torch.nn as nn
from module.basic_module import *



## timestep t -> 삼각함수 주파수 임베딩→ mlp 투영 → hidden_size 차원의 벡터

class TimeEmbedding(nn.Module):
    def __init__(self,f):
        super().__init__()
        
        self.module = nn.Sequential(
            nn.SiLU()
            
        )
        
         
    def forward(self,x):
        



## 기본 틀은 VAE에서 사용한 Unet과 매우 유사하게 진행.

class DiffusionUnet(nn.Module):
    def __init__(self,):
        super().__init__()
        






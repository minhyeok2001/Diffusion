import torch 
import torch.nn as nn
import torch.nn.functional as F

import math

## timestep t -> 삼각함수 주파수 임베딩 → mlp 투영 → hidden_size 차원의 벡터
# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py 참고.

class TimeEmbedding(nn.Module):
    def __init__(self,hidden_size=128, frequency_embedding_size = 128):
        # dim은 output dim
        super().__init__()
        
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
    
    def timestep_Embedding(self, timestep, dim, max_period = 10000):
        ## timestep은 [N] 꼴로 입력을 받음 
        if dim %2 != 0 :
            raise RuntimeError("DIM 2로 안나눠짐 -> cos sin embedding 불균형")
        device = timestep.device
        half = dim //2 # 반으로 나눠서 cos / sin 사용 
        #  exp ( -log(10000) * 0/half, -log(10000) * 1/half, -log(10000) * 2/half , ... )
        freqs = torch.exp(-math.log(max_period)* torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device)     ## 그러니까 이게 timestep에 의존적인게 아니라, 세로로.. 즉 dim 축을 구성하기 위한 frequency
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # [N, dim]
        return embedding         
    
    def forward(self,timestep):
        ## timestep 넣으면 embedding으로 치환됨
        if timestep.ndim == 0:
            timestep= timestep.unsqueeze(-1)
        t_freq = self.timestep_Embedding(timestep, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb    


class ClassEmbedding(nn.Module):
    def __init__(self,num_cls=4,hidden_dim=128):    ## 0 인덱스를 고려하여 4개. cfg를 위해..
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_cls,embedding_dim=hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        
    def forward(self,x):
        x = self.embedding(x)
        x = self.mlp(x)
        return x
        

class ResnetBlock2D(nn.Module):
    def __init__(self,c_in,c_out,shortcut=False,time_embedding=False, cfg = False, emb_dim= 128):
        super().__init__()
        
        self.module = nn.Sequential(
            nn.GroupNorm(32,c_in),
            nn.Conv2d(c_in,c_out,kernel_size=3,padding=1),
            nn.SiLU(),
            nn.GroupNorm(32,c_out),
            nn.Conv2d(c_out,c_out,kernel_size=3,padding=1),
            nn.SiLU()
        )
        self.shortcut = shortcut
        
        if self.shortcut :
            if c_in != c_out :
                self.shortcut_module = nn.Conv2d(c_in,c_out,kernel_size=3,padding=1)    
            else : 
                self.shortcut_module = nn.Identity()
                
        if time_embedding :
            self.handling_embedding_dim = nn.Linear(emb_dim,c_in)    

    def forward(self,x,embedding=None):
        origin = x
        if embedding is not None:
            emb = self.handling_embedding_dim(embedding) ## [B, C]
            x += emb[:,:,None,None]
            
        x = self.module(x)
        if self.shortcut:
            temp = self.shortcut_module(origin)
            x += temp
        
        return x 
        

## UP DOWN 둘다 이걸로 적용
class Sample2D(nn.Module):  
    def __init__(self,c,type):
        super().__init__()
        if type == "upsampling":
            self.module = nn.Conv2d(c,c,kernel_size=3,padding=1,stride=1)
        elif type == "downsampling":
            self.module = nn.Conv2d(c,c,kernel_size=3,padding=1,stride=2)
        else :
            raise RuntimeError("Invalid up/down sampling type !!")
                
        self._type = type
        
    def forward(self,x):
        if self._type =="upsampling" :
            x = F.interpolate(x, scale_factor=2, mode="nearest")  
        x = self.module(x)
        return x
        
        
class Attention(nn.Module):
    def __init__(self,c_hidden):
        super().__init__()
        
        ## 여기에는 입력 512 채널 기준으로, 512//8 = 64 이므로 C_in * 64 * 64가 만들어짐
        self.groupnorm = nn.GroupNorm(32,c_hidden)
        self.to_k = nn.Linear(c_hidden,c_hidden)
        self.to_q = nn.Linear(c_hidden,c_hidden)
        self.to_v = nn.Linear(c_hidden,c_hidden)
        self.mlp = nn.Linear(c_hidden,c_hidden)

    def forward(self,x):
        B,C,H,W = x.shape
        
        identity = x
        x = self.groupnorm(x)
        temp = x.permute(0,3,1,2).reshape(B,H*W,C)
        key = self.to_k(temp)
        query = self.to_q(temp)
        value = self.to_v(temp)
        
        scale = math.sqrt(key.shape[-1])
        
        attention_score = torch.softmax(query@key.transpose(1,2)/scale,dim=-1)    ## 가로로 더해서 softmax 하므로..
        x = attention_score @ value
        
        x = self.mlp(x)
        
        x = x.reshape(B,H,W,C).permute(0,3,1,2)
        
        x += identity
        return x
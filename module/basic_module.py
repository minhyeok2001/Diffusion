import torch 
import torch.nn as nn

class ResnetBlock2D(nn.Module):
    def __init__(self,c_in,c_out,shortcut=False):
        super().__init__()
        
        self.module = nn.Sequential(
            nn.GroupNorm(32,c_in),
            nn.Conv2d(c_in,c_out,kernel_size=3,padding=1),
            nn.GroupNorm(32,c_out),
            nn.Conv2d(c_out,c_out,kernel_size=3,padding=1),
            nn.SiLU()
        )
        self.shortcut = shortcut
        
        if self.shortcut :
            self.shortcut_module = nn.Conv2d(c_in,c_out,kernel_size=3,padding=1)    
    
    def forward(self,x):
        origin = x
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
        
    def forward(self,x):
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
        x = self.groupnorm(x)
        
        temp = x.permute(0,3,1,2).reshape(B,H*W,C)
        key = self.to_k(temp)
        query = self.to_q(temp)
        value = self.to_v(temp)
        
        attention_score = torch.softmax(query@key.transpose(1,2),dim=-1)    ## 가로로 더해서 softmax 하므로..
        x = attention_score @ value
        
        x = self.mlp(x)
        
        x = x.reshape(B,H,W,C).permute(0,3,1,2)
        return x
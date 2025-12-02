## 이거 처음부터 짜자니까 너무 Redundant하다고 느낌. 이참에 class 덮어씌우기 연습해보기 + attention 짜면서 복습한번 고고
## 1. 기존 DDPM에서 사용한 Unet을 가져온다
## 2. Attention block 값 뒤에다가 Cross attention block 값도 덮어씌우기

import math
import torch
import torch.nn as nn
from ddpm_ddim_cfg.diffusion_model import DiffusionUnet
from module.basic_module import Attention

def get_parent_module(model: nn.Module, module_name: str):
    ## 해당 모듈의 str형식말고 쓸 수 있는 형식을 찾아내주는 함수
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:  
        parent = getattr(parent, p) ## parent = model.unet.down[0].resblock[1] -> 이거 원래 module_name으로 호출하면 문자열로 되어있는데, 이거를 이렇게 바꿔줌
    return parent, parts[-1]   ## 이렇게 하면 parent는 마지막 레이어 딱 직전까지 나옴

            
def inject_Attention2CrossAttention(model: nn.Module,clip_dim=768,target_class=Attention):
    ## 부모모듈 탐색하면서 isinstance로 이름이 Attention인거 찾아서 뒤에다가 주입
    
    modules = list(model.named_modules())
    
    for name, module in modules:
        if not isinstance(module,target_class):
            pass
        else :
            parent, child_name = get_parent_module(model, name)
            new_instance = Attention2CrossAttention(module,clip_dim)
            setattr(parent,child_name,new_instance) ## 이게 의미적으로, parent.child_name = new_instance 이거랑 같음 !!
    

class CrossAttention(Attention):
    def __init__(self,c_hidden,clip_dim,num_head=2):
        super().__init__(c_hidden=c_hidden,num_head=num_head)
        
        ## cross attention에서는 query만 실제 타겟이어야하죠~
        ## 이떄 clip 결과는 B,token_len,dim 이런식으로 들어옴
        
        ## 얘네만 오버라이드 고고 끝 dim은 맞춰야하니까~. clip은 768을 받음 
        self.to_k = nn.Linear(clip_dim,c_hidden)
        self.to_v = nn.Linear(clip_dim,c_hidden)
        
    def forward(self, x, clip):
        
        identity = x
        
        x = self.groupnorm(x)
        ## 1. 만약 clip이 들어왔을떄 우선 cross att를 위한 dim이 맞는지 한번 학인하기 B L D -> 여기서는 D가 hidden_dim 부분이겠죠
        ## 지금 x의 dim은 어떻지? -> B C H W 꼴. -> 여기서 C가 hidden_dim 부분이겟죠 

        B,C,H,W = x.shape
        Bc,L,D = clip.shape
        
        assert B==Bc, "hidden 채널하고 CLIP하고 배치가 달라요~~~~~"
        
        ## 2. 우선 각각 파라미터 만들기 
        ## 내가 기대하는 x는 B H*W C 꼴임

        x = x.view(B,C,-1).permute(0,2,1)    # 어차피 얘는 무조건 contiguous해서 ㄱㄴ
        query = self.to_q(x)
        key = self.to_k(clip)
        value = self.to_v(clip)
        
        # 지금기준으로 query B HW C, key B L D
        # 이때 head는 배치 다음으로 오셔야함. 그래야 그 HW랑 L간의 att가 생기지 !! 
        query = query.view(B,H*W,self.num_head,-1).transpose(1,2)
        key = key.view(B,L,self.num_head,-1).transpose(1,2)
        value = value.view(B,L,self.num_head,-1).transpose(1,2)
    
        attention_score = torch.softmax(query@key.transpose(-1,-2)/ math.sqrt(key.shape[-1]),dim=-1)
        attention = attention_score @ value
        
        ## 지금 attention의 차원은 B num_head HW C//head개수
        attention = attention.transpose(1,2).reshape(B,H,W,-1)
        ## 이렇게하면 B HW C 
        x = self.mlp(attention)
        
        x = x.permute(0,3,1,2)
        #print(x.shape)
        #print(identity.shape)
        x += identity
        
        return x 
    
"""
LEGACY version...
class StableDiffusionUnet(DiffusionUnet):
    ## 내생각에는, forward부분에서도 super().forward() 이거 부른다음에 그 뒤에 바로 붙이면 될거같은데 ?? hasattr이나 isinstance 이거 써가지고 !!
    def __init__(self, channels = [128,256,512], clip_dim = 768, cfg=False, cfg_dropout=0.2):
        super().__init__(channels, cfg, cfg_dropout)
        self.parent_att = [m for m in self.modules() if isinstance(m,Attention)]
        self.att_len = len(self.parent_att)
        self.cross_att_list = nn.ModuleList([])

        for i in range(self.att_len):
            self.cross_att_list.append(CrossAttention(self.parent_att[i].c_hidden,clip_dim=clip_dim))
        
        
    def forward(self,x,t,clip,cls=None):
        parent_result = super().forward(x,t,cls)
"""
        
class Attention2CrossAttention(nn.Module):
    def __init__(self, base_att : Attention, clip_dim):
        super().__init__()
        self.base_att = base_att
        self.cross_att = CrossAttention(base_att.c_hidden,clip_dim)
        
        self.clip = None
        
    def set_clip(self,clip):
        self.clip = clip
    
    def forward(self,x):
        x = self.base_att(x)
        if self.clip is None:
            raise RuntimeError("NO CLIP INJECTION!!!")
        else : 
            x = self.cross_att(x,self.clip)
        
        return x
        
class StableDiffusionUnet(DiffusionUnet):
    ## 기본 diffusion은 B C H W 꼴이 들어왔는데, VAE를 거친거는 아마 음.. B C H W 똑같네 ㅇㅇ 
    def __init__(self, channels=[128,256,512], clip_dim=768, cfg=True, cfg_dropout=0.2):
        super().__init__(channels, cfg, cfg_dropout)
        ## 전부다 만든 att로 갈아끼우고
        inject_Attention2CrossAttention(self, clip_dim=clip_dim)

    def set_clip(self, clip):
        for m in self.modules():
            if isinstance(m, Attention2CrossAttention):
                m.set_clip(clip)

    def forward(self, x, t, clip=None, cls=None):
        self.set_clip(clip)

        return super().forward(x, t, cls)


def test_cross_att():
    cross_att = CrossAttention(768,20,2)

    img = torch.randn(3,768,32,32)
    clip = torch.randn(3,16,20)

    print(cross_att(img,clip).shape)    
    

def test_model():
    hola = StableDiffusionUnet()
    img = torch.randn(3,3,64,64)
    clip = torch.randn(3,16,768)
    
    t = torch.randint(0,1000,(3,))
    cls = torch.full((3,),1)
    
    print(hola(img,t,clip,cls).shape)
    

test_model()
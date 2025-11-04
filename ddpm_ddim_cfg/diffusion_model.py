import torch.nn as nn
import torch.nn.functional as F
from module.basic_module import *



class UnetDown(nn.Module):
    def __init__(self,channels,cfg):
        super().__init__()
        
        assert len(channels) == 3 , "Check length of channel list !!"
        assert all(channels[i] % 32 == 0 for i in range(3)), "Each channel must be multiple of 32 !! "
        
        self.conv1 = nn.Conv2d(3,channels[0],kernel_size=3,padding=1,stride=1)
        
        ## 0,0 2개
        ## 1,1 2개
        ## 2,2 3개
        
        list_resblock_params_change = [
            {"c_in": channels[0], "c_out" :channels[1],"shortcut" : True,"time_embedding" : True,"cfg" : cfg},
            {"c_in": channels[1], "c_out" :channels[2],"shortcut" : True,"time_embedding" : True,"cfg" : cfg},
        ]
        
        list_resblock_params_00 = [
            {"c_in": channels[0], "c_out" :channels[0],"shortcut" : True,"time_embedding" : True,"cfg" : cfg} for i in range(2)
        ] 
            
        list_resblock_params_11 = [
            {"c_in": channels[1], "c_out" :channels[1],"shortcut" : True,"time_embedding" : True,"cfg" : cfg} for i in range(2)
        ]
        
        list_resblock_params_22 = [
            {"c_in": channels[2], "c_out" :channels[2],"shortcut" : True,"time_embedding" : True,"cfg" : cfg} for i in range(3)
        ]
        
        list_sample_params = [
            {"c" : channels[0], "type" : "downsampling"},
            {"c" : channels[1], "type" : "downsampling"},
            #{"c" : channels[2], "type" : "downsampling"},
        ]

        DownBlock = []

        for params in list_resblock_params_00:
            DownBlock.append(ResnetBlock2D(**params))
        
        DownBlock.append(Sample2D(**list_sample_params[0]))
        DownBlock.append(ResnetBlock2D(**list_resblock_params_change[0]))
        
        for params in list_resblock_params_11:
            DownBlock.append(ResnetBlock2D(**params))
        
        DownBlock.append(Sample2D(**list_sample_params[1]))    
        DownBlock.append(ResnetBlock2D(**list_resblock_params_change[1]))
        
        for params in list_resblock_params_22:
            DownBlock.append(ResnetBlock2D(**params))
        
        #DownBlock.append(Sample2D(**list_sample_params[2]))
        
        self.layers = nn.ModuleList(DownBlock)

        
    def forward(self,x,time=None):    
        x = self.conv1(x)
        skip_connection_list = []
        for layer in self.layers :
            if isinstance(layer,Sample2D):
                skip_connection_list.append(x)
                
            if (time is not None) and isinstance(layer,ResnetBlock2D):
                x = layer(x,time)
            else :
                x = layer(x)
        return x ,skip_connection_list
        
        
class UnetMid(nn.Module):
    def __init__(self,channel,cfg):
        super().__init__()
        
        list_resblock_params = [
            {"c_in": channel, "c_out" :channel,"shortcut" : True,"time_embedding" : True,"cfg" : cfg},
        ]
        
        MidBlock = []
        
        for i in range(3) :
            MidBlock.append(Attention(channel)) 
            MidBlock.append(ResnetBlock2D(**list_resblock_params[0]) ) 
            MidBlock.append(ResnetBlock2D(**list_resblock_params[0])) 
        
        self.layers = nn.ModuleList(MidBlock)
    
    def forward(self,x,time=None):        
        for layer in self.layers :
            if (time is not None) and isinstance(layer,ResnetBlock2D):
                x = layer(x,time)
            else :
                x = layer(x)
        return x 
        
        
class UnetUp(nn.Module):
    def __init__(self,channels,cfg):
        super().__init__()
        
        assert len(channels) == 3 , "Check length of channel list !!"
        assert all(channels[i] % 32 == 0 for i in range(3)), "Each channel must be multiple of 32 !! "
        
        self.conv1 = nn.Conv2d(in_channels=channels[0],out_channels=channels[0],kernel_size=3,stride=1,padding=1)
        
        
        list_resblock_params_change = [
            {"c_in": channels[0], "c_out" :channels[1],"shortcut" : True,"time_embedding" : True,"cfg" : cfg},
            {"c_in": channels[1], "c_out" :channels[2],"shortcut" : True,"time_embedding" : True,"cfg" : cfg},
        ]
        
        list_resblock_params_00 = [
            {"c_in": channels[0], "c_out" :channels[0],"shortcut" : True,"time_embedding" : True,"cfg" : cfg} for i in range(3)
        ] 
            
        list_resblock_params_11 = [
            {"c_in": channels[1], "c_out" :channels[1],"shortcut" : True,"time_embedding" : True,"cfg" : cfg} for i in range(2)
        ]
        
        list_resblock_params_22 = [
            {"c_in": channels[2], "c_out" :channels[2],"shortcut" : True,"time_embedding" : True,"cfg" : cfg} for i in range(2)
        ]
        
        list_sample_params = [
            {"c" : channels[0], "type" : "upsampling"},
            {"c" : channels[1], "type" : "upsampling"},
            #{"c" : channels[2], "type" : "upsampling"},
        ]

        
        UpBlock = []
        
        for params in list_resblock_params_00:
            UpBlock.append(ResnetBlock2D(**params))
            
        UpBlock.append(Sample2D(**list_sample_params[0]))
        UpBlock.append(ResnetBlock2D(**list_resblock_params_change[0]))
        
        for params in list_resblock_params_11:
            UpBlock.append(ResnetBlock2D(**params))
            
        UpBlock.append(Sample2D(**list_sample_params[1]))    
        UpBlock.append(ResnetBlock2D(**list_resblock_params_change[1]))
        
        for params in list_resblock_params_22:
            UpBlock.append(ResnetBlock2D(**params))
            
        #UpBlock.append(Sample2D(**list_sample_params[2]))
             
        self.layers = nn.ModuleList(UpBlock)
        self.conv2 = nn.Conv2d(in_channels=channels[-1],out_channels=3,kernel_size=3,stride=1,padding=1)
        
        self.fusion1 = nn.Conv2d(in_channels=2*channels[1],out_channels=channels[1],kernel_size=3,stride=1,padding=1)
        self.fusion2 = nn.Conv2d(in_channels=2*channels[2],out_channels=channels[2],kernel_size=3,stride=1,padding=1)
        
        # fusion 과정은 sample2 의 다음 resblock이 끝난 후에 수행 
            
    def forward(self,x,time=None,skip_connection_list=None):    
        assert skip_connection_list is not None, "NO SKIP CONNECTION LIST !!"
        x = self.conv1(x)
        check_flag = False
        fusion1_flag = False
        timing_flag = False
        
        for layer in self.layers :
            if check_flag and timing_flag:
                x = layer(x)
                #print("x.shape",x.shape)
                skip = skip_connection_list.pop()
                x = torch.cat([x, skip], dim=1)
                if fusion1_flag :
                    x = self.fusion2(x)
                else :
                    x = self.fusion1(x)
                    fusion1_flag = True
                    
                check_flag = False
                timing_flag = False
                
            if check_flag :
                timing_flag = True
            if isinstance(layer, Sample2D):
                check_flag = True
                
            if (time is not None) and isinstance(layer,ResnetBlock2D):
                #print(x.shape)
                #print(time.shape)
                x = layer(x,time)
            else :
                x = layer(x)
        x = self.conv2(x)
        return x 

## 기본 틀은 VAE에서 사용한 Unet과 매우 유사하게 진행.
class DiffusionUnet(nn.Module):
    def __init__(self,channels : list = [128,256,512],cfg=False, cfg_dropout=0.2):
        super().__init__()

        self.down = UnetDown(channels,cfg)
        self.mid = UnetMid(channels[-1],cfg)
        self.up = UnetUp(channels[::-1],cfg)
        
        ## BASIC BLOCK의 dim handling을 위해서, hidden_size는 128로 고정.
        self.time_embedding = TimeEmbedding(hidden_size=128,frequency_embedding_size=128)
        self.cfg = cfg
        self.cfg_dropout = cfg_dropout
        if cfg :
            self.cls_embedding = ClassEmbedding(num_cls=4,hidden_dim=128)
        
    def forward(self,x,t,cls=None):  ## 여기에는 single timestep이 B 차원으로 들어감
        emb = self.time_embedding(t) ## [B,dim] 
        if self.cfg and cls is not None:
            if self.training :  ## 파이토치에서 자동으로 제공해주는 training 여부 flag 
                mask = torch.rand(x.shape[0],device=x.device) < self.cfg_dropout
                cls_emb = cls.clone()
                cls_emb[mask] = 0
                cls = cls_emb
            emb += self.cls_embedding(cls) ## [B,dim]  
        
        x ,skip_connection = self.down(x,emb)
        x = self.mid(x,emb)
        x = self.up(x,emb,skip_connection)
        return x 
import torch.nn as nn
from module.basic_module import *

class UnetDown(nn.Module):
    def __init__(self,channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,channels[0],kernel_size=3,stride=1,padding=1)
        
        assert len(channels) == 3 , "Check length of channel list !!"
        assert all(channels[i] % 32 == 0 for i in range(3)), "Each channel must be multiple of 32 !! "
        
        self.conv1 = nn.Conv2d(3,channels[0],kernel_size=3,padding=1,stride=1)
        
        list_resblock_params = [
            {"c_in": channels[0], "c_out" :channels[0]},
            {"c_in": channels[0], "c_out" :channels[0]},
            
            {"c_in": channels[0], "c_out" :channels[1],"shortcut" : True},
            {"c_in": channels[1], "c_out" :channels[1]},
            
            {"c_in": channels[1], "c_out" :channels[2],"shortcut" : True},
            {"c_in": channels[2], "c_out" :channels[2]},
            
            {"c_in": channels[2], "c_out" :channels[2],"shortcut" : True},
            {"c_in": channels[2], "c_out" :channels[2]},
            ]
        
        list_sample_params = [
            {"c" : channels[0], "type" : "downsampling"},
            {"c" : channels[1], "type" : "downsampling"},
            {"c" : channels[2], "type" : "downsampling"},
        ]

        DownEncoderBlock = []
        for i in range(len(list_sample_params)):
            DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[2*i]))
            DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[2*i+1]))
            DownEncoderBlock.append(Sample2D(**list_sample_params[i]))
        
        DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[-2]))
        DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[-1]))
        
        self.DownEncoderBlocks = nn.Sequential(*DownEncoderBlock)
    
        self.layers = nn.ModuleList([*self.DownEncoderBlocks])
        
    def forward(self,x,time=None):        
        x = self.conv1(x)
        for layer in self.layers :
            if (time is not None) and isinstance(layer,ResnetBlock2D):
                x = layer(x,time)
            else :
                x = layer(x)
        return x 
        
        
class UnetMid(nn.Module):
    def __init__(self,channel):
        super().__init__()
        
        list_resblock_params = [
            {"c_in": channel, "c_out" :channel,"shortcut" : True}
        ]
        
        MidBlock = []
        
        for i in range(3) :
            MidBlock.append(Attention(channel)) 
            MidBlock.append(ResnetBlock2D(**list_resblock_params[0]) ) 
            MidBlock.append(ResnetBlock2D(**list_resblock_params[0])) 
            
        self.Midblocks = nn.Sequential(*MidBlock)
        
        self.layers = nn.ModuleList([*self.Midblocks])
    
    def forward(self,x,time=None):        
        for layer in self.layers :
            if (time is not None) and isinstance(layer,ResnetBlock2D):
                x = layer(x,time)
            else :
                x = layer(x)
        return x 
        
        
class UnetUp(nn.Module):
    def __init__(self,channels):
        super().__init__()
        
        assert len(channels) == 3 , "Check length of channel list !!"
        assert all(channels[i] % 32 == 0 for i in range(3)), "Each channel must be multiple of 32 !! "
        
        self.conv1 = nn.Conv2d(in_channels=channels[0],out_channels=channels[0],kernel_size=3,stride=1,padding=1)
        
        list_resblock_params = [
            
            {"c_in": channels[0], "c_out" :channels[0]},
            {"c_in": channels[0], "c_out" :channels[0]},
            
            {"c_in": channels[0], "c_out" :channels[0],"shortcut" : True},
            {"c_in": channels[0], "c_out" :channels[0]},
            {"c_in": channels[0], "c_out" :channels[0]},
            
            {"c_in": channels[0], "c_out" :channels[1],"shortcut" : True},
            {"c_in": channels[1], "c_out" :channels[1]},
            {"c_in": channels[1], "c_out" :channels[1]},
            
            {"c_in": channels[1], "c_out" :channels[2],"shortcut" : True},
            {"c_in": channels[2], "c_out" :channels[2]},
            {"c_in": channels[2], "c_out" :channels[2]},
        ]
        
        list_sample_params = [
            {"c" : channels[0], "type" : "upsampling"},
            {"c" : channels[1], "type" : "upsampling"},
            {"c" : channels[2], "type" : "upsampling"},
        ]

        
        UpDecoderBlock = []
        for i in range(len(list_sample_params)):
            UpDecoderBlock.append(ResnetBlock2D(**list_resblock_params[3*i+2]))
            UpDecoderBlock.append(ResnetBlock2D(**list_resblock_params[3*i+3]))
            UpDecoderBlock.append(ResnetBlock2D(**list_resblock_params[3*i+4]))
            UpDecoderBlock.append(Sample2D(**list_sample_params[i]))
        
        self.UpDecoderBlocks = nn.Sequential(*UpDecoderBlock)
        
        self.layers = nn.ModuleList([*self.UpDecoderBlocks])
        
            
    def forward(self,x,time=None):        
        x = self.conv1(x)
        for layer in self.layers :
            if (time is not None) and isinstance(layer,ResnetBlock2D):
                x = layer(x,time)
            else :
                x = layer(x)
        return x 
        
             
## 기본 틀은 VAE에서 사용한 Unet과 매우 유사하게 진행.
class DiffusionUnet(nn.Module):
    def __init__(self,channels : list = [128,256,512]):
        super().__init__()

        self.down = UnetDown(channels)
        self.mid = UnetMid(channels[-1])
        self.up = UnetUp(channels[::-1])
        
    def forward(self,x,time):
        x = self.down(x,time)
        x = self.mid(x,time)
        x = self.up(x,time)
        return x 






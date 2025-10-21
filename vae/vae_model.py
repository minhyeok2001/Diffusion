from module.basic_module import *
import torch.nn as nn

#device = "cuda"

## [64,128,256] for default 
class VaeEncoder(nn.Module):
    def __init__(self,channels : list = [64,128,256]):
        super().__init__()
        
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
            
            {"c_in": channels[2], "c_out" :channels[2]},
            {"c_in": channels[2], "c_out" :channels[2]},
            ]
        
        list_sample_params = [
            {"c" : channels[0], "type" : "downsampling"},
            {"c" : channels[1], "type" : "downsampling"},
            {"c" : channels[2], "type" : "downsampling"},
        ]
        
        
        ## 1. DownBlocks
        
        DownEncoderBlock = []
        for i in range(len(list_sample_params)):
            DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[2*i]))
            DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[2*i+1]))
            DownEncoderBlock.append(Sample2D(**list_sample_params[i]))
        
        DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[-2]))
        DownEncoderBlock.append(ResnetBlock2D(**list_resblock_params[-1]))
        
        self.DownEncoderBlocks = nn.Sequential(*DownEncoderBlock)
        
        ## 2. MidBlocks
                
        self.MidBlocks = nn.Sequential(
            Attention(c_hidden=channels[2]),
            ResnetBlock2D(**list_resblock_params[-2]),
            ResnetBlock2D(**list_resblock_params[-1])
        )
        
        ## 3. FinalBlocks
        self.FinalBlocks = nn.Sequential(
            nn.GroupNorm(32,channels[2]),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels[2],out_channels=8,kernel_size=3,stride=1,padding=1)
        )

        
    def forward(self,x):
        """
        1. conv 3 -> 128
        2. resblock 2개
        3. downsample

        128 245 512
        2-3번 3번 반복 후 midLayer( attention ) 거치고 8채널로 축소
        """
        
        x = self.conv1(x)
        x = self.DownEncoderBlocks(x)
        x = self.MidBlocks(x)
        x = self.FinalBlocks(x)
        return x 
    
    
class LatentHandler(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
        #self.conv2 = nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
    
    def reparameterization(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def forward(self,x):
        #x = self.conv1(x)      ## Mu, Sigma concat result
        mu = x[:,:4]
        sigma = x[:,4:]
        z = self.reparameterization(mu,sigma)
        #z = self.conv2(z)  -> 허깅페이스에는 있는데 도대체 이건 왜 있는건지 이해가 안가서 pass
        return z, mu, sigma
        

class VaeDecoder(nn.Module):
    def __init__(self,channels : list):
        super().__init__()
        
        assert len(channels) == 3 , "Check length of channel list !!"
        assert all(channels[i] % 32 == 0 for i in range(3)), "Each channel must be multiple of 32 !! "
        
        self.conv1 = nn.Conv2d(in_channels=4,out_channels=channels[0],kernel_size=3,stride=1,padding=1)
        
        list_resblock_params = [
            
            {"c_in": channels[0], "c_out" :channels[0]},
            {"c_in": channels[0], "c_out" :channels[0]},
            
            {"c_in": channels[0], "c_out" :channels[0]},
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

        ## 1. UpBlocks

        UpDecoderBlock = []
        for i in range(len(list_sample_params)):
            UpDecoderBlock.append(ResnetBlock2D(**list_resblock_params[3*i+2]))
            UpDecoderBlock.append(ResnetBlock2D(**list_resblock_params[3*i+3]))
            UpDecoderBlock.append(ResnetBlock2D(**list_resblock_params[3*i+4]))
            UpDecoderBlock.append(Sample2D(**list_sample_params[i]))
        
        self.UpDecoderBlocks = nn.Sequential(*UpDecoderBlock)
             
        ## 2. MidBlocks
        
        self.MidBlocks = nn.Sequential(
            Attention(c_hidden=channels[0]),
            ResnetBlock2D(**list_resblock_params[0]),
            ResnetBlock2D(**list_resblock_params[1])
        )
            
        ## 3. FinalBlocks
        
        self.FinalBlocks = nn.Sequential(
            nn.GroupNorm(32,channels[-1]),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels[-1],out_channels=3,kernel_size=3,stride=1,padding=1)
        )
            
    def forward(self,x):
        x = self.conv1(x)
        x = self.MidBlocks(x)
        x = self.UpDecoderBlocks(x)
        x = self.FinalBlocks(x)
        return x
    
    
class VAE(nn.Module):
    def __init__(self,channels :list = [64,128,256]):
        super().__init__()

        self.encoder = VaeEncoder(channels=channels)
        self.latent_handler = LatentHandler()
        self.decoder = VaeDecoder(channels=channels[: : -1])
        
    def forward(self,x):
        x = self.encoder(x)
        z,mu,sigma = self.latent_handler(x)
        z = self.decoder(z)
        return z, mu, sigma
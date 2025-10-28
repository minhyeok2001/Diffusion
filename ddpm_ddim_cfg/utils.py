import torch
import torch.nn as nn

class BaseScheduler(nn.Module):
    def __init__(self,inference_step):
        super().__init__()
        ## 스케줄러에는 시간을 넣으면 해당 시점에서의 cumprod들이 나와줘야하는데
        beta = torch.linspace(1e-4,2e-2,inference_step) ## 공식 논문 베타 값 기준
        alpha = 1-beta
        cumprod_alpha = torch.cumprod(alpha)
        
        ## register buffer 활용하여, 이후 체크포인트에서도 사용 가능하게
        self.register_buffer("alpha",alpha)
        self.register_buffer("cumprod_alpha",cumprod_alpha)
        
    def teeth(self,const,t):
        ## timestep이랑 alpha, cumprod_alpha같은거 넣으면 거기에 맞는거 뽑아주는 함수
        ## const : [ linspace 한거만큼의 dim ],  t : [ B ]
        const = const.to(t.device)
        return torch.gather(const,t).reshape(-1,1,1,1)



class DDPMScheduler(nn.Module):
    def __init__(self,):
        
    def tweedie():
    

    def forward_process(t,x_0,eps=None):
        """
        <add noise 과정>
        timestep , x_0, eps  필요
        """
        x_t = x_0 
        return 
    
    def reverse_process():
        


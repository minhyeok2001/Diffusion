import torch
import torch.nn as nn
import random

class BaseScheduler(nn.Module):
    def __init__(self,inference_step,device,num_timestep=1000):
        super().__init__()
        ## 스케줄러에는 시간을 넣으면 해당 시점에서의 cumprod들이 나와줘야하는데
        ## timesteps = torch.arange(num_timestep,0,-1) -> 이렇게 해버리면, DDIM에서는 재활용하기 힘들 수 있을 것 같은데? 그치만 일단 진행
        timesteps = torch.arange(num_timestep,0,-1,device=device)
        beta = torch.linspace(1e-4,2e-2,inference_step,device=device) ## 공식 논문 베타 값 기준
        alpha = 1-beta
        cumprod_alpha = torch.cumprod(alpha,-1)
        
        ## register buffer 활용하여, 이후 체크포인트에서도 사용 가능하게
        self.register_buffer("timesteps",timesteps)
        self.register_buffer("alpha",alpha)
        self.register_buffer("cumprod_alpha",cumprod_alpha)
        
    def teeth(self,const,t):
        ## timestep이랑 alpha, cumprod_alpha같은거 넣으면 거기에 맞는거 뽑아주는 함수
        ## const : [ linspace 한거만큼의 dim ],  t : [ B ]
        const = const.to(t.device)
        return const.gather(-1,t).reshape(-1,1,1,1)


class DDPMScheduler(BaseScheduler):
    def __init__(self,inference_step,device):
        super().__init__(inference_step,device)
        
    def forward_process(self,t,x_0,eps=None):
        """
        <add noise 과정>
        timestep , x_0, eps  필요
        """
            
        alpha_bar = self.teeth(self.cumprod_alpha,t)
    
        if eps is None :
            eps = torch.randn_like(x_0)
        
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps 
        
        return x_t, eps
    
    def reverse_process(self,t,x_t,eps,noise=None):
        """
        base -> eps predictor network, not mu predictor 
        
        eps : network prediction
        
        mu(x_t,eps) 은 mu(x_t,x_0)에 forwardstep x_0를 대입하여 얻어냄
        그래서 mu = 1 /sqrt alpha  * ( x_t - 1-alpha_t / sqrt 1-alpha_bar_t ) * eps
        
        prev : 한스텝 reverse ver
        """
        
        alpha_bar = self.teeth(self.cumprod_alpha,t)
        alpha = self.teeth(self.alpha,t)
        
        mu = 1 / torch.sqrt(alpha) * (x_t - (1-alpha) / torch.sqrt(1-alpha_bar) * eps)
        
        t_prev = torch.cat([torch.tensor([1],device=x_t.device),self.timesteps[:-1].to(x_t.device)],dim=-1)[t]
        
        alpha_bar_prev = self.teeth(self.cumprod_alpha,t_prev)
        
        sigma_square = ((1-alpha_bar_prev) / (1-alpha_bar)) * (1-alpha)
        ## t_prev 구하는 법은, ddim의 경우에는 t-1이 아닐 수 있으므로 그렇게 하면 안되고, timestep을 한칸 밀어서 거기서 t 추출하도록 해야함
        
        if noise is None:
            noise = torch.randn_like(x_t)
            
        sample_prev = mu + torch.sqrt(sigma_square) * noise

        return mu, sample_prev, noise
        
def test():
    scheduler = DDPMScheduler(50,None)
    x_t = torch.randn(4,3,128,128,device="mps")
    eps = torch.randn(4,3,128,128,device="mps")
    
    t = torch.tensor(random.sample(range(0, 11), 4), device="mps")
    
    print(t.shape)
    print(scheduler.forward_process(x_0=x_t,t=t,eps=eps))
    print(scheduler.reverse_process(x_t=x_t,t=t,eps=eps))
    
#test()
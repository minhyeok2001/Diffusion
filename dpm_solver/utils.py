import torch
import torch.nn as nn

class DpmSolver(nn.Module):
    def __init__(self,device="cuda",inference_step=100):
        super().__init__()
        num_timestep=1000
        timesteps = torch.arange(num_timestep-1,-1,-1,device=device) 
        beta = torch.linspace(1e-4,2e-2,num_timestep,device=device) 
        alpha = 1-beta
        cumprod_alpha = torch.cumprod(alpha,-1)
        
        dpm_alphas = torch.sqrt(cumprod_alpha)
        dpm_sigmas = torch.sqrt(1 - cumprod_alpha)
        
        ## 람다는 sde 식에서 log(alpha)-log(sigma) 임 
        dpm_lambdas = torch.log(dpm_alphas/dpm_sigmas)
        
        ## 지금 개인적인 생각으로는, DDIM case와는 다르게 그냥 쌩 alpha를 쓰지는 않고 SNR이 전부다 cumprod로 구성되어있으므로,
        # 굳이 그 alpha를 계산하지 않아줘도 될듯. cumprod끼리 나눠서 ..
        #self.register_buffer("steps",num_timestep)
        self.register_buffer("dpm_alphas",dpm_alphas)
        self.register_buffer("dpm_sigmas",dpm_sigmas)
        self.register_buffer("dpm_lambdas",dpm_lambdas)
        
        self.inference_step = inference_step
        self.ratio = num_timestep//inference_step

    def teeth(self,const,t):
        const = const.to(t.device)
        return const.gather(-1,t).reshape(-1,1,1,1)
        
    def basic_function(self,t,s,x_s,integral_term):
        ## t,s : timestep. s가 더 노이지한 step이여야함. 즉 t<s
        
        alpha_t = self.teeth(self.dpm_alphas,t)
        alpha_s = self.teeth(self.dpm_alphas,s)
        x_t = (alpha_t/alpha_s) * x_s - integral_term 
        return x_t
        
    def first_order(self,t,s,x_s,eps):
        
        sigma_t = self.teeth(self.dpm_sigmas,t)
        lambda_t = self.teeth(self.dpm_lambdas,t)
        lambda_s = self.teeth(self.dpm_lambdas,s)
        
        integral_term = sigma_t * (torch.exp(lambda_t-lambda_s)-1) * eps
        
        x_t = self.basic_function(t,s,x_s,integral_term)
        return x_t

    def second_order(self,):
        pass
    def third_order(self,):
        pass

        
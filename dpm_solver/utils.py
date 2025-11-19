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
        
        self.num_timestep = num_timestep 
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

    def second_order(self,t,s,x_s,cls,model,cfg_weight):
        ## 다 좋은데, 만약 반띵한걸 어케 구하지 ??
        ## 그러니까... lambda가 반띵되는 부분의 timestep을 구해야함.

        sigma_t = self.teeth(self.dpm_sigmas,t)
        sigma_s = self.teeth(self.dpm_sigmas,s)
        lambda_t = self.teeth(self.dpm_lambdas,t)
        lambda_s = self.teeth(self.dpm_lambdas,s)
        
        
        lambda_mid = (lambda_t + lambda_s)/2
        
        ## CS492 구현과는 조금 다르게 진행해보자.
        ## mid값을 lambda table에서 뺀 다음에, 이거 abs가 가장 작은 값의 인덱스를 사용
        
        distance = torch.abs(self.dpm_lambdas - lambda_mid)
        
        mid = torch.argmin(distance)
        
        mid = torch.full((x_s.shape[0],),mid,dtype=torch.long,device=x_s.device)
        
        sigma_mid = self.teeth(self.dpm_sigmas,mid)
        lambda_mid = self.teeth(self.dpm_lambdas,mid)
        
        cond_noise = model(x_s,s,cls)
        uncond_noise = model(x_s,s,torch.zeros_like(cls))
        noise1 = (1+cfg_weight)*cond_noise - cfg_weight * uncond_noise
        
        integral_term1 = sigma_mid * (torch.exp(lambda_mid-lambda_s)-1) * noise1
        
        x_mid = self.basic_function(mid,s,x_s,integral_term1)
        
        cond_noise = model(x_mid,mid,cls)
        uncond_noise = model(x_mid,mid,torch.zeros_like(cls))
        noise2 = (1+cfg_weight)*cond_noise - cfg_weight * uncond_noise
        
        integral_term2 = sigma_t * (torch.exp(lambda_t-lambda_mid)-1) * noise2
        
        x_t = self.basic_function(t,mid,x_mid,integral_term2)
        
        return x_t
        
        
    def third_order(self,):
        pass

        
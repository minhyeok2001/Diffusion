import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,pred,gt):
        return F.mse_loss(pred,gt)

class FlowMatchingScheduler(nn.Module):
    def __init__(self, model_type ="optimal transport"):
        super().__init__()
        self.model_type = model_type
        
    def mu_t(self,t,x_1):
        if self.model_type == "optimal transport":
            mu = t*x_1 
        else : 
            raise RuntimeError("Not Implemented yet !!!")
        return mu
    
    def sigma_t(self,t,sigma_min):
        if self.model_type == "optimal transport":
            sigma = 1 - ( 1 - sigma_min) * t
        else : 
            raise RuntimeError("Not Implemented yet !!!")
        return sigma
    
    def probability_path(self,x):
        ## notational concept
        pass
    
    def flow_map(self,t,x_1,sigma_min,x_0=None):
        if t.dim() != x_1.dim():
            t = t.view(x_0.shape[0],1,1,1)
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        return self.mu_t(t,x_1)+self.sigma_t(t,sigma_min) * x_0
    
    def vector_field(self,t,x_1,sigma_min,x_0=None):
        if t.dim() != x_1.dim():
            t = t.view(x_0.shape[0],1,1,1)
            
        if self.model_type == "optimal transport":
            x_t = self.flow_map(t,x_1,sigma_min,x_0)
            return (x_1 -(1-sigma_min)*x_t) / (1-(1-sigma_min)*t)

        else :
            raise RuntimeError("Not Implemented yet !!!")
    
    def sampling(self,network,t,x_t,delta_t):
        return x_t + network(x_t,t) * delta_t

    
    

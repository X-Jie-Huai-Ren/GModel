"""
Building the Diffusion Model, Step1: The forward process: Noise Scheduler

First, we need to bulid the inputs for our model, which are more and more noisy images.
Instead of doing this sequentially, we can use the closed form provided in the papers to calculate the image for any of the timesteps individually

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-03-22
"""

import torch
import torch.nn.functional as F
from torch import nn


class ForwardProcess(nn.Module):
    """
    """
    def __init__(self, T, start=0.0001, end=0.02) -> None:
        """
        Params:
            T: The total number of timesteps in the forward process
            start: initial beta
            end: final beta
        """
        super(ForwardProcess, self).__init__()

        self.T = T
        self.start = start
        self.end = end

        # betas
        self.betas = self.linear_beta_schedule()
        # alphas
        self.alphas = 1. - self.betas
        # 累乘的alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # [0.9999, 0.9997, 0.9995, ..., 0.0500, 0.0490, 0.0481]
        # 上一时间步对应的累乘alphas
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)  # [1.0000, 0.9999, 0.9997, ..., 0.0511, 0.0500, 0.0490]
        # sqrt(1/a)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # 根号下累乘alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # 根号下(1-累乘alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # 新正态分布下的方差是一个定值，是直接由前向过程中的beta和alpha确定的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def linear_beta_schedule(self):
        """
        """
        return torch.linspace(self.start, self.end, self.T)
    
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0] # 16
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward(self, x_0, t):
        """
        Take image and timestep t as input
        return the noisy version of t
        """
        # device
        device = x_0.device
        # create a random noise
        noise = torch.randn_like(x_0).to(device)
        #
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t.to(device) * x_0 + sqrt_one_minus_alphas_cumprod_t.to(device) * noise, noise



if __name__ == '__main__':

    FP = ForwardProcess(T=300)

    for idx in range(0, 300, 30):

        t = torch.Tensor([idx]).type(torch.int64)

        img, noise = FP.forward_diffusion_sample(t)

        break


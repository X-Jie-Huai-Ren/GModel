"""
Generate images based on a trained model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-03-27
"""


import torch
import matplotlib.pyplot as plt

from backward import Unet
from forward import ForwardProcess
from utils import rebuild_imgs, bulid_log_dir



class Generator:
    """
    Generate Manager
    """
    def __init__(self, arg_dict, model, fp) -> None:
        
        self.arg_dict = arg_dict
        self.model = model

        # betas parameters of forward process
        self.betas = fp.betas
        self.sqrt_one_minus_alphas_cumprod = fp.sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = fp.sqrt_recip_alphas
        # p[x(t-1)|x(t)]下的方差
        self.posterior_variance = fp.posterior_variance

        # stepsize
        self.stepsize = 30


    @ torch.no_grad()
    def generate(self):
        # create the random xt, we need to transform the noise to generated image step by step
        xt = torch.randn(size=(self.arg_dict['num'], 3, 128, 128))

        buffer = torch.tensor([])

        # generate based on backward model
        for t in range(0, self.arg_dict['T'])[::-1]:
            # the β, sqrt(1-α_cum), sqrt(1/α) of timestep t
            betas_t = self.betas[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
            # 根据论文, p[x(t-1)|x(t)]下的方差是一个定值，是直接由前向过程中的beta和alpha确定的
            variance_t = self.posterior_variance[t]
            # based on backward model, predict the noise of current timestep t
            noise_pred = self.model(xt, torch.full((self.arg_dict['num'],), t))
            # p[x(t-1)|x(t)]下的均值
            mean_t = sqrt_recip_alphas_t * (xt - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

            """
            个人对于扩散模型的理解：
                目前, 我们已经知道后验概率分布p[x(t-1)|x(t)]的均值和方差, 可以从该分布中采样得到x(t-1)
            但需要注意的是(以本项目为例, 生成3x128x128的图像), 这里并不是从一个已知均值和标准差的正态分布中
            采样一个(3, 128, 128) 的数据。
                要知道, 我们前面得到的均值和标准差均是 (3, 128, 128) 的张量(这里假设batch size 为1), 这意味着我们有
            3x128x128个正态分布, 也就是说从一个正态分布中采样一个数据来生成一个像素点, 3x128x128个像素点便组成一张3x128x128的图像。
            但怎么采样呢?  
                我们知道, 随机变量X服从N(μ, σ), 那X的大概范围即 μ ± σ [这对于后面生成x(t-1)及其重要]
            """
            # when timestep t=0, there is no standard-deviation offset
            if t == 0:
                xt = mean_t
            else:
                xt = mean_t + torch.sqrt(variance_t) * torch.randn_like(xt)

            # clip for maintaining natural range of the preprocessed image (-1~1)
            xt = torch.clamp(xt, -1.0, 1.0)

            if t % self.stepsize == 0:
                buffer = torch.concat([buffer, xt.unsqueeze(0)], axis=0)

        return buffer
    

def plot(imgs, cols=5):
    """
    Params:
        imgs: (steps, num, 3, 128, 128)
    """
    # the number of images which you want to be showed on a picture
    num_images = imgs.shape[0]
    # the number of generated image
    num = imgs.shape[1]

    for idx in range(num):
        fig, axes = plt.subplots(num_images//cols, cols)
        for t in range(num_images):
            img = imgs[t, idx]
            img = rebuild_imgs(img).transpose(0, 2)
            axes[t//cols, t%cols].imshow(img)
        # 调整子图之间的间距
        plt.tight_layout()
        # 保存图形为文件
        plt.savefig(arg_dict['result_dir'] + f'/{idx+1}.png')
        plt.show()




def main(arg_dict):
    """"""
    # load the model checkpoints
    checkpoints = torch.load(arg_dict['checkpoints'])
    backwardModel = Unet()
    backwardModel.load_state_dict(checkpoints)

    # create result dir
    arg_dict['result_dir'] = bulid_log_dir(dir='./results')

    # forward process
    fp = ForwardProcess(arg_dict['T'])

    # generate manager
    generator = Generator(arg_dict, backwardModel, fp)

    generated_imgs = generator.generate()

    # plot
    plot(generated_imgs)




if __name__ == '__main__': 

    arg_dict = {
        "checkpoints": './logs/[03-29]20.28.08/minloss.tar', 
        "num": 2,  # the number of images you want to generate
        "T": 300
    }

    main(arg_dict)


import torch
from torch import nn



class Generator(nn.Module):
    """
        生成器
    """
    def __init__(self, z_dim, img_dim) -> None:
        """
        Params:
            z_dim: 输入噪音的维度
            img_dim: 输出图像的维度, 对于mnist数据集, 这里的输出是784
        """
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()    # 在原图像中像素值的范围在0-1之间, 这里使用Tanh函数使输出在-1~1之间, 在训练的时候要对原始图像处理, 后续可以比较一下sigmoid
        )
    
    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    """
        判别器
    """
    def __init__(self, img_dim) -> None:
        """
        Params:
            img_dim: 将生成的图像输入进Discriminator
        """
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)
    
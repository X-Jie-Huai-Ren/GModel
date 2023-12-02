"""
MLP-Structure for RenewablePowerGAN, including Generator and Discriminator

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-10-30
"""

from torch import nn

class Generator(nn.Module):
    """
    the Generator for GAN
    """
    def __init__(self, input_dim, output_dim) -> None:
        """
        :param input_dim: the dimension of input noise
        :param output_dim: the dimension of generate data
        """
        super(Generator, self).__init__()

        # input/output dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # build the model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        ) 

    def forward(self, x):

        return self.model(x)
    



class Discriminator(nn.Module):
    """
    the discriminator for GAN
    """
    def __init__(self, input_dim=288) -> None:
        """
        :param input_dim: the dimension of generated data by Generator
        """
        super(Discriminator, self).__init__()

        # the input dim
        self.input_dim = input_dim

        # build the model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)

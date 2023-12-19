"""
MLP-Structure for RenewablePowerGAN, including Generator and Discriminator.
And Discriminator's last layer cancel Sigmoid layer in WGAN
For C-GAN, add labels as input to the model, including Generator and Discriminator

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-11
"""

import torch
from torch import nn

class Generator(nn.Module):
    """
    the Generator for GAN
    """
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            num_classes=4,
            embedding_size=28
        ) -> None:
        """
        :param input_dim: the dimension of input noise
        :param output_dim: the dimension of generate data
        """
        super(Generator, self).__init__()

        # input/output dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        # build the model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim + self.embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        ) 

        # Embedding layer
        self.embed = nn.Embedding(self.num_classes, self.embedding_size)

    def forward(self, z, labels):
        """
        Params:
            z: the randomly generated noise
            labels: the label corresponding to the real sample
        """

        embedding = self.embed(labels)
        x = torch.cat([z, embedding], dim=1)

        return self.model(x)
    

class Discriminator(nn.Module):
    """
    the discriminator for GAN
    """
    def __init__(
            self, 
            input_dim=24,
            num_classes=4,
            embedding_size=12
        ) -> None:
        """
        :param input_dim: the dimension of generated data by Generator
        """
        super(Discriminator, self).__init__()

        # the input dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        # build the model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim + self.embedding_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

        # Embedding layer
        self.embed = nn.Embedding(self.num_classes, self.embedding_size)

    def forward(self, x, labels):
        """
        Params:
            x: the generated data
            labels: the label corresponding to the real sample
        """

        embedding = self.embed(labels)
        x = torch.cat([x, embedding], dim=1)

        return self.model(x)
    

# Initialize the weights: all weights need to be initialized from a 0-centered Normal distribution with standard deviation 0.02
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    


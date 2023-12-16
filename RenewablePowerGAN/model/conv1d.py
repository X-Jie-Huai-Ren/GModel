"""
Conv1d-Structure for RenewablePowerGAN, including Generator and Discriminator.
And Discriminator's last layer cancel Sigmoid layer in WGAN

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-4 
"""

from torch import nn

class Generator(nn.Module):
    """
    the Conv1d-Structure for Generator
    """
    def __init__(self, input_dim, output_dim) -> None:
        """
        Params:
        input_dim: the dimension of input noise
        output_dim: the dimension of generate data
        """
        super(Generator, self).__init__()

        # input/output dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(self.input_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, padding='same')

    def forward(self, x):

        embedding = self.bn(self.fc(x)).unsqueeze(1)

        return self.conv(embedding)

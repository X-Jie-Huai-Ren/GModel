
import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import torch
from torch.utils.data import DataLoader
from model.conv1d import Generator
import config
from dataset import ChangChuanDataset



if __name__ == '__main__':

    # load the dataset
    solardataset = ChangChuanDataset('./data/changchun.xlsx', normed='standard')
    # make a dataloader
    train_loader = DataLoader(
        dataset=solardataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # Generator
    generator = Generator(input_dim=config.Z_DIM, output_dim=config.OUTPUT_DIM)

    for x in train_loader:
        z = torch.randn(size=(config.BATCH_SIZE, config.Z_DIM))
        fake = generator(z)
        print(fake.shape)
        break 
    
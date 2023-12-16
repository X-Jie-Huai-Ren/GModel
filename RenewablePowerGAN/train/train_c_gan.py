"""
Train script for C-GAN

* @author: xuan
* @email: 1920425406@qq.com
* @date: 203=23-11-15
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

from torch.utils.data import DataLoader
from datasets.c_gan_dataset import SeasonDataset
import config


def main(arg_dict):

    # Read the dataset
    solardataset = SeasonDataset('./data/changchun.xlsx', normed=arg_dict["norm"])

    # make a dataloader
    train_loader = DataLoader(
        dataset=solardataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    for real, label in train_loader:

        print(real.shape)
        print(label.shape)
        print(label)
        break






if __name__ == '__main__':

    arg_dict = {
        "data": './data/changchun.xlsx',
        "norm": 'standard',   # norm or standard 
        "model": 'mlp_wgan_bt',   # mlp, mlp_wgan, mlp_wgan_bt
    }
    
    main(arg_dict)
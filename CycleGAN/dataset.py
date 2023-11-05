"""
@author: xuansd
@email: 1920425405@qq.com
@date: 2023-10-22
"""

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CycleGANDataset(Dataset):
    """
    vangogh2photo dataset
    """
    def __init__(self, x_dir, y_dir, transform=None) -> None:
        """
        :param x_dir: the path of input imgs
        :param y_fir: the path of output(label) imgs
        :param transforms: default None, Image tranform mode
        """
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.transform = transform

        # loop over the x_dir, y_dir
        self.x_imgs = os.listdir(self.x_dir)
        self.y_imgs = os.listdir(self.y_dir)
        # the length of the x_imgs, y_imgs
        self.x_length = len(self.x_imgs)
        self.y_length = len(self.y_imgs)
        # the length of dataset
        self.length = max(self.x_length, self.y_length)

    def __len__(self):
        """
        return the length of dataset
        """
        return self.length
    
    def __getitem__(self, index):
        # the x/y image
        x_img_name = self.x_imgs[index % self.x_length]
        y_img_name = self.y_imgs[index % self.y_length]

        # the path of x/y img
        x_img_path = os.path.join(self.x_dir, x_img_name)
        y_img_path = os.path.join(self.y_dir, y_img_name)

        # x/y image
        x_img = np.array(Image.open(x_img_path).convert('RGB'))
        y_img = np.array(Image.open(y_img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=y_img, image0=x_img)
            x_img = augmentations['image0']
            y_img = augmentations['image']

        return x_img, y_img

    
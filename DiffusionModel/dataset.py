"""
dataset for diffusion model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-03-21
"""



import os
import numpy as np
from typing import Any
from PIL import Image
from torch.utils.data import Dataset

from utils import transform



class DMDataset(Dataset):
    """
    dataset
    """
    def __init__(self, img_dir, transform=None) -> None:
        """
        Params: 
            data_path: train data or test data
        """
        super().__init__()

        self.img_dir = img_dir
        self.transform = transform

        # loop over the img_dir
        self.imgs = os.listdir(self.img_dir)
        # the length of the imgs
        self.length = len(self.imgs)

    def __len__(self):
        """
        return the length of dataset
        """
        return self.length
    
    def __getitem__(self, index) -> Any:

        img_path = os.path.join(self.img_dir, self.imgs[index])

        img = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations['image']

        return img
    

if __name__ == '__main__':

    transforms = transform()

    dataset = DMDataset('./data/test', transform=transforms)

    print(len(dataset))

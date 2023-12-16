"""
The dataset is power dataset which it's site is in ChangChun
For C-GAN, we divide the dataset into four categories by season

* @author: xuan
* @email: 1920425406@qq.com
* @date: 203=23-12-11
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import normalize, normalize1


class SeasonDataset(Dataset):
    """
    the power dataset in ChangChun, 
    """
    def __init__(self, file_path, normed='norm', **kwargs) -> None:
        """
        :param file_path: thr path of data file
        """
        # Read data
        data = pd.read_excel(file_path, **kwargs)["G"].values

        # Format data: (8760, ) --> (-1, 24)
        self.data = data.reshape((-1, 24))

        # Normalize the data or not
        if normed == 'norm':
            self.data, self.max, self.min = normalize1(self.data)
        elif normed == 'standard':
            self.data, self.mean, self.std = normalize(self.data)

    def __len__(self):
        """
        the number of samples
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        :param index: the id of sample
        """
        data = torch.FloatTensor(self.data[index])
        # Spring --> label:0
        if index < (31 + 28 + 31):
            return data, 0
        # Summer --> label:1
        elif (31 + 28 + 31) <= index < (90 + 31 + 31 + 30):
            return data, 1
        # Autumn --> label:2
        elif (90 + 31 + 31 + 30) <= index < (182 + 30 + 31 +30):
            return data, 2
        # Winter --> label:3
        else:
            return data, 3
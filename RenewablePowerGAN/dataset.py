

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class SolarDataset(Dataset):
    """Solar Power Dataset"""
    def __init__(self, file_path, **kwargs) -> None:
        """
        :param file_path: thr path of data file
        """
        # Read data
        data = pd.read_csv(file_path, **kwargs).values
        # data sequence's len of a year(time resolution is 5 min), the number of sites
        datapoint_number, site_number = data.shape

        ## Format data: (datapoint_number, size)-->(sample_number, 1*24*12)
        alldata = np.array([])
        # loop over the sites
        for idx in range(site_number):
            alldata = np.concatenate([alldata, data[:, idx]], axis=0)
        # reshape to (sample_number, 1*24*12)
        self.data = alldata.reshape((-1, 1*24*12))

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

        return data


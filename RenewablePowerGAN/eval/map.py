"""
For the fake and real data, Due to the diversity of the generated data, we need to match the most similar pairs

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-11-27
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import pandas as pd
import numpy as np
from utils import plot

def mapdata(arg_dict):
    """
    """
    # Read the data
    real_data = pd.read_excel(arg_dict["file1"], index_col=0).values
    fake_data = pd.read_excel(arg_dict["file2"], index_col=0).values

    # the number of real samples
    real_num = real_data.shape[0]

    # data pair
    data_pairs = {}

    # loop the fake data
    for (id, data) in enumerate(fake_data):
        # reshape the data
        data = data.reshape((-1, data.shape[0]))
        # repeat the single data
        data = np.repeat(data, real_num, axis=0)
        
        # Calculate the RMSE of fake data and all real data
        delta = data - real_data
        delta_2 = np.power(delta, 2)
        rmse = np.sqrt(np.sum(delta_2, axis=1))

        # get the index respect to min rmse
        index = np.argmin(rmse)

        # data pair
        pair = {}
        pair["fake"] = fake_data[id]
        pair["real"] = real_data[index]
        pair["rmse"] = rmse[index]
        data_pairs[f'{id}'] = pair
    
    # Select the data pair with the smallest RMSE
    # Sort
    data_pairs = sorted(data_pairs.items(), key=lambda x:x[1]["rmse"], reverse=False)
    if arg_dict["show"]:
        for i in range(arg_dict["pair_num"]):
            pair = data_pairs[i][1]
            real_data = pair["real"].reshape((-1, real_data.shape[1]))
            fake_data = pair["fake"].reshape((-1, real_data.shape[1]))
            plot(real_data, fake_data)
    
    return [data_pairs[i] for i in range(arg_dict["pair_num"])]    
    



if __name__ == '__main__':

    arg_dict = {
        "file1": './results/mlp_wgan_bt/bt_norm_nodisc_gp/real.xlsx',
        "file2": './results/mlp_wgan_bt/bt_norm_nodisc_gp/fake.xlsx',
        "show": False,
        "pair_num": 5,   # the number of pairs
        "number": 5      # One fake data corresponds to the number of similar real-world data
    }
    data = mapdata(arg_dict)
    print(data)
    
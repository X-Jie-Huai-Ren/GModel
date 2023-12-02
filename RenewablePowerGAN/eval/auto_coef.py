"""
For the fake and real data, the autocorrelation coefficient compares the temporal correlation of the scene

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-11-27
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import numpy as np
import matplotlib.pyplot as plt
from eval.map import mapdata
from statsmodels.graphics.tsaplots import plot_acf




if __name__ == '__main__':

    arg_dict = {
        "file1": './results/mlp_wgan_bt/bt_norm_nodisc_gp/real.xlsx',
        "file2": './results/mlp_wgan_bt/bt_norm_nodisc_gp/fake.xlsx',
        "number": 5,
        "show": False
    }
    data_pairs = mapdata(arg_dict)

    for i in range(arg_dict["number"]):
        pair = data_pairs[i][1]
        real_data = pair["real"]
        fake_data = pair["fake"]
        plot_acf(real_data)
        plot_acf(fake_data)
        plt.show()
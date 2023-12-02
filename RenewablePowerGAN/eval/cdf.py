"""
For the fake and real data, Draw a cumulative distribution plot

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-11-27
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import make_interp_spline
from eval.map import mapdata



# 曲线平滑
def smooth(x, y):
    model = make_interp_spline(x, y)
    xs = np.linspace(min(x), max(x), 500)
    ys = model(xs)
    
    return xs, ys




if __name__ == '__main__':

    arg_dict = {
        "file1": './results/mlp_wgan_bt/bt_norm_nodisc_gp/real.xlsx',
        "file2": './results/mlp_wgan_bt/bt_norm_nodisc_gp/fake.xlsx',
        "show": False,
        "pair_num": 5,   # the number of pairs
        "number": 5      # One fake data corresponds to the number of similar real-world data
    }
    data_pairs = mapdata(arg_dict)
    
    for i in range(arg_dict["number"]):
        pair = data_pairs[i][1]
        real_data = pair["real"]
        fake_data = pair["fake"]

        # real
        real_res = stats.relfreq(real_data, numbins=5)
        realx = real_res.lowerlimit + np.linspace(0, real_res.binsize*real_res.frequency.size,real_res.frequency.size)
        realy = np.cumsum(real_res.frequency)
        real_xs, real_ys = smooth(realx, realy)

        # fake
        fake_res = stats.relfreq(fake_data, numbins=5)
        fakex = fake_res.lowerlimit + np.linspace(0, fake_res.binsize*fake_res.frequency.size,fake_res.frequency.size)
        fakey = np.cumsum(fake_res.frequency)
        fake_xs, fake_ys = smooth(fakex, fakey)
        
        plt.plot(real_xs, real_ys, label='real')
        plt.plot(fake_xs, fake_ys, label='fake')
        plt.title('Figure6 CDF')
        plt.legend()
        plt.show()
        break
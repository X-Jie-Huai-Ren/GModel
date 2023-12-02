"""
check the weight distribution

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-11-20
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def main(arg_dict):

    # load the checkpoints
    checkpoints = torch.load(arg_dict["checkpoints"])

    # the discrininator's weight
    disc_weight = checkpoints["discriminator"]

    all_weights = np.zeros([0, ])
    for _, weight in disc_weight.items():
        weight_flatten = weight.cpu().flatten()
        all_weights = np.concatenate([all_weights, weight_flatten], axis=0)

    # plot the weight distribution
    plt.figure()
    plt.hist(all_weights, bins=100, color='b') 
    plt.title('the disc weight distribution')
    plt.savefig(f'{arg_dict["save_path"]}/disc_distribution.png')   
    plt.show()




if __name__ == '__main__':

    arg_dict = {
        "checkpoints": './logs/WGAN-GP/bt_norm_nodisc/model_2999.tar',
        "save_path": './results/mlp_wgan_bt/bt_norm_nodisc_gp'
    }

    main(arg_dict)

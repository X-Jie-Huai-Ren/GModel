"""
plot the generated data according to trained model

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-11-11
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import torch
import matplotlib.pyplot as plt
import random
import importlib
import numpy as np
import os
from datetime import datetime, timedelta

# from model.mlp import Generator
# from model.mlp_wgan import Generator
from dataset import ChangChuanDataset
import config
from utils import save_data


def build_folder(arg_dict):
    """build the folder to save the results"""

    cur_time = datetime.now() + timedelta(hours=0)  # hours参数是时区
    # join the path
    result_dir = os.path.join(arg_dict["root_dir"], arg_dict["model"])
    result_dir = os.path.join(result_dir, cur_time.strftime(f"[%m-%d]%H.%M.%S"))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def lineArg():
    """
    define the plot format
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
    markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    linestyle = ['--', '-.', '-']
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    # line_arg['linewidth'] = random.randint(1, 4)
    return line_arg



def plot(real_data, fake_data, arg_dict):
    """
    plot the data
    
    :param real_data: the data need to be ploted
    :param fake_data: the data need to be ploted
    """
    # set the random seed
    random.seed(0)
    
    assert len(real_data.shape) == 2, 'the real data to be ploted need multiple'
    assert len(fake_data.shape) == 2, 'the fake data to be ploted need multiple'

    ## pl0t the data
    plt.figure(figsize=(15, 5))

    # plot the real data
    plt.subplot(1, 2, 1)
    x = range(real_data.shape[1])
    for id in range(real_data.shape[0]):
        plt.plot(x, real_data[id], **lineArg())
    plt.title('real')
    plt.xlabel('time')
    plt.ylabel('power')

    # plot the fake data
    plt.subplot(1, 2, 2)
    x = range(fake_data.shape[1])
    for id in range(fake_data.shape[0]):
        plt.plot(x, fake_data[id], **lineArg())
    plt.title('fake')
    plt.xlabel('time')
    plt.ylabel('power')

    # save the result or not
    if arg_dict["savefig"]:
        # build the folder
        result_dir = build_folder(arg_dict)
        plt.savefig(f'{result_dir}/result.png')

    # show the plot or not
    if arg_dict["show"]:
        plt.show()

    return result_dir



def main(arg_dict):

    # load the trained model 
    impoted_model = importlib.import_module("model." + arg_dict["model"])
    generator = impoted_model.Generator(input_dim=config.Z_DIM, output_dim=config.OUTPUT_DIM)
    checkpoints = torch.load(arg_dict["checkpoints_path"])
    generator.load_state_dict(checkpoints["generator"])

    # Randomly generate the noise(input data)
    input_z = torch.randn(size=(arg_dict["number"], config.Z_DIM))

    # For Test
    generator.eval()

    # Generate the fake data
    fake_data = generator(input_z).detach().numpy()

    # load the real data 
    solardataset = ChangChuanDataset(arg_dict["data_file"], normed=arg_dict["normed"])
    real_data = solardataset[np.random.randint(0, len(solardataset), size=arg_dict["number"])]

    # for normalize, if normed, the data should be midified
    if arg_dict["normed"] == 'norm':
        fake_data = fake_data * (solardataset.max - solardataset.min) + solardataset.min
        real_data = real_data * (solardataset.max - solardataset.min) + solardataset.min
    elif arg_dict["normed"] == 'standard':
        fake_data = fake_data * solardataset.std + solardataset.mean
        real_data = real_data * solardataset.std + solardataset.mean
    
    # plot
    result_dir = plot(real_data, fake_data, arg_dict)

    # save the generated data to excel
    if arg_dict["save"]:
        save_data(file_path=f'{result_dir}/real.xlsx', data=real_data)
        save_data(file_path=f'{result_dir}/fake.xlsx', data=fake_data)



if __name__ == '__main__':

    arg_dict = {
        "number": 64,    # 32, 64, 128 ...  the number of generated data
        "model": 'mlp_wgan_bt',   # mlp, mlp_wgan, mlp_wgan_bt
        "checkpoints_path": './logs/WGAN-GP/bt_standard_nodisc/model_9999.tar',
        "data_file": './data/changchun.xlsx',
        "show": True,
        "savefig": True,
        "save": True,    # save data to excel or not
        "root_dir": './results',   # if "save" is True, the results will be saved to this dir
        "normed": 'standard'   # norm or standard
    }

    main(arg_dict)
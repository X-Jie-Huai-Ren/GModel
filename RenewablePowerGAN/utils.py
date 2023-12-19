import os
import torch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import config



# 生成记录日志的文件夹
def build_log_folder():
    cur_time = datetime.now() + timedelta(hours=0)  # hours参数是时区
    log_path_dir = os.path.join(config.LOAD_MDEOL_FILE, cur_time.strftime(f"[%m-%d]%H.%M.%S"))
    # 若文件夹不存在，则创建
    if not os.path.exists(log_path_dir):
        os.makedirs(log_path_dir)
    return log_path_dir


# Save the checkpoints
def save_checkpoints(checkpoints, log_dir, epoch):
    """
    Params:
        checkpoints: 模型权重
        log_dir: 日志目录
        epoch: 当前训练的轮数
    """
    # 若文件夹不存在，则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path = log_dir + f'/model_{epoch}.tar'
    print('==> Saving checkpoints')
    torch.save(checkpoints, path)


# save data to excel
def save_data(file_path, data):
    """
    :param file_path: the file path to save data
    :param data: the data need to be saved
    """
    data = pd.DataFrame(data)
    data.to_excel(file_path)



# Standardize the data
def normalize(data):
    """
    :param data: shape(num_samples, output_dim)
    """
    # Calculate the mean and std
    mean = np.mean(data)
    std = np.std(data)
    # Normalize
    data_normed = (data - mean) / (std + 1e-6)

    return data_normed, mean, std

# 归一化数据
def normalize1(data):
    """
    :param data: shape(num_samples, output_dim)
    """
    # the maximum/minimum
    maximum = np.max(data)
    minimum = np.min(data)
    # Normalize
    data_normed = (data - minimum) / (maximum - minimum)

    return data_normed, maximum, minimum
    

# gradient penalty
def gradient_penalty(critic, real, fake, device="cpu", labels=None):
    batch, data_len = real.shape
    # randomly generate the epsilon
    epsilon = torch.rand((batch, 1)).repeat(1, data_len).to(device)
    interpolated_data = real*epsilon + fake*(1-epsilon)

    # calculate critic scores
    if labels is not None:
        mixed_scores = critic(interpolated_data, labels)
    else:
        mixed_scores = critic(interpolated_data)

    gradient = torch.autograd.grad(
        inputs=interpolated_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)         # in the wgan-gp paper, take the L2-Norm for gradient
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty


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

def plot(real_data, fake_data):
    """
    plot the data
    
    :param real_data: the data need to be ploted
    :param fake_data: the data need to be ploted
    """
    # set the random seed
    random.seed(0)
    
    assert len(real_data.shape) == 2, 'the real data to be ploted need multiple'
    assert len(fake_data.shape) == 2, 'the fake data to be ploted need multiple'

    ## plot the data
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

    plt.show()
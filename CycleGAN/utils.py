"""
utils for CycleGan

* @author: xuansd
* @email: 1920425405@qq.com
* @date: 2023-10-22
"""

import os
import torch
from datetime import datetime, timedelta 
from config import Arg


# 生成记录日志的文件夹
def build_log_folder():
    cur_time = datetime.now() + timedelta(hours=0)  # hours参数是时区
    log_path_dir = os.path.join(Arg["Load_model_path"], cur_time.strftime(f"[%m-%d]%H.%M.%S"))
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



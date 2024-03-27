"""
the utils for diffusion model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-03-21
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime, timedelta


# plot samples
def show_imgs(img_dir: str, num_samples=20, cols=5):
    """ 
    plot some samples from the dataset 
    Params:
        img_dir: the directory of dataset
        num_samples: the number to be showed
    """
    img_paths = os.listdir(img_dir)

    # 随机生成num_samples个整数
    ids = [random.randint(0, len(img_paths)) for i in range(num_samples)]
    # 根据索引获取图像路径
    imgs_path = [os.path.join(img_dir, img_paths[index]) for index in ids]

    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(imgs_path):
        img = np.array(Image.open(img_path).convert('RGB'))
        plt.subplot(int(num_samples/cols), cols, i+1)
        plt.imshow(img)
    plt.show()

# rebuild image
def rebuild_imgs(imgs):
    """
    before the training, we preprocessed the images, 
    for outputs, we need to rebuild those to images
    """
    return imgs * 0.5 + 0.5

def bulid_log_dir():
    curtime = datetime.now() + timedelta(hours=0)  # hours参数是时区
    log_path_dir = os.path.join('./logs', curtime.strftime(f"[%m-%d]%H.%M.%S"))
    # 若文件夹不存在，则创建
    if not os.path.exists(log_path_dir):
        os.makedirs(log_path_dir)
    return log_path_dir


# image transform
def transform():
    return A.Compose(
        [
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2()
        ]
    )



if __name__ == '__main__':

    show_imgs(img_dir='./data/train')
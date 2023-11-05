"""
@author: xuansd
@email: 1920425405@qq.com
@date: 2023-10-22
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

Arg = {
    "trainx_dir": './data/vangogh2photo/trainB',
    "trainy_dir": './data/vangogh2photo/trainA',
    "testx_dir": './data/vangogh2photo/testB',
    "testy_dir": './data/vangogh2photo/testA',
    "epochs": 100,
    "batch_size": 8,
    "lr": 2e-6,
    "num_workers": 1,
    "traind_model_path": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "transform": A.Compose(
        [
            A.Resize(width=224, height=224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2()
        ],
        additional_targets={"image0": "image"}
    ),
    "Load_model_path": './logs'
}
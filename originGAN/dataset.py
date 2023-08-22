"""
    预处理数据集
"""
import os
from torchvision import datasets
from torch.utils import data
from utils import transform_to_image
from PIL import Image

class MNISTDataset(data.Dataset):
    """
        先将数据转为图片格式保存在本地
        再从本地读取成dataset
    """
    def __init__(self, train_dir, transform=None) -> None:
        # 下载数据集(非图片格式)
        self.train_data = datasets.MNIST('../dataset/mnist', train=True, download=True)
        self.train_dir = train_dir
        self.transform = transform

        # 将其转为图片格式
        transform_to_image(self.train_data, save_path=train_dir)

        # 获取目录下图片的路径
        self.img_paths = os.listdir(train_dir)


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, self.img_paths[index])
        image = Image.open(image_path)
        # 对图像数据进行处理，由于mnist图片均为28x28, 无须Resize, 这里只将其转为tensor形式
        if self.transform:
            image = self.transform(image)

        return image    
    
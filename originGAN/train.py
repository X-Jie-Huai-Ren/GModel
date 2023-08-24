
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from tqdm import tqdm
from dataset import MNISTDataset
import config
from utils import Compose
from model import Generator, Discriminator


# 单周期训练
def train(train_loader, generator, discriminator, gen_opt, dis_opt, criterion):
    """
    Params:
        ...
    Return:

    """
    loop = tqdm(train_loader, leave=False)  # 设置参数leave=False, 结果只显示在一行
    for real in loop:
        # 将真实图像的数据拷贝到device上
        real = real.view(-1, 784).to(config.DEVICE)
        # 生成输入噪声
        batch_size = real.shape[0]
        z = torch.randn(size=(batch_size, config.Z_DIM)).to(config.DEVICE)
        # 将噪声输入Generator, 生成假图像
        fake = generator(z)
        
        # Train the Discriminator: maximize log(D(real)) + log(1-D(G(z)))
        discri_real = discriminator(real)
        real_loss = criterion(discri_real, torch.ones_like(discri_real))
        print(real_loss)
        print(real_loss.shape)


        break 


def main():
    # 数据转换形式
    transform = Compose([transforms.ToTensor()])
    # 加载数据集(从本地加载)
    train_dataset = MNISTDataset(train_dir=config.IMAGE_DIR, transform=transform)
    # 小批量加载数据集
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True
    )
    mean_loss = []
    # Model
    generator = Generator(z_dim=config.Z_DIM, img_dim=config.IMG_DIM).to(config.DEVICE)
    discriminator = Discriminator(img_dim=config.IMG_DIM).to(config.DEVICE)
    # OPtimizer
    gen_optimizer = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE)
    # loss function: -w[ylogx+(1-y)log(1-x)]
    criterion = nn.BCELoss()

    for _ in range(config.NUM_EPOCHS):
        train(train_loader, generator, discriminator, gen_optimizer, dis_optimizer, criterion)
        break



if __name__ == '__main__':

    main()
    
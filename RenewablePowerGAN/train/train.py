"""
Train for RenewablePowerGAN, the dataset is ChangChun dataset, because the dataset that was previously used has too many samples, and include
multiple sites, the trained model is badly bad

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-10-30
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm import tqdm

from dataset import ChangChuanDataset
from model.mlp import Generator, Discriminator
import config
from utils import build_log_folder, save_checkpoints


def train(gen, disc, opt_gen, opt_disc, train_loader, criterion, epoch):
    
    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    loop.set_description(f'epoch:{epoch}')
    dis_loss_epoch = []
    gen_loss_epoch = []
    fake_score_epoch = []
    real_score_epoch = []

    for real in loop:

        real = real.to(config.DEVICE)
        # randomly generate the input noise
        z = torch.randn(size=(config.BATCH_SIZE, config.Z_DIM)).to(config.DEVICE)
        # fake
        fake = gen(z)

        # Train the Discriminator: maximize log(D(real)) + log(1-D(G(z)))
        real_score = disc(real)
        real_loss = criterion(real_score, torch.ones_like(real_score))
        fake_score = disc(fake)
        fake_loss = criterion(fake_score, torch.zeros_like(fake_score))
        dis_loss = (real_loss + fake_loss) / 2  # 不用加负号, 因为BCELoss中自带一个负号，最大化转最小化
        opt_disc.zero_grad()
        dis_loss.backward(retain_graph=True)  # 如果对某一变量有第二次backward, 需要保持计算图
        opt_disc.step()
        # 记录日志
        dis_loss_epoch.append(round(float(dis_loss.detach().cpu().numpy()), 3))
        real_score_epoch.append(float(torch.mean(real_score)))
        fake_score_epoch.append(float(torch.mean(fake_score)))

        # Train the Genarator: minimize log(D(real)) + log(1-D(G(z)))
        real_score1 = disc(real)
        real_loss1 = criterion(real_score1, torch.ones_like(real_score1))
        fake_score1 = disc(fake)
        fake_loss1 = criterion(fake_score1, torch.zeros_like(fake_score1))
        gen_loss = -(real_loss1 + fake_loss1)   # 加负号, 因为BCELoss中自带一个负号
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        # 记录日志
        gen_loss_epoch.append(round(float(gen_loss.detach().cpu().numpy()), 3))

    return (
        sum(dis_loss_epoch) / len(dis_loss_epoch), 
        sum(gen_loss_epoch) / len(gen_loss_epoch), 
        sum(real_score_epoch) / len(real_score_epoch), 
        sum(fake_score_epoch) / len(fake_score_epoch)
    )

def main():

    # Read the dataset
    solardataset = ChangChuanDataset('./data/changchun.xlsx')
    # dataloader
    train_loader = DataLoader(
        dataset=solardataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # Generator/Discriminator
    generator = Generator(config.Z_DIM, config.OUTPUT_DIM).to(config.DEVICE)
    discriminator = Discriminator(input_dim=config.OUTPUT_DIM).to(config.DEVICE)

    # optimizer
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G)
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE_D)

    # loss function: -w[ylogx+(1-y)log(1-x)]
    criterion = nn.BCELoss()

    # 日志和参数保存
    log_dir = build_log_folder()

    writer = SummaryWriter(log_dir)

    # Start training
    for epoch in range(config.NUM_EPOCHS):

        # train for epoch
        dis_loss, gen_loss, real_score, fake_score = train(generator, discriminator, opt_gen, opt_disc, train_loader, criterion, epoch)

        # 保存模型参数
        if epoch % 200 == 0:
            checkpoints = {
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'generator_optimizer': opt_gen.state_dict(),
                'discriminator_optimizer': opt_disc.state_dict()
            }
            save_checkpoints(checkpoints, log_dir, epoch)

        writer.add_scalar(tag='ModelLoss/generator loss', scalar_value=gen_loss, global_step=epoch)
        writer.add_scalar(tag='ModelLoss/discriminator loss', scalar_value=dis_loss, global_step=epoch)
        writer.add_scalar(tag='Score/real score', scalar_value=real_score, global_step=epoch)
        writer.add_scalar(tag='Score/fake score', scalar_value=fake_score, global_step=epoch)


if __name__ == '__main__':
    
    main()

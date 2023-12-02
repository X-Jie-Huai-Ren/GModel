"""
Train for RenewablePowerHAN, the algorithm to be used is WGAN, is weight-clipping, not Gradient penalty

* @author: xuan
* @email: 1920425406@qq.com
* @date: 203=23-11-15
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\GANS\RenewablePowerGAN')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm import tqdm
import importlib

from dataset import ChangChuanDataset
from utils import build_log_folder, save_checkpoints
import config


def train(gen, disc, opt_gen, opt_disc, train_loader, epoch, interval):

    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    loop.set_description(f'epoch:{epoch}')
    dis_loss_epoch = []
    gen_loss_epoch = []
    fake_score_epoch = []
    real_score_epoch = []

    for real in loop:
        real = real.to(config.DEVICE)

        for _ in range(interval):
            # randomly generate the input noise
            z = torch.randn(size=(config.BATCH_SIZE, config.Z_DIM)).to(config.DEVICE)
            fake = gen(z)

            # Train the Discriminator: maximize E[D(real)]-E[D(fake)]
            real_score = disc(real)
            fake_score = disc(fake)
            disc_loss = -(torch.mean(real_score) - torch.mean(fake_score))
            disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()

            # weight clipping
            for p in disc.parameters():
                p.data.clamp_(-config.WEIGHT_CLIP, config.WEIGHT_CLIP)

        # 记录日志
        dis_loss_epoch.append(round(float(disc_loss.detach().cpu().numpy()), 3))
        real_score_epoch.append(float(torch.mean(real_score)))
        fake_score_epoch.append(float(torch.mean(fake_score)))

        # Train the Generator: minimize -E[D(fake)] 
        output = disc(fake).reshape(-1)
        gen_loss = -torch.mean(output)
        gen.zero_grad()
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


def main(arg_dict):

    # Read the dataset
    solardataset = ChangChuanDataset('./data/changchun.xlsx', normed=arg_dict["norm"])
    # make a dataloader
    train_loader = DataLoader(
        dataset=solardataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # Generator and Discriminator
    impoted_model = importlib.import_module("model." + arg_dict["model"])
    generator = impoted_model.Generator(input_dim=config.Z_DIM, output_dim=config.OUTPUT_DIM).to(config.DEVICE)
    discriminator = impoted_model.Discriminator(input_dim=config.OUTPUT_DIM).to(config.DEVICE)
    # initialize the weights
    if config.INIT_MODEL:
        impoted_model.initialize_weights(generator)
        impoted_model.initialize_weights(discriminator)

    # optimizer
    opt_gen = optim.RMSprop(generator.parameters(), lr=config.LEARNING_RATE_G)
    opt_disc = optim.RMSprop(discriminator.parameters(), lr=config.LEARNING_RATE_D)

    # 日志和参数保存
    log_dir = build_log_folder()
    writer = SummaryWriter(log_dir)

    # Training mode
    generator.train()
    discriminator.train()

    # Start training
    for epoch in range(config.NUM_EPOCHS):

        # train for epoch
        dis_loss, gen_loss, real_score, fake_score = train(generator, discriminator, opt_gen, opt_disc, train_loader, epoch, config.CRITIC_ITERATIONS)

        # 保存模型参数
        if (epoch+1) % 1000 == 0:
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

    arg_dict = {
        "data": './data/changchun.xlsx',
        "standardize": True, 
        "model": 'mlp_wgan_bt',   # mlp, mlp_wgan, mlp_wgan_bt
    }
    
    main(arg_dict)

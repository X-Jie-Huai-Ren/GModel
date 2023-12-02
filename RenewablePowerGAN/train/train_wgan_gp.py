"""
Train for RenewablePowerHAN, the algorithm to be used is WGAN, is Gradient penalty, not weight-clipping
Compare the weight-clipping, the difference includes:
    first, while optimizing the discriminator, not weight-clipping, is gradient-penalty
    second, the optimizer is Adam, not RMSprop

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
from eval.mmd import MMDLoss
from utils import build_log_folder, save_checkpoints, gradient_penalty
import config


def train(gen, disc, opt_gen, opt_disc, train_loader, MMD, epoch, interval):

    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    loop.set_description(f'epoch:{epoch}')
    dis_loss_epoch = []
    gen_loss_epoch = []
    fake_score_epoch = []
    real_score_epoch = []
    mmdloss_epoch = []

    for real in loop:
        real = real.to(config.DEVICE)
        # batch_size, when calculate the gradient penalty, The dimensions of the fake data must be the same as the dimensions of the real data
        batch_size = real.shape[0]

        for _ in range(interval):
            # randomly generate the input noise
            z = torch.randn(size=(batch_size, config.Z_DIM)).to(config.DEVICE)
            fake = gen(z)

            # Train the Discriminator: maximize E[D(real)]-E[D(fake)]
            real_score = disc(real)
            fake_score = disc(fake)
            # gradient penalty
            gp = gradient_penalty(disc, real, fake, device=config.DEVICE)
            disc_loss = (
                -(torch.mean(real_score) - torch.mean(fake_score)) + config.LAMBDA_GP * gp
            )
            disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()

            # n0 weight clipping
            # for p in disc.parameters():
                # p.data.clamp_(-config.WEIGHT_CLIP, config.WEIGHT_CLIP)

        # 记录日志
        dis_loss_epoch.append(round(float(disc_loss.detach().cpu().numpy()), 3))
        real_score_epoch.append(float(torch.mean(real_score)))
        fake_score_epoch.append(float(torch.mean(fake_score)))
        mmdloss_epoch.append(MMD(fake, real))

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
        sum(fake_score_epoch) / len(fake_score_epoch),
        sum(mmdloss_epoch) / len(mmdloss_epoch),
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
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G, betas=(0, 0.9))  # in the wgan-gp, the optimizer is Adam, and the momentem is the same as paper
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE_D, betas=(0, 0.9))

    # MMDLoss
    MMD = MMDLoss()

    # 日志和参数保存
    log_dir = build_log_folder()
    writer = SummaryWriter(log_dir)

    # Training mode
    generator.train()
    discriminator.train()

    # Start training
    for epoch in range(config.NUM_EPOCHS):

        # train for epoch
        dis_loss, gen_loss, real_score, fake_score, mmdloss = train(generator, discriminator, opt_gen, opt_disc, train_loader, MMD, epoch, config.CRITIC_ITERATIONS)

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
        writer.add_scalar(tag='MMDLoss', scalar_value=mmdloss, global_step=epoch)



if __name__ == '__main__':

    arg_dict = {
        "data": './data/changchun.xlsx',
        "norm": 'standard',   # norm or standard 
        "model": 'mlp_wgan_bt',   # mlp, mlp_wgan, mlp_wgan_bt
    }
    
    main(arg_dict)

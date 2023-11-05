"""
Training for CycleGAN

* @author: xuansd
* @email: 1920425405@qq.com
* @date: 2023-10-22
"""

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import CycleGANDataset
from config import Arg
from Discriminator import Discriminator
from Generator import Generator
from utils import build_log_folder, save_checkpoints


def train(discriminator_x, discriminator_y, generator_y, generator_x, train_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    """
    Training for epoch
    """
    x_reals = 0
    x_fakes = 0
    loop = tqdm(train_loader, leave=False)
    loop.set_description(f'Epoch:{epoch}')

    x_to_y_score = []
    y_to_y_score = []
    y_to_x_score = []
    x_to_x_score = []
    D_loss_lst = []
    G_loss_lst = []

    for idx, (x, y) in enumerate(loop):
        x = x.to(Arg["device"])
        y = y.to(Arg["device"])

        # Train Discriminators x and y
        # with torch.cuda.amp.autocast():
        fake_x = generator_x(y)
        D_x_real = discriminator_x(x)
        D_x_fake = discriminator_x(fake_x.detach())
        x_reals += D_x_real.mean().item()
        x_fakes += D_x_fake.mean().item()
        D_x_real_loss = mse(D_x_real, torch.ones_like(D_x_real))
        D_x_fake_loss = mse(D_x_fake, torch.zeros_like(D_x_fake))
        D_x_loss = D_x_real_loss + D_x_fake_loss

        fake_y = generator_y(x)
        D_y_real = discriminator_y(y)
        D_y_fake = discriminator_y(fake_y.detach())
        D_y_real_loss = mse(D_y_real, torch.ones_like(D_y_real))
        D_y_fake_loss = mse(D_y_fake, torch.zeros_like(D_y_fake))
        D_y_loss = D_y_real_loss + D_y_fake_loss

        # put it togethor
        D_loss = (D_x_loss + D_y_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # 记录日志
        D_loss_lst.append(round(float(D_loss.detach().cpu().numpy()), 3))
        x_to_y_score.append(float(torch.mean(D_y_fake)))
        y_to_y_score.append(float(torch.mean(D_y_real)))
        y_to_x_score.append(float(torch.mean(D_x_fake)))
        x_to_x_score.append(float(torch.mean(D_x_real)))

        # Train Generators x and y
        # with torch.cuda.amp.autocast():
            # adversarial loss for both generators
        D_x_fake = discriminator_x(fake_x)
        D_y_fake = discriminator_y(fake_y)
        loss_G_x = mse(D_x_fake, torch.ones_like(D_x_fake))
        loss_G_y = mse(D_y_fake, torch.ones_like(D_y_fake))

        # cycle loss
        cycle_y = generator_y(fake_x)
        cycle_x = generator_x(fake_y)
        cycle_y_loss = L1(y, cycle_y)
        cycle_x_loss = L1(x, cycle_x)

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        identity_y = generator_y(y)
        identity_x = generator_x(x)
        identity_y_loss = L1(y, identity_y)
        identity_x_loss = L1(x, identity_x)

        # add all togethor
        G_loss = (
            loss_G_y
            + loss_G_x
            + cycle_y_loss * 10
            + cycle_x_loss * 10
            + identity_x_loss * 0.0
            + identity_y_loss * 0.0
        )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # 记录日志
        G_loss_lst.append(round(float(G_loss.detach().cpu().numpy()), 3))

        if idx % 200 == 0:
            save_image(fake_x * 0.5 + 0.5, f"saved_images/x_{idx}.png")
            save_image(fake_y * 0.5 + 0.5, f"saved_images/y_{idx}.png")

        # loop.set_postfix(H_real=x_reals / (idx + 1), H_fake=x_fakes / (idx + 1))

        return (
            sum(D_loss_lst) / len(D_loss_lst),
            sum(G_loss_lst) / len(G_loss_lst),
            sum(x_to_y_score) / len(x_to_y_score),
            sum(y_to_y_score) / len(y_to_y_score),
            sum(y_to_x_score) / len(y_to_x_score),
            sum(x_to_x_score) / len(x_to_x_score)
        )
    


if __name__ == '__main__':

    # create the Vangogh dataset
    vangoghdataset_train = CycleGANDataset(Arg["trainx_dir"], Arg["trainy_dir"], transform=Arg["transform"])
    vangoghdataset_test = CycleGANDataset(Arg["testx_dir"], Arg["testy_dir"], transform=Arg["transform"])

    # the train dataloader
    train_loader = DataLoader(
        dataset=vangoghdataset_train,
        batch_size=Arg["batch_size"],
        num_workers=Arg["num_workers"],
        pin_memory=True,
        shuffle=True
    )
    # the test dataloader
    test_loader = DataLoader(
        dataset=vangoghdataset_test,
        batch_size=Arg["batch_size"],
        num_workers=Arg["num_workers"],
        pin_memory=True,
        shuffle=False
    )

    # device
    device = Arg["device"]

    # the generator and discriminator
    discriminator_x = Discriminator(in_channels=3).to(device)
    discriminator_y = Discriminator(in_channels=3).to(device)
    generator_y = Generator(img_channels=3, num_residuals=9).to(device)
    generator_x = Generator(img_channels=3, num_residuals=9).to(device)
    # the optimizer
    opt_disc = optim.Adam(
        list(discriminator_x.parameters()) + list(discriminator_y.parameters()),
        lr=Arg["lr"],
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(generator_y.parameters()) + list(generator_x.parameters()),
        lr=Arg["lr"],
        betas=(0.5, 0.999)
    )

    # the loss func
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # 日志和模型保存
    log_dir = build_log_folder()
    writer = SummaryWriter(log_dir)

    # TO TRAIN
    for epoch in range(Arg["epochs"]):

        D_loss, G_loss, xtoy_score, ytoy_score, ytox_score, xtox_score = train(discriminator_x, discriminator_y, generator_y, generator_x, train_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        # 保存模型参数
        if epoch % 50 == 0:
            checkpoints = {
                'generator_x': generator_x.state_dict(),
                'generator_y': generator_y.state_dict(),
                'discriminator_x': discriminator_x.state_dict(),
                'discriminator_y': discriminator_y.state_dict(),
                'generator_optimizer': opt_gen.state_dict(),
                'discriminator_optimizer': opt_disc.state_dict()
            }
            save_checkpoints(checkpoints, log_dir, epoch)

        writer.add_scalar(tag='ModelLoss/generator loss', scalar_value=G_loss, global_step=epoch)
        writer.add_scalar(tag='ModelLoss/discriminator loss', scalar_value=D_loss, global_step=epoch)
        writer.add_scalar(tag='Score/x_to_y_score', scalar_value=xtoy_score, global_step=epoch)
        writer.add_scalar(tag='Score/y_to_y_score', scalar_value=ytoy_score, global_step=epoch)
        writer.add_scalar(tag='Score/y_to_x_score', scalar_value=ytox_score, global_step=epoch)
        writer.add_scalar(tag='Score/x_to_x_score', scalar_value=xtox_score, global_step=epoch)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset import MNISTDataset
import config
from utils import Compose, save_checkpoints, build_log_folder
from model import Generator, Discriminator
from surgery import Surgery

# 单周期训练
def train(train_loader, generator, discriminator, gen_opt, dis_opt, criterion, epoch, writer):
    """
    Params:
        ...
    Return:

    """
    loop = tqdm(train_loader, leave=False)  # 设置参数leave=False, 结果只显示在一行
    loop.set_description(f'Epoch:{epoch}')
    dis_loss_epoch = []
    gen_loss_epoch = []
    fake_score_epoch = []
    real_score_epoch = []
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
        discri_fake = discriminator(fake)
        fake_loss = criterion(discri_fake, torch.zeros_like(discri_fake))
        dis_loss = (real_loss + fake_loss) / 2  # 不用加负号, 因为BCELoss中自带一个负号，最大化转最小化
        dis_opt.zero_grad()
        dis_loss.backward(retain_graph=True)  # 如果对某一变量有第二次backeard, 需要保持计算图
        dis_opt.step()
        # 记录日志
        dis_loss_epoch.append(round(float(dis_loss.detach().cpu().numpy()), 3))
        real_score_epoch.append(float(torch.mean(discri_real, dim=0)))
        fake_score_epoch.append(float(torch.mean(discri_fake, dim=0)))

        # Train the Genarator: minimize log(D(real)) + log(1-D(G(z)))
        discri_real1 = discriminator(real)
        real_loss1 = criterion(discri_real1, torch.ones_like(discri_real1))
        discri_fake1 = discriminator(fake)
        fake_loss1 = criterion(discri_fake1, torch.zeros_like(discri_fake1))
        gen_loss = -(real_loss1 + fake_loss1)   # 加负号, 因为BCELoss中自带一个负号
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        # 记录日志
        gen_loss_epoch.append(round(float(gen_loss.detach().cpu().numpy()), 3))

    # 将生成的图片展示出来
    with torch.no_grad():
        fake_fixed = generator(config.Z_FIXED).reshape((-1, 1, 28, 28))
        real = real.reshape((-1, 1, 28, 28))
        img_grid_fake = make_grid(fake_fixed, normalize=True)
        img_grid_real = make_grid(real, normalize=True)
        writer.add_image(
            "Mnist Fake Image", img_grid_fake, global_step = epoch
        )
        writer.add_image(
            "Mnist Real Image", img_grid_real, global_step = epoch
        )

    return (sum(dis_loss_epoch) / len(dis_loss_epoch), sum(gen_loss_epoch) / len(gen_loss_epoch), 
            sum(real_score_epoch) / len(real_score_epoch), sum(fake_score_epoch) / len(fake_score_epoch))

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
    # Model
    generator = Generator(z_dim=config.Z_DIM, img_dim=config.IMG_DIM).to(config.DEVICE)
    discriminator = Discriminator(img_dim=config.IMG_DIM).to(config.DEVICE)
    # OPtimizer
    gen_optimizer = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE)
    # loss function: -w[ylogx+(1-y)log(1-x)]
    criterion = nn.BCELoss()

    # 日志和参数保存
    log_dir = build_log_folder()

    writer = SummaryWriter(log_dir)

    for epoch in range(config.NUM_EPOCHS):

        dis_loss, gen_loss, real_score, fake_score = train(train_loader, generator, discriminator, gen_optimizer, dis_optimizer, criterion, epoch, writer)

        # 保存模型参数
        if epoch % 50 == 0:
            checkpoints = {
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'generator_optimizer': gen_optimizer.state_dict(),
                'discriminator_optimizer': dis_optimizer.state_dict()
            }
            save_checkpoints(checkpoints, log_dir, epoch)

        writer.add_scalar(tag='ModelLoss/generator loss', scalar_value=gen_loss, global_step=epoch)
        writer.add_scalar(tag='ModelLoss/discriminator loss', scalar_value=dis_loss, global_step=epoch)
        writer.add_scalar(tag='Score/real score', scalar_value=real_score, global_step=epoch)
        writer.add_scalar(tag='Score/fake score', scalar_value=fake_score, global_step=epoch)



if __name__ == '__main__':

    main()
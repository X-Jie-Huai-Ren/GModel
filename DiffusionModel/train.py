"""
the main script for diffusion model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-03-21
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from tqdm import tqdm



from forward import ForwardProcess
from backward import Unet
from dataset import DMDataset
from utils import transform, bulid_log_dir


class Trainer:
    """ the train manager """
    def __init__(self, arg_dict, forwardModel, backwardModel, optimizer, train_loader) -> None:
        self.arg_dict = arg_dict
        self.fp = forwardModel
        self.backwardModel = backwardModel
        self.optimizer = optimizer
        self.train_loader = train_loader
        # loss func
        self.loss_func = nn.L1Loss()
        # device
        self.device = self.arg_dict['device']
        # tensorboard writer
        self.writer = SummaryWriter(self.arg_dict['log_dir'])


    def _train_for_epoch(self, epoch):
        
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)
        loop.set_description(f'epoch {epoch}')
        loss_list = []

        for imgs in loop:
            imgs = imgs.to(self.device)
            t = torch.randint(0, self.arg_dict['T'], size=(self.arg_dict['batch_size'],)).to(self.device)
            noisy_imgs, noises = self.fp(imgs, t)
            noise_pred = self.backwardModel(noisy_imgs, t)
            loss = self.loss_func(noises, noise_pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            loss_list.append(round(float(loss.detach().cpu().numpy()), 3))

        return sum(loss_list) / len(loss_list)


    
    def train(self):

        loss_max = 1000

        for epoch in range(self.arg_dict['num_epochs']):

            # train for epoch
            loss = self._train_for_epoch(epoch)

            # save the model
            if loss < loss_max:
                loss_max = loss
                self._save_model(epoch, min=True)
            if epoch > 200 and (epoch+1) % 100 == 0:
                self._save_model(epoch)

            # 记录日志
            self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=epoch)



    def _save_model(self, epoch, min=False):
        """
        """
        if not os.path.exists(self.arg_dict['log_dir']):
            os.makedirs(self.arg_dict['log_dir'])
        checkpoints = self.backwardModel.state_dict()
        if min:
            path = self.arg_dict['log_dir'] + '/minloss.tar'
        else:
            path = self.arg_dict['log_dir'] + f'/model_{epoch}.tar'
        print(f'==> Saving checkpoints: {epoch}')
        torch.save(checkpoints, path)



def main(arg_dict):
    """
    Params:
        arg_dict: the hyper-parameters
    """
    # create the transform mode
    transforms = transform()

    # load the dataset
    train_dataset = DMDataset(arg_dict['train'], transform=transforms)
    test_dataset = DMDataset(arg_dict['test'], transform=transforms)
    # create the dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=arg_dict['batch_size'],
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=arg_dict['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    # device
    device = arg_dict['device']
    # create log dir
    arg_dict['log_dir'] = bulid_log_dir(dir='./logs')

    # create the ForwardProcess Model
    fp = ForwardProcess(arg_dict['T']).to(device)
    # create backward Model to predict the noise of timestep t
    UnetModel = Unet().to(device)
    # optimizer
    optimizer = optim.Adam(UnetModel.parameters(), lr=arg_dict['lr'])

    # Training mode
    UnetModel.train()

    # create trainer
    trainer = Trainer(arg_dict, fp, UnetModel, optimizer, train_loader)
    trainer.train()

    

if __name__ == '__main__':

    arg_dict = {
        "train": './data/train',
        "test": './data/test',
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "T": 300,  # timesteps
        "lr": 2e-5,
        "num_epochs": 5000
    }

    main(arg_dict)
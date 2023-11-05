"""
generate vangogh images from origin images

* @author: xuansd
* @email: 1920425405@qq.com
* @date: 2023-10-31
"""

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from dataset import CycleGANDataset
from config import Arg
from Generator import Generator







if __name__ == '__main__':

    # Read the test data
    vangoghdataset_test = CycleGANDataset(Arg["testx_dir"], Arg["testy_dir"], transform=Arg["transform"])

    # test loader
    test_loader = DataLoader(
        dataset=vangoghdataset_test,
        batch_size=Arg["batch_size"],
        shuffle=False,
        pin_memory=True
    )

    # load the generator_x and the generator_y
    generator_x = Generator(img_channels=3)
    generator_y = Generator(img_channels=3)

    # load the weight
    checkpoints = torch.load('./logs/[11-02]20.06.51/model_0.tar')
    generator_x.load_state_dict(checkpoints["generator_x"])
    generator_y.load_state_dict(checkpoints["generator_y"])

    for x, y in test_loader:
        # x --> y
        fake_y = generator_y(x)
        # y --> x
        fake_x = generator_x(y)

        # fake_x = fake_x.permute(0, 2, 3, 1)
        # fake_y = fake_y.permute(0, 2, 3, 1)
        # fake_y0 = fake_y[0].detach().numpy() * 0.5 + 0.5
        save_image(fake_y*0.5 + 0.5, './saved_images/fakey/fake_y0.png')
        save_image(fake_x*0.5 + 0.5, './saved_images/fakex/fake_x0.png')
        # plt.imshow(fake_y[0].numpy())
        # plt.show()
        break


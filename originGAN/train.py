
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import MNISTDataset
import config
from utils import Compose
from model import Generator
from model import Discriminator




def train():

    pass


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
    generator = Generator()

    for _ in range(config.NUM_EPOCHS):
        pass



if __name__ == '__main__':

    main()
    
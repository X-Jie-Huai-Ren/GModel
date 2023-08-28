"""
    Hyper Parameters
"""
import torch

# Hyper Parameters etc.


NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
Z_DIM = 128
IMG_DIM = 28 * 28 * 1
IS_TRAINING = True
LOAD_MDEOL = False
LOAD_MDEOL_FILE = './logs'
IMAGE_DIR = '../dataset/mnist/train'
NUM_WORKERS = 0
PIN_MEMORY = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_FIXED= torch.randn(size=(BATCH_SIZE, Z_DIM)).to(DEVICE)


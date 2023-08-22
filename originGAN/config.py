"""
    Hyper Parameters
"""
import torch

# Hyper Parameters etc.


NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
Z_DIM = 128
IMG_DIM = 28 * 28 * 1
IS_TRAINING = True
LOAD_MDEOL = False
LOAD_MDEOL_FILE = 'model.pth.tar'
IMAGE_DIR = '../dataset/mnist/train'
NUM_WORKERS = 0
PIN_MEMORY = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

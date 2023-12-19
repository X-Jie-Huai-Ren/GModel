"""
Hyper Parameters for RenewablePowerGAN

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-10-30
"""

import torch

# Hyper Parameters
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_DIM = 100
OUTPUT_DIM = 24   # The temporal resolution is 1h
NUM_EPOCHS = 10000
LEARNING_RATE_D = 2e-4
LEARNING_RATE_G = 1e-3
LOAD_MDEOL_FILE = './logs/CGAN'
TRAINED_MODEL = False
WEIGHT_CLIP = 0.1      # WGAN for weight clipping
CRITIC_ITERATIONS = 5  # training trick
INIT_MODEL = False     # initialize the model or not
LAMBDA_GP = 10         # according to the paper wgan-gp
EMBEDDING_SIZE = 28    # for c-gan, labels embedding

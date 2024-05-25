import torch

HIDDEN_SIZE = 512
RECURRENT_LAYERS = 1
LEARNING_RATE = 0.0003
GAMMA = 0.99
C1 = 0.95
LAMBDA = 0.95
ROLL_OUT = 3600
BATCH_SIZE = 64
BATCH_LEN = 32
PPO_CLIP = 0.2
ENTROPY_FACTOR = 0.01
VF_FACTOR = 1
MAX_GRAD_NORM = 1.0
ACTION_NUM = 40

BASE_CHECKPOINT_PATH = 'ppo_pytorch/checkpoints'
GATHER_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

STATE_DIM = {
    1: {
        'conv1d': 160,
        'fft': 512,
        'mel': 2560
    },
    4: {
        'conv1d': 64,
        'fft': 512,
        'mel': 1280
    }
}

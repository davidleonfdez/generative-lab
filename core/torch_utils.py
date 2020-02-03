import torch.nn as nn


def get_relu(leaky:float=None):
    return nn.ReLU() if leaky is None else nn.LeakyReLU(leaky)

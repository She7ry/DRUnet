import torch
import torch.nn as nn
from module.EMA import EMA
from module.ELA import ELA


class MLDA(nn.Module):
    def __init__(self, channels, phi="T", ema_factor=32):
        super().__init__()
        self.ema = EMA(channels=channels, factor=ema_factor)
        self.ela = ELA(in_channels=channels, phi=phi)

    def forward(self, x):
        identity = x
        x_ema = self.ema(x)
        x_ema_ela = self.ela(x_ema)
        x = x_ema_ela + x_ema + identity
        return x


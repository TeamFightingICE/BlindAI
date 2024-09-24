from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(BaseEncoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.FPS = fps
        self.frame_skip = frame_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left side
        left = x[:, :, 0]
        left = self.encode_single_channel(left)
        # right side
        right = x[:, :, 1]
        right = self.encode_single_channel(right)
        return torch.cat((left, right), dim=1)

    @abstractmethod
    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        pass
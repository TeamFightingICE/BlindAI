import torch
import torch.nn.functional as F

from src.encoders.base import BaseEncoder


class RawEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(RawEncoder, self).__init__(sampling_rate, fps, frame_skip)
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.frame_skip
        assert int(self.num_samples) == self.num_samples

        # Encoder (small 1D conv)
        self.pool = torch.nn.MaxPool1d(2)
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=16, stride=8)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=16, stride=8)

    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        """Shape of x: [batch_size, num_samples]"""
        # Subsample
        x = data[:, ::self.num_to_subsample]

        # Add channel dimension
        x = x[:, None, :]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if x.shape[2] >= 24:
            x = self.conv2(x)
            x = self.pool(x)
        return x
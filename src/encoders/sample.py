import torch

from src.encoders.base import BaseEncoder


class SampleEncoder(BaseEncoder):
    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        return data
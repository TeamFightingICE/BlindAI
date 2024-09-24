from dataclasses import dataclass

import torch


@dataclass
class TrajectorBatch():
    """
    Dataclass for storing data batch.
    """
    states: torch.Tensor
    actions: torch.Tensor
    action_probabilities: torch.Tensor
    advantages: torch.Tensor
    discounted_returns: torch.Tensor
    batch_size: torch.Tensor

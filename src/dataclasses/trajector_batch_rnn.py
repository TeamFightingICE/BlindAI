from dataclasses import dataclass

import torch


@dataclass
class TrajectorBatchRNN():
    """
    Dataclass for storing data batch.
    """
    states: torch.Tensor
    actions: torch.Tensor
    action_probabilities: torch.Tensor
    advantages: torch.Tensor
    discounted_returns: torch.Tensor
    batch_size: torch.Tensor
    actor_hidden_states: torch.Tensor
    critic_hidden_states: torch.Tensor

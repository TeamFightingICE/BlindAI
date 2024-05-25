import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 256


class FeedForwardCritic(nn.Module):
    def __init__(self, state_dim, hidden_size, recurrent_layers, encoder=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.value_layer = nn.Linear(HIDDEN_SIZE_2, 1)
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, state, terminal=None) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device
        # perform audio encoder
        if self.encoder is not None:
            state_encoded = self.relu(self.flatten(self.encoder(state)))
        else:
            state_encoded = state
        hidden1 = F.elu(self.fc1(state_encoded))
        hidden2 = F.elu(self.fc2(hidden1))
        value_out = self.value_layer.forward(hidden2)
        return value_out
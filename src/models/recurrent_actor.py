import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 256


class RecurrentActor(nn.Module):
    def __init__(self, state_dim, hidden_size, recurrent_layers, encoder=None, action_num=40):
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.gru = nn.GRU(state_dim, hidden_size, recurrent_layers)
        self.fc1 = nn.Linear(hidden_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.action_layer = nn.Linear(HIDDEN_SIZE_2, action_num)
        self.hidden_cell = None
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def get_init_state(self, device):
        self.hidden_cell = torch.zeros(self.recurrent_layers, self.hidden_size).to(device)

    def forward(self, state, terminal=None) -> distributions.Categorical:
        # batch_size = state.shape[1]
        device = state.device
        # perform audio encoder
        if self.encoder is not None:
            state_encoded = self.relu(self.flatten(self.encoder(state)))

        if self.hidden_cell is None:  # or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(device)
        if terminal is not None:
            self.hidden_cell = (self.hidden_cell * (1. - terminal))#.reshape(-1,)

        try:
            rnn_output, self.hidden_cell = self.gru(state_encoded, self.hidden_cell)
        except Exception as ex:
            raise ex
        # hidden1 = F.elu(self.fc1(self.hidden_cell[0][-1]))
        # hidden1 = F.elu(self.fc1(self.hidden_cell[-1]))
        hidden1 = F.elu(self.fc1(rnn_output))
        hidden2 = F.elu(self.fc2(hidden1))
        policy_logits_out = self.action_layer(hidden2)
        policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to(device))
        return policy_dist

    def act(self, state, terminal=None):
        # batch_size = state.shape[1]
        device = state.device
        # perform audio encoder
        if self.encoder is not None:
            state_encoded = self.relu(self.flatten(self.encoder(state)))

        if self.hidden_cell is None:  # or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(device)
        if terminal is not None:
            self.hidden_cell = (self.hidden_cell * (1. - terminal))#.reshape(-1,)

        try:
            rnn_output, self.hidden_cell = self.gru(state_encoded, self.hidden_cell)
        except Exception as ex:
            raise ex
        # hidden1 = F.elu(self.fc1(self.hidden_cell[0][-1]))
        # hidden1 = F.elu(self.fc1(self.hidden_cell[-1]))
        hidden1 = F.elu(self.fc1(rnn_output))
        hidden2 = F.elu(self.fc2(hidden1))
        policy_logits_out = self.action_layer(hidden2)
        return F.softmax(policy_logits_out, dim=1)
        # policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to(device))
        # return policy_dist
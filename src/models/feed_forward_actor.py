import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 256


class FeedForwardActor(nn.Module):
    def __init__(self, state_dim, hidden_size, recurrent_layers, encoder=None, action_num=40):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.action_layer = nn.Linear(HIDDEN_SIZE_2, action_num)
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, state, terminal=None):
        # batch_size = state.shape[1]
        device = state.device
        # perform audio encoder
        if self.encoder is not None:
            state_encoded = self.relu(self.flatten(self.encoder(state)))
        else:
            state_encoded = state
        # print(state_encoded.shape)
        hidden1 = F.elu(self.fc1(state_encoded))
        hidden2 = F.elu(self.fc2(hidden1))
        policy_logits_out = self.action_layer(hidden2)
        policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to(device))
        # print(F.softmax(policy_logits_out, dim=0))
        return policy_dist

    def act(self, state):
        # print(state.sum())
        device = state.device
        # perform audio encoder
        if self.encoder is not None:
            state_encoded = self.relu(self.flatten(self.encoder(state)))
        else:
            state_encoded = state
        # print(state_encoded.shape)
        hidden1 = F.elu(self.fc1(state_encoded))
        hidden2 = F.elu(self.fc2(hidden1))
        policy_logits_out = self.action_layer(hidden2)
        # print(hidden2)
        # policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=0).to(device))
        # print(policy_logits_out)
        policy_logits_out = F.softmax(policy_logits_out, dim=1)
        # print(policy_logits_out)
        # action = torch.argmax(policy_logits_out)
        return policy_logits_out
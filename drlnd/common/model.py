import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer_width = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_layer_width)
        self.fc2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.fc3 = nn.Linear(hidden_layer_width, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, seed, state_size: int, action_size: int, hidden_layer_size: int = 256, output_activation = F.tanh):
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = torch.nn.Linear(state_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = torch.nn.Linear(hidden_layer_size, action_size)

        self.output_activation = output_activation

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.output_activation(x)
        return x
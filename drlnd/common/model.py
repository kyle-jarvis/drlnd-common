import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List


# TODO: Remove references in navigation project in favaour of SimpleFCNetwork
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


class SimpleFCNetwork(nn.Module):
    def __init__(
        self, 
        seed, 
        input_size: int, 
        output_size: int, 
        hidden_layer_size: Union[int, List[int]] = 256, 
        output_activation = F.tanh):

        super(SimpleFCNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        if isinstance(hidden_layer_size, int):
            n_hidden_layers = 2
            dims = [input_size, *n_hidden_layers*[hidden_layer_size], output_size]

        elif isinstance(hidden_layer_size, list):
            dims = [input_size, *hidden_layer_size, output_size]

        self.layers = nn.ModuleList(
            [nn.Linear(x1, x2) for x1, x2 in zip(dims[:-1], dims[1:])]
            )
        nn.init.uniform_(self.layers[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layers[-1].bias, -3e-3, 3e-3)

        self.output_activation = output_activation
        self.gates = [F.relu for x in dims[1:-1]] + [self.output_activation]

    def forward(self, state):
        x = state
        for gate, layer in zip(self.gates, self.layers):
            x = gate(layer(x))
        return x
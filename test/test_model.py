import unittest
import torch.nn.functional as F
import torch
import numpy as np
from drlnd.common.model import SimpleFCNetwork
from drlnd.common.utils import gumbel_softmax_fn


class TestModel(unittest.TestCase):
    def test_simple_fc_network(self):
        network = SimpleFCNetwork(1234, 1, 1, [10, 10])
        print(type(network))
        self.assertIsInstance(network, torch.nn.Module)

    def test_simple_fc_network_activations(self):
        network = SimpleFCNetwork(
            1234,
            10,
            10,
            [10, 10],
            output_activation=lambda x: torch.argmax(x),
            scale_init_output_coef=1.0,
        )
        print(type(network))
        self.assertIsInstance(network, torch.nn.Module)

        # Apply the network to some random inputs:
        outputs = [
            network.forward(torch.from_numpy(x).float())
            for x in np.random.randn(20, 10)
        ]
        print(outputs)

        network = SimpleFCNetwork(
            1234,
            10,
            10,
            [10, 10],
            output_activation=lambda x: F.gumbel_softmax(x.unsqueeze(0)).squeeze(),
            scale_init_output_coef=1.0,
        )
        print(type(network))
        self.assertIsInstance(network, torch.nn.Module)

        # Apply the network to some random inputs:
        outputs = [
            network.forward(torch.from_numpy(x).float())
            for x in np.random.randn(20, 10)
        ]
        print(outputs)

        outputs = [
            network(torch.from_numpy(x).float())
            for x in np.random.randn(20, 10)
        ]
        print(outputs)

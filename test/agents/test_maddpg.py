import unittest
import torch
import numpy as np
from drlnd.common.agents.utils import BrainAgentSpec, AgentInventory, MultiAgentSpec
from drlnd.common.agents.maddpg import AgentSpec, MADDPGAgent2


class TestMADDPG(unittest.TestCase):
    def test_maddpg(self):
        brain_agent_1 = BrainAgentSpec(
            "default_brain", num_agents=2, state_size=32, action_size=4
        )

        agent_inventory = AgentInventory([brain_agent_1])

        maddpg = MADDPGAgent2(agent_inventory, hidden_layer_size=256)

        self.assertIsInstance(maddpg, MADDPGAgent2)

        actions = maddpg.act(
            states={
                "default_brain": [
                    torch.from_numpy(np.zeros(32)).float() for i in range(2)
                ]
            },
            policy_suppression=1.0,
        )

        self.assertIn("default_brain", actions.keys())
        self.assertEqual(2, len(actions["default_brain"]))

    def test_discrete_maddpg(self):
        brain_agent_1 = BrainAgentSpec(
            "default_brain", num_agents=2, state_size=32, action_size=4
        )

        agent_inventory = AgentInventory([brain_agent_1])

        maddpg = MADDPGAgent2(
            agent_inventory,
            hidden_layer_size=256,
            policy_network_kwargs={"output_activation": lambda x: torch.argmax(x)},
        )

        self.assertIsInstance(maddpg, MADDPGAgent2)

        actions = maddpg.act(
            states={
                "default_brain": [
                    torch.from_numpy(np.zeros(32)).float() for i in range(2)
                ]
            },
            policy_suppression=1.0,
        )

        self.assertIn("default_brain", actions.keys())
        self.assertEqual(2, len(actions["default_brain"]))
        self.assertEqual(actions["default_brain"][0].dtype, torch.int64)

        joined_state_tensor = torch.cat(
            [torch.from_numpy(np.random.randn(32)).float() for i in range(2)]
        ).unsqueeze(0)
        results = maddpg._evaluate_actions_using_target_policies(joined_state_tensor)
        print(results)

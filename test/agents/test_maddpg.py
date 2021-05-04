import unittest
import torch
import numpy as np
from drlnd.common.agents.utils import BrainAgentSpec, AgentInventory, MultiAgentSpec
from drlnd.common.agents.maddpg import AgentSpec, MADDPGAgent2
from shutil import rmtree
import os


class TestMADDPG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("./tmp_test_dir"):
            os.mkdir("./tmp_test_dir")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./tmp_test_dir"):
            rmtree("./tmp_test_dir")

    def test_maddpg(self):
        brain_agent_1 = BrainAgentSpec(
            "default_brain", num_agents=2, state_size=32, action_size=4
        )

        agent_inventory = AgentInventory([brain_agent_1])

        maddpg = MADDPGAgent2(agent_inventory, hidden_layer_size=256)

        self.assertIsInstance(maddpg, MADDPGAgent2)

        actions = maddpg.act(
            states={"default_brain": [np.random.randn(32) for i in range(2)]},
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

        # Test we can easily get the actions, given the set of observations, by
        # using the maddpg agent to access and evaluate the policy networks.
        actions = maddpg.act(
            states={"default_brain": [np.random.randn(32) for i in range(2)]},
            policy_suppression=1.0,
        )

        self.assertIn("default_brain", actions.keys())
        self.assertEqual(2, len(actions["default_brain"]))

        # Test we can easily get the actions from the policy/target policy networks
        joined_state_tensor = torch.cat(
            [torch.from_numpy(np.random.randn(32)).float() for i in range(2)]
        ).unsqueeze(0)
        results = maddpg._evaluate_actions_using_target_policies(joined_state_tensor)
        print(results)

        # Test we have the network names we expect
        self.assertIn("default_brain_agent_0_policy", maddpg.networks.keys())
        self.assertIn("default_brain_agent_1_critic", maddpg.networks.keys())
        print(f"INFO: Agent network keys: {maddpg.networks.keys()}\n")

        # Test we can save the networks
        maddpg.save_weights("./tmp_test_dir")
        self.assertTrue(os.path.exists("./tmp_test_dir/model_weights"))
        self.assertTrue(
            os.path.exists(
                "./tmp_test_dir/model_weights/default_brain_agent_0_policy_weights.pth"
            )
        )
        print(f"INFO: Files in directory: {os.listdir('./tmp_test_dir/model_weights')}")

        # Test we can load the netowrks
        maddpg.load_weights("./tmp_test_dir/model_weights")

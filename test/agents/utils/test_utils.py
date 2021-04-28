import unittest
from unittest import skipIf
import torch
import numpy
from drlnd.common.agents.utils import (
    BrainAgentSpec,
    AgentInventory,
    MultiAgentSpec,
    get_unity_env,
    ReplayBuffer,
    ActionType,
    UnityEnvWrapper,
)
from drlnd.common.agents.maddpg import AgentSpec
import os
from unityagents import UnityEnvironment


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if "UNITY_MA_ENV" in os.environ:
            self.env, self.brain_spec = get_unity_env(os.environ["UNITY_MA_ENV"])

    @skipIf("UNITY_MA_ENV" not in os.environ.keys(), "No path to unity executable")
    def test_unity_multiagent_env(self):
        env, brain_spec = self.env, self.brain_spec
        for brain_name in env.brain_names:
            self.assertIn(brain_name, brain_spec.keys())

        get_states = lambda env: {k: env[k].vector_observations for k in env.keys()}
        states = get_states(env.reset(train_mode=False))
        print(states)

    @skipIf("UNITY_MA_ENV" not in os.environ.keys(), "No path to unity executable")
    def test_unity_env_wrapper(self):
        env, brain_spec = self.env, self.brain_spec
        brain_spec_list = list(brain_spec.values())
        print(brain_spec_list)
        env_wrapper = UnityEnvWrapper(env, brain_spec_list)
        env_wrapper.reset(train_mode=False)
        for nsteps in range(2):
            actions = {
                brain_spec_i.name: [
                    numpy.random.randint(brain_spec_i.action_size)
                    for x in range(brain_spec_i.num_agents)
                ]
                for brain_spec_i in brain_spec_list
            }
            env_wrapper.step(vector_action=actions)
            states, actions, rewards, next_states, dones = env_wrapper.sars()
            self.assertIsInstance(states, dict)
            self.assertIsInstance(actions, dict)
            self.assertIsInstance(rewards, dict)
            self.assertIsInstance(next_states, dict)
            self.assertIsInstance(dones, dict)

        buffer = ReplayBuffer(
            None,
            int(5e6),
            1,
            1234,
            action_dtype=ActionType.DISCRETE,
            brain_agents=brain_spec_list,
        )
        print(buffer.brain_order)
        buffer.add_from_dicts(states, actions, rewards, next_states, dones)

        s, a, r, ns, d = buffer.sample()
        print(f"states = \n{s}")
        print(f"actions = \n{a}")
        print(f"rewards = \n{r}")
        print(f"dones = \n{d}")

    def test_brain_agent_spec(self):
        state_size = 32
        sample_brain_agents = BrainAgentSpec(
            "default_brain", num_agents=2, state_size=state_size, action_size=4
        )
        self.assertTrue(hasattr(sample_brain_agents, "name"))
        self.assertTrue(hasattr(sample_brain_agents, "state_size"))
        self.assertEqual(sample_brain_agents.state_size, state_size)

    def test_agent_inventory(self):
        action_size = 4
        brain_agent_1 = BrainAgentSpec(
            "default_brain", num_agents=2, state_size=32, action_size=action_size
        )

        agent_inventory = AgentInventory([brain_agent_1])
        print(agent_inventory)
        self.assertIn(
            "default_brain_agent_0",
            [a.agent_name for a in agent_inventory.agents["default_brain"]],
        )
        self.assertIsInstance(agent_inventory.default_brain_agent_0, MultiAgentSpec)

        test_tensor = torch.zeros((100, 36))
        print(
            f"Action slice sample: {agent_inventory.default_brain_agent_0.action_slice}"
        )
        tensor_slice = test_tensor[agent_inventory.default_brain_agent_0.action_slice]
        self.assertEqual(tensor_slice.shape, torch.Size([100, action_size]))
        tensor_slice = test_tensor[agent_inventory.default_brain_agent_1.action_slice]
        self.assertEqual(tensor_slice.shape, torch.Size([100, action_size]))

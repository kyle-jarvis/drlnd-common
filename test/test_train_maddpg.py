import unittest
from unittest import skipIf
import torch
import numpy as np
from drlnd.common.agents.utils import (
    BrainAgentSpec,
    AgentInventory,
    MultiAgentSpec,
    UnityEnvWrapper,
    ReplayBuffer,
    ActionType,
    get_unity_env,
)
from drlnd.common.agents.maddpg import AgentSpec, MADDPGAgent2
import os
import re


class TestMADDPGInUnityEnv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if "UNITY_MA_ENV" in os.environ:
            unity_exe = os.path.basename(os.environ["UNITY_MA_ENV"])
            self.env, self.brain_spec = get_unity_env(os.environ["UNITY_MA_ENV"])
            if re.compile("Tennis.*").match(unity_exe) is not None:
                self.env_action_type = ActionType.CONTINUOUS
            elif re.compile("Soccer.*").match(unity_exe) is not None:
                self.env_action_type = ActionType.DISCRETE
            else:
                raise Exception(f"Don't know how to deal with env: {unity_exe}")

    @skipIf("UNITY_MA_ENV" not in os.environ.keys(), "No path to unity executable")
    def test_unity_env_wrapper(self):
        # Get the environment
        env, brain_spec = self.env, self.brain_spec
        brain_spec_list = list(brain_spec.values())

        # Create the wrapper
        print(brain_spec_list)
        env_wrapper = UnityEnvWrapper(env, brain_spec_list, self.env_action_type)
        env_wrapper.reset(train_mode=False)

        agent_inventory = AgentInventory(brain_spec_list)

        buffer = ReplayBuffer(
            int(5e6),
            1,
            1234,
            action_dtype=self.env_action_type,
            brain_agents=brain_spec_list,
        )

        # Check we can add observations
        print(buffer.brain_order)

        maddpg = MADDPGAgent2(
            agent_inventory,
            hidden_layer_size=256,
            policy_network_kwargs={"output_activation": lambda x: x},
            replay_buffer=buffer,
        )

        self.assertIsInstance(maddpg, MADDPGAgent2)

        for i in range(10):
            # Test we can easily get the actions, given the set of observations, by
            # using the maddpg agent to access and evaluate the policy networks.
            print(env_wrapper.get_states())
            actions = maddpg.act(
                states=env_wrapper.get_states(),
                policy_suppression=1.0,
            )

            # Check that we can take a step based on the resulting actions.
            env_wrapper.step(vector_action=actions)

            # Check that we can get the SARS' tuples easily.
            states, actions, rewards, next_states, dones = env_wrapper.sars()
            self.assertIsInstance(states, dict)
            self.assertIsInstance(actions, dict)
            self.assertIsInstance(rewards, dict)
            self.assertIsInstance(next_states, dict)
            self.assertIsInstance(dones, dict)

            buffer.add_from_dicts(states, actions, rewards, next_states, dones)

        s, a, r, ns, d = buffer.sample()
        print(f"states = \n{s}")
        print(f"actions = \n{a}")
        print(f"rewards = \n{r}")
        print(f"dones = \n{d}")

        # See if the agent can learn
        maddpg.learn(0.99)

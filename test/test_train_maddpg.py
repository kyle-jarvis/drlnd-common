import unittest
from unittest import skipIf
import torch
import torch.nn.functional as F
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
from drlnd.common.utils import OrnsteinUhlenbeckProcess
from drlnd.common.agents.maddpg import MADDPGAgent2
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
                self.policy_output_activation = lambda x: x
            elif re.compile("Soccer.*").match(unity_exe) is not None:
                self.env_action_type = ActionType.DISCRETE
                def policy_activation(self, x):
                    if len(x.shape) == 1:
                        return F.gumbel_softmax(x.unsqueeze(0)).squeeze()
                    return F.gumbel_softmax(x)

                self.policy_output_activation = policy_activation
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
            policy_network_kwargs={"output_activation": self.policy_output_activation},
            replay_buffer=buffer,
        )

        self.assertIsInstance(maddpg, MADDPGAgent2)

        for i in range(10):
            # Test we can easily get the actions, given the set of observations, by
            # using the maddpg agent to access and evaluate the policy networks.
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

        # Check that we can add noise to the agent's actions
        def make_noise_generator(size):
            noise_generator = OrnsteinUhlenbeckProcess(
                [size], 1.0, dt=1.0, theta=0.5
                )
            noise_callable = lambda: torch.from_numpy(noise_generator.sample()).float()

            return noise_callable

        noise_generators = {brain.name: make_noise_generator(brain.action_size) for brain in brain_spec_list}

        def no_noise_callable(size):
            return lambda: torch.zeros(size)

        no_noise = {brain.name: no_noise_callable(brain.action_size) for brain in brain_spec_list}

        actions = maddpg.act(
                    states, policy_suppression=1.0, noise_func=noise_generators
                )

        actions = maddpg.act(
                    states, policy_suppression=1.0, noise_func=no_noise
                )


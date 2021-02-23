import torch
import torch.nn.functional as F
import torch.optim as optim
from ..model import SimpleFCNetwork
from .config import LR, BUFFER_SIZE, GAMMA, TAU, device
from .utils import hard_update, soft_update, ReplayBuffer
from .base import BaseAgent
from typing import List
from copy import deepcopy
from collections import OrderedDict
from itertools import starmap

LR = 5e-5

class AgentSpec:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size


class MADDPGAgent(BaseAgent):
    def __init__(
        self, 
        agent_specs: List[AgentSpec],
        replay_buffer: ReplayBuffer = None, 
        hidden_layer_size = None):

        super().__init__()
        network_kwargs = {'hidden_layer_size': hidden_layer_size} if hidden_layer_size is not None else {}

        self.agent_specs = agent_specs
        self.agents = OrderedDict({})
        for i, agent in enumerate(self.agent_specs):
            policy_i = SimpleFCNetwork(1234, agent.state_size, agent.action_size, **network_kwargs)
            policy_i_target = SimpleFCNetwork(1234, agent.state_size, agent.action_size, **network_kwargs)
            hard_update(policy_i, policy_i_target)

            critic_i = SimpleFCNetwork(
                1234, 
                sum([a.state_size + a.action_size for a in self.agent_specs]), 
                1, 
                output_activation=lambda x: x, 
                **network_kwargs
                )
            critic_i_target = SimpleFCNetwork(
                1234, 
                sum([a.state_size + a.action_size for a in self.agent_specs]), 
                1, 
                output_activation=lambda x: x, 
                **network_kwargs
                )
            hard_update(critic_i, critic_i_target)

            agent_items = {
                'networks': {
                    'policy' : policy_i,
                    'policy_target' : policy_i_target,
                    'critic': critic_i, 
                    'critic_target': critic_i_target
                }, 
                'optimizers' : {
                    'critic': optim.Adam(critic_i.parameters(), lr=LR), 
                    'policy': optim.Adam(policy_i.parameters(), lr=LR), 
                }
            }
            self.agents.update({f'agent_{i}': agent_items})

        self.replay_buffer = replay_buffer

        self.networks = dict(
            [
                ("_".join([agent_name, k]), v) 
                for agent_name, agent_items in self.agents.items()
                for k, v in agent_items['networks'].items() 
            ]
        )

        self.action_size = agent_specs[0].action_size
        self.state_size = agent_specs[0].state_size

        if self.replay_buffer is not None:
            self.learn = lambda *args, **kwargs: self._learn(*args, **kwargs)


    def act(self, states, policy_suppression: 1.0, noise_func = None,):
        agent_actions = []
        with torch.no_grad():
            for state, (agent_name, agent_inventory) in zip(states, self.agents.items()):
                policy = agent_inventory['networks']['policy']
                actions = policy_suppression*policy(state)
                if noise_func is not None:
                    action_noise = noise_func()
                    actions = torch.clamp(actions + action_noise, -1.0, 1.0)
                agent_actions.append(actions)
        return agent_actions


    def evaluate_networks_from_sliced_tensor(self, joined_tensor: torch.Tensor, networks: list, slice_length: int):
        # Default
        result = starmap(lambda i, network: network(joined_tensor[slice(None), slice(slice_length*i, (slice_length*(i+1)))]), enumerate(networks))
        return list(result)

    def learn(self):
        raise NotImplementedError("Agents without replay buffers cant learn")

    def _learn(self, gamma):
        if len(self.replay_buffer.memory) > self.replay_buffer.batch_size:
            for i, (agent_name, agent_items) in enumerate(self.agents.items()):
                networks, optimizers = agent_items['networks'], agent_items['optimizers']
                
                states, actions, rewards, next_states, dones = self.replay_buffer.sample()

                reward = rewards[:, i].unsqueeze(-1)

                # Update the critic
                target_policy_action_selections = \
                    self.evaluate_networks_from_sliced_tensor(states, self.policy_target_networks, self.state_size)

                with torch.no_grad():
                    TD = gamma*networks["critic_target"](torch.cat([next_states, *target_policy_action_selections], 1))
                    yi = reward + TD

                y = networks["critic"](torch.cat([states, actions], 1))

                critic_loss = F.mse_loss(yi, y)

                optimizers['critic'].zero_grad()
                critic_loss.backward()
                optimizers['critic'].step()

                # Update the actor
                state_indices = [slice(None), slice((i*self.state_size), ((i+1)*self.state_size))]
                action_indices = [slice(None), slice((i*self.action_size), ((i+1)*self.action_size))]
                actions[action_indices] \
                    = networks["policy"](states[state_indices])

                policy_loss = -networks['critic'](torch.cat([states, actions], 1)).mean()
                optimizers['policy'].zero_grad()
                policy_loss.backward()
                optimizers['policy'].step()

            for i, (_, agent_items) in enumerate(self.agents.items()):
                networks = agent_items['networks']
                soft_update(networks['policy'], networks['policy_target'], TAU)
                soft_update(networks['critic'], networks['critic_target'], TAU)



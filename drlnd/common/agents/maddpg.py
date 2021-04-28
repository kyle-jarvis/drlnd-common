import torch
import torch.nn.functional as F
import torch.optim as optim
from ..model import SimpleFCNetwork
from .config import LR, BUFFER_SIZE, GAMMA, TAU, device
from .utils import (
    hard_update,
    soft_update,
    ReplayBuffer,
    AgentSpec,
    AgentInventory,
    MultiAgentSpec,
)
from .base import BaseAgent
from typing import List, Dict, NamedTuple
from copy import deepcopy
from collections import OrderedDict, defaultdict
from itertools import starmap, chain

LR = 5e-5


class MADDPGAgent(BaseAgent):
    def __init__(
        self,
        agent_specs: List[Dict[str, List[AgentSpec]]],
        replay_buffer: ReplayBuffer = None,
        hidden_layer_size=None,
    ):

        super().__init__()
        network_kwargs = (
            {"hidden_layer_size": hidden_layer_size}
            if hidden_layer_size is not None
            else {}
        )

        self.agent_specs = agent_specs
        self.agents = OrderedDict({})
        for i, agent in enumerate(self.agent_specs):
            policy_i = SimpleFCNetwork(
                1234, agent.state_size, agent.action_size, **network_kwargs
            )
            policy_i_target = SimpleFCNetwork(
                1234, agent.state_size, agent.action_size, **network_kwargs
            )
            hard_update(policy_i, policy_i_target)

            critic_i = SimpleFCNetwork(
                1234,
                sum([a.state_size + a.action_size for a in self.agent_specs]),
                1,
                output_activation=lambda x: x,
                **network_kwargs,
            )
            critic_i_target = SimpleFCNetwork(
                1234,
                sum([a.state_size + a.action_size for a in self.agent_specs]),
                1,
                output_activation=lambda x: x,
                **network_kwargs,
            )
            hard_update(critic_i, critic_i_target)

            agent_items = {
                "networks": {
                    "policy": policy_i,
                    "policy_target": policy_i_target,
                    "critic": critic_i,
                    "critic_target": critic_i_target,
                },
                "optimizers": {
                    "critic": optim.Adam(critic_i.parameters(), lr=LR),
                    "policy": optim.Adam(policy_i.parameters(), lr=LR),
                },
            }
            self.agents.update({f"agent_{i}": agent_items})

        self.replay_buffer = replay_buffer

        self.networks = dict(
            [
                ("_".join([agent_name, k]), v)
                for agent_name, agent_items in self.agents.items()
                for k, v in agent_items["networks"].items()
            ]
        )

        self.action_size = agent_specs[0].action_size
        self.state_size = agent_specs[0].state_size

        if self.replay_buffer is not None:
            self.learn = lambda *args, **kwargs: self._learn(*args, **kwargs)

    def act(
        self,
        states,
        policy_suppression: 1.0,
        noise_func=None,
    ):
        agent_actions = []
        with torch.no_grad():
            for state, (agent_name, agent_inventory) in zip(
                states, self.agents.items()
            ):
                policy = agent_inventory["networks"]["policy"]
                actions = policy_suppression * policy(state)
                if noise_func is not None:
                    action_noise = noise_func()
                    actions = torch.clamp(actions + action_noise, -1.0, 1.0)
                agent_actions.append(actions)
        return agent_actions

    def evaluate_networks_from_sliced_tensor(
        self, joined_tensor: torch.Tensor, networks: list, slice_length: int
    ):
        # Default
        result = starmap(
            lambda i, network: network(
                joined_tensor[
                    slice(None), slice(slice_length * i, (slice_length * (i + 1)))
                ]
            ),
            enumerate(networks),
        )
        return list(result)

    def learn(self):
        raise NotImplementedError("Agents without replay buffers cant learn")

    def _learn(self, gamma):
        if len(self.replay_buffer.memory) > self.replay_buffer.batch_size:
            for i, (agent_name, agent_items) in enumerate(self.agents.items()):
                networks, optimizers = (
                    agent_items["networks"],
                    agent_items["optimizers"],
                )

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ) = self.replay_buffer.sample()

                reward = rewards[:, i].unsqueeze(-1)

                # Update the critic
                target_policy_action_selections = (
                    self.evaluate_networks_from_sliced_tensor(
                        states, self.policy_target_networks, self.state_size
                    )
                )

                with torch.no_grad():
                    TD = gamma * networks["critic_target"](
                        torch.cat([next_states, *target_policy_action_selections], 1)
                    )
                    yi = reward + TD

                y = networks["critic"](torch.cat([states, actions], 1))

                critic_loss = F.mse_loss(yi, y)

                optimizers["critic"].zero_grad()
                critic_loss.backward()
                optimizers["critic"].step()

                # Update the actor
                state_indices = [
                    slice(None),
                    slice((i * self.state_size), ((i + 1) * self.state_size)),
                ]
                action_indices = [
                    slice(None),
                    slice((i * self.action_size), ((i + 1) * self.action_size)),
                ]
                actions[action_indices] = networks["policy"](states[state_indices])

                policy_loss = -networks["critic"](
                    torch.cat([states, actions], 1)
                ).mean()
                optimizers["policy"].zero_grad()
                policy_loss.backward()
                optimizers["policy"].step()

            for i, (_, agent_items) in enumerate(self.agents.items()):
                networks = agent_items["networks"]
                soft_update(networks["policy"], networks["policy_target"], TAU)
                soft_update(networks["critic"], networks["critic_target"], TAU)


class MADDPGAgentNetworks(NamedTuple):
    policy: torch.nn.Module
    policy_target: torch.nn.Module
    critic: torch.nn.Module
    critic_target: torch.nn.Module


class MADDPGAgentOptimizers(NamedTuple):
    policy: torch.optim.Adam
    critic: torch.optim.Adam


class AgentItems(NamedTuple):
    networks: MADDPGAgentNetworks
    optimizers: MADDPGAgentOptimizers
    spec: MultiAgentSpec


class MADDPGAgent2(BaseAgent):
    def __init__(
        self,
        agent_inventory: AgentInventory,
        replay_buffer: ReplayBuffer = None,
        hidden_layer_size=None,
        policy_network_kwargs=None,
    ):

        super().__init__()
        network_kwargs = (
            {"hidden_layer_size": hidden_layer_size}
            if hidden_layer_size is not None
            else {}
        )
        policy_network_kwargs = (
            {} if policy_network_kwargs is None else policy_network_kwargs
        )
        policy_network_kwargs.update(network_kwargs)

        self.agent_inventory = agent_inventory
        self.agents = defaultdict(list)
        for agent_spec in chain(*self.agent_inventory.agents.values()):
            policy = SimpleFCNetwork(
                1234,
                agent_spec.state_size,
                agent_spec.action_size,
                **policy_network_kwargs,
            )
            policy_target = SimpleFCNetwork(
                1234,
                agent_spec.state_size,
                agent_spec.action_size,
                **policy_network_kwargs,
            )
            hard_update(policy, policy_target)

            critic_i = SimpleFCNetwork(
                1234,
                agent_inventory.critic_input_size,
                1,
                output_activation=lambda x: x,
                **network_kwargs,
            )
            critic_i_target = SimpleFCNetwork(
                1234,
                agent_inventory.critic_input_size,
                1,
                output_activation=lambda x: x,
                **network_kwargs,
            )
            hard_update(critic_i, critic_i_target)

            agent_networks = MADDPGAgentNetworks(
                policy=policy,
                policy_target=policy_target,
                critic=critic_i,
                critic_target=critic_i_target,
            )

            agent_optimizers = MADDPGAgentOptimizers(
                policy=optim.Adam(policy.parameters(), lr=LR),
                critic=optim.Adam(critic_i.parameters(), lr=LR),
            )

            agent_items = AgentItems(
                networks=agent_networks, optimizers=agent_optimizers, spec=agent_spec
            )

            self.agents[agent_spec.brain_name].append(agent_items)

        self.replay_buffer = replay_buffer

        if self.replay_buffer is not None:
            self.learn = lambda *args, **kwargs: self._learn(*args, **kwargs)

    def act(
        self,
        states,
        policy_suppression: 1.0,
        noise_func=None,
    ):
        #
        # Assume that states are received like: dict(zip(brain_names, states))
        # Actions are fed to env like: dict(zip(brain_names, actions))

        agent_actions = {}
        with torch.no_grad():
            for brain_name, observed_states in states.items():
                brain_name_actions = []
                for i, state in enumerate(observed_states):
                    state = torch.from_numpy(state).float()
                    policy_network = self.agents[brain_name][i].networks.policy
                    action = policy_suppression * policy_network(state)
                    if noise_func is not None:
                        action_noise = noise_func()
                        action = torch.clamp(action + action_noise, -1.0, 1.0)
                    brain_name_actions.append(action)
                agent_actions.update({brain_name: brain_name_actions})
        return agent_actions

    def _evaluate_actions_using_target_policies(self, joined_tensor: torch.Tensor):
        # Default
        results = []
        for brain_name in self.agent_inventory.brain_order:
            for agent_items in self.agents[brain_name]:
                results.append(
                    agent_items.networks.policy_target(
                        joined_tensor[agent_items.spec.state_slice]
                    )
                )

        return results

    def learn(self):
        raise NotImplementedError("Agents without replay buffers cant learn")

    def _learn(self, gamma):
        if len(self.replay_buffer.memory) > self.replay_buffer.batch_size:
            # for agent in self.agent_inventory.agents.keys():
            # networks, optimizers = self.agents[agent]['networks'], self.agents[agent]['optimizers']
            # for i, (agent_name, agent_items) in enumerate(self.agents.items()):
            for brain_name, agent_items_list in self.agents.items():
                for agent_items in agent_items_list:

                    networks, optimizers, agent_spec = (
                        agent_items["networks"],
                        agent_items["optimizers"],
                        agent_items["spec"],
                    )

                    (
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                    ) = self.replay_buffer.sample()

                    reward = rewards[:, agent_items.spec.agent_index].unsqueeze(-1)

                    # Update the critic
                    target_policy_action_selections = (
                        self._evaluate_actions_using_target_policies(states)
                    )

                    with torch.no_grad():
                        TD = gamma * networks["critic_target"](
                            torch.cat(
                                [next_states, *target_policy_action_selections], 1
                            )
                        )
                        yi = reward + TD

                    y = networks["critic"](torch.cat([states, actions], 1))

                    critic_loss = F.mse_loss(yi, y)

                    optimizers["critic"].zero_grad()
                    critic_loss.backward()
                    optimizers["critic"].step()

                    # Update the actor

                    actions[agent_spec.action_indices] = networks["policy"](
                        states[agent_spec.state_indices]
                    )

                    policy_loss = -networks["critic"](
                        torch.cat([states, actions], 1)
                    ).mean()
                    optimizers["policy"].zero_grad()
                    policy_loss.backward()
                    optimizers["policy"].step()

            for agent_items in chain(*self.agents.values()):
                networks = agent_items["networks"]
                soft_update(networks["policy"], networks["policy_target"], TAU)
                soft_update(networks["critic"], networks["critic_target"], TAU)

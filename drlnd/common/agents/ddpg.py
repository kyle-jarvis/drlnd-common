import torch
import torch.nn.functional as F
import torch.optim as optim
from ..model import SimpleFCNetwork
from .config import LR, BUFFER_SIZE, GAMMA, TAU, device
from .utils import hard_update, soft_update, ReplayBuffer
from .base import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(
        self, 
        state_size:int, 
        action_size:int, 
        replay_buffer: ReplayBuffer = None, 
        hidden_layer_size = None):

        super().__init__()
        network_kwargs = {'hidden_layer_size': hidden_layer_size} if hidden_layer_size is not None else {}
        self.actor_local = SimpleFCNetwork(1234, state_size, action_size, **network_kwargs)
        self.actor_target = SimpleFCNetwork(1234, state_size, action_size, **network_kwargs)
        hard_update(self.actor_local, self.actor_target)

        self.critic_local = SimpleFCNetwork(1234, (state_size + action_size), 1, output_activation=lambda x: x, **network_kwargs)
        self.critic_target = SimpleFCNetwork(1234, (state_size + action_size), 1, output_activation=lambda x: x, **network_kwargs)
        hard_update(self.critic_local, self.critic_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)
        self.replay_buffer = replay_buffer

        self.networks = {"critic_local": self.critic_local, "actor_local": self.actor_local}

        self.action_size = action_size

        if self.replay_buffer is not None:
            self.learn = lambda *args, **kwargs: self._learn(*args, **kwargs)

    def act(self, state, policy_suppression: 1.0, noise_func = None,):
        with torch.no_grad():
            actions = policy_suppression*self.actor_local.forward(state)
        if noise_func is not None:
            action_noise = noise_func()
            actions = torch.clamp(actions+action_noise, -1.0, 1.0)
        return actions

    def learn(self):
        raise NotImplementedError("Agents without replay buffers cant learn")

    def _learn(self, gamma):
        if len(self.replay_buffer.memory) > self.replay_buffer.batch_size:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()

            # Update the critic
            action_selection = self.actor_target(next_states).detach()
            yi = rewards + gamma*self.critic_target(torch.cat([next_states, action_selection], 1)).detach()
            y = self.critic_local(torch.cat([states, actions], 1))
            critic_loss = F.mse_loss(yi, y)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the actor
            actor_actions = self.actor_local(states)
            policy_loss = -self.critic_local(torch.cat([states, actor_actions], 1)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor_local, self.actor_target, TAU)
            soft_update(self.critic_local, self.critic_target, TAU)



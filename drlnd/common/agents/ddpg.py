import torch
import torch.nn.functional as F
import torch.optim as optim
from .model import SimpleFCNetwork
from .utils import hard_update, soft_update, ReplayBuffer
from . import LR, BUFFER_SIZE, GAMMA, TAU, device


class DDPGAgent:
    def __init__(self, state_size:int, action_size:int, replay_buffer: ReplayBuffer):
        self.actor_local = SimpleFCNetwork(1234, state_size, action_size)
        self.actor_target = SimpleFCNetwork(1234, state_size, action_size)
        hard_update(self.actor_local, self.actor_target)

        self.critic_local = SimpleFCNetwork(1234, (state_size + action_size), 1, output_activation=lambda x: x)
        self.critic_target = SimpleFCNetwork(1234, (state_size + action_size), 1, output_activation=lambda x: x)
        hard_update(self.critic_local, self.critic_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)
        self.replay_buffer = replay_buffer

    def act(self, state):
        with torch.no_grad():
            actions = self.actor_local.forward(state)
        return actions

    def learn(self, gamma):
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



from enum import Enum
from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from .model import QNetwork, PolicyNetwork


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
COPY_WEIGHTS_EVERY = 50 # how often to update the target network using local weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LearningStrategy(Enum):
    DQN = 1
    DDQN = 2

class TargetNetworkUpdateStrategy(Enum):
    SOFT = 1
    HARD = 2

class ActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self, 
        state_size: int, 
        action_size: int, 
        seed,
        learning_strategy: LearningStrategy, 
        target_network_update_strategy: TargetNetworkUpdateStrategy,
        hidden_layer_width: int = 64):
        """Initialise an Agent object.

        :param state_size: Dimension of input state to DQN.
        :type state_size: int
        :param action_size: Dimension of output action vector from DQN.
        :type action_size: int
        :param seed: Seed for random state intialisation.
        :type seed: numeric
        :param learning_strategy: Whether to use DQN or DDQN learning algorithms.
        :type learning_strategy: LearningStrategy
        :param target_network_update_strategy: Whether to use hard or soft updates when updating the fixed-Q target networks.
        :type target_network_update_strategy: TargetNetworkUpdateStrategy
        :param hidden_layer_width: Size of intermediate , defaults to 64
        :type hidden_layer_width: int, optional
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layer_width=hidden_layer_width).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layer_width=hidden_layer_width).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #self.loss = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.tt_step = 0

        self.learning_strategy = learning_strategy

        if self.learning_strategy == LearningStrategy.DQN:
            self.learn = self.learn_dqn
        elif self.learning_strategy == LearningStrategy.DDQN:
            self.learn = self.learn_ddqn

        self.target_network_update_strategy = target_network_update_strategy

        if self.target_network_update_strategy == TargetNetworkUpdateStrategy.SOFT:
            self.update_target_network = self.soft_update
        elif self.target_network_update_strategy == TargetNetworkUpdateStrategy.HARD:
            self.update_target_network = self.hard_update

        self.loss = torch.nn.MSELoss()
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
            if self.target_network_update_strategy == TargetNetworkUpdateStrategy.SOFT:
                # ------------------- update target network ------------------- #
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   

        if self.target_network_update_strategy == TargetNetworkUpdateStrategy.HARD:
            self.tt_step = (self.tt_step + 1) % COPY_WEIGHTS_EVERY
            if self.tt_step == 0:
                self.update_target_network(self.qnetwork_local, self.qnetwork_target)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn_dqn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Q(State, Action) evaluation in the DQN network is done using the target
        # network. Because we don't update this network every step, the computation
        # is detatched from the graph before calling .backward().

        yj = rewards + (1-dones)*(gamma*torch.max(self.qnetwork_target(next_states).detach(), dim=1)[0].unsqueeze(1))
        output = F.mse_loss(yj, torch.gather(self.qnetwork_local(states), 1, actions))
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()


    def learn_ddqn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Q(State, Action) evaluation in the DQN network is separated into action
        # selection using the target, offline network, and evaluation of the
        # chosen action and state using the online network. Because we don't 
        # update this network every step, the computation all computations
        # done using the local network remain attached to the computational graph 
        # whilst calling .backward().

        action_selection = torch.max(self.qnetwork_local(next_states), dim=1)[1].unsqueeze(1)
        action_evaluation = torch.gather(self.qnetwork_target(next_states).detach(), 1, action_selection)

        yj = rewards + (1-dones)*(gamma*action_evaluation)
        output = F.mse_loss(torch.gather(self.qnetwork_local(states), 1, actions), yj)#torch.sum((yj - torch.gather(self.qnetwork_local(states), 1, actions))**2)
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()
                  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, action_dtype: ActionType = ActionType.DISCRETE):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        if action_dtype == ActionType.CONTINUOUS:
            self.get_action_as = lambda tensor: tensor.float()
        elif action_dtype == ActionType.DISCRETE:
            self.get_action_as = lambda tensor: tensor.long()
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = self.get_action_as(torch.from_numpy(np.vstack([e.action for e in experiences if e is not None]))).to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def hard_update(local_model, target_model):
    """Hard update model parameters.

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class DDPGAgent:
    def __init__(self, state_size:int, action_size:int, replay_buffer: ReplayBuffer):
        self.actor_local = PolicyNetwork(1234, state_size, action_size)
        self.actor_target = PolicyNetwork(1234, state_size, action_size)
        hard_update(self.actor_local, self.actor_target)

        self.critic_local = PolicyNetwork(1234, (state_size + action_size), 1, output_activation=lambda x: x)
        self.critic_target = PolicyNetwork(1234, (state_size + action_size), 1, output_activation=lambda x: x)
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



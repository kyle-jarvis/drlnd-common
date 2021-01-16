from enum import Enum
from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork


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

    def __init__(self, action_size, buffer_size, batch_size, seed):
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
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
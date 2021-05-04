from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from ..model import QNetwork, SimpleFCNetwork
from .utils import (
    soft_update,
    hard_update,
    ReplayBuffer,
    LearningStrategy,
    TargetNetworkUpdateStrategy,
)
from .config import (
    LR,
    BUFFER_SIZE,
    GAMMA,
    TAU,
    UPDATE_EVERY,
    COPY_WEIGHTS_EVERY,
    device,
)
from .base import BaseAgent


class Agent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed,
        learning_strategy: LearningStrategy,
        target_network_update_strategy: TargetNetworkUpdateStrategy,
        hidden_layer_width: int = 64,
    ):
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
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed, hidden_layer_width=hidden_layer_width
        ).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed, hidden_layer_width=hidden_layer_width
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.networks = {"qnetwork_local": self.qnetwork_local}

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
            self.update_target_network = soft_update
        elif self.target_network_update_strategy == TargetNetworkUpdateStrategy.HARD:
            self.update_target_network = hard_update

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
                soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        if self.target_network_update_strategy == TargetNetworkUpdateStrategy.HARD:
            self.tt_step = (self.tt_step + 1) % COPY_WEIGHTS_EVERY
            if self.tt_step == 0:
                self.update_target_network(self.qnetwork_local, self.qnetwork_target)

    def act(self, state, eps=0.0):
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

        yj = rewards + (1 - dones) * (
            gamma
            * torch.max(self.qnetwork_target(next_states).detach(), dim=1)[0].unsqueeze(
                1
            )
        )
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

        action_selection = torch.max(self.qnetwork_local(next_states), dim=1)[
            1
        ].unsqueeze(1)
        action_evaluation = torch.gather(
            self.qnetwork_target(next_states).detach(), 1, action_selection
        )

        yj = rewards + (1 - dones) * (gamma * action_evaluation)
        output = F.mse_loss(
            torch.gather(self.qnetwork_local(states), 1, actions), yj
        )  # torch.sum((yj - torch.gather(self.qnetwork_local(states), 1, actions))**2)
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()

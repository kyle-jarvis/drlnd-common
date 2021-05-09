import re
import random
from collections import deque, namedtuple
from typing import NamedTuple, List
from enum import Enum
import numpy as np
import torch
from .config import device
from unityagents import UnityEnvironment


def get_unity_env(path: str):
    env = UnityEnvironment(file_name=path)

    brain_names = env.brain_names
    brains = env.brains

    set_env = env.reset(train_mode=False)
    brain_spec = {}
    for x in brain_names:
        brain = brains[x]
        env_brain = set_env[x]
        num_agents, observation_size = env_brain.vector_observations.shape
        brain_spec.update(
            {
                x: BrainAgentSpec(
                    x,
                    num_agents=num_agents,
                    state_size=observation_size,
                    action_size=brain.vector_action_space_size,
                )
            }
        )
    return env, brain_spec


class LearningStrategy(Enum):
    DQN = 1
    DDQN = 2


class TargetNetworkUpdateStrategy(Enum):
    SOFT = 1
    HARD = 2


class ActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2


class BrainAgentSpec(NamedTuple):
    "Container for Unity Brain specification details."
    name: str
    num_agents: int
    state_size: int
    action_size: int


class MultiAgentSpec(NamedTuple):
    "Container for agent specification in a multi-agent setting."
    brain_name: str
    agent_name: str
    state_size: int
    action_size: int
    counter: int
    agent_index: int
    action_slice: List[slice]
    state_slice: List[slice]


def hard_update(local_model, target_model):
    """Hard update model parameters.

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
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
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )


class UnityEnvWrapper:
    """Wrapper around Unity Environment objects that tracks information to be
    loaded into a ReplayBuffer

    :return: [description]
    :rtype: [type]
    """

    # USe getattr to pass calls to unity env by default
    # add step call to manage updating / recording sars tuple, then invoking underlying step
    def __init__(self, unity_env, brain_spec_list, action_type: ActionType):
        self.unity_env = unity_env
        self.brain_spec_list = brain_spec_list
        self.env_state = None
        self.action_type = action_type
        if self.action_type == ActionType.DISCRETE:
            self.action_transformer = lambda x: dict(
                [(k, np.argmax(v, axis=-1)) for k, v in x.items()]
            )
        else:
            self.action_transformer = lambda x: x
        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.next_states = None

    def _apply_brain_spec_func(self, function_to_apply):
        return {
            brain_spec_i.name: function_to_apply(brain_spec_i.name)
            for brain_spec_i in self.brain_spec_list
        }

    def get_states(self):
        return self._apply_brain_spec_func(
            lambda x: np.array(self.env_state[x].vector_observations)
        )

    def get_rewards(self):
        return self._apply_brain_spec_func(
            lambda x: np.array(self.env_state[x].rewards)
        )

    def get_dones(self):
        return self._apply_brain_spec_func(
            lambda x: np.array(self.env_state[x].local_done)
        )

    def step(self, *args, **kwargs):
        # Do my stuff
        assert "vector_action" in kwargs.keys()
        assert self.env_state is not None
        self.actions = kwargs["vector_action"]
        kwargs["vector_action"] = self.action_transformer(kwargs["vector_action"])
        self.states = self.get_states()
        self.env_state = self.unity_env.step(*args, **kwargs)
        self.next_states = self.get_states()
        self.rewards = self.get_rewards()
        self.dones = self.get_dones()

    def reset(self, *args, **kwargs):
        self.env_state = self.unity_env.reset(*args, **kwargs)

    def sars(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones

    def __getattr__(self, attr):
        return getattr(self.unity_env, attr)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self,
        buffer_size,
        batch_size,
        seed=1234,
        action_dtype: ActionType = ActionType.DISCRETE,
        brain_agents: List[BrainAgentSpec] = None,
    ):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.brain_order = [brain_spec.name for brain_spec in brain_agents]
        self.total_agents = sum([brain_spec.num_agents for brain_spec in brain_agents])
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        random.seed(seed)
        self.latest_reward = [0.0 for _ in range(self.total_agents)]
        self.latest_done = [0.0 for _ in range(self.total_agents)]
        
        self.get_action_as = lambda tensor: tensor.float()
        #if action_dtype == ActionType.CONTINUOUS:
        #    self.get_action_as = lambda tensor: tensor.float()
        #elif action_dtype == ActionType.DISCRETE:
        #    self.get_action_as = lambda tensor: tensor.long()

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_from_dicts(self, state, action, reward, next_state, done):
        # print("state before", state)
        state = np.concatenate(
            [np.hstack(state[brain_name]) for brain_name in self.brain_order]
        )
        # print("state after", np.concatenate(state))
        action = np.concatenate(
            [np.hstack(action[brain_name]) for brain_name in self.brain_order]
        )
        # print("reward before", reward)
        reward = np.concatenate(
            [np.hstack(reward[brain_name]) for brain_name in self.brain_order]
        )
        # print("reward after", reward)
        next_state = np.concatenate(
            [np.hstack(next_state[brain_name]) for brain_name in self.brain_order]
        )
        done = np.concatenate(
            [np.hstack(done[brain_name]) for brain_name in self.brain_order]
        )

        self.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.latest_done = done
        self.latest_reward = reward

        return state, action, reward, next_state, done

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = self.get_action_as(
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
        ).to(device)
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class AgentInventory:
    # A container that stores collections of agents controlled by their respective
    # brains. Returns slices corresponding to the actions / states for this agent
    # from the concatenated actions / states
    def __init__(self, brain_agents: List[BrainAgentSpec]):
        self.brain_agents = brain_agents

        self.agents = {}

        action_offset = 0
        state_offset = 0

        self.critic_input_size = sum(
            [(a.state_size + a.action_size) * a.num_agents for a in brain_agents]
        )

        self.brain_order = []
        self.agent_order = []
        agent_index = 0

        for brain in brain_agents:
            self.brain_order.append(brain.name)
            this_brains_agents = []
            for i in range(brain.num_agents):
                agent_name = "_".join([brain.name, "agent", str(i)])
                self.agent_order.append(agent_name)
                ma_spec = MultiAgentSpec(
                    brain_name=brain.name,
                    agent_name=agent_name,
                    state_size=brain.state_size,
                    action_size=brain.action_size,
                    counter=i,
                    agent_index=agent_index,
                    action_slice=[
                        slice(None),
                        slice(action_offset, action_offset + brain.action_size),
                    ],
                    state_slice=[
                        slice(None),
                        slice(state_offset, state_offset + brain.state_size),
                    ],
                )
                this_brains_agents.append(ma_spec)
                action_offset += brain.action_size
                state_offset += brain.state_size
                agent_index += 1
            self.agents.update({brain.name: this_brains_agents})
        self.num_agents = agent_index

    def __repr__(self):
        return repr(self.agents)

    def __getattr__(self, attr):
        if attr in self.agents.keys():
            return self.agents[attr]
        elif attr in self.agent_order:
            brain, agent_index = re.compile("(.*)_agent_(\d+)").match(attr).groups()
            print(f"trying to get: {brain}, agent {agent_index}")
            return self.agents[brain][int(agent_index)]
        else:
            raise Exception("Cannot access any member")

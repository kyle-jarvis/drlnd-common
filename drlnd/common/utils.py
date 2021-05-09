"""
A collection of utilites and helper functions called from the main script.
"""

import os
import re
import yaml
from enum import Enum
from typing import Tuple
import numpy as np
import pandas
import torch.nn.functional as F
from unityagents import UnityEnvironment


class BananaEnv(Enum):
    STANDARD = "Banana_Linux/Banana.x86_64"
    HEADLESS = "Banana_Linux_NoVis/Banana.x86_64"
    VISUAL = "VisualBanana_Linux/Banana.x86_64"


def path_from_project_home(path: str, or_else: str = None):
    origin = os.environ["PROJECT_HOME"] if "PROJECT_HOME" in os.environ else or_else
    assert origin is not None
    return os.path.join(origin, path)


def get_environment_executable(environment: BananaEnv) -> str:
    """Returns a path to a specified environment executable based upon selection
    from an enum.

    :param environment: Which environment executable to return, selected from the
    environment enum.
    :type environment: BananaEnv
    :return: String to the environment executable.
    :rtype: str
    """
    assert isinstance(environment, BananaEnv)
    base_directory = path_from_project_home("unity_environments")
    return os.path.join(base_directory, environment.value)


def get_env(headless: bool) -> UnityEnvironment:
    """Returns an object representing a running unity environment that agents
    can be trained in and explore.

    :param headless: Whether to get the headless environment or the standard
    environment.
    :type headless: bool
    :return: The running unity environment.
    :rtype: UnityEnvironment
    """
    env = BananaEnv.HEADLESS if headless else BananaEnv.STANDARD
    env_location = get_environment_executable(env)
    print(f"Getting environment from {env_location}")
    env = UnityEnvironment(file_name=env_location)
    return env


def get_env_properties(env: UnityEnvironment) -> Tuple[int, int]:
    """Inspects the default brain from the unity environment to extract the
    action size, and the state size given by the vector observations property
    of the agent.

    :param env: The unity environment to inspect.
    :type env: UnityEnvironment
    :return: Action size, state size.
    :rtype: Tuple[int, int]
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)
    print("State size:", state_size)
    return action_size, state_size


def get_next_results_directory() -> str:
    """Creates and returns the path to the next valid sub-directory where results
    can be stored.

    :return: Path to the next results subdirectory.
    :rtype: str
    """
    pattern = re.compile("run([0-9]*)")
    results_directory = path_from_project_home("results")
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)
    contents = os.listdir(results_directory)
    matches = []
    for x in contents:
        try:
            index = pattern.match(x).group(1)
            matches.append(int(index))
        except:
            pass
    if len(matches) == 0:
        next_index = "000"
    else:
        next_index = f"{(max(matches) + 1):03}"
    return os.path.join(results_directory, "run" + next_index)


def _load_results(learning_strategy: str, n_points: int):
    solution_directory = os.environ["PROJECT_HOME"]
    checkpoint_directory = f"checkpoints/{learning_strategy}"
    scores_file = os.path.join(solution_directory, checkpoint_directory, "scores.txt")
    scores = pandas.Series(np.loadtxt(scores_file)[:n_points])
    rolling_window = 100
    ma = scores.rolling(rolling_window, center=True).mean()
    x = np.arange(len(ma))
    return scores, ma, x


def load_results(directory: str):
    results_directory = path_from_project_home("results")
    conf_path = os.path.join(results_directory, directory, "parameters.yml")
    with open(conf_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.BaseLoader)

    scores_file = os.path.join(results_directory, directory, "scores.txt")
    scores = pandas.Series(np.loadtxt(scores_file)[:1500])
    rolling_window = 100
    ma = scores.rolling(rolling_window, center=True).mean()
    x = np.arange(len(ma))

    return scores, ma, x, conf


class LinearSequence:
    def __init__(self, initial_value, steps, min_value=None):
        self.x = initial_value
        self.step = 0.0
        self.steps = steps
        if min_value is not None:
            self.out = lambda y: max(y, min_value)
        else:
            self.out = lambda y: y

    def _sample(self):
        result = self.out(self.x * (1.0 - (self.step / self.steps)))
        self.step += 1.0
        return result

    def __call__(self):
        return self._sample()


# Below snippet is taken from the following location:
# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/random_process.py


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, std, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.std * np.sqrt(self.dt) * np.random.randn(*self.size)
        )
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

def gumbel_softmax_fn(logits, tau: float = 1.0, dim: int = -1):
    gumbels = (
        -torch.empty_like(logits).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = F.gumbel_softmaxgumbels.softmax(dim)
    return y_soft

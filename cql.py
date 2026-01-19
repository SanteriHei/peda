# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf
import os
import pickle
import random
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import numpy.typing as npt
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

import environments  # import to register environments for multi-objective

# imports from modt
from modt.utils import (
    compute_hypervolume,
    compute_sparsity,
    pref_grid,
    undominated_indices,
)
from state_norm_params import state_norm_params

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # W&B project name
    project: str = "CORL"
    # W&B group name. If "", set automatically
    group: str = ""
    # W&B run name. If "", set automatically
    name: str = ""
    # W&B run mode. Should be either "disabled", "offline" or "online"
    mode: str = "disabled"
    # Device where the operations are executed.
    device: str = "cuda"

    # Environment name. One of {MO-Ant-v2, MO-HalfCheetah-v2, MO-Hopper-v2, MO-Hopper-v3, MO-Walker2d-v2 }
    env: str = "MO-Hopper-v2"
    # The used dataset One of {amateur_uniform, amateur_narrow, amateur_wide, expert_uniform, expert_narrow, expert_wide}
    dataset: str = "expert_uniform"
    # Path to the directory that contains the dataset
    data_path: Path = Path(__file__).parents[0] / "data"
    # Determine if the preferences are concatenated to the observations
    concat_state_pref: int = 0
    # Determines the number of evaluation preferences
    granularity: int = 50
    # Sets Gym, PyTorch and Numpy seeds
    seed: int = 0  
    # How often (time steps) we evaluate
    eval_freq: int = int(5e3)  
    # How many episodes run during evaluation
    num_episodes: int = 10  
    # Max time steps to run environment
    max_timesteps: int = int(1e6)  
    # Save path
    checkpoints_path: Optional[str] = None  
    # Model load file name, "" doesn't load
    load_model: str = ""  

    # Replay buffer size
    buffer_size: int = 2_000_000  
    # Batch size for all networks
    batch_size: int = 256  
    # Discount factor
    discount: float = 0.99  
    # Multiplier for alpha in loss
    alpha_multiplier: float = 1.0  
    # Tune entropy
    use_automatic_entropy_tuning: bool = True  
    # Use backup entropy
    backup_entropy: bool = False  
    # Policy learning rate
    policy_lr: float = 3e-5  
    # Critics learning rate
    qf_lr: float = 3e-4  
    # Target network update rate
    soft_target_update_rate: float = 5e-3
    # Frequency of target nets updates
    target_update_period: int = 1  
    # Number of sampled actions
    cql_n_actions: int = 10  
    # Use importance sampling
    cql_importance_sample: bool = True  
    # Use Lagrange version of CQL
    cql_lagrange: bool = False  
    # Action gap
    cql_target_action_gap: float = -1.0  
    # CQL temperature
    cql_temp: float = 1.0  
    # Minimal Q weight
    cql_alpha: float = 10.0  
    # Use max target backup
    cql_max_target_backup: bool = False  
    # Q-function lower loss clipping
    cql_clip_diff_min: float = -np.inf  
    # Q-function upper loss clipping
    cql_clip_diff_max: float = np.inf  
    # Orthogonal initialization
    orthogonal_init: bool = True  
    # Normalize states
    normalize_states: bool = True  
    # Normalize reward
    normalize_reward: bool = False  
    # Number of hidden layers in Q networks
    q_n_hidden_layers: int = 3  
    # Number of BC steps at start
    bc_steps: int = int(0)
    # Reward scale for normalization
    reward_scale: float = 5.0  
    # Reward bias for normalization
    reward_bias: float = -1.0  
    # Stochastic policy std multiplier
    policy_log_std_multiplier: float = 1.0  

    def __post_init__(self):
        if len(self.name) == 0:
            self.name = f"cql-{self.env.lower()}-{self.dataset}-{str(uuid.uuid4())[:8]}"

        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        today = datetime.now().strftime("%d-%m-%Y")

        # Setup group name automatically
        if len(self.group) == 0:
            self.group = f"cql-{self.env.lower()}-{today}"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(
    config: TrainConfig, dataset: Dict[str, npt.NDArray], pref_dim: int, eps: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    # Use precomputed statistics
    state_mean = state_norm_params[config.env]["mean"]
    state_std = np.sqrt(state_norm_params[config.env]["var"])

    # If the preference has been concated to the state, add normalization
    # statistics for those as well.
    if config.concat_state_pref != 0:
        state_mean = np.concatenate(
            (state_mean, np.zeros(config.concat_state_pref * pref_dim))
        )
        state_std = np.concatenate(
            (state_mean, np.ones(config.concat_state_pref * pref_dim))
        )

    return state_mean, state_std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        pref_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )

        self._preferences = torch.zeros(
            (buffer_size, pref_dim), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._preferences[:n_transitions] = self._to_tensor(data["preferences"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int, resample_prefs: bool = False) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]

        if resample_prefs:
            prefs = None  # TODO: sample from a dirichlet
            # TODO: Compute the scalarized returns by using the just sampled prefs
            rewards = self._rewards[indices]
        else:
            prefs = self._preferences[indices]
            rewards = self._rewards[indices]

        # rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, prefs, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    cfg = {
        key: val
        for key, val in config.items()
        if key not in ["project", "group", "name", "mode"]
    }

    env_dataset = f"{config['env'].lower()}-{config['dataset']}"
    tags = ["cql", config["env"].lower(), config["dataset"], env_dataset]
    wandb.init(
        project=config["project"],
        group=config["group"],
        name=config["name"],
        mode=config["mode"],
        id=str(uuid.uuid4()),
        tags=tags,
        config=cfg,
    )


@torch.no_grad()
def eval_mo_actor(
    env: gym.Env,
    actor: nn.Module,
    device: str,
    num_episodes: int,
    seed: int,
    preferences: torch.Tensor,
) -> None:
    actor.eval()

    all_rewards = []
    for _ in range(num_episodes):
        rewards = []
        for i, pref in enumerate(preferences):
            seed = np.random.randint(0, 10000)
            env.seed(seed)
            state, done = env.reset(), False
            ep_reward = np.zeros(env.obj_dim)
            while not done:
                action = actor.act(state, pref, device)
                state, _, done, info = env.step(action)
                reward = info["obj"]
                ep_reward += reward
            rewards.append(ep_reward)
        all_rewards.append(rewards)
    actor.train()

    # Compute average hypervolume & sparsity
    hvs = []
    sps = []
    for rewards in all_rewards:
        hvs.append(compute_hypervolume(rewards))

        indices_wanted_strict = undominated_indices(rewards, tolerance=0)
        print(indices_wanted_strict)
        front_return_batch = rewards[indices_wanted_strict]
        sps.append(compute_sparsity(rewards))
    return {"hypervolume": np.mean(hvs), "sparsity": np.mean(sps)}


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        pref_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim + pref_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            print(f"Before {observations.shape=}, {actions.shape=}")
            observations = extend_and_repeat(observations, 1, actions.shape[1])

            print(f"After {observations.shape=}, {actions.shape=}")
        input_tensor = torch.cat([observations, prefs], dim=1)
        base_network_output = self.base_network(input_tensor)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        prefs: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            prefs = extend_and_repeat(prefs, 1, repeat)

        input_tensor = torch.cat([observations, prefs], dim=-1)
        base_network_output = self.base_network(input_tensor)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, prefs: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        prefs = torch.tensor(prefs.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, prefs, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        pref_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim + pref_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            prefs = extend_and_repeat(prefs, 1, actions.shape[1]).reshape(
                -1, prefs.shape[1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions, prefs], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        prefs: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions, prefs),
                self.critic_2(observations, new_actions, prefs),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        prefs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions, prefs)
        q2_predicted = self.critic_2(observations, actions, prefs)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, prefs, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions, prefs),
                    self.target_critic_2(next_observations, new_next_actions, prefs),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations, prefs)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions, prefs),
                self.target_critic_2(next_observations, new_next_actions, prefs),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, prefs, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, prefs, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions, prefs)
        cql_q2_rand = self.critic_2(observations, cql_random_actions, prefs)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions, prefs)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions, prefs)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions, prefs)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions, prefs)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            prefs,
            dones,
        ) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations, prefs)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, prefs, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations,
            actions,
            next_observations,
            prefs=prefs,
            rewards=rewards,
            dones=dones,
            alpha=alpha,
            log_dict=log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


def load_dataset(config: TrainConfig):
    """Loads & preprocesses the offline trajectory dataset"""
    dataset_paths = [
        Path(config.data_path) / config.env / f"{config.env}_50000_{config.dataset}.pkl"
    ]
    trajectories = []
    for data_path in dataset_paths:
        with open(data_path, "rb") as f:
            trajectories.extend(pickle.load(f))

    obs, actions, next_obs, rewards, mo_rewards, preferences, terminals = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    traj_lens, returns, returns_mo = [], [], []

    min_each_obj_step = np.min(
        np.vstack([np.min(traj["raw_rewards"], axis=0) for traj in trajectories]),
        axis=0,
    )
    max_each_obj_step = np.max(
        np.vstack([np.max(traj["raw_rewards"], axis=0) for traj in trajectories]),
        axis=0,
    )

    for traj in trajectories:
        if config.concat_state_pref != 0:
            traj["observations"] = np.concatenate(
                (
                    traj["observations"],
                    np.tile(traj["preference"], config.concat_state_pref),
                ),
                axis=1,
            )
            # TODO: Do I need to also integate this into the next preference?

        if config.normalize_reward:
            traj["raw_rewards"] = (traj["raw_rewards"] - min_each_obj_step) / (
                max_each_obj_step - min_each_obj_step
            )

        traj["rewards"] = np.sum(
            np.multiply(traj["raw_rewards"], traj["preference"]), axis=1
        )

        obs.append(traj["observations"])
        actions.append(traj["actions"])
        next_obs.append(traj["next_observations"])
        terminals.append(traj["terminals"])
        rewards.append(traj["rewards"])
        mo_rewards.append(traj["raw_rewards"])
        # TODO: Could just add one preference, as these are the same for the whole trajectory
        preferences.append(traj["preference"])

        # Some extra stuff that is not maybe needed?
        traj_lens.append(len(traj["observations"]))
        returns.append(traj["rewards"].sum())
        returns_mo.append(traj["raw_rewards"].sum(axis=0))

    traj_lens, returns, returns_mo = (
        np.array(traj_lens),
        np.array(returns),
        np.array(returns_mo),
    )

    obs = np.concatenate(obs, axis=0)
    next_obs = np.concatenate(next_obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    mo_rewards = np.concatenate(mo_rewards, axis=0)
    terminals = np.concatenate(terminals, axis=0)
    preferences = np.concatenate(preferences, axis=0)

    out = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "mo_rewards": mo_rewards,
        "preferences": preferences,
        "terminals": terminals,
    }
    return out


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    pref_dim = env.obj_dim

    dataset = load_dataset(config)
    # dataset = d4rl.qlearning_dataset(env)

    # if config.normalize_reward:
    #     modify_reward(
    #         dataset,
    #         config.env,
    #         reward_scale=config.reward_scale,
    #         reward_bias=config.reward_bias,
    #     )

    if config.normalize_states:
        state_mean, state_std = compute_mean_std(
            config, dataset, pref_dim=pref_dim, eps=1e-3
        )
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        pref_dim,
        buffer_size=config.buffer_size,
        device="cpu",  # config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        pref_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(
        state_dim, action_dim, pref_dim, config.orthogonal_init
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        pref_dim=pref_dim,
        max_action=max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    eval_prefs = pref_grid(pref_dim, granularity=config.granularity)
    print(f"Using {eval_prefs.shape[0]} preferences for evaluation!")

    wandb_init(asdict(config))

    evaluations = []
    _start_time = time.perf_counter()
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            _dur = time.perf_counter() - _start_time
            _prop = (t + 1) / int(config.max_timesteps)
            print(
                f"Timesteps: {t + 1} / {config.max_timesteps} "
                f"({100 * _prop:.2f}%)-> {config.eval_freq} "
                f"steps took {_dur:.2f}s"
            )
            eval_data = eval_mo_actor(
                env,
                actor,
                device=config.device,
                num_episodes=config.num_episodes,
                seed=config.seed,
                preferences=eval_prefs,
            )
            evaluations.append(eval_data)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.num_episodes} episodes: "
                f"HV {eval_data['hypervolume']:.3f} , Sparsity: {eval_data['sparsity']:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {f"eval/{key}": val for key, val in eval_data.items()},
                step=trainer.total_it,
            )
            _start_time = time.perf_counter()


if __name__ == "__main__":
    train()

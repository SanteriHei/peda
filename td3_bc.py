# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import copy
import os
import pathlib
import pickle
import pprint
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gym
import numpy as np
import numpy.typing as npt
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

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
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "TD3_BC-D4RL"
    # wandb run name
    name: str = "TD3_BC"
    # wandb mode
    mode: str = "disabled"
    # Environment name. One of {MO-Ant-v2, MO-HalfCheetah-v2, MO-Hopper-v2, MO-Hopper-v3, MO-Walker2d-v2 }
    env: str = "MO-Hopper-v2"
    # The used dataset One of {amateur_uniform, amateur_narrow, expert_uniform, expert_narrow}
    dataset: str = "expert_uniform" 
    
    # Path to the directory that contains the dataset
    data_path: pathlib.Path = pathlib.Path(__file__).parents[0] / "data"
    # Determine if the preferences are concatenated to the observations
    concat_state_pref: int = 0

    # coefficient for the Q-function in actor loss
    alpha: float = 2.5
    # discount factor
    discount: float = 0.99
    # standard deviation for the gaussian exploration noise
    expl_noise: float = 0.1
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # scalig coefficient for the noise added to
    # target actor during critic update
    policy_noise: float = 0.2
    # range for the target actor noise clipping
    noise_clip: float = 0.5
    # actor update delay
    policy_freq: int = 2
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize_states: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    num_episodes: int = 10
    # maximum episode length
    max_ep_len: int = 500

    granularity: int = 50
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


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
                f"Replay buffer is smaller ({self._buffer_size}) than the "
                f"dataset you are trying to load! ({n_transitions})"
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

    wandb.init(
        project=config["project"],
        group=config["group"],
        name=config["name"],
        mode=config["mode"],
        id=str(uuid.uuid4()),
        config=cfg
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

        # indices_wanted_strict = undominated_indices(rewards, tolerance=0)
        # print(indices_wanted_strict)
        # front_return_batch = rewards[indices_wanted_strict]
        sps.append(compute_sparsity(rewards))
    return {"hypervolume": np.mean(hvs), "sparsity": np.mean(sps)}


def return_reward_range(dataset, max_episode_steps):
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


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, pref_dim: int, max_action: float
    ):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + pref_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, pref], dim=1)
        return self.max_action * self.net(x)

    @torch.no_grad()
    def act(
        self, state: npt.NDArray, pref: npt.NDArray, device: str = "cpu"
    ) -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        pref = torch.tensor(pref.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state, pref).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pref_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + pref_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, pref: torch.Tensor
    ) -> torch.Tensor:
        saw = torch.cat([state, action, pref], 1)
        return self.net(saw)


class TD3_BC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, pref, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state, pref) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action, pref)
            target_q2 = self.critic_2_target(next_state, next_action, pref)
            target_q = torch.min(target_q1, target_q2)
            target_q = (
                reward + not_done * self.discount * target_q
            )  # NOTE: Uses scalarized reward!

        # Get current Q estimates
        current_q1 = self.critic_1(state, action, pref)
        current_q2 = self.critic_2(state, action, pref)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state, pref)
            q = self.critic_1(state, pi, pref)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


def load_dataset(config: TrainConfig):
    """Loads & preprocesses the offline trajectory dataset"""
    dataset_paths = [
        pathlib.Path(config.data_path)
        / config.env
        / f"{config.env}_50000_{config.dataset}.pkl"
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
    scale = 100

    state_dim += pref_dim * config.concat_state_pref
    if not config.normalize_reward:
        scale *= 10

    dataset = load_dataset(config)

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
        device=config.device,
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

    actor = Actor(state_dim, action_dim, pref_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim, pref_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim, pref_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    eval_prefs = pref_grid(pref_dim, granularity=config.granularity)

    evaluations = []

    _start_time = time.perf_counter()
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        # pprint.pprint(log_dict, indent=4)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            _dur = time.perf_counter() - _start_time
            print(f"Time steps: {t + 1}: {config.eval_freq} steps took {_dur:.2f}s")
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
                f"HV {eval_data['hypervolume']:.3f} , Sparsity : {eval_data['sparsity']:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
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

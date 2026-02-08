from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch as th
from typing import NamedTuple, Optional, Any

class GoalReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor
    indices: np.ndarray
    relabel_flags: th.Tensor
    priorities: th.Tensor

class PrioritizedGoalReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: th.device,
        n_envs: int = 1,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0
        self.relabel_flags = np.zeros((buffer_size,), dtype=np.int8)

    def set_alpha(self, alpha: float):
        self.alpha = max(0.0, float(alpha))

    def set_beta_schedule(self, beta_start: float = None, beta_frames: int = None):
        if beta_start is not None:
            self.beta_start = float(beta_start)
        if beta_frames is not None:
            self.beta_frames = max(1, int(beta_frames))

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        # 新样本赋予最大优先级
        idx = (self.pos - 1) % self.buffer_size
        self.priorities[idx] = self.max_priority
        self.max_priority = max(self.max_priority, self.priorities[idx])
        self.relabel_flags[idx] = 0

    def _beta(self):
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + (1.0 - self.beta_start) * progress

    # [Fix] 增加 env 参数以兼容 SB3 接口
    def sample(self, batch_size: int, env: Optional[Any] = None) -> GoalReplayBufferSamples:
        self.frame += 1
        current_size = self.buffer_size if self.full else self.pos
        if current_size == 0:
            raise RuntimeError("ReplayBuffer empty")
        actual_batch = min(batch_size, current_size)

        prios = self.priorities if self.full else self.priorities[: self.pos]
        if prios.max() <= 0:
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = prios ** self.alpha
            probs /= probs.sum()

        replace_flag = len(probs) < actual_batch
        indices = np.random.choice(len(probs), size=actual_batch, p=probs, replace=replace_flag)

        beta = self._beta()
        total = current_size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        env_idx = 0
        obs = self.observations[indices, env_idx, :].astype(np.float32)
        next_obs = self.next_observations[indices, env_idx, :].astype(np.float32)
        acts = self.actions[indices, env_idx, :]
        rews = self.rewards[indices, env_idx]
        dones = self.dones[indices, env_idx]

        obs_t = th.as_tensor(obs, device=self.device)
        next_obs_t = th.as_tensor(next_obs, device=self.device)
        acts_t = th.as_tensor(acts, device=self.device)
        rews_t = th.as_tensor(rews, dtype=th.float32, device=self.device).unsqueeze(-1)
        dones_t = th.as_tensor(dones, dtype=th.float32, device=self.device).unsqueeze(-1)
        weights_t = th.as_tensor(weights, dtype=th.float32, device=self.device).unsqueeze(-1)
        relabel_flags_t = th.as_tensor(self.relabel_flags[indices], dtype=th.int32, device=self.device)
        priorities_t = th.as_tensor(prios[indices], dtype=th.float32, device=self.device)

        return GoalReplayBufferSamples(
            observations=obs_t,
            actions=acts_t,
            next_observations=next_obs_t,
            dones=dones_t,
            rewards=rews_t,
            weights=weights_t,
            indices=indices,
            relabel_flags=relabel_flags_t,
            priorities=priorities_t
        )

    def update_priorities(self, indices, td_errors):
        if len(indices) == 0:
            return
        td_errors = np.asarray(td_errors, dtype=np.float32)
        # 防御：若传入多维或长度不匹配，做聚合
        if td_errors.ndim > 1:
            td_errors = td_errors.reshape(td_errors.shape[0], -1).mean(axis=1)
        if td_errors.shape[0] != len(indices):
            if td_errors.shape[0] > len(indices):
                td_errors = td_errors[:len(indices)]
            else:
                pad_val = float(td_errors.mean()) if td_errors.size > 0 else 0.0
                td_errors = np.pad(td_errors, (0, len(indices) - td_errors.shape[0]), constant_values=pad_val)
        td = np.abs(td_errors).astype(np.float32) + 1e-6
        self.priorities[indices] = td
        self.max_priority = max(self.max_priority, float(td.max()))
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import List


# --- Helper ---
def _act(name: str):
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }.get(name, nn.ReLU())


# --- Shared Features (用于多任务/多目标特征共享) ---

class SharedFeatureBackbone(nn.Module):
    def __init__(self, input_dim: int, arch: List[int], activation="relu", dropout=0.0,
                 adapter_mode: str = "pad"):
        super().__init__()
        self.expected_input_dim = input_dim
        layers = []
        prev = input_dim
        for h in arch:
            lin = nn.Linear(prev, h)
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
            nn.init.constant_(lin.bias, 0.0)
            layers.append(lin)
            layers.append(_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = arch[-1]
        self.adapter_mode = adapter_mode
        self.adapter_linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, shared_backbone: SharedFeatureBackbone):
        super().__init__(observation_space, features_dim=shared_backbone.output_dim)
        self.backbone = shared_backbone

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.float64:
            obs = obs.float()
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        in_dim = obs.size(-1)
        target_dim = self.backbone.expected_input_dim

        # 维度适配 (处理某些 Gym wrapper 可能改变 obs 维度的情况)
        if in_dim != target_dim:
            if self.backbone.adapter_mode == "linear":
                if self.backbone.adapter_linear is None:
                    self.backbone.adapter_linear = nn.Linear(in_dim, target_dim, bias=True).to(obs.device)
                    nn.init.xavier_uniform_(self.backbone.adapter_linear.weight)
                    nn.init.constant_(self.backbone.adapter_linear.bias, 0.0)
                obs = self.backbone.adapter_linear(obs)
            else:
                if in_dim < target_dim:
                    pad = torch.zeros(obs.size(0), target_dim - in_dim, dtype=obs.dtype, device=obs.device)
                    obs = torch.cat([obs, pad], dim=-1)
                else:
                    obs = obs[..., :target_dim]
        return self.backbone(obs)


# --- Goal Conditioned (用于 HIRO 高层) ---

class GoalConditionedExtractor(BaseFeaturesExtractor):
    """
    GoalConditionedExtractor
    Input decomposition:
      obs = [state_base | proxy_goal | goal_diff]
    Provides fused representation for actor/critic.
    """

    def __init__(self, observation_space: spaces.Box, proxy_slice, diff_slice,
                 state_hidden=256, goal_hidden=96, diff_hidden=96, fused_dim=256):
        super().__init__(observation_space, features_dim=fused_dim)
        self.proxy_slice = proxy_slice
        self.diff_slice = diff_slice
        self.obs_dim = int(np.prod(observation_space.shape))
        self.state_indices = list(range(0, proxy_slice.start))
        self.goal_indices = list(range(proxy_slice.start, proxy_slice.stop))
        self.diff_indices = list(range(diff_slice.start, diff_slice.stop))

        def mlp(in_dim, h_dim):
            return nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU()
            )

        self.state_net = mlp(len(self.state_indices), state_hidden)
        self.goal_net = mlp(len(self.goal_indices), goal_hidden)
        self.diff_net = mlp(len(self.diff_indices), diff_hidden)

        self.fuse = nn.Sequential(
            nn.Linear(state_hidden + goal_hidden + diff_hidden, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
        state_part = obs[:, self.state_indices]
        goal_part = obs[:, self.goal_indices]
        diff_part = obs[:, self.diff_indices]

        hs = self.state_net(state_part)
        hg = self.goal_net(goal_part)
        hd = self.diff_net(diff_part)
        fused = torch.cat([hs, hg, hd], dim=-1)
        return self.fuse(fused)
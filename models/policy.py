import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==========================================
# [High-Level] 市场策略特征提取器
# ==========================================

class CustomMarketExtractor(BaseFeaturesExtractor):
    """
    [High-Level] 用于处理混合观测空间：
    1. 标量特征 (Scalars): SoC, SOH, 温度, 系统状态, 时间特征
    2. 序列特征 (Sequence): 未来 96 点电价预测 (1D CNN 处理)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # 解析 HighLevelEnv 的观测结构
        # 结构: [Cluster(12), Sys(3), Mkt(2), Future(96), Time(6)]
        self.total_dim = observation_space.shape[0]
        self.seq_len = 96  # 未来电价长度
        self.scalar_dim = self.total_dim - self.seq_len

        # 检查维度合法性
        if self.scalar_dim <= 0:
            raise ValueError(f"Observation dimension {self.total_dim} too small for sequence length {self.seq_len}")

        # --- 1. 序列特征提取 (1D CNN) ---
        # Input: (Batch, 1, 96)
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算 CNN 输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.seq_len)
            cnn_out_dim = self.cnn_encoder(dummy_input).shape[1]

        # --- 2. 标量特征提取 (MLP) ---
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_dim, 64),
            nn.ReLU()
        )

        # --- 3. 特征融合 ---
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 按照 HighLevelEnv._get_obs 的拼接顺序进行切分
        # Scalars Part 1: Index 0~17 [Cluster(12) + Sys(3) + Mkt(2)]
        # Sequence:       Index 17~113 [Future(96)]
        # Scalars Part 2: Index 113~End [Time(6)]

        seq_start = 17
        seq_end = 17 + 96  # 113

        # 提取标量
        scalar_part_1 = observations[:, :seq_start]
        scalar_part_2 = observations[:, seq_end:]
        scalars = torch.cat([scalar_part_1, scalar_part_2], dim=1)

        # 提取序列 (并增加 Channel 维度用于 Conv1d)
        sequence = observations[:, seq_start:seq_end]
        sequence = sequence.unsqueeze(1)  # (Batch, 1, 96)

        # 前向传播
        cnn_out = self.cnn_encoder(sequence)
        scalar_out = self.scalar_encoder(scalars)

        # 拼接融合
        features = torch.cat([cnn_out, scalar_out], dim=1)
        return self.fusion(features)


# ==========================================
# [Low-Level] 电池物理控制策略组件 (Missing Fixed)
# ==========================================

class BatteryFeatureExtractor(BaseFeaturesExtractor):
    """
    [Low-Level] 专用的电池物理特征提取器。
    用于处理电压、温度、SoC 等物理状态向量。
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128,
                 num_cells: int = 24, use_layer_norm: bool = True, dropout_rate: float = 0.0):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        layers = []
        # 第一层映射
        layers.append(nn.Linear(input_dim, 64))
        if use_layer_norm:
            layers.append(nn.LayerNorm(64))
        layers.append(nn.ReLU())

        # Dropout (可选，增加鲁棒性)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # 中间层
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(128, features_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class BatterySACPolicy(SACPolicy):
    """
    [Low-Level] 结构化 SAC 策略。
    默认绑定 BatteryFeatureExtractor，确保低层 Agent 使用正确的网络结构。
    """

    def __init__(self, *args, **kwargs):
        # 如果未指定提取器，强制使用 BatteryFeatureExtractor
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = BatteryFeatureExtractor
            # 可以在这里注入默认 kwargs，但通常由外部 pretrainer 传入

        super().__init__(*args, **kwargs)
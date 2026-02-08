import numpy as np
from typing import Tuple, Union


class RunningMeanStd:
    """
    Running Mean and Standard Deviation Calculator.
    Uses Welford's online algorithm for numerical stability.

    用于维护数据流的实时统计特性 (Mean, Var)。
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with a new batch of data.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """
        Combine current stats with batch stats.
        """
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class RewardNormalizer:
    """
    [#34] Reward Normalizer with Anti-Leakage Mechanism.

    功能：
    1. 维护折扣累计回报 (Discounted Return) 的运行均值和方差。
    2. 将原始 Reward 归一化，使其通过原点且单位方差，利于神经网络收敛。
    3. [#34] 提供 freeze (training=False) 模式，在评估时仅使用训练集的统计量。
    """

    def __init__(self,
                 clip_range: float = 10.0,
                 gamma: float = 0.99,
                 epsilon: float = 1e-8):
        """
        Args:
            clip_range: 归一化后的截断范围 (防止离群值破坏梯度)
            gamma: 折扣因子 (用于计算 Return)
            epsilon: 防止除零
        """
        self.rms = RunningMeanStd(shape=())
        self.clip_range = clip_range
        self.gamma = gamma
        self.epsilon = epsilon

        # 内部状态：当前的折扣回报估计
        self.ret = 0.0

        # [#34] 训练模式开关
        # True: 更新统计量 (训练时)
        # False: 冻结统计量 (评估时)
        self.training = True

    def __call__(self, reward: float, done: bool = False) -> float:
        """
        Process a reward.

        Args:
            reward: 原始奖励值
            done: Episode 结束标志 (用于重置内部 Return)

        Returns:
            Normalized Reward
        """
        # 1. 更新当前的折扣回报估计 (R_t = r_t + gamma * R_{t-1})
        # 注意：即使在 eval 模式下，ret 也需要根据当前轨迹更新，才能正确计算当前的 scale
        # 但我们通常使用 Global 的统计量来归一化
        self.ret = self.ret * self.gamma + reward

        # 2. 更新全局统计量 (仅在 Training 模式下!) [#34]
        if self.training:
            self.rms.update(np.array([self.ret]))

        # 3. 归一化
        # Rew_norm = Rew / Std(Return)
        # 这里的逻辑是：我们将 Reward 缩放，使得 Return 的方差为 1
        var = self.rms.var
        std = np.sqrt(var) + self.epsilon
        normalized_reward = reward / std

        # 4. 截断保护 [#43]
        if self.clip_range > 0:
            normalized_reward = np.clip(normalized_reward, -self.clip_range, self.clip_range)

        # Episode 结束时重置 Return 估计
        # 注意：这里并不重置 rms (全局统计量)
        if done:
            self.ret = 0.0

        return float(normalized_reward)

    def reset(self):
        """仅重置当前 Episode 的 Return 累加器，不重置全局统计量"""
        self.ret = 0.0
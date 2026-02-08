import numpy as np
from typing import Dict, Any, Optional


class ConstraintManager:
    """
    Constraint Management for CMDP (Final Logic)

    核心职责：
    1. 监控物理边界违规 (Temperature, SoC, Power).
    2. 维护和更新拉格朗日乘子 (Lambdas).
    3. 为 Agent 提供额外的 Reward Penalty (用于训练中的软约束).
    """

    def __init__(self, full_config: Dict[str, Any]):
        """
        Args:
            full_config: 完整的配置字典
        """
        self.cfg = full_config.get('constraints', {})
        self.enabled = self.cfg.get('enabled', True)

        # 1. 核心参数
        self.cost_limit = self.cfg.get('cost_limit', 0.05)  # 允许的违反概率 (e.g. 5%)
        self.lr = self.cfg.get('lagrangian_lr', 0.05)  # Lambda 学习率
        self.lambda_init = self.cfg.get('lambda_init', 0.0)
        self.penalty_max = self.cfg.get('penalty_soft_cap', 100.0)

        # 2. 定义受控的约束项
        # keys 对应 HighLevelEnv info['violation_metrics'] 中的键
        self.constraint_keys = [
            'temp_violation',  # 温度超限程度
            'soc_violation',  # SoC 过充/过放程度
            'power_violation'  # 功率超限程度
        ]

        # 3. 初始化乘子
        self.lambdas = {k: self.lambda_init for k in self.constraint_keys}

        # 统计数据 (用于日志)
        self.episode_violation_accum = {k: 0.0 for k in self.constraint_keys}
        self.episode_steps = 0

    def reset(self):
        """每个 Episode 开始时重置累加器"""
        self.episode_violation_accum = {k: 0.0 for k in self.constraint_keys}
        self.episode_steps = 0

    def compute_penalty(self, info: Dict[str, Any]) -> float:
        """
        [Step-wise] 根据当前步的违规情况计算惩罚项
        Reward_Total = Reward_Task - Penalty_Lagrangian
        """
        if not self.enabled:
            return 0.0

        # 从 Info 中获取违规度量 (HighLevelEnv 必须提供这个字典)
        metrics = info.get('violation_metrics', {})

        total_penalty = 0.0

        for k in self.constraint_keys:
            # 获取违规值 (必须 >= 0)
            # 例如: max(0, current_temp - max_temp)
            val = max(0.0, metrics.get(k, 0.0))

            # 累加统计
            if val > 1e-6:
                self.episode_violation_accum[k] += 1.0  # 记录违规步数

            # 计算惩罚: Lambda * Violation
            # 加上一个巨大的系数或非线性项可以让约束更硬，但 CMDP 标准做法是线性
            penalty = self.lambdas[k] * val
            total_penalty += penalty

        self.episode_steps += 1

        # 软截断，防止梯度爆炸
        return min(total_penalty, self.penalty_max)

    def update_lambdas(self):
        """
        [Episode-End] 更新拉格朗日乘子 (Dual Ascent)
        逻辑：
        违反率 (Violation Rate) = 违规步数 / 总步数
        Diff = 违反率 - 允许限制 (cost_limit)
        Lambda = Lambda + lr * Diff
        """
        if not self.enabled or self.episode_steps == 0:
            return

        for k in self.constraint_keys:
            # 计算本回合的违规率
            violation_rate = self.episode_violation_accum[k] / self.episode_steps

            # 误差: 实际违规率 - 允许违规率
            # 如果 rate > limit (违规太多), diff > 0, lambda 增加, 惩罚变大
            # 如果 rate < limit (很安全), diff < 0, lambda 减小, 鼓励探索
            diff = violation_rate - self.cost_limit

            # 更新并投影回 >= 0
            new_val = self.lambdas[k] + self.lr * diff
            self.lambdas[k] = max(0.0, new_val)

    def get_status(self) -> Dict[str, float]:
        """返回当前的 Lambda 值用于 Log"""
        status = {}
        for k, v in self.lambdas.items():
            status[f"lambda/{k}"] = v

        # 也记录一下当前的违规率
        if self.episode_steps > 0:
            for k in self.constraint_keys:
                rate = self.episode_violation_accum[k] / self.episode_steps
                status[f"violation_rate/{k}"] = rate

        return status

    # --- 序列化接口 (用于模型保存/加载) ---
    def save_state(self) -> Dict:
        return self.lambdas

    def load_state(self, state: Dict):
        if state:
            self.lambdas = state
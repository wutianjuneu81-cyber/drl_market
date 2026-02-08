import numpy as np
import cvxpy as cp
from typing import Dict, Any
from baselines.base_direct_controller import BaseDirectController


class DirectMPCController(BaseDirectController):
    """
    [Direct MPC Controller]
    单层 MPC：直接输出总功率指令 (kW)
    - 追踪目标功率 (target_power_kw)
    - 简化老化成本：L1 throughput
    - 具备平滑约束
    """

    def __init__(
        self,
        config: Dict[str, Any],
        horizon: int = 8,
        dt_seconds: float = 1.0,
        weight_tracking: float = 100.0,
        weight_aging: float = 0.1,
        weight_smooth: float = 0.01,
    ):
        super().__init__(config)

        self.horizon = int(horizon)
        self.dt_hours = float(dt_seconds) / 3600.0

        self.weight_tracking = float(weight_tracking)
        self.weight_aging = float(weight_aging)
        self.weight_smooth = float(weight_smooth)

        # 缓存上一次功率，用于平滑项
        self.last_cmd = 0.0

    def get_action(
        self,
        current_soc: float,
        current_max_temp: float,
        target_power_kw: float,
        timestamp: int = 0,
        **kwargs
    ) -> float:
        # --- 1) 软限制：基于 SoC 保护 ---
        if current_soc >= self.soc_max and target_power_kw < 0:
            target_power_kw = 0.0
        elif current_soc <= self.soc_min and target_power_kw > 0:
            target_power_kw = 0.0

        # --- 2) 构建 MPC 优化问题 ---
        P = cp.Variable(self.horizon, name="P_cmd")

        cost = 0.0
        constraints = []

        for t in range(self.horizon):
            # Tracking
            cost += cp.sum_squares(P[t] - target_power_kw) * self.weight_tracking

            # Aging (简化 L1 throughput)
            cost += cp.abs(P[t]) * self.weight_aging

            # Physical bounds
            constraints.append(P[t] <= self.max_power_kw)
            constraints.append(P[t] >= -self.max_power_kw)

            # Smoothness
            if t > 0:
                cost += cp.sum_squares(P[t] - P[t - 1]) * self.weight_smooth

        # 软约束加入 last_cmd（防止跳变）
        cost += cp.sum_squares(P[0] - self.last_cmd) * (self.weight_smooth * 5.0)

        prob = cp.Problem(cp.Minimize(cost), constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
            except Exception:
                return float(np.clip(target_power_kw, -self.rated_power_kw, self.rated_power_kw))

        if P.value is None:
            return float(np.clip(target_power_kw, -self.rated_power_kw, self.rated_power_kw))

        cmd_kw = float(P.value[0])
        cmd_kw = float(np.clip(cmd_kw, -self.max_power_kw, self.max_power_kw))

        self.last_cmd = cmd_kw
        return cmd_kw
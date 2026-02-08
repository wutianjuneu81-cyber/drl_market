import numpy as np
from typing import Dict, Any
from baselines.base_direct_controller import BaseDirectController


class DirectRuleController(BaseDirectController):
    """
    [Direct Rule-Based Controller]
    单层规则控制器：直接输出总功率指令 (kW)，不经过 shadow price。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 温度降额参数（硬阈值 + 线性降额区间）
        self.temp_soft_limit = self.temp_max - 5.0

    def get_action(
        self,
        current_soc: float,
        current_max_temp: float,
        target_power_kw: float,
        timestamp: int = 0,
        **kwargs
    ) -> float:
        # 1) 功率裁剪到额定范围
        cmd_kw = float(np.clip(target_power_kw, -self.max_power_kw, self.max_power_kw))

        # 2) SoC 边界保护
        if current_soc >= self.soc_max and cmd_kw < 0:
            cmd_kw = 0.0
        elif current_soc <= self.soc_min and cmd_kw > 0:
            cmd_kw = 0.0

        # 3) 温度降额（线性）
        if current_max_temp > self.temp_soft_limit:
            overheat = current_max_temp - self.temp_soft_limit
            span = max(self.temp_max - self.temp_soft_limit, 1e-6)
            derate = max(0.0, 1.0 - overheat / span)
            cmd_kw *= derate

        return float(cmd_kw)
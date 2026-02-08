from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDirectController(ABC):
    """
    [单层控制器统一接口 - 强制 ABC]
    职责：接收系统状态 -> 输出总功率指令 (kW)
    注意：不负责电芯之间的功率分配（由外部执行器做平均分配）
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

        market_cfg = self.cfg.get("market", {})
        if "rated_power_kw" not in market_cfg:
            raise ValueError("BaseDirectController: 'rated_power_kw' missing in config['market']")
        if "rated_capacity_kwh" not in market_cfg:
            raise ValueError("BaseDirectController: 'rated_capacity_kwh' missing in config['market']")

        self.rated_power_kw = float(market_cfg["rated_power_kw"])
        self.rated_capacity_kwh = float(market_cfg["rated_capacity_kwh"])

        # ✅ 统一硬阈值：与 HighLevelEnv 对齐
        self.soc_min = 0.02
        self.soc_max = 0.98

        # 温度阈值仍可从配置读取
        constraints = self.cfg.get("constraints", {}).get("limits", {})
        self.temp_max = float(constraints.get("temp_max", 45.0))  # 摄氏度

        env_limits = self.cfg.get("env", {}).get("limits", {})
        cons_limits = self.cfg.get("constraints", {}).get("limits", {})
        power_max = float(env_limits.get("power_max", cons_limits.get("power_max", 1.2)))
        self.max_power_kw = self.rated_power_kw * power_max

    @abstractmethod
    def get_action(
        self,
        current_soc: float,
        current_max_temp: float,
        target_power_kw: float,
        timestamp: int = 0,
        **kwargs
    ) -> float:
        """
        Args:
            current_soc: 电池组当前平均 SoC (0.0 ~ 1.0)
            current_max_temp: 当前最大温度 (°C)
            target_power_kw: AGC 目标功率 (kW, +放电 / -充电)
            timestamp: 当前时间步（可选）
            **kwargs: 扩展字段（如电价、SoH 等）

        Returns:
            cmd_kw: 实际执行功率 (kW)
        """
        raise NotImplementedError
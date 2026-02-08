import numpy as np
from typing import Dict, Optional, Any


class LithiumIonAging:
    """
    [创新核心] 物理感知应力累积模型 (Physics-Aware Stress Accumulation Model, PASAM)

    Paper Methodology:
    Delta_Q = |I| * dt * Psi(T, SOC, I)

    Psi 包含了:
    1. 热应力 (Arrhenius) -> 限制高温运行
    2. 机械应力 (Volume Expansion) -> 限制深充深放
    3. 动力学应力 (Polarization) -> 限制过大倍率
    """

    def __init__(self, config_or_dt: Any):
        self.dt = 1.0

        # 默认物理参数 (LFP)
        self.params = {
            'alpha_base': 2.0e-5,
            'beta_T': 0.069,
            'gamma_soc': 1.5,
            'k_soc': 2.0,
            'lambda_rate': 0.1,
            'C_cap': 280.0,
            'T_ref': 25.0
        }

        if isinstance(config_or_dt, dict):
            self.dt = config_or_dt.get('env', {}).get('low_level_step_seconds', 1.0)
            aging_cfg = config_or_dt.get('aging', {})
            # 覆盖默认参数
            for key in self.params:
                if key in aging_cfg:
                    self.params[key] = aging_cfg[key]
        elif isinstance(config_or_dt, (float, int)):
            self.dt = float(config_or_dt)

    def _compute_stress_factor(self, current_a: float, soc: float, temp_c: float) -> float:
        """
        计算综合物理应力因子 Psi (Severity Factor)
        """
        p = self.params

        # 1. 热应力 (Thermal Stress) - 指数增长
        # 物理含义: SEI 膜副反应速率
        thermal_stress = np.exp(p['beta_T'] * (temp_c - p['T_ref']))

        # 2. 机械应力 (Mechanical Stress) - 边缘区膨胀
        # 物理含义: 电极颗粒体积变化导致的疲劳
        soc_dev = abs(soc - 0.5)
        # SOC=0.5 -> stress=1.0; SOC=0/1 -> stress=1+gamma
        mech_stress = 1.0 + p['gamma_soc'] * (soc_dev ** p['k_soc'])

        # 3. 动力学应力 (Kinetic Stress) - 大倍率
        # 物理含义: 浓度极化与析锂风险
        c_rate = abs(current_a) / p['C_cap']
        kinetic_stress = 1.0 + p['lambda_rate'] * c_rate

        # 总应力因子 (乘法耦合)
        return thermal_stress * mech_stress * kinetic_stress

    def compute_aging_step(self, cluster_id: int, current_soc: float, temperature_c: float, current_a: float,
                           dt: Optional[float] = None) -> Dict[str, float]:
        step_dt = dt if dt is not None else self.dt

        # 1. 基础吞吐量 (Ah)
        throughput_ah = abs(current_a) * step_dt / 3600.0

        # 2. 计算应力因子
        stress_factor = self._compute_stress_factor(current_a, current_soc, temperature_c)

        # 3. 计算实际物理损伤 (统一模型)
        # Loss = Base_Rate * Stress * Throughput
        step_loss_ah = self.params['alpha_base'] * stress_factor * throughput_ah

        return {
            'total_loss_ah': step_loss_ah,  # 真实物理损失 (SOH update)
            'stress_factor': stress_factor,  # 记录应力水平 (Info, 可视化用)

            # [接口兼容] 无论叫 reward_loss 还是 proxy_loss，现在都是同一个物理值
            'stress_weighted_loss': step_loss_ah
        }

    def reset(self, cluster_id: Optional[int] = None):
        pass  # 无需状态重置
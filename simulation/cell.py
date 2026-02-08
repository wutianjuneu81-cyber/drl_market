import numpy as np
from typing import Dict, Tuple


class LithiumIronPhosphateCell:
    """
    LFP Battery Cell Model (Refactored for Campaign 2: Physics Fidelity)

    物理模型升级：
    1. [#48] Thevenin Equivalent Circuit (1-RC Model): 模拟极化效应和电压回升(Relaxation)。
    2. [#17] IR Drop: 考虑欧姆内阻压降。
    3. [#50] Aging Coupling: 这里的物理参数会随 SOH 动态变化。
    """

    def __init__(self, config: Dict):
        # 基础参数
        self.capacity_nominal_ah = config['env'].get('cell_capacity_ah', 280.0)
        self.nominal_voltage = 3.2

        # 物理限制
        self.v_max = 3.65
        self.v_min = 2.50

        # --- 电路参数 (Thevenin 1-RC) ---
        # R0: 欧姆内阻 (Ohm)
        # R1: 极化内阻 (Ohm)
        # C1: 极化电容 (Farad)
        # 这些通常是 SoC 和 温度 的函数，这里简化为常数或简单线性关系
        self.r0_base = 0.5e-3  # 0.5 mOhm
        self.r1_base = 0.2e-3  # 0.2 mOhm
        self.c1_base = 5000.0  # Farad (Tau = R1*C1 = 1s, 实际上 LFP Tau 较大，约 10-60s)
        self.tau_1 = 30.0  # 设定时间常数为 30秒

        # 导出 C1
        self.c1_base = self.tau_1 / self.r1_base

        # --- 状态变量 ---
        self.u_p = 0.0  # 极化电压 (Polarization Voltage across C1)
        self.soc = config['env'].get('initial_soc', 0.5)

        # LFP OCV 曲线 (Soc -> Voltage)
        # 简化的特征点插值：0%, 10%, ..., 100%
        self.ocv_soc_points = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
        self.ocv_vol_points = np.array([2.50, 2.95, 3.10, 3.20, 3.25, 3.28, 3.30, 3.32, 3.34, 3.45, 3.60])

    def reset(self, initial_soc: float = 0.5):
        self.soc = np.clip(initial_soc, 0.0, 1.0)
        self.u_p = 0.0  # 初始极化电压归零 (假设静置很久)

    def get_ocv(self, soc: float) -> float:
        """查表获取开路电压"""
        return np.interp(soc, self.ocv_soc_points, self.ocv_vol_points)

    def step(self, current_a: float, dt: float, soh_capacity: float = 1.0, soh_resistance: float = 1.0) -> Dict[
        str, float]:
        """
        单体电池步进

        Args:
            current_a: 电流 (Amps), 正为放电, 负为充电
            dt: 时间步长 (Seconds)
            soh_capacity: 容量健康度 (0~1), 影响 SoC 变化率
            soh_resistance: 内阻健康度 (通常 >= 1.0), 老化后内阻增加

        Returns:
            Dict: Voltage, Heat, SoC_next
        """
        # 1. 更新参数 (基于老化) [#50]
        # 真实容量 = 标称容量 * SOH_C
        capacity_actual = self.capacity_nominal_ah * soh_capacity

        # 真实内阻 = 标称内阻 * SOH_R (通常 SOH_R 定义为 R_now / R_new，所以是乘)
        # 注意：这里假设输入的 soh_resistance 随着老化是增加的 (e.g. 1.2, 1.5)
        # 如果输入是 0.8 (表示变差), 则需要取倒数，这里按 >1 增加处理
        r0_actual = self.r0_base * soh_resistance
        r1_actual = self.r1_base * soh_resistance
        # 时间常数 Tau = R1 * C1，假设 C1 衰减较小，主要由 R1 决定 Tau 变大
        tau = r1_actual * self.c1_base

        # 2. 更新 SoC (Coulomb Counting)
        # dSoC = - I * dt / (3600 * Cap)
        # 放电(I>0) -> SoC 减少
        delta_soc = -(current_a * dt) / (3600.0 * capacity_actual)
        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)

        # 3. 更新极化电压 (RC Circuit Dynamics) [#48]
        # 离散化公式: U_p[k+1] = U_p[k] * exp(-dt/tau) + I[k] * R1 * (1 - exp(-dt/tau))
        # 放电时 (I>0)，极化电压 U_p 增加 (表现为压降)
        exp_factor = np.exp(-dt / tau)
        self.u_p = self.u_p * exp_factor + current_a * r1_actual * (1 - exp_factor)

        # 4. 计算端电压 (Terminal Voltage) [#17]
        # V = OCV(SoC) - I*R0 - U_p
        ocv = self.get_ocv(self.soc)
        v_terminal = ocv - current_a * r0_actual - self.u_p

        # 5. 计算产热 (Heat Generation)
        # Q = I^2 * R0 + U_p^2 / R1 (焦耳热 + 极化热)
        # 或者简化为总内阻产热: Q approx I * (OCV - V) = I^2 * R_total
        # 使用 RC 模型计算更精确：
        heat_watts = (current_a ** 2) * r0_actual + (self.u_p ** 2) / r1_actual

        return {
            'voltage_v': v_terminal,
            'ocv_v': ocv,
            'current_a': current_a,
            'soc': self.soc,
            'heat_watts': heat_watts,
            'polarization_v': self.u_p,
            'r0_ohm': r0_actual
        }

    def get_state(self) -> np.ndarray:
        return np.array([self.soc, self.u_p])
import numpy as np
from collections import deque
from typing import Dict, Any


class HealthMetricsProvider:
    """
    物理层健康指标传感器。
    在 HighLevelEnv 的 Step 循环中被调用，用于计算 Reward 所需的各种统计量。
    """

    def __init__(self, num_cells, dt_seconds, temp_ref=35.0, window_size=10, q_ref=3.4, history_maxlen=5000):
        self.num_cells = num_cells
        self.dt = dt_seconds
        self.temp_ref = temp_ref
        self.window_size = window_size
        self.q_ref = q_ref  # 参考容量(Ah)用于归一化

        self.history_keys = [
            "soc_std", "temp_peak", "cycle_equiv_increment", "thermal_stress_increment",
            "violation_count", "utilization", "soh_std", "delta_soh", "steps"
        ]
        self.history = {k: deque(maxlen=history_maxlen) for k in self.history_keys}
        self.prev_avg_soh = 1.0
        self.reset_window()

    def reset_window(self):
        self.steps = 0
        self.soc_records = []
        self.temp_records = []
        self.curr_records = []
        self.violation_count = 0
        self.util_acc = 0.0
        self.soh_records = []

    def update_step(self, socs, temps_c, currents, violation_flags, utilization, sohs):
        self.steps += 1
        self.soc_records.append(socs)
        self.temp_records.append(temps_c)
        self.curr_records.append(currents)
        self.soh_records.append(sohs)

        if any(violation_flags.values()):
            self.violation_count += 1
        self.util_acc += utilization

    def finalize_window(self) -> Dict[str, Any]:
        """计算当前窗口内的聚合统计量"""
        if self.steps == 0:
            return {k: 0.0 for k in self.history_keys if k != "steps"} | {"steps": 0}

        soc_arr = np.array(self.soc_records)
        temp_arr = np.array(self.temp_records)
        curr_arr = np.array(self.curr_records)
        soh_arr = np.array(self.soh_records)

        # 1. SoC 一致性 (取窗口末端)
        soc_std = float(np.std(soc_arr[-1]))

        # 2. 温度峰值 (窗口内最大值)
        temp_peak = float(np.max(temp_arr))

        # 3. 等效循环增量 (Ah Throughput)
        ah_inc = np.sum(np.abs(curr_arr)) * self.dt / 3600.0
        cycle_equiv_inc = float(ah_inc / (self.q_ref + 1e-6))

        # 4. 热应力增量 (超过参考温度的积分)
        thermal_excess = np.clip(temp_arr - self.temp_ref, 0, None)
        thermal_stress_inc = float(np.sum(thermal_excess) * self.dt)

        # 5. 利用率
        util = float(self.util_acc / max(self.steps, 1))

        # 6. SOH 统计
        soh_std = float(np.std(soh_arr[-1]))
        avg_soh = float(np.mean(soh_arr[-1]))
        delta_soh = max(0.0, self.prev_avg_soh - avg_soh)
        self.prev_avg_soh = avg_soh

        result = {
            "soc_std": soc_std,
            "temp_peak": temp_peak,
            "cycle_equiv_increment": cycle_equiv_inc,
            "thermal_stress_increment": thermal_stress_inc,
            "violation_count": self.violation_count,
            "utilization": util,
            "soh_std": soh_std,
            "delta_soh": delta_soh,
            "steps": self.steps,
        }

        # 存入历史
        for k in self.history_keys:
            if k in result:
                self.history[k].append(result[k])

        self.reset_window()
        return result

    def get_trend_features(self):
        """返回历史均值，用于观测空间的 Trend 部分"""

        def avg(k):
            return float(np.mean(self.history[k])) if self.history[k] else 0.0

        if self.history["steps"]:
            vr_list = [(vc / max(st, 1)) for vc, st in zip(self.history["violation_count"], self.history["steps"])]
            violation_rate_mean = float(np.mean(vr_list))
        else:
            violation_rate_mean = 0.0

        return {
            "soc_std_mean": avg("soc_std"),
            "temp_peak_mean": avg("temp_peak"),
            "cycle_equiv_mean": avg("cycle_equiv_increment"),
            "thermal_stress_mean": avg("thermal_stress_increment"),
            "violation_rate_mean": violation_rate_mean,
            "utilization_mean": avg("utilization"),
            "soh_std_mean": avg("soh_std"),
            "delta_soh_mean": avg("delta_soh"),
        }
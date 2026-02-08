import numpy as np
from gymnasium import Env as GymEnv, spaces
from .battery import Battery
from .data_loader import PowerProfileLoader
from typing import Optional, Dict, Any, Union, Sequence, Callable


class BaseSimulationEnv(GymEnv):
    """
    基础物理仿真环境 (Simulation Layer)
    """

    def __init__(
            self,
            num_cells=24,
            time_passed_by_step=1.0,
            randomization=True,
            scenario=0,
            charging=True,
            length_reference_run=86400,
            quiet_init=True,
            safety_limits=None,
            aging_cfg=None,
            external_power_profile: Optional[Union[Sequence[float], np.ndarray, Callable[[int], float]]] = None,
            enable_internal_drift: bool = False,
            internal_drift_sigma: float = 0.5,
            internal_action_coupling_scale: float = 2.0,
            aging_weight_current_scale: float = 0.6,
            cluster_variation: dict | None = None,
            cluster_variation_seed: int | None = None,
            low_level_reward_weights: Dict[str, float] | None = None,
            allow_relaxation: bool = False,
            data_split: Dict[str, int] | None = None,
            subset: str = "full",
            **kwargs
    ):
        super().__init__()
        self.num_cells = num_cells
        self.time_passed_by_step = time_passed_by_step
        self.randomization = randomization
        self.scenario = scenario
        self.charging = charging
        self.quiet_init = quiet_init
        self.length_reference_run = length_reference_run

        profile_type = kwargs.get('profile_type', 'load')

        # 数据加载器
        self.profile_loader = PowerProfileLoader(external_power_profile, data_split, subset, profile_type=profile_type)
        self.profile_index = 0

        # 动作/观察空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_cells * 4 + 8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_cells + 1,), dtype=np.float32
        )

        # 电池组配置
        self.cluster_variation_cfg = cluster_variation or {}
        rng = np.random.default_rng(cluster_variation_seed if cluster_variation_seed is not None else None)
        capacity_range = self.cluster_variation_cfg.get("capacity_scale_range", [0.98, 1.02])
        rint_range = self.cluster_variation_cfg.get("rint_scale_range", [0.97, 1.03])
        self.cluster_capacity_scales = [float(rng.uniform(capacity_range[0], capacity_range[1])) for _ in range(24)]
        self.cluster_rint_scales = [float(rng.uniform(rint_range[0], rint_range[1])) for _ in range(24)]

        self.battery = Battery(
            num_cell=num_cells,
            Time_passed_by_step=time_passed_by_step,
            randomization=randomization,
            scenario=scenario,
            charging=charging,
            aging_cfg=aging_cfg,
            cluster_capacity_scales=self.cluster_capacity_scales,
            cluster_rint_scales=self.cluster_rint_scales,
            num_clusters=24,
        )

        self.enable_internal_drift = enable_internal_drift
        self.internal_drift_sigma = internal_drift_sigma
        self.internal_action_coupling_scale = internal_action_coupling_scale

        # [MODIFIED] 初始化为 0.0，不再是 -45.0
        self.target_power = 0.0
        self.prev_target_power = self.target_power
        self.power_error = 0.0
        self.prev_actual_power_kw = 0.0

        self._degradation_budget = 1.0
        self._tracking_slack = 0.05
        self.num_clusters = 24
        self.cluster_map = self._assign_clusters(num_cells, self.num_clusters)
        self.aging_weights = [1.0] * self.num_clusters

        self.aging_weight_current_scale = aging_weight_current_scale
        self.allow_relaxation = allow_relaxation
        self.relax_cap = 1.05

        self.safety = safety_limits or {
            "soc_min": 0.10, "soc_max": 0.95, "temp_min": 10.0, "temp_max": 50.0,
        }

        self.step_counter = 0
        self._prev_total_scale = 0.0

        self.rw = low_level_reward_weights or {
            "tracking_bonus": 0.10, "soc_coeff_base": 0.10, "soc_coeff_scale": 0.25,
            "temp_coeff": 0.01, "soh_coeff_base": 0.40, "soh_coeff_scale": 0.30
        }

    def _assign_clusters(self, num_cells, num_clusters):
        clusters = [[] for _ in range(num_clusters)]
        for i in range(num_cells):
            clusters[i % num_clusters].append(i)
        return clusters

    def consume_goal(self, goal_dict: dict):
        if not goal_dict: return
        aw = goal_dict.get("aging_weights", goal_dict.get("aging_weight_schedule", self.aging_weights))
        self.aging_weights = [float(np.clip(x, 0.0, 2.0)) for x in aw]
        self._degradation_budget = float(np.clip(goal_dict.get("degradation_budget", 1.0), 0.0, 1.0))
        self._tracking_slack = float(np.clip(goal_dict.get("tracking_slack", self._tracking_slack), 0.0, 0.25))

    def _load_command(self):
        # [MODIFIED] 逻辑优化：只有当 profile_loader 有真实数据时才覆盖 target_power
        # 如果 profile_loader 返回 None 或 0.0 (且我们认为0是无效值时)，则保持原样(可能是wrapper注入的)
        # 这里简化为：只要 profile_loader 有 active_profile，就尝试读取
        if self.profile_loader.raw_profile is not None:
            val = self.profile_loader.get_value_at(self.profile_index) if hasattr(self.profile_loader,
                                                                                  'get_value_at') else self.profile_loader.get_current_target()
            self.target_power = val
            return True
        return False

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # [FIX #10] 连续性保护
        # 如果 options['reset_physics'] 为 False，则保留上一时刻的功率状态
        soft_reset = options and not options.get('reset_physics', False)

        self.step_counter = 0

        pl_len = 0
        if hasattr(self.profile_loader, 'active_profile') and self.profile_loader.active_profile is not None:
            pl_len = len(self.profile_loader.active_profile)

        if self.profile_loader.subset == "train" and self.randomization and pl_len > self.length_reference_run:
            self.profile_loader.reset(seed)  # Use internal reset
        else:
            self.profile_loader.current_index = 0

        # [MODIFIED] 如果没加载到命令，归零，而不是 -45
        if not self._load_command():
            self.target_power = 0.0

        if not soft_reset:
            self.prev_target_power = self.target_power
            self.power_error = 0.0
            self._prev_total_scale = 0.0
            self.prev_actual_power_kw = 0.0  # Hard Reset
        else:
            # Soft Reset: Keep prev_actual_power_kw from last step
            pass

        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1
        total_scale, ratios = self._decode_action(action)

        external_used = self._load_command()
        if external_used:
            self.profile_loader.step()
        else:
            # 如果没有外部 Profile，也没有 Wrapper 注入（target_power保持不变），加一点点漂移模拟
            drift = np.random.normal(0, self.internal_drift_sigma) if self.enable_internal_drift else 0.0
            # 保持 target_power 不变 (由 Wrapper 控制)，只加内部噪音
            # candidate = self.target_power + drift
            pass

        max_current_base = (
            self.battery.get_max_charging_current()
            if self.charging else self.battery.get_max_discharging_current()
        )
        applied_total = total_scale * max_current_base
        cell_currents = applied_total * ratios

        # Cluster Aging Constraints
        cluster_utilizations = []
        for ci, cluster in enumerate(self.cluster_map):
            cluster_currs = cell_currents[cluster]
            avg_mag = np.mean(np.abs(cluster_currs))
            w = self.aging_weights[ci]
            limit_factor_raw = 1.0 / (1.0 + self.aging_weight_current_scale * (w - 1.0))
            if self.allow_relaxation and w < 1.0 and limit_factor_raw > 1.0:
                limit_factor = min(self.relax_cap, limit_factor_raw)
            else:
                limit_factor = min(1.0, limit_factor_raw)
            limit = limit_factor * max_current_base / self.num_cells
            cluster_utilizations.append(float(avg_mag / (limit + 1e-8)))
            if avg_mag > limit:
                factor = limit / (avg_mag + 1e-8)
                cell_currents[cluster] *= factor

        self.battery.step(cell_currents.tolist(), 0.0)

        actual_power_watts = self.battery.get_total_output_power()
        actual_power_kw = actual_power_watts / 1000.0
        step_mileage = abs(actual_power_kw - self.prev_actual_power_kw)
        self.prev_actual_power_kw = actual_power_kw

        self.power_error = actual_power_kw - self.target_power
        reward, components = self._calc_reward(actual_power_kw, total_scale)

        terminated = False
        truncated = self.step_counter >= self.length_reference_run

        info = self._build_info(actual_power_kw, step_mileage, cluster_utilizations, components, external_used)
        return self._get_obs(), reward, terminated, truncated, info

    def _decode_action(self, action):
        a = np.asarray(action, dtype=np.float32)
        total_scale = float(np.tanh(a[0]))
        raw_weights = a[1:]
        z = raw_weights - np.max(raw_weights)
        exp_z = np.exp(z)
        ratios = exp_z / (np.sum(exp_z) + 1e-12)
        return total_scale, ratios

    def _get_obs(self):
        socs = self.battery.get_SoC_values()
        temps_norm = [self.battery.scale_temp(t) for t in self.battery.get_temp_values()]
        sohs = self.battery.get_soh_values()
        obs_cells = []
        for i in range(self.num_cells):
            obs_cells.extend([socs[i], temps_norm[i], sohs[i], 0.5])

        obs_global = [
            np.tanh(self.target_power / 30.0) * 0.5 + 0.5,
            np.tanh(self.power_error / 10.0) * 0.5 + 0.5,
            min(self.step_counter / 1000.0, 1.0),
            float(np.mean(sohs)),
            self._degradation_budget,
            self._tracking_slack,
            np.mean(self.aging_weights),
            np.std(self.aging_weights),
        ]
        return np.array(obs_cells + obs_global, dtype=np.float32)

    def _calc_reward(self, actual_kw, total_scale_current):
        relative_error = abs(self.power_error) / max(abs(self.target_power), 1.0)
        tracking_component = 1.0 - relative_error
        if relative_error < self._tracking_slack:
            tracking_component += self.rw["tracking_bonus"]

        soc_std = np.std(self.battery.get_SoC_values())
        temp_std = np.std([t - 273.15 for t in self.battery.get_temp_values()])
        avg_soh = np.mean(self.battery.get_soh_values())
        w_mean = np.mean(self.aging_weights)
        weight_amp = w_mean ** 1.2

        soc_penalty = soc_std * (self.rw["soc_coeff_base"] + self.rw["soc_coeff_scale"] * weight_amp)
        soh_penalty = (1.0 - avg_soh) * (self.rw["soh_coeff_base"] + self.rw["soh_coeff_scale"] * weight_amp)
        temp_penalty = temp_std * self.rw["temp_coeff"] * (1.0 - self._degradation_budget) * (
                1.0 + 0.3 * (w_mean - 1.0))

        delta_scale = total_scale_current - self._prev_total_scale
        self._prev_total_scale = total_scale_current
        delta_penalty = 0.03 * (delta_scale ** 2)

        health_component = -(soc_penalty + soh_penalty + temp_penalty) - delta_penalty
        final_reward = float(np.clip(tracking_component + health_component, -5.0, 5.0))

        return final_reward, {
            "tracking_component": tracking_component,
            "health_component": health_component,
            "final_reward": final_reward
        }

    def _build_info(self, actual_kw, step_mileage, cluster_utilizations, components, external_used):
        socs = self.battery.get_SoC_values()
        temps_c = [t - 273.15 for t in self.battery.get_temp_values()]

        # [NEW] Get Aux Power
        aux_watts = self.battery.aux_powers[-1] if self.battery.aux_powers else 0.0

        return {
            "TotalOutputPower": actual_kw,
            "TargetPower": self.target_power,
            "GridCommand": self.target_power,
            "power_error": self.power_error,
            "SoC": socs,
            "Temperature": temps_c,
            "SOH": self.battery.get_soh_values(),
            "aging_info": self.battery.get_latest_aging_info(),
            "violation_flags": {
                "soc": any(s < self.safety["soc_min"] or s > self.safety["soc_max"] for s in socs),
                "temp": any(t < self.safety["temp_min"] or t > self.safety["temp_max"] for t in temps_c),
            },
            "cluster_budget_utilization": cluster_utilizations,
            "external_profile_used": external_used,
            "reward_components": components,
            "step_mileage_kw": step_mileage,
            "cluster_capacity_scales": self.cluster_capacity_scales,
            "cluster_rint_scales": self.cluster_rint_scales,
            "cell_currents": [],
            # [NEW] Explicit Aux Power
            "aux_power_kw": aux_watts / 1000.0,
            "cooling_active": self.battery.cooling_active
        }

    def close(self):
        pass
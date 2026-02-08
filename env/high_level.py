import os
import gymnasium as gym
import numpy as np
import torch
import warnings
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, Callable

# 鲁棒导入
try:
    from DRL_market.env.low_level import LowLevelEnv
    from DRL_market.common.config import load_config
    from DRL_market.simulation.data_loader import MarketDataLoader
    from DRL_market.market.mechanism import MarketEngine
    from DRL_market.core.constraints.manager import ConstraintManager
    from DRL_market.market.scheduler import DayAheadScheduler, SchedulerConfig
    from DRL_market.market.clearing import (
        FrequencyMarketClearing,
        FrequencyBid,
        EnergyMarketResult,
    )

    try:
        from DRL_market.core.math_utils.stats import RewardNormalizer
    except ImportError:
        from DRL_market.core.math.stats import RewardNormalizer
except ImportError:
    from env.low_level import LowLevelEnv
    from common.config import load_config
    from simulation.data_loader import MarketDataLoader
    from market.mechanism import MarketEngine
    from core.constraints.manager import ConstraintManager
    from market.scheduler import DayAheadScheduler, SchedulerConfig
    from market.clearing import FrequencyMarketClearing, FrequencyBid, EnergyMarketResult
    from core.math.stats import RewardNormalizer


class HighLevelEnv(gym.Env):
    """
    High-Level Environment (Market Layer, Fully Refactored)

    职责：
    1) 市场交互：报价、出清、结算
    2) 生成调度指令 (plan_p_base / plan_reg_cap)
    3) 结算只基于 LowLevelEnv 实际执行结果
    """

    def __init__(
            self,
            config_or_factory: Any = "DRL_market/config.yaml",
            is_eval: bool = False,
            interval: int = 900,
            goal_interface: Any = None,
            low_model: Any = None,
            reward_weights: Dict = None,
            training_mode: bool = True,
            **kwargs
    ):
        super().__init__()

        # --- 1. 配置加载 ---
        if isinstance(config_or_factory, (str, dict)):
            self.cfg = load_config(config_or_factory) if isinstance(config_or_factory, str) else config_or_factory
            self.make_low_env_fn = lambda: LowLevelEnv(self.cfg)
        elif callable(config_or_factory):
            self.make_low_env_fn = config_or_factory
            self.cfg = load_config("DRL_market/config.yaml")
        else:
            self.cfg = config_or_factory
            self.make_low_env_fn = lambda: LowLevelEnv(self.cfg)

        self.env_cfg = self.cfg.get("env") if self.cfg.get("env") else self.cfg.get("environment")
        self.market_cfg = self.cfg.get("market")
        self.is_eval = is_eval or (not training_mode)

        # 时间与物理参数
        self.interval = int(self.env_cfg.get("high_level_step_minutes", 15) * 60)
        self.dt_low = float(self.env_cfg.get("low_level_step_seconds", 1.0))

        # ✅ 确保 market.time_step_hours 与 env.high_level_step_minutes 对齐
        if "time_step_hours" not in self.market_cfg:
            self.market_cfg["time_step_hours"] = self.interval / 3600.0

        self.steps_per_episode = 96
        self.steps_per_hour = int(3600 / self.interval)

        self.n_clusters = int(self.env_cfg.get("n_clusters", 24))
        self.capacity_nominal = float(self.market_cfg.get("rated_capacity_kwh", 5000.0))
        self.rated_power_kw = float(self.market_cfg.get("rated_power_kw", 2000.0))

        # 奖励参数
        self.reward_scaler = float(self.market_cfg.get("reward_scaler", 1e-4))
        self.penalty_price_dev = float(self.market_cfg.get("penalty_price_deviation", 2.0))

        # [考核门槛]
        self.k_threshold = float(self.market_cfg.get("k_threshold", 2.5))
        self.k_threshold_cap = float(self.market_cfg.get("k_threshold_cap", 1.8))

        # ✅ [FIX] 固定市场容忍度（禁止 action 篡改规则）
        self.market_tolerance = float(self.market_cfg.get("default_tolerance", 0.02))

        # ✅ [FIX] 真实老化成本（CNY/Ah）换算
        self.cost_per_ah = float(self.market_cfg.get("cost_per_ah_cny", 0.0))
        if self.cost_per_ah <= 0.0:
            battery_capex_cny = float(self.market_cfg.get("battery_capex_cny", 3.6e6))
            nominal_voltage_v = float(self.env_cfg.get("nominal_voltage_v", 768.0))
            total_capacity_ah = (self.capacity_nominal * 1000.0) / max(nominal_voltage_v, 1e-6)
            self.cost_per_ah = battery_capex_cny / max(total_capacity_ah, 1e-6)

        # --- 2. 初始化日前调度器 ---
        sched_cfg = SchedulerConfig(
            rated_power_kw=self.rated_power_kw,
            rated_capacity_kwh=self.capacity_nominal,
            dt_hours=self.interval / 3600.0,
            horizon=self.steps_per_episode
        )
        self.scheduler = DayAheadScheduler(sched_cfg)
        self.current_da_plan = None

        # --- 3. 空间定义 ---
        self.act_dim = self.n_clusters + 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
        self.obs_dim = 12 + 3 + 2 + 96 + 6 + 2
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(self.obs_dim,), dtype=np.float32)

        # --- 4. 模块初始化 ---
        self.low_env = self.make_low_env_fn()

        if low_model is not None:
            self.low_model = low_model
        else:
            default_path = "DRL_market/models/low_level_sac.zip"
            if os.path.exists(default_path):
                try:
                    from stable_baselines3 import SAC
                    self.low_model = SAC.load(default_path)
                    print(f"✅ HighLevelEnv: Auto-loaded Low-Level Model from {default_path}")
                except Exception as e:
                    self.low_model = None
                    print(f"⚠️ HighLevelEnv: Failed to load Low-Level Model: {e}")
            else:
                self.low_model = None
                if not self.is_eval:
                    print("⚠️ HighLevelEnv: No Low-Level Model provided. Using dummy actions (zeros).")

        self.data_loader = MarketDataLoader(self.cfg, is_eval=self.is_eval)
        self.market = MarketEngine(self.cfg)
        self.reward_normalizer = RewardNormalizer()
        self.constraint_manager = ConstraintManager(self.cfg)

        # 出清引擎
        self.clearing_engine = FrequencyMarketClearing(
            market_cfg=self.market_cfg,
            assumption_mode=self.market_cfg.get("reg_capacity_mode", "simplified"),
            duration_req_hours=1.0
        )

        # 状态与缓存
        self.current_step = 0
        self.current_price_series = None
        self.current_agc_series = None
        self.injected_lambdas = {}
        self.suggested_power_chunk = None

        # 出清/结算缓存
        self.current_market_period = -1
        self.last_clearing_result = None
        self.current_reg_mileage_price = 0.0
        self.current_reg_cap_kw = 0.0
        self.last_k_value = 1.0

        # 小时级缓冲（必须按小时结算）
        self.hourly_mileage_kw = 0.0
        self.hourly_actual_curve = []
        self.hourly_target_curve = []
        self.hourly_target_base = []
        self.hourly_price_list = []
        self.hourly_aging_ah = 0.0
        self.hourly_aging_cost = 0.0
        self.hourly_aux_cost = 0.0
        self.hourly_max_temp = -999.0
        self.price_unit = str(self.market_cfg.get("price_unit", "mwh")).lower()

        # 限制
        constraints_cfg = self.cfg.get("constraints", {})
        self.limits = constraints_cfg.get("limits", {
            "temp_max": 45.0,
            "soc_min": 0.05,
            "soc_max": 0.95,
            "power_max": 1.2
        })
        self._mileage_warned = False

    def _price_to_kwh(self, price: float) -> float:
        """
        将价格统一转换为 CNY/kWh。
        默认假设输入为 CNY/MWh（市场通用），因此 /1000。
        若已是 CNY/kWh，则直接返回。
        """
        if self.price_unit in ("mwh", "cny/mwh", "yuan/mwh"):
            return price / 1000.0
        return price

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.low_env.reset(seed=seed, options=options)
        self.data_loader.reset(seed=seed)
        self.current_step = 0
        self.current_market_period = -1
        self.constraint_manager.reset()

        self.current_price_series, self.current_agc_series = self.data_loader.sample_day_data()

        # 预测数据生成
        forecast_prices = self.data_loader.get_forecast_with_noise(
            self.current_price_series,
            noise_level=0.05
        )
        forecast_reg_prices = np.ones(96) * 6.0

        # 日前计划生成
        init_soc = float(self.low_env.battery.get_mean_soc())
        plan_result = self.scheduler.solve(
            price_energy=forecast_prices,
            price_reg=forecast_reg_prices,
            initial_soc=init_soc
        )
        self.current_da_plan = plan_result

        # 清空缓存
        self._reset_hourly_buffers()

        obs = self._get_obs()
        return obs, {}

    def _reset_hourly_buffers(self):
        self.hourly_mileage_kw = 0.0
        self.hourly_actual_curve = []
        self.hourly_target_curve = []
        self.hourly_target_base = []
        self.hourly_price_list = []
        self.hourly_aging_ah = 0.0
        self.hourly_aging_cost = 0.0
        self.hourly_aux_cost = 0.0
        self.hourly_max_temp = -999.0

    def _should_clear_market(self) -> bool:
        market_period = self.current_step // self.steps_per_hour
        return market_period != self.current_market_period

    def _run_clearing(self, plan_p_base_kw: float, plan_reg_bid_kw: float):
        market_period = self.current_step // self.steps_per_hour
        self.current_market_period = market_period

        bid_capacity_mw = max(0.0, plan_reg_bid_kw / 1000.0)
        bid_price = float(self.market_cfg.get("reg_mileage_bid_price", 8.0))
        soc_now = float(self.low_env.battery.get_mean_soc())

        bids = [
            FrequencyBid(
                unit_id="main_storage",
                time_period=market_period,
                unit_type="storage",
                area="GD",
                capacity_mw=bid_capacity_mw,
                mileage_price=bid_price,
                k_value=self.last_k_value,
                is_independent_storage=True,
                rated_power_mw=self.rated_power_kw / 1000.0,
                rated_capacity_mwh=self.capacity_nominal / 1000.0
            )
        ]

        energy_results = {
            "main_storage": EnergyMarketResult(
                time_period=market_period,
                base_power_mw=plan_p_base_kw / 1000.0,
                energy_price=0.0
            )
        }

        system_demand_mw = float(
            self.market_cfg.get("system_reg_demand_mw", self.rated_power_kw / 1000.0)
        )

        res = self.clearing_engine.clear_frequency_market(
            bids=bids,
            system_demand_mw=system_demand_mw,
            energy_results=energy_results,
            soc_snapshot={"main_storage": soc_now},
            area_demands={"GD": system_demand_mw},
            area_min_ratios={"GD": self.market_cfg.get("area_min_ratio", 0.0)}
        )

        cleared_cap = 0.0
        for u in res.cleared_units:
            if u.unit_id == "main_storage":
                cleared_cap = u.cleared_capacity_mw
                break

        self.current_reg_cap_kw = cleared_cap * 1000.0
        self.current_reg_mileage_price = res.unified_price
        self.last_clearing_result = res

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # --- A. 获取日前计划 ---
        t = self.current_step
        idx_plan = min(t, len(self.current_da_plan["p_base"]) - 1)
        plan_p_base = float(self.current_da_plan["p_base"][idx_plan])
        plan_reg_bid = float(self.current_da_plan["c_reg"][idx_plan])

        # --- B. 动作解析 ---
        hierarchy = self.cfg.get("hierarchy", {})
        lambda_max = float(hierarchy.get("lambda_max", 3.0))
        shadow_prices = (action[:self.n_clusters] + 1.0) / 2.0 * lambda_max

        power_max = float(self.limits.get("power_max", 1.2))
        p_real = float(action[self.n_clusters] * self.rated_power_kw * power_max)

        # ✅ [FIX] 禁止 action 修改市场规则，使用固定容忍度
        self.market.update_tolerance(self.market_tolerance)

        # --- 出清（每小时强制执行） ---
        if self._should_clear_market():
            self._reset_hourly_buffers()
            self._run_clearing(plan_p_base, plan_reg_bid)

        plan_reg_cap = self.current_reg_cap_kw

        # [Safety] 物理熔断检查
        mean_soc = float(self.low_env.battery.get_mean_soc())
        if not np.isfinite(mean_soc) or mean_soc > 2.0 or mean_soc < -1.0:
            return np.zeros(self.obs_dim, dtype=np.float32), -10.0, False, True, {"error": "Physics Collapse"}

        # 硬约束保护
        if mean_soc > 0.98:
            p_real = max(0.0, p_real)
            phys_reg_cap = 0.0
        elif mean_soc < 0.02:
            p_real = min(0.0, p_real)
            phys_reg_cap = 0.0
        else:
            phys_reg_cap = plan_reg_cap

        # --- C. 执行仿真 ---
        start_idx = self.current_step * self.interval
        end_idx = start_idx + self.interval
        agc_segment = self.current_agc_series[start_idx: min(end_idx, len(self.current_agc_series))]

        price_idx = min(self.current_step, len(self.current_price_series) - 1)
        current_price_raw = float(self.current_price_series[price_idx])
        current_price_kwh = self._price_to_kwh(current_price_raw)

        # 统一记录 kWh 价
        self.hourly_price_list.append(current_price_kwh)

        # --- 物理仿真循环（只保留一份） ---
        for step_t in range(len(agc_segment)):
            agc_val = float(agc_segment[step_t])
            grid_target_kw = plan_p_base + agc_val * plan_reg_cap
            agent_req_kw = p_real + agc_val * phys_reg_cap

            if self.low_model:
                ll_obs = self.low_env._get_obs(agent_req_kw)
                ll_action, _ = self.low_model.predict(ll_obs, deterministic=True)
            else:
                ll_action = np.zeros(self.n_clusters)

            _, _, _, _, info_ll = self.low_env.step(
                action=ll_action,
                power_request_kw=agent_req_kw,
                shadow_prices=shadow_prices
            )

            actual_p = float(info_ll.get("actual_power_kw", 0.0))

            if "step_mileage_kw" not in info_ll and not self._mileage_warned:
                warnings.warn(
                    "HighLevelEnv: step_mileage_kw missing in LowLevelEnv info. "
                    "Mileage will be zero. Check LowLevelEnv.step output.",
                    RuntimeWarning
                )
                self._mileage_warned = True

            step_mileage_kw = float(info_ll.get("step_mileage_kw", 0.0))
            self.hourly_mileage_kw += step_mileage_kw

            self.hourly_actual_curve.append(actual_p)
            self.hourly_target_curve.append(grid_target_kw)
            self.hourly_target_base.append(plan_p_base)

            # ✅ [FIX] 使用真实老化损耗（Ah）计成本
            real_aging_ah = float(info_ll.get("aging_cost_real", 0.0))
            self.hourly_aging_ah += real_aging_ah
            self.hourly_aging_cost += real_aging_ah * self.cost_per_ah

            # ✅ [FIX] 辅机成本统一用 kWh 价格
            aux_kw = float(info_ll.get("aux_load_kw", 0.0))
            self.hourly_aux_cost += aux_kw * (self.dt_low / 3600.0) * current_price_kwh

            current_temps = self.low_env.battery.temps
            self.hourly_max_temp = max(self.hourly_max_temp, float(np.max(current_temps)))

        # --- D. 每小时结算 ---
        m_coeff = 0.0
        capacity_penalty = 0.0
        is_hour_end = ((self.current_step + 1) % self.steps_per_hour) == 0
        duration_hours = 1.0

        if is_hour_end:
            actual_curve_kw = np.array(self.hourly_actual_curve, dtype=float)
            target_curve_kw = np.array(self.hourly_target_curve, dtype=float)

            actual_power_kw = float(np.mean(actual_curve_kw)) if len(actual_curve_kw) else 0.0

            metrics = self.market.calculate_metrics_from_series(
                actual_p=actual_curve_kw / 1000.0,
                instruct_p=target_curve_kw / 1000.0
            )
            k_value = self.market.calculate_sorting_performance_k(
                metrics["speed"], metrics["delay"], metrics["error"]
            )
            m_value = self.market.calculate_performance_m(
                metrics["speed"], metrics["delay"], metrics["error"]
            )

            self.last_k_value = k_value
            m_coeff = m_value if k_value >= self.k_threshold else 0.0

            # ✅ [FIX] 容量收益不再倒扣（归零 + 固定罚）
            pass_k_check = (k_value >= self.k_threshold_cap)
            pass_temp_check = (self.hourly_max_temp < self.limits["temp_max"])
            capacity_ok = pass_k_check and pass_temp_check
            capacity_availability = 1.0 if capacity_ok else 0.0

            base_cap_rev = self.market.calculate_capacity_revenue_split(
                installed_cap_mw=self.rated_power_kw / 1000.0,
                availability=1.0
            )
            capacity_penalty_ratio = float(self.market_cfg.get("capacity_penalty_ratio", 2.0))
            capacity_penalty = 0.0 if capacity_ok else base_cap_rev * capacity_penalty_ratio

            avg_hour_price_kwh = float(np.mean(self.hourly_price_list)) if self.hourly_price_list else 0.0
            avg_hour_price_mwh = avg_hour_price_kwh * 1000.0

            settlement = self.market.calculate_settlement(
                target_power_kw=float(np.mean(self.hourly_target_base)),
                actual_power_kw=actual_power_kw,
                price=avg_hour_price_mwh,
                is_reg_mode=(plan_reg_cap > 0.1),
                reg_capacity_kw=plan_reg_cap,
                accumulated_mileage_kw=self.hourly_mileage_kw,
                price_reg_mileage=self.current_reg_mileage_price,
                duration_hours=duration_hours,
                m_coeff=m_coeff,
                capacity_availability=capacity_availability,
                installed_capacity_mw=self.rated_power_kw / 1000.0,
            )

            revenue_energy = settlement["revenue_energy"]
            revenue_reg = settlement["revenue_regulation"]
            revenue_capacity = settlement["revenue_capacity"]
            penalty = settlement["penalty"]

            net_profit = (
                    revenue_energy
                    + revenue_reg
                    + revenue_capacity
                    - penalty
                    - capacity_penalty
                    - self.hourly_aging_cost
                    - self.hourly_aux_cost
            )

            violation_metrics = {
                "temp_violation": max(0.0, self.hourly_max_temp - self.limits["temp_max"]),
                "soc_violation": max(0.0, self.limits["soc_min"] - mean_soc)
                                 + max(0.0, mean_soc - self.limits["soc_max"]),
                "power_violation": 0.0
            }
            cmdp_penalty = float(self.constraint_manager.compute_penalty({"violation_metrics": violation_metrics}))

            reward_scaled = net_profit * self.reward_scaler
            reward_final = reward_scaled - cmdp_penalty

            if not self.is_eval:
                reward_final = self.reward_normalizer(reward_final)
                reward_final = np.clip(reward_final, -100.0, 100.0)

            self._reset_hourly_buffers()
        else:
            settlement = {
                "revenue_energy": 0.0,
                "revenue_regulation": 0.0,
                "revenue_capacity": 0.0,
                "penalty": 0.0,
                "base_reward": 0.0,
                "scaled_reward": 0.0
            }
            net_profit = 0.0
            cmdp_penalty = float(self.constraint_manager.compute_penalty({"violation_metrics": {}}))
            reward_final = -cmdp_penalty

        # --- E. 状态推进 ---
        self.current_step += 1
        terminated = (self.current_step >= self.steps_per_episode)
        truncated = False

        if terminated:
            self.constraint_manager.update_lambdas()

        obs = self._get_obs()

        info = {
            "profit/net": net_profit,
            "profit/revenue_energy": settlement.get("revenue_energy", 0.0),
            "profit/revenue_reg": settlement.get("revenue_regulation", 0.0),
            "profit/revenue_capacity": settlement.get("revenue_capacity", 0.0),
            "profit/penalty_dev": settlement.get("penalty", 0.0),
            "market/k_factor": self.last_k_value,
            "market/m_factor": m_coeff if is_hour_end else 0.0,
            "market/reg_mileage_price": self.current_reg_mileage_price,
            "market/cleared_reg_cap_kw": plan_reg_cap,
            "market/hourly_mileage_kw": self.hourly_mileage_kw,
            "cost/aging_real": self.hourly_aging_cost,
            "cost/aux": self.hourly_aux_cost,
            "cost/capacity_penalty": capacity_penalty if is_hour_end else 0.0,
            "state/mean_soc": mean_soc,
            "state/max_temp": self.hourly_max_temp,
            "da_plan_p": plan_p_base,
            "da_plan_reg": plan_reg_cap,
            "action/p_real": p_real,
            "action/p_plan": plan_p_base,
            "settlement_pending": not is_hour_end
        }

        return obs, float(reward_final), terminated, truncated, info

    # ==========================================
    # Internal Methods
    # ==========================================

    def _get_obs(self) -> np.ndarray:
        socs = self.low_env.battery.socs
        temps = self.low_env.battery.temps
        sohs = self.low_env.battery.sohs

        cluster_stats = np.array([
            np.min(socs), np.max(socs), np.mean(socs), np.std(socs),
            np.min(temps), np.max(temps), np.mean(temps), np.std(temps),
            np.min(sohs), np.max(sohs), np.mean(sohs), np.std(sohs),
        ])

        soe = self.low_env.battery.get_total_energy() / self.capacity_nominal
        sys_state = np.array([soe, 0.0, 1.0])

        if self.current_price_series is not None and len(self.current_price_series) > 0:
            idx = min(self.current_step, len(self.current_price_series) - 1)
            curr_price = self.current_price_series[idx]
            future_prices = np.zeros(96)
            remain = len(self.current_price_series) - idx
            valid = min(remain, 96)
            future_prices[:valid] = self.current_price_series[idx:idx + valid]
        else:
            curr_price = 0.5
            future_prices = np.zeros(96)

        if self.current_da_plan is not None:
            t = min(self.current_step, len(self.current_da_plan["p_base"]) - 1)
            cur_plan_p = self.current_da_plan["p_base"][t]
            cur_plan_reg = self.current_da_plan["c_reg"][t]
        else:
            cur_plan_p = 0.0
            cur_plan_reg = 0.0

        plan_state = np.array([cur_plan_p, cur_plan_reg])

        obs = np.concatenate([
            cluster_stats,
            sys_state,
            [curr_price, 0.0],
            future_prices,
            np.zeros(6),
            plan_state
        ])

        if len(obs) != self.obs_dim:
            obs = np.resize(obs, (self.obs_dim,))

        return np.clip(np.nan_to_num(obs), -1e6, 1e6).astype(np.float32)

    def inject_lambdas(self, lambda_dict: Dict[str, float]):
        self.injected_lambdas = lambda_dict

    def set_power_profile_chunk(self, chunk: np.ndarray):
        self.suggested_power_chunk = chunk

    def update_obs_with_current_lambdas(self, obs: np.ndarray) -> np.ndarray:
        return obs
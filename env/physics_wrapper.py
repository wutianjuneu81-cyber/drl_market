import gymnasium as gym
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Literal, List

# [FIX] 引用正确的 MarketEngine 类
try:
    from DRL_market.market.mechanism import MarketEngine
except ImportError:
    from market.mechanism import MarketEngine


class RunningStat:
    """Helper for Adaptive Normalization with Warm Start"""

    def __init__(self, maxlen=1000, warm_start_vals=None):
        self.history = deque(maxlen=maxlen)
        if warm_start_vals:
            self.history.extend(warm_start_vals)
        else:
            self.history.extend([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    def push(self, val):
        self.history.append(val)

    def get_percentile(self, p=95):
        if not self.history: return 1.0
        return np.percentile(list(self.history), p)


class MarketPhysicsKernel(gym.Wrapper):
    """
    MHC Physics Kernel (Scientifically Rigorous Version)
    [Updates]:
    1. Full Cost Accounting: Energy, Reg, Penalty, Aging (Real), HVAC.
    2. Zero-Lag Observation.
    3. [New] Enhanced Logging for "Price-Life" Narrative Visualization.
    4. [Fix] Adapted to new MarketEngine interface.
    """

    def __init__(self,
                 env: gym.Env,
                 market_config: Any = None,  # 可能是 MarketConfig 对象或 Dict
                 obs_mode: Literal["augment", "transparent"] = "augment",
                 step_interval_seconds: int = 900,
                 scheduler: Any = None):
        super().__init__(env)

        # --- [FIX: 适配 MarketEngine 初始化] ---
        # 新的 MarketEngine 期望一个 {'market': ...} 的字典
        # 我们尝试从 env.unwrapped 获取全局配置，或者使用传入的 market_config

        full_cfg = {}

        # 1. 优先尝试从 wrapped env 获取完整配置
        if hasattr(env.unwrapped, 'cfg') and isinstance(env.unwrapped.cfg, dict):
            full_cfg = env.unwrapped.cfg

        # 2. 如果传入了特定的 market_config (覆盖)
        if market_config is not None:
            # 如果是对象 (MarketConfig)，转为字典
            if hasattr(market_config, '__dict__'):
                mkt_dict = market_config.__dict__
            elif isinstance(market_config, dict):
                mkt_dict = market_config
            else:
                mkt_dict = {}

            # 覆盖或创建 'market' 键
            if 'market' not in full_cfg:
                full_cfg['market'] = {}
            full_cfg['market'].update(mkt_dict)

        # 实例化引擎
        self.market_engine = MarketEngine(full_cfg)

        self.obs_mode = obs_mode
        self.market_interval_s = step_interval_seconds
        self.scheduler = scheduler

        if hasattr(self.env.unwrapped, "set_training_mode"):
            self.env.unwrapped.set_training_mode(False)

        self.da_schedule = None
        self.agc_base_signal = None
        self.real_time_prices = None

        self.accumulated_market_mileage = 0.0
        self.accumulated_physical_mileage = 0.0
        self.current_market_idx = 0
        self.current_agc_val = 0.0

        self.lambda_risk_stat = RunningStat()
        self.lambda_aging_stat = RunningStat()

        self._last_info_cache = None
        self.last_iro_step = -1
        self.iro_interval_steps = 4

        self._prev_day_soh = 1.0
        self._daily_throughput_accum = 0.0
        self._daily_aging_cost_accum = 0.0

    def set_daily_plan(self, plan: Dict[str, np.ndarray], agc_signal: np.ndarray, prices: Dict[str, np.ndarray]):
        self.da_schedule = plan
        self.agc_base_signal = agc_signal
        self.real_time_prices = prices
        self._reset_accumulators()

    def _reset_accumulators(self):
        self.current_market_idx = 0
        self.accumulated_market_mileage = 0.0
        self.current_agc_val = 0.0
        self.last_iro_step = -1

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_info_cache = info
        self._reset_accumulators()

        if options and options.get('reset_physics'):
            self.accumulated_physical_mileage = 0.0
            self._prev_day_soh = 1.0
            self._daily_throughput_accum = 0.0
            self._daily_aging_cost_accum = 0.0
        else:
            self.accumulated_physical_mileage = 0.0
            if 'aging_info' in info:
                self._prev_day_soh = info['aging_info'].get('soh_estimate', 1.0)

        if self.da_schedule is not None:
            self._run_iro_correction()
            self._inject_state()
            if hasattr(self.env.unwrapped, "update_obs_with_current_lambdas"):
                obs = self.env.unwrapped.update_obs_with_current_lambdas(obs)

        return obs, info

    def _map_shadow_price_adaptive(self, raw: float, stat_tracker: RunningStat, update_stats: bool = True) -> float:
        val_log = np.log1p(max(0.0, raw))
        if self.obs_mode != 'transparent' and update_stats:
            stat_tracker.push(val_log)

        scale = stat_tracker.get_percentile(95)
        if scale < 1e-6: scale = 1.0
        return float(np.tanh((val_log / scale) * 2.0))

    def _generate_agc_trajectory(self, base_agc: float, c_reg: float, steps: int) -> np.ndarray:
        traj = []
        val = self.current_agc_val
        integral_error = 0.0

        for _ in range(steps):
            noise = np.random.normal(0, 0.08)
            val = 0.95 * val + 0.05 * base_agc + noise
            correction = -0.01 * integral_error
            correction = np.clip(correction, -0.05, 0.05)
            val += correction
            val = np.clip(val, -1.0, 1.0)
            traj.append(val)
            integral_error += val

        self.current_agc_val = val
        return np.array(traj)

    def _run_iro_correction(self):
        if self.scheduler is None or self.da_schedule is None:
            return

        current_soc = 0.5
        if self._last_info_cache and 'SoC' in self._last_info_cache:
            current_soc = np.mean(self._last_info_cache['SoC'])
        elif hasattr(self.env.unwrapped, 'battery'):
            current_soc = np.mean(self.env.unwrapped.battery.get_mean_soc())

        t_idx = self.current_market_idx
        current_soh = 1.0
        current_rint = 1.0
        if self._last_info_cache and 'aging_info' in self._last_info_cache:
            current_soh = self._last_info_cache['aging_info'].get('soh_estimate', 1.0)
            current_rint = self._last_info_cache['aging_info'].get('rint_factor', 1.0)

        result = self.scheduler.solve_rolling(
            current_step=t_idx,
            current_soc=current_soc,
            original_plan=self.da_schedule,
            current_soh=current_soh,
            current_rint_factor=current_rint
        )

        if result.get('status') == 'optimal':
            new_p = result['p_base']
            length = len(new_p)
            end_idx = min(t_idx + length, len(self.da_schedule['p_base']))
            copy_len = end_idx - t_idx

            if copy_len > 0:
                self.da_schedule['p_base'][t_idx: t_idx + copy_len] = new_p[:copy_len]
                if 'soc_plan' in result:
                    new_soc = result['soc_plan']
                    self.da_schedule['soc_plan'][t_idx: t_idx + copy_len] = new_soc[:copy_len]
                if 'c_reg' in result:
                    new_c_reg = result['c_reg']
                    self.da_schedule['c_reg'][t_idx: t_idx + copy_len] = new_c_reg[:copy_len]

    def _inject_state(self):
        if self.da_schedule is None: return

        if self.current_market_idx > 0 and \
                self.current_market_idx % self.iro_interval_steps == 0 and \
                self.current_market_idx != self.last_iro_step:
            self._run_iro_correction()
            self.last_iro_step = self.current_market_idx

        idx_max = len(self.da_schedule['p_base']) - 1
        t = min(self.current_market_idx, idx_max)

        p_base = self.da_schedule['p_base'][t]
        c_reg = self.da_schedule['c_reg'][t]

        soc_target = self.da_schedule['soc_plan'][t]
        if hasattr(self.env.unwrapped, "set_soc_target"):
            self.env.unwrapped.set_soc_target(soc_target)

        base_agc = self.agc_base_signal[t] if self.agc_base_signal is not None else 0.0
        low_steps = getattr(self.env.unwrapped, "interval", 900)

        agc_traj_norm = self._generate_agc_trajectory(base_agc, c_reg, low_steps)
        agc_delta_sum = np.sum(np.abs(np.diff(agc_traj_norm, prepend=self.current_agc_val)))
        # 这里记录的是归一化 agc 的里程，后面会乘 reg_cap
        self._current_step_mileage = agc_delta_sum  # Note: not used directly, calculated in step

        target_profile = p_base + c_reg * agc_traj_norm

        if hasattr(self.env.unwrapped, "set_power_profile_chunk"):
            self.env.unwrapped.set_power_profile_chunk(target_profile)
        elif hasattr(self.env.unwrapped, "target_power"):
            self.env.unwrapped.target_power = np.mean(target_profile)

        sched_status = self.da_schedule.get('status', 'optimal')
        is_reliable = (sched_status == 'optimal')

        l_risk = self.da_schedule['lambda_risk'][t]
        l_aging = self.da_schedule['lambda_aging'][t]

        norm_risk = self._map_shadow_price_adaptive(l_risk, self.lambda_risk_stat, update_stats=is_reliable)
        norm_aging = self._map_shadow_price_adaptive(l_aging, self.lambda_aging_stat, update_stats=is_reliable)

        lambda_dict = {"violation_rate": norm_risk, "delta_soh": norm_aging}
        if hasattr(self.env.unwrapped, "inject_lambdas"):
            self.env.unwrapped.inject_lambdas(lambda_dict)

    def _calc_response_time(self, target_series: np.ndarray, actual_series: np.ndarray) -> float:
        if len(target_series) < 10: return 0.0
        t_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-6)
        a_norm = (actual_series - np.mean(actual_series)) / (np.std(actual_series) + 1e-6)
        correlation = np.correlate(a_norm, t_norm, mode='full')
        lags = np.arange(-len(a_norm) + 1, len(a_norm))
        lag = lags[np.argmax(correlation)]
        return max(0.0, float(lag))

    def _update_scheduler_params(self):
        if not self.scheduler: return
        if self._daily_throughput_accum > 1.0:
            self.scheduler.update_aging_parameters(self._daily_aging_cost_accum, self._daily_throughput_accum)

        self._daily_throughput_accum = 0.0
        self._daily_aging_cost_accum = 0.0

    def step(self, action):
        # 1) 直接调用 HighLevelEnv（权威结算）
        obs, reward, term, trunc, info = self.env.step(action)
        self._last_info_cache = info

        # 2) 仅记录日志与注入，无结算逻辑
        # (保留 _inject_state 调度注入)
        if not (term or trunc):
            self._inject_state()
            if hasattr(self.env.unwrapped, "update_obs_with_current_lambdas"):
                obs = self.env.unwrapped.update_obs_with_current_lambdas(obs)

        return obs, reward, term, trunc, info
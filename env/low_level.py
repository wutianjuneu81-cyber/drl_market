import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

# 尝试导入，防止路径问题
try:
    from DRL_market.simulation.battery import BatteryPack
    from DRL_market.common.config import load_config
except ImportError:
    from simulation.battery import BatteryPack
    from common.config import load_config


class LowLevelEnv(gym.Env):
    """
    Low-Level Environment (Final Cleaned & Robust Version)
    物理层：接收功率指令 + 影子价格，输出物理状态。
    """

    def __init__(self, config_path: Any = "config.yaml"):
        """
        Args:
            config_path: 配置文件路径 (str) 或 配置对象 (dict/object)
        """
        super().__init__()

        # --- 1. 配置加载 ---
        if isinstance(config_path, str):
            self.cfg = load_config(config_path)
        else:
            self.cfg = config_path

        self.env_cfg = self.cfg.get('env') or self.cfg.get('environment')
        if self.env_cfg is None:
            raise ValueError("LowLevelEnv: Config missing 'env' or 'environment' section.")

        # --- 2. 参数提取 ---
        def get_val(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.num_clusters = get_val(self.env_cfg, 'n_clusters', 24)
        self.dt = get_val(self.env_cfg, 'low_level_step_seconds', 1.0)
        self.topology = get_val(self.env_cfg, 'topology', 'active_dcdc')
        self.dcdc_eff = 0.97
        self.max_current_a = 500.0
        self.min_voltage_sys = 100.0

        # --- 3. 状态与动作空间 ---
        # 动作: log_ratios (24维) -> Softmax -> 分配比例
        self.action_space = spaces.Box(
            low=-20.0, high=20.0,
            shape=(self.num_clusters,),
            dtype=np.float32
        )

        # 观测: [Base(10) + ShadowPrices(N)]
        self.base_obs_dim = 10
        self.total_obs_dim = self.base_obs_dim + self.num_clusters

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )

        # --- 4. 物理子模块 ---
        self.battery = BatteryPack(self.cfg)

        # 运行时状态
        self.current_step = 0
        self.prev_power_sign = 0
        self.aux_load_kw = 0.0
        self.current_shadow_prices = np.zeros(self.num_clusters, dtype=np.float32)
        self.prev_actual_power_kw = 0.0

    # [Compatibility] 为 Wrapper 提供 .config 属性别名
    @property
    def config(self):
        return self.cfg

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if options and options.get('reset_physics', True):
            self.battery.reset()

        self.current_step = 0
        self.prev_power_sign = 0
        self.current_shadow_prices = np.zeros(self.num_clusters, dtype=np.float32)

        # ✅ 新增：重置里程基准
        self.prev_actual_power_kw = 0.0

        return self._get_obs(0.0), {}

    def step(self, action: np.ndarray, power_request_kw: float, shadow_prices: np.ndarray) -> Tuple[
        np.ndarray, float, bool, bool, Dict]:
        # [Safety] 输入清洗
        action = np.nan_to_num(action, nan=0.0)
        power_request_kw = float(np.nan_to_num(power_request_kw))
        shadow_prices = np.nan_to_num(shadow_prices, nan=0.0)

        self.current_shadow_prices = shadow_prices

        # --- A. 动作解析 (Softmax) ---
        action_clipped = np.clip(action, -20, 20)
        exps = np.exp(action_clipped - np.max(action_clipped))
        ratios = exps / np.sum(exps)

        # --- B. 物理计算 ---
        voltages = self.battery.get_voltages()
        cluster_powers_req = power_request_kw * ratios
        actual_cluster_powers = np.zeros(self.num_clusters)
        cluster_currents = np.zeros(self.num_clusters)

        for i in range(self.num_clusters):
            p_req = cluster_powers_req[i]
            v_clus = max(0.1, voltages[i])

            # [硬约束] 低电压保护
            if v_clus < 2.5 * self.battery.cells_per_cluster and p_req > 0:
                p_req = 0.0

            # 考虑 DC/DC 效率
            if p_req > 0:
                p_cell_side = p_req / self.dcdc_eff
            else:
                p_cell_side = p_req * self.dcdc_eff

            # 计算电流 I = P / V
            i_cell = np.clip((p_cell_side * 1000.0) / v_clus, -self.max_current_a, self.max_current_a)
            cluster_currents[i] = i_cell

            # 反算实际功率
            p_actual_cell = (i_cell * v_clus) / 1000.0
            if p_actual_cell > 0:
                actual_cluster_powers[i] = p_actual_cell * self.dcdc_eff
            else:
                actual_cluster_powers[i] = p_actual_cell / self.dcdc_eff

        # --- C. 物理步进 ---
        phys_info = self.battery.step(cluster_currents, self.dt)

        # --- D. 奖励与成本计算 ---

        # [关键] 统一获取 aging_costs (现在是加权后的)
        # 从 BatteryPack 返回的字典中获取 'aging_costs' (其实就是 stress_weighted_loss)
        step_damages_weighted = phys_info.get('aging_costs', np.zeros(self.num_clusters))

        # 优化成本 = 加权损伤 * 影子价格
        aging_costs_optimization = step_damages_weighted * shadow_prices

        total_actual_power = np.sum(actual_cluster_powers)
        step_mileage_kw = abs(total_actual_power - self.prev_actual_power_kw)
        self.prev_actual_power_kw = total_actual_power
        power_error = abs(power_request_kw - total_actual_power)
        cost_penalty = power_error * (self.dt / 3600.0) * 10.0

        # 3. 切换损耗
        curr_sign = np.sign(total_actual_power)
        cost_switch = 0.0
        if curr_sign != 0 and self.prev_power_sign != 0 and curr_sign != self.prev_power_sign:
            cost_switch = 0.5
        self.prev_power_sign = curr_sign

        # 4. 辅助功耗
        self.aux_load_kw = 0.5 + 0.01 * abs(total_actual_power)

        # 总奖励 (基于优化路径)
        total_cost = np.sum(aging_costs_optimization) + cost_penalty + cost_switch
        reward = np.clip(-total_cost, -100.0, 0.0)

        # --- E. 观测与结束条件 ---
        obs = self._get_obs(power_request_kw)
        truncated = self.battery.check_safety_constraints()

        # 获取真实的物理损伤 (用于日志和评估)
        real_damages = phys_info.get('total_loss_ah', np.zeros(self.num_clusters))

        # [修复] 字典键唯一化，逻辑清晰
        info = {
            'actual_power_kw': total_actual_power,
            'aux_load_kw': self.aux_load_kw,
            'step_mileage_kw': step_mileage_kw,

            # [关键指标]
            'aging_cost_reward': float(np.sum(aging_costs_optimization)),  # 优化用的虚拟成本 (带影子价格)
            'aging_cost_real': float(np.sum(real_damages)),  # 真实的物理成本 (SOH下降原因)
            'aging_cost_sum': float(np.sum(aging_costs_optimization)),  # 兼容旧代码 Key

            'penalty_cost': cost_penalty,
            'switch_cost': cost_switch,
            'cluster_powers': actual_cluster_powers,
            'soe_total': self.battery.get_total_energy(),
            'soc_mean': self.battery.get_mean_soc()
        }

        return obs, float(reward), False, truncated, info

    def _get_obs(self, power_request: float) -> np.ndarray:
        """
        构造观测向量:
        [0:10] 基础物理状态
        [10:]  影子价格 (关键信号)
        """
        obs = np.zeros(self.total_obs_dim, dtype=np.float32)
        obs[0] = power_request
        obs[1] = self.battery.get_mean_soc()
        obs[2] = np.mean(self.battery.voltages)
        obs[3] = np.mean(self.battery.temps)
        obs[4] = np.mean(self.battery.sohs)

        # [CRITICAL] 注入 Shadow Prices
        obs[self.base_obs_dim: self.base_obs_dim + self.num_clusters] = self.current_shadow_prices
        return np.nan_to_num(obs)
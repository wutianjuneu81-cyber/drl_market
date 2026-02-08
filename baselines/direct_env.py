import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Optional

# 鲁棒导入
try:
    from DRL_market.simulation.battery import BatteryPack
    from DRL_market.common.config import load_config
    from DRL_market.simulation.data_loader import MarketDataLoader
except ImportError:
    from simulation.battery import BatteryPack
    from common.config import load_config
    from simulation.data_loader import MarketDataLoader


class DirectControlEnv(gym.Env):
    """
    [严谨版单层 SAC 训练环境]
    Direct Control Environment (Strict Physical Consistency)

    核心保证：
    1) 真实 AGC：MarketDataLoader (1s) 逐点读取
    2) 物理一致：P = U * I，电流限幅
    3) dt 对齐：来自 config.env.low_level_step_seconds
    4) 老化对齐：使用 total_loss_ah
    5) 观测归一化：硬截断与缩放
    """

    def __init__(self, config_or_path: Any = "config.yaml", episode_seconds: int = 3600):
        super().__init__()

        # --- 1. 配置加载 ---
        if isinstance(config_or_path, str):
            self.cfg = load_config(config_or_path)
        else:
            self.cfg = config_or_path

        self.market_cfg = self.cfg.get("market", {})
        self.env_cfg = self.cfg.get("env") or self.cfg.get("environment")

        # 关键参数读取
        self.rated_power_kw = float(self.market_cfg.get("rated_power_kw", 2580.0))
        self.rated_capacity_kwh = float(self.market_cfg.get("rated_capacity_kwh", 5000.0))

        # [Fix] dt 严格从配置读取
        self.dt = float(self.env_cfg.get("low_level_step_seconds", 1.0))

        # [Fix] 电流限幅（与 low_level.py 一致）
        self.max_current_a = float(self.env_cfg.get("max_current_a", 500.0))

        # [Optional] DC/DC 效率
        self.dcdc_eff = float(self.env_cfg.get("dcdc_efficiency", 0.97))

        # --- 2. 物理内核与数据 ---
        self.battery = BatteryPack(self.cfg)
        self.data_loader = MarketDataLoader(self.cfg)

        # Episode 长度（秒级）
        self.max_steps = int(np.ceil(episode_seconds / self.dt))

        # --- 3. 空间定义 ---
        # Action: [-1, 1] -> [-rated, +rated]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation (归一化)
        # [0] Target Power (norm)
        # [1] Mean SoC (centered)
        # [2] Max Temp (norm)
        # [3] Mean SOH (norm)
        # [4] Last Error (norm)
        self.obs_dim = 5
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)

        # 运行时状态
        self.current_agc_series = None
        self.current_price_series = None
        self.episode_idx = 0
        self.current_step = 0

        self.target_power_kw = 0.0
        self.current_soc = 0.5
        self.last_error_norm = 0.0
        self.current_base_bias = 0.0

        env_limits = self.cfg.get("env", {}).get("limits", {})
        cons_limits = self.cfg.get("constraints", {}).get("limits", {})
        power_max = float(env_limits.get("power_max", cons_limits.get("power_max", 1.2)))
        self.max_power_kw = self.rated_power_kw * power_max

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.battery.reset()
        self.data_loader.reset(seed=seed)

        # [Fix] 采样真实数据
        self.current_price_series, self.current_agc_series = self.data_loader.sample_day_data()

        # ✅ 修复：max_steps 与实际数据长度对齐，防止越界
        self.max_steps = min(self.max_steps, len(self.current_agc_series))

        data_len = len(self.current_agc_series)
        if data_len > self.max_steps:
            self.episode_idx = self.np_random.integers(0, data_len - self.max_steps)
        else:
            self.episode_idx = 0

        self.current_step = 0
        self.current_soc = self.battery.get_mean_soc()

        # ✅ 新增：随机基点偏置（模拟日前基点）
        self.current_base_bias = self.np_random.uniform(-0.5, 0.5) * self.rated_power_kw

        # 初始化目标
        self._update_target()
        self.last_error_norm = 0.0

        return self._get_obs(), {}

    def _update_target(self):
        idx = self.episode_idx + self.current_step
        agc_val = self.current_agc_series[idx] if idx < len(self.current_agc_series) else 0.0

        # ✅ Target = Base Bias + AGC
        raw_target = self.current_base_bias + (agc_val * self.rated_power_kw)

        # 物理截断
        self.target_power_kw = float(np.clip(raw_target, -self.rated_power_kw, self.rated_power_kw))

    def step(self, action: np.ndarray):
        # 1) 动作解析
        p_cmd_norm = float(np.clip(action[0], -1.0, 1.0))
        p_cmd_kw = p_cmd_norm * self.max_power_kw

        # ✅ SoC 硬熔断（对齐 HighLevelEnv）
        if self.current_soc > 0.98:
            p_cmd_kw = max(0.0, p_cmd_kw)
        elif self.current_soc < 0.02:
            p_cmd_kw = min(0.0, p_cmd_kw)

        # 2) 平均分配（单层劣势）
        num_clusters = self.battery.n_clusters
        voltages = self.battery.get_voltages()

        currents = []
        actual_cluster_powers = []

        p_req_per_cluster = p_cmd_kw / num_clusters
        currents = []
        actual_cluster_powers = []

        for i in range(num_clusters):
            v = max(0.1, float(voltages[i]))

            # ✅ 低电压放电保护（对齐 low_level.py）
            if v < 2.5 * self.battery.cells_per_cluster and p_req_per_cluster > 0:
                p_req = 0.0
            else:
                p_req = p_req_per_cluster

            # ✅ DC/DC 补偿（对齐 low_level.py）
            if p_req > 0:
                p_cell_side = p_req / self.dcdc_eff
            else:
                p_cell_side = p_req * self.dcdc_eff

            curr = (p_cell_side * 1000.0) / v
            curr = float(np.clip(curr, -self.max_current_a, self.max_current_a))
            currents.append(curr)

            p_cell = (curr * v) / 1000.0
            if p_cell >= 0:
                p_actual = p_cell * self.dcdc_eff
            else:
                p_actual = p_cell / max(self.dcdc_eff, 1e-6)
            actual_cluster_powers.append(p_actual)

        # 3) 物理步进
        phys_info = self.battery.step(np.array(currents), self.dt)

        # 实际功率
        actual_power_kw = float(np.sum(actual_cluster_powers))

        # 4) 状态更新
        self.current_soc = self.battery.get_mean_soc()

        # 5) 奖励计算
        err_kw = abs(actual_power_kw - self.target_power_kw)
        self.last_error_norm = err_kw / max(self.rated_power_kw, 1e-6)
        r_track = - (err_kw / max(self.rated_power_kw, 1e-6)) * 5.0

        loss_ah_vec = phys_info.get("total_loss_ah", np.zeros(num_clusters))
        total_loss_ah = float(np.sum(loss_ah_vec))
        r_aging = - total_loss_ah * 20000.0

        r_soc = 0.0
        if self.current_soc < 0.05 or self.current_soc > 0.95:
            r_soc = -10.0

        reward = r_track + r_aging + r_soc

        # 6) 下一步
        self.current_step += 1
        self._update_target()

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "p_target": self.target_power_kw,
            "p_actual": actual_power_kw,
            "loss_ah": total_loss_ah,
            "soc": self.current_soc,
            "reward/track": r_track,
            "reward/aging": r_aging
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Target Power
        obs_target = self.target_power_kw / max(self.rated_power_kw, 1e-6)

        # SoC (centered)
        obs_soc = (self.current_soc - 0.5) * 2.0

        # Max Temp -> normalized [25,45] -> [0,1]
        max_t = float(np.max(self.battery.temps))
        obs_temp = (max_t - 25.0) / 20.0
        obs_temp = float(np.clip(obs_temp, 0.0, 2.0))

        # SOH -> [0.8,1.0] -> approx [-1,1]
        mean_soh = float(np.mean(self.battery.sohs))
        obs_soh = (mean_soh - 0.9) * 10.0

        # Last Error
        obs_err = float(np.clip(self.last_error_norm, -1.0, 1.0))

        obs = np.array([obs_target, obs_soc, obs_temp, obs_soh, obs_err], dtype=np.float32)
        return obs
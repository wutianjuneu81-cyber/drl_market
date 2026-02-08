import numpy as np
import os
from typing import Tuple, Optional, Dict, Any, Union


class MarketDataLoader:
    """
    Market Data Loader (Enhanced for Two-Stage Architecture)

    [Modifications]:
    1. Added `get_forecast_with_noise`: Generates realistic day-ahead forecasts
       by injecting noise/bias into actual data.
    2. Robust Path Loading: Handles various file path structures.
    3. Synthetic Fallback: Generates synthetic data if files are missing.
    """

    def __init__(self, full_config: Any, is_eval: bool = False):
        # 1. 鲁棒配置读取
        def get_cfg_val(obj, key, default):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        self.data_cfg = get_cfg_val(full_config, 'data', {})
        self.env_cfg = get_cfg_val(full_config, 'env', None)
        if not self.env_cfg:
            self.env_cfg = get_cfg_val(full_config, 'environment', {})

        self.is_eval = is_eval

        # 2. 路径搜索策略
        possible_paths = []
        # A. 优先从配置读取
        config_path = get_cfg_val(self.env_cfg, 'external_power_profile_path', None)
        if config_path:
            possible_paths.append(config_path)
            possible_paths.append(os.path.join("DRL_market", config_path))

        # B. 默认路径
        possible_paths.extend([
            "data/guanghe_30days_scaled.npz",
            "DRL_market/data/guanghe_30days_scaled.npz",
            "../data/guanghe_30days_scaled.npz"
        ])

        self.data_path = None
        for p in possible_paths:
            p = p.replace("\\", "/")
            if os.path.exists(p):
                self.data_path = p
                break

        if self.data_path:
            print(f"[{'Eval' if is_eval else 'Train'}] Loading Data from: {self.data_path}")
        else:
            print(f"[{'Eval' if is_eval else 'Train'}] Warning: Data file not found. Using SYNTHETIC generator.")

        # 3. 加载数据
        self.high_res_prices, self.high_res_agc = self._load_data()

        # 4. 数据预处理 (降采样)
        # Low Level: 1s
        # High Level: 15min = 900s
        self.steps_per_day_high = 96
        self.steps_per_day_low = 86400
        self.downsample_rate = 900

        n_samples = len(self.high_res_prices)
        n_valid = (n_samples // self.downsample_rate) * self.downsample_rate

        if n_valid == 0:
            print("Error: Data too short. Using fallback synthetic.")
            self.prices_15min = np.ones(96, dtype=np.float32) * 0.5
            self.agc_1s = np.zeros(86400, dtype=np.float32)
            self.total_days = 1
        else:
            # 截断并降采样 Price (1s -> 15min)
            prices_truncated = self.high_res_prices[:n_valid]
            # Reshape to (N_steps, 900) then mean
            self.prices_15min = prices_truncated.reshape(-1, self.downsample_rate).mean(axis=1)

            # AGC 保持 1s 分辨率
            self.agc_1s = self.high_res_agc[:n_valid]

            self.total_days = len(self.prices_15min) // self.steps_per_day_high

        # 5. 数据集划分
        split_ratios = get_cfg_val(self.data_cfg, 'split_ratio', [0.7, 0.15, 0.15])
        train_end = int(self.total_days * split_ratios[0])

        if not is_eval:
            # Training Set
            self.day_indices = np.arange(0, max(1, train_end))
        else:
            # Validation/Test Set
            # If explicit test set is needed, can use split_ratios[1]
            start = train_end if train_end < self.total_days else 0
            end = self.total_days
            self.day_indices = np.arange(start, end)
            if len(self.day_indices) == 0:
                self.day_indices = np.arange(0, 1)  # Fallback

        self.rng = np.random.default_rng(42)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load NPZ or generate synthetic"""
        if self.data_path:
            try:
                raw = np.load(self.data_path)
                # Flexible key access
                prices = raw['price'] if 'price' in raw else raw['arr_0']
                agc = raw['agc'] if 'agc' in raw else (raw['arr_1'] if 'arr_1' in raw else np.zeros_like(prices))
                return prices.astype(np.float32), agc.astype(np.float32)
            except Exception as e:
                print(f"NPZ Load Failed: {e}")

        # Synthetic Data Generator
        # 30 days of data
        total_seconds = 30 * 24 * 3600
        t = np.linspace(0, 30 * 24, total_seconds)
        # Price: Daily cycle + Random walk
        base_price = 0.6 + 0.4 * np.sin(2 * np.pi * t / 24.0 - np.pi / 2)
        noise = np.random.normal(0, 0.05, total_seconds)
        prices = np.clip(base_price + noise, 0.1, 1.5).astype(np.float32)

        # AGC: Fast sine waves + noise
        agc = (0.3 * np.sin(2 * np.pi * t * 60) + 0.1 * np.random.randn(total_seconds)).astype(np.float32)
        agc = np.clip(agc, -1.0, 1.0)

        return prices, agc

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def sample_day_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random day from the split.
        Returns: (price_15min_actual, agc_1s_actual)
        """
        if len(self.day_indices) == 0:
            day_idx = 0
        else:
            day_idx = self.rng.choice(self.day_indices)

        # Slice Price (15min)
        p_start = day_idx * self.steps_per_day_high
        p_end = p_start + self.steps_per_day_high
        price_series = self.prices_15min[p_start:p_end].copy()

        # Slice AGC (1s)
        a_start = day_idx * self.steps_per_day_low
        a_end = a_start + self.steps_per_day_low
        agc_series = self.agc_1s[a_start:a_end].copy()

        # Safety padding
        if len(price_series) < 96:
            price_series = np.pad(price_series, (0, 96 - len(price_series)), mode='edge')
        if len(agc_series) < 86400:
            agc_series = np.pad(agc_series, (0, 86400 - len(agc_series)), mode='edge')

        return price_series, agc_series

    def get_forecast_with_noise(self, actual_prices: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        [NEW] Generate Day-Ahead Forecast based on Actual Data

        Args:
            actual_prices: The ground truth prices (96 steps)
            noise_level: Std dev of noise (relative to price magnitude or absolute)

        Returns:
            forecast_prices: Prices with simulated prediction error
        """
        n = len(actual_prices)

        # 1. Random Gaussian Noise (Measurement error)
        jitter = self.rng.normal(0, noise_level, n)

        # 2. Systematic Bias (Forecast tendency, e.g., overestimating peak)
        # Randomly choose a bias type for the day
        bias_type = self.rng.choice(['none', 'shift', 'scale'])

        forecast = actual_prices.copy()

        if bias_type == 'shift':
            # Temporal shift (e.g., peak predicted 15 min early)
            shift = self.rng.choice([-1, 1])
            forecast = np.roll(forecast, shift)
            # Fix edge artifacts
            forecast[0] = forecast[1]
            forecast[-1] = forecast[-2]

        elif bias_type == 'scale':
            # Magnitude error (e.g., predicted 10% higher)
            scale = self.rng.uniform(0.9, 1.1)
            forecast = forecast * scale

        forecast = forecast + jitter

        # Ensure non-negative
        return np.maximum(forecast, 0.0).astype(np.float32)
import logging
import argparse
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

# Ensure Root Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# [Correct Imports]
from DRL_market.env.low_level import LowLevelEnv
# 引入我们修复后的混合配置类
from DRL_market.common.config import load_config, FinalRLBMSConfig
from DRL_market.common.paths import RunPaths
from DRL_market.common.logging_utils import init_root_logger, get_logger
from DRL_market.models.policy import BatteryFeatureExtractor, BatterySACPolicy

logger = logging.getLogger("core.pretrainer")

VALID_SAC_KEYS = {
    "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau", "gamma", "train_freq",
    "gradient_steps", "action_noise", "ent_coef", "target_update_interval", "target_entropy",
    "use_sde", "sde_sample_freq", "use_sde_at_warmup", "device", "verbose", "tensorboard_log",
}


class DataLoaderWrapper(gym.Wrapper):
    """
    [适配器 Wrapper]
    负责为 LowLevelEnv 注入数据 (Power Profile) 并随机注入 Shadow Prices (Domain Randomization)。
    """

    def __init__(self, env, profile_path, config=None, active=True, noise_std=0.5):
        """
        Args:
            config: FinalRLBMSConfig 对象 (或字典)，包含 hierarchy.lambda_max 等参数
        """
        super().__init__(env)
        self.active = active
        self.noise_std = noise_std
        self.data = None

        # [Fix] 保存配置。由于 FinalRLBMSConfig 支持字典操作，后续访问非常安全。
        self.config = config if config is not None else {}

        if profile_path and active:
            try:
                loaded = np.load(profile_path)
                # 鲁棒加载 .npz 或 .npy
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    if 'power' in loaded:
                        self.data = loaded['power']
                    elif 'arr_0' in loaded:
                        self.data = loaded['arr_0']
                else:
                    self.data = loaded
            except Exception as e:
                logger.warning(f"DataLoaderWrapper: Failed to load profile from {profile_path}: {e}")
                self.data = None

        # Fallback 数据
        if self.data is None:
            self.data = np.random.uniform(-20, 20, 100000).astype(np.float32)

        self.idx = 0
        # 此时 env.unwrapped 肯定是 LowLevelEnv，它有 num_clusters 属性
        self.n_clusters = getattr(self.env.unwrapped, 'num_clusters', 24)
        self.current_target = 0.0

    def reset(self, **kwargs):
        if 'options' not in kwargs: kwargs['options'] = {}
        kwargs['options']['reset_physics'] = True

        # 随机选择数据起点
        if self.active and len(self.data) > 86400:
            self.idx = np.random.randint(0, len(self.data) - 86400)
        else:
            self.idx = 0

        if self.active:
            target = float(self.data[self.idx])
            if self.noise_std > 0: target += np.random.normal(0, self.noise_std)
            self.current_target = target
        else:
            self.current_target = 0.0

        return self.env.reset(**kwargs)

    def step(self, action):
        # 1. 更新功率指令
        if self.active:
            self.idx = (self.idx + 1) % len(self.data)
            target = float(self.data[self.idx])
            if self.noise_std > 0: target += np.random.normal(0, self.noise_std)
            self.current_target = target

        power_req = self.current_target

        # 2. 生成随机 Shadow Price (关键预训练逻辑)
        # 使用 .get() 安全获取层级参数
        hierarchy = self.config.get('hierarchy', {})
        # 如果 hierarchy 也是 ConfigNode，它同样支持 .get()
        lambda_max = hierarchy.get('lambda_max', 5.0) if hasattr(hierarchy, 'get') else 5.0

        # 注入随机价格，训练 Agent 对价格的敏感度
        shadow_prices = np.random.uniform(0.0, lambda_max, size=self.n_clusters).astype(np.float32)

        # 3. 调用底层环境
        return self.env.step(action, power_request_kw=power_req, shadow_prices=shadow_prices)


class LowLevelPretrainer:
    def __init__(self, config_obj, run_paths):
        # 这里的 config_obj 是修复后的 FinalRLBMSConfig (Hybrid)
        self.config = config_obj
        self.run_paths = run_paths
        self.models_dir = run_paths.models_dir
        self.low_model = None

        # 检查是否使用结构化策略
        hierarchy = self.config.get('hierarchy', {})
        self.low_level_structured_policy = hierarchy.get("low_level_structured_policy", True)

    def _filter_sac_kwargs(self, raw: dict, overrides: dict = None):
        sac = {k: v for k, v in raw.items() if k in VALID_SAC_KEYS}
        if overrides: sac.update(overrides)
        return sac

    def _make_env(self):
        """
        [多核工厂函数]
        SubprocVecEnv 会序列化这个函数到子进程执行。
        """
        # 1. 获取环境配置和数据路径
        # 由于 FinalRLBMSConfig 同时支持 env 和 environment 键，且支持 .get()，这里非常安全
        env_cfg = self.config.get('env')
        if not env_cfg: env_cfg = self.config.get('environment', {})

        profile_path = env_cfg.get('external_power_profile_path')

        # 2. 实例化基础环境
        # 直接传入 Hybrid Config 对象
        base_env = LowLevelEnv(self.config)

        # 3. 实例化 Wrapper
        # 显式传入 config=self.config
        return DataLoaderWrapper(base_env, profile_path, config=self.config, active=True, noise_std=0.5)

    def train(self, steps: int = None):
        # 获取总步数
        env_cfg = self.config.get('env', {})
        total_ts = env_cfg.get('total_timesteps', 200000)
        if steps is None: steps = min(120000, total_ts // 2)

        # 16核并行
        n_envs = 16
        logger.info(f"[LowLevel] Pretraining: {steps} steps using {n_envs} Parallel Envs...")

        env = SubprocVecEnv([self._make_env for _ in range(n_envs)])

        # 准备 SAC 参数
        train_cfg = self.config.get('training', {})
        # 兼容 ConfigNode 转字典
        raw_kwargs = train_cfg.get('default_params', {})
        if hasattr(raw_kwargs, 'to_dict'): raw_kwargs = raw_kwargs.to_dict()

        sac_low_overrides = {
            "buffer_size": 200000,
            "batch_size": 256,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "ent_coef": "auto",
            "tau": 0.005,
            "learning_starts": 1000,
            "target_update_interval": 1
        }
        sac_kwargs = self._filter_sac_kwargs(raw_kwargs, sac_low_overrides)

        # 选择策略网络
        if self.low_level_structured_policy:
            logger.info("✅ Using CUSTOM Structured Policy (BatterySACPolicy)")
            policy_cls = BatterySACPolicy
            policy_kwargs = {
                "features_extractor_class": BatteryFeatureExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 128,
                    "use_layer_norm": True,
                    "dropout_rate": 0.1,
                    "num_cells": 24
                }
            }
        else:
            logger.info("ℹ️ Configuration requested Standard MlpPolicy.")
            policy_cls = "MlpPolicy"
            policy_kwargs = {}

        # 开始训练
        self.low_model = SAC(policy_cls, env, policy_kwargs=policy_kwargs, **sac_kwargs)
        self.low_model.learn(total_timesteps=steps)
        self._save_model()

        env.close()
        logger.info(f"[LowLevel] Pretraining complete.")
        return self.low_model

    def _save_model(self):
        local_low = self.run_paths.low_level_model_path()
        global_low = self.run_paths.global_low_level_model_path()

        targets = [local_low, global_low]
        for tgt in targets:
            try:
                tgt.parent.mkdir(parents=True, exist_ok=True)
                self.low_model.save(tgt)
                logger.info(f"✅ Saved Low-Level Model to: {tgt}")
            except Exception as e:
                logger.warning(f"[LowLevelSaveError] Failed to save to {tgt}: {e}")


if __name__ == "__main__":
    init_root_logger()
    logger = get_logger("PretrainMain")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()

    # 使用修复后的配置类加载
    cfg_path = PROJECT_ROOT / args.config
    cfg = FinalRLBMSConfig(str(cfg_path))

    # 路径管理
    run_paths = RunPaths(str(PROJECT_ROOT), seed=42, strategy="low_level", category="pretrain").ensure()

    target = run_paths.global_low_level_model_path()
    if target.exists():
        logger.info(
            f"⏭️  Low-level model already exists at {target}. Skipping (Use --force logic in scripts if needed).")
    else:
        trainer = LowLevelPretrainer(cfg, run_paths)
        trainer.train(steps=args.steps)
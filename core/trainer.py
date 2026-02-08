import os
import torch
import numpy as np
import yaml
import pickle
from typing import Optional, Dict, Callable
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from DRL_market.common.config import load_config
from DRL_market.common.logging_utils import get_logger
from DRL_market.env.high_level import HighLevelEnv
from DRL_market.common.paths import RunPaths  # [新增] 引入路径管理

# 鲁棒导入
try:
    from DRL_market.core.math_utils.stats import RewardNormalizer
except ImportError:
    try:
        from DRL_market.core.math.stats import RewardNormalizer
    except ImportError:
        from core.math.stats import RewardNormalizer

from DRL_market.core.constraints.manager import ConstraintManager
from DRL_market.models.policy import CustomMarketExtractor


# [修改] 并行环境工厂函数：支持传入 Low-Level 模型路径
def make_parallel_env(cfg, rank: int, seed: int = 0, low_level_path: str = None) -> Callable:
    """
    用于 SubprocVecEnv 的工厂函数。
    为每个子进程设置独立的随机种子，并在进程内部加载 Low-Level 模型。
    """

    def _init():
        # [关键修复] 在子进程内部加载模型，避免 Pickle 错误
        low_model = None
        if low_level_path:
            try:
                # 只需要 CPU 推理，map_location='cpu' 至关重要
                low_model = SAC.load(low_level_path, device='cpu')
            except Exception as e:
                print(f"⚠️ [Worker {rank}] Failed to load low-level model from {low_level_path}: {e}")

        # HighLevelEnv 接收 loaded model
        env = HighLevelEnv(cfg, is_eval=False, low_model=low_model)
        env.reset(seed=seed + rank)
        return env

    return _init


class SACTrainer:
    """
    SAC Trainer (Parallelized for Speed)
    """

    def __init__(self, config_path: str, output_dir: str):
        self.cfg = load_config(config_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = get_logger("Trainer", os.path.join(output_dir, "train.log"))

        # 保存本次运行的配置副本
        with open(os.path.join(output_dir, "config_final.yaml"), "w") as f:
            yaml.dump(self.cfg, f)

        # --- [CRITICAL] 并行环境设置 ---
        self.n_envs = self.cfg['training'].get('n_envs', 16)
        seed = self.cfg['training'].get('seed', 42)

        # [新增] 自动寻找预训练好的 Low-Level 模型
        run_paths = RunPaths(os.path.dirname(config_path), strategy="low_level")
        low_model_path = run_paths.global_low_level_model_path()

        # 检查模型是否存在
        final_low_path_str = None
        if low_model_path.exists():
            self.logger.info(f"✅ Found Pretrained Low-Level Model: {low_model_path}")
            final_low_path_str = str(low_model_path)
        else:
            self.logger.warning(
                f"⚠️ Low-level model NOT found at {low_model_path}. High-level training will use fallback (Naive) logic!")

        self.logger.info(f"⚡ Initializing {self.n_envs} Parallel Environments (SubprocVecEnv)...")

        # [修改] 创建并行环境时传入路径
        self.env = SubprocVecEnv([
            make_parallel_env(self.cfg, i, seed, low_level_path=final_low_path_str)
            for i in range(self.n_envs)
        ])

        self.env = VecMonitor(self.env, filename=os.path.join(output_dir, "monitor.csv"))

        # 评估环境保持单进程
        # 注意：评估环境也需要加载 Low-Level 模型，这里为了简单直接实例化
        # 如果内存允许，也可以加载一次模型对象传进去
        eval_low_model = None
        if final_low_path_str:
            eval_low_model = SAC.load(final_low_path_str, device='cpu')

        self.eval_env = HighLevelEnv(self.cfg, is_eval=True, low_model=eval_low_model)

        # 辅助模块
        self.cmdp_manager = ConstraintManager(self.cfg)
        self.reward_norm = RewardNormalizer()

        # 初始化 SAC 模型
        agent_cfg = self.cfg['agent']
        self.model = SAC(
            "MlpPolicy",
            self.env,
            gamma=agent_cfg['gamma'],
            learning_rate=agent_cfg['learning_rate'],
            buffer_size=agent_cfg['buffer_size'],
            learning_starts=agent_cfg['learning_starts'],
            batch_size=agent_cfg['batch_size'],
            tau=agent_cfg['tau'],
            ent_coef=agent_cfg['ent_coef'],
            train_freq=agent_cfg['train_freq'],
            gradient_steps=agent_cfg['gradient_steps'],
            policy_kwargs=dict(
                net_arch=agent_cfg['hidden_sizes'],
                features_extractor_class=CustomMarketExtractor,
                features_extractor_kwargs=dict(features_dim=256)
            ),
            verbose=1,
            tensorboard_log=os.path.join(output_dir, "tb_logs"),
            seed=seed
        )

        self.best_eval_score = -np.inf
        self.rolling_score_window = []

    def warmup_buffer(self):
        """
        [向量化 Warm-up]

        [FIX]:
        之前使用了 for 循环逐个添加环境数据，导致了 'cannot reshape array' 错误。
        现在的实现完全符合 VecEnv 的 Batch 操作规范。
        """
        total_warmup_steps = self.cfg['agent'].get('learning_starts', 10000)
        # 这里的 step 是指 environment steps，因为是并行的，所以循环次数要除以 n_envs
        loops = int(np.ceil(total_warmup_steps / self.n_envs))

        self.logger.info(f"🔥 Starting Parallel Warm-up: {loops} loops x {self.n_envs} envs...")

        obs = self.env.reset()

        for _ in range(loops):
            # 构造 Batch Action (16, ActionDim)
            actions = np.zeros((self.n_envs, self.env.action_space.shape[0]), dtype=np.float32)

            # 提取 SoC (假设 index 12 是 System SoC)
            sys_socs = obs[:, 12]

            p_idx = self.env.action_space.shape[0] - 3

            # 简单的启发式规则 (向量化操作)
            mask_high = sys_socs > 0.6
            mask_low = sys_socs < 0.4

            actions[mask_high, p_idx] = 0.5  # Discharge
            actions[mask_low, p_idx] = -0.5  # Charge

            actions[:, p_idx + 1] = -1.0  # 0 Regulation
            actions[:, p_idx + 2] = 1.0  # Max Slack

            # 环境交互 (Batch Step)
            next_obs, rewards, dones, infos = self.env.step(actions)

            # [关键修复] 向量化处理 Terminal Observation
            # VecEnv 会自动 Reset 结束的环境，真实的 next_obs 藏在 infos 里
            real_next_obs = next_obs.copy()
            for i, done in enumerate(dones):
                if done and infos[i].get("terminal_observation") is not None:
                    real_next_obs[i] = infos[i]["terminal_observation"]

            # [关键修复] 批量添加至 ReplayBuffer
            # SB3 的 add 方法当 n_envs > 1 时，期望输入 shape 为 (n_envs, ...)
            self.model.replay_buffer.add(
                obs,
                real_next_obs,
                actions,
                rewards,
                dones,
                infos
            )

            obs = next_obs

        self.logger.info("✅ Warm-up Complete.")

    def train(self):
        # 1. 执行 Warm-up
        self.warmup_buffer()

        # 2. 开始正式训练
        total_timesteps = self.cfg['training']['total_timesteps']
        eval_interval = self.cfg['training']['eval_interval']

        callback = CustomCallback(
            self,
            eval_env=self.eval_env,
            eval_freq=eval_interval
        )

        self.logger.info("🚀 Starting Main Training Loop (Multi-Core)...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=self.cfg['training']['log_interval']
        )

        self.logger.info("Training Finished.")
        self.save_checkpoint("final_model")

    def save_checkpoint(self, name: str):
        save_path = os.path.join(self.output_dir, name)
        self.model.save(f"{save_path}.zip")

        # [关键修复] 从子进程获取真实的 Normalizer 状态
        # self.env 是 VecEnv, get_attr 会返回一个列表 [norm_0, norm_1, ...]
        try:
            # 获取所有子环境的 Normalizer
            norm_list = self.env.get_attr("reward_normalizer")

            if norm_list and len(norm_list) > 0:
                # 策略 A: 直接取第一个 (因为它们是独立同分布的，且通常同步)
                # 策略 B: 聚合所有 (更严谨，但这里取第一个通常足够)
                real_norm = norm_list[0]

                # 将真实的统计量覆盖到主进程的 self.reward_norm
                # 这样下次 self.reward_norm 就是有数据的了
                self.reward_norm = real_norm

                # 打印日志确认
                self.logger.info(f"🔄 Synced Normalizer from SubprocEnv (Count={real_norm.rms.count:.1f})")
            else:
                self.logger.warning("⚠️ Failed to retrieve reward_normalizer from envs.")

        except Exception as e:
            self.logger.error(f"❌ Error syncing normalizer: {e}")

        # 保存
        norm_path = f"{save_path}_normalizer.pkl"
        with open(norm_path, "wb") as f:
            pickle.dump(self.reward_norm, f)  # 现在保存的是有数据的对象了

        cmdp_path = f"{save_path}_cmdp.pkl"
        with open(cmdp_path, "wb") as f:
            pickle.dump(self.cmdp_manager, f)

        self.logger.info(f"💾 Checkpoint saved: {name}")


class CustomCallback(BaseCallback):
    """
    SB3 回调：处理评估、课程更新和模型保存
    """

    def __init__(self, trainer_instance, eval_env, eval_freq: int = 10000):
        super().__init__(verbose=1)
        self.trainer = trainer_instance
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True

    def _run_evaluation(self):
        avg_reward = 0.0
        n_eval_episodes = 5

        for _ in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.trainer.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = self.eval_env.step(action)
                ep_reward += reward
                done = term or trunc
            avg_reward += ep_reward

        avg_reward /= n_eval_episodes

        window = self.trainer.rolling_score_window
        window.append(avg_reward)
        if len(window) > 5:
            window.pop(0)

        rolling_score = np.mean(window)

        self.trainer.logger.info(
            f"📈 Step {self.num_timesteps}: Eval Reward = {avg_reward:.2f}, Rolling = {rolling_score:.2f}")

        if rolling_score > self.trainer.best_eval_score:
            self.trainer.best_eval_score = rolling_score
            self.trainer.save_checkpoint("best_rolling_model")

        if self.num_timesteps % self.trainer.cfg['training']['save_interval'] == 0:
            self.trainer.save_checkpoint(f"checkpoint_{self.num_timesteps}")
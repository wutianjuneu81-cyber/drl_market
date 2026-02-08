import os
import argparse
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from DRL_market.common.config import load_config
from baselines.direct_env import DirectControlEnv


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def make_env(cfg, rank: int, seed: int = 0):
    def _init():
        env = DirectControlEnv(cfg)
        env.reset(seed=seed + rank)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train Direct SAC (Single-Layer)")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "logs" / "direct_sac"),
                        help="Tensorboard/log output directory")
    parser.add_argument("--model_dir", type=str, default=str(PROJECT_ROOT / "models"),
                        help="Output model directory")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # 并行环境
    env = SubprocVecEnv([make_env(cfg, i, args.seed) for i in range(args.n_envs)])
    env = VecMonitor(env, filename=os.path.join(args.log_dir, "monitor.csv"))

    # SAC 模型（单层）
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        ent_coef="auto",
        seed=args.seed,
    )

    # ✅ 统一 checkpoint 前缀
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=args.model_dir,
        name_prefix="baseline_direct_sac"
    )

    print("🚀 Training Direct SAC (Single-Layer)...")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # ✅ 统一最终模型命名
    final_path = os.path.join(args.model_dir, "baseline_direct_sac")
    model.save(final_path)
    print(f"✅ Training complete. Model saved at {final_path}.zip")


if __name__ == "__main__":
    main()
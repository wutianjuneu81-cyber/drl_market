#!/usr/bin/env python3
"""
MHC Architecture Smoke Test (Fixed)
功能：快速验证重构后的代码结构是否完整，Import 是否正确，环境能否启动。
不进行实质性训练，只跑通 1 个 Step。
"""

import sys
import os
import shutil
from pathlib import Path
import numpy as np

# 1. 路径修正：确保能导入根目录的包
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

print(f"🚀 [SmokeTest] Project Root: {PROJECT_ROOT}")


def test_imports():
    print("\n--- Step 1: Testing Imports ---")
    try:
        from common.config import FinalRLBMSConfig
        print("✅ common.config")
        from simulation.battery import Battery
        print("✅ simulation.battery")
        from env.low_level import LowLevelEnv
        print("✅ env.low_level")
        from env.high_level import HighLevelEnv
        print("✅ env.high_level")
        # [FIX] Class name updated to HRLTrainer
        from core.trainer import HRLTrainer
        print("✅ core.trainer (HRLTrainer)")
        from core.pretrainer import LowLevelPretrainer
        print("✅ core.pretrainer")
        from models.policy import BatterySACPolicy
        print("✅ models.policy")
        from market.scheduler import DayAheadScheduler
        print("✅ market.scheduler")
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        print("💡 提示：请检查 __init__.py 文件是否遗漏，或者 sys.path 是否正确。")
        sys.exit(1)


def test_config_and_data():
    print("\n--- Step 2: Config & Data ---")
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        # 尝试找旧名
        old_cfg = PROJECT_ROOT / "config_simplified.yml"
        if old_cfg.exists():
            print(f"⚠️ Found {old_cfg}, copying to config.yaml")
            shutil.copy(old_cfg, cfg_path)
        else:
            print(f"❌ Config not found at {cfg_path}")
            sys.exit(1)

    from common.config import FinalRLBMSConfig
    cfg = FinalRLBMSConfig(str(cfg_path))
    print("✅ Config loaded")

    # 检查数据文件，如果不存在则生成假数据，防止环境报错
    data_path = PROJECT_ROOT / cfg.environment.external_power_profile_path
    if not data_path.exists():
        print(f"⚠️ Data missing at {data_path}, generating dummy data...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        # 生成 .npz 格式的假数据 (匹配 process_data.py 的输出)
        dummy_power = np.random.uniform(-20, 20, 100000).astype(np.float32)
        dummy_price = np.random.uniform(0.2, 1.5, 100000).astype(np.float32)
        dummy_agc = np.random.uniform(-1, 1, 100000).astype(np.float32)

        np.savez(data_path, power=dummy_power, price=dummy_price, agc=dummy_agc)
        print(f"✅ Dummy data generated at {data_path}")

    return cfg


def test_environments(cfg):
    print("\n--- Step 3: Environment Initialization ---")
    from env.interface import GoalInterface
    from env.low_level import LowLevelEnv
    from env.high_level import HighLevelEnv
    from stable_baselines3 import SAC

    gi = GoalInterface()

    # Low Level
    try:
        # 构造 kwargs
        env_kwargs = cfg.get_env_kwargs()
        # 确保移除不支持的参数
        env_kwargs.pop("quiet_init", None)
        env_kwargs['subset'] = 'train'

        low_env = LowLevelEnv(env_kwargs, gi)
        obs, _ = low_env.reset()
        action = low_env.action_space.sample()
        _, _, _, _, _ = low_env.step(action)
        print("✅ LowLevelEnv: Reset & Step OK")
    except Exception as e:
        print(f"❌ LowLevelEnv Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Mock Low Level Model
    print("   Creating Mock Low Level Model...")
    low_model = SAC("MlpPolicy", low_env)

    # High Level
    try:
        def make_low():
            return LowLevelEnv(env_kwargs, gi)

        high_env = HighLevelEnv(
            make_low_env_fn=make_low,
            interval=5,  # Short interval for test
            goal_interface=gi,
            low_model=low_model,
            reward_weights=cfg.reward_high.weights.to_dict(),
            normalization_cfg=cfg.reward_high.normalization.to_dict(),
            hiro_cfg=cfg.hierarchy.hiro.to_dict(),
            training_mode=True
        )
        obs, _ = high_env.reset()
        action = high_env.action_space.sample()
        _, _, _, _, _ = high_env.step(action)
        print("✅ HighLevelEnv: Reset & Step OK")
    except Exception as e:
        print(f"❌ HighLevelEnv Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return low_model


def test_trainer(cfg, low_model):
    print("\n--- Step 4: Trainer Integration ---")
    from core.trainer import HRLTrainer
    from core.pretrainer import LowLevelPretrainer
    from common.paths import RunPaths

    # 使用临时路径
    run_paths = RunPaths(str(PROJECT_ROOT), seed=999, strategy="smoke_test", category="test").ensure()

    try:
        # 1. Test Low Level Pretrainer
        print("   [1/2] Testing LowLevelPretrainer...")
        pretrainer = LowLevelPretrainer(config=cfg, run_paths=run_paths)
        # 只跑极少的步数验证流程
        pretrainer.train(steps=100)
        print("   ✅ Low level pretraining loop OK")

        # 2. Test High Level Trainer
        print("   [2/2] Testing HRLTrainer...")
        trainer = HRLTrainer(config=cfg, run_paths=run_paths, low_model=low_model)
        print("   ✅ HRLTrainer initialized")

        # 尝试极短的训练 (2个窗口)
        trainer.train(total_windows=2)
        print("   ✅ High level training loop OK")

    except Exception as e:
        print(f"❌ Trainer Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理垃圾
        shutil.rmtree(run_paths.run_root, ignore_errors=True)
        print("   (Cleaned up test logs)")


def main():
    print("🔥 Starting MHC Smoke Test 🔥")
    test_imports()
    cfg = test_config_and_data()
    low_model = test_environments(cfg)
    test_trainer(cfg, low_model)

    print("\n🎉🎉🎉 SMOKE TEST PASSED! The architecture is sound. 🎉🎉🎉")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MHC-Architecture Orchestrator
Wraps Training and Evaluation pipelines.
[FIXED]: Now dynamically reads step counts from config.yaml instead of hardcoding.
"""

import sys
import subprocess
from pathlib import Path
import os
import yaml

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()


def load_simple_config(config_path):
    """简单读取 yaml，不需要复杂的 ConfigNode"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config: {e}")
        return {}


def run_script(script_name, args=None):
    """
    Helper to run a script with optional arguments.
    """
    if args is None:
        args = []

    # 构造命令：python script_name arg1 arg2 ...
    script_path = PROJECT_ROOT / script_name
    cmd = [sys.executable, str(script_path)] + args

    print(f"\n{'=' * 50}")
    print(f"🎬 Executing: {script_name} {' '.join(args)}")
    print(f"{'=' * 50}")

    # check=True 会在脚本返回非零状态码时抛出异常
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Success: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {script_name} failed with exit code {e.returncode}")
        # 如果任何一个步骤失败，整个实验应该停止
        sys.exit(e.returncode)


def main():
    print("🚀 Starting Full MHC Experiment Sequence (Pretrain -> Train -> Eval)")

    # 1. 读取配置
    cfg = load_simple_config(PROJECT_ROOT / "config.yaml")

    # 提取参数 (优先从 root 获取，其次尝试 env/training 节点)
    # 你的 config 结构可能是 low_level_steps 在根节点，也可能在 env 下
    ll_steps = cfg.get("low_level_steps")
    if ll_steps is None:
        ll_steps = cfg.get("training", {}).get("low_level_steps", 100000)  # 默认兜底

    print(f"📋 Configuration Loaded: Low-Level Steps = {ll_steps}")

    # --- Step 0: Pretraining Low-Level Model (CRITICAL) ---
    # 检查预训练模型是否存在，如果不存在则运行预训练
    low_model_path = PROJECT_ROOT / "models" / "low_level_pretrained.zip"

    if not low_model_path.exists():
        print(f"⚠️ Low-Level Model not found. Running Pretrainer for {ll_steps} steps...")
        # [FIX] 使用配置中的步数
        run_script("core/pretrainer.py", args=["--steps", str(ll_steps)])
    else:
        print(f"✅ Low-Level Model found at {low_model_path}. Skipping pretraining.")

    # --- Step 1: Training High-Level Model ---
    # High-Level 的步数 (total_timesteps) 是由 run_training.py 内部读取 config 完成的
    # 所以这里不需要传参，只要 config.yaml 改了，run_training.py 就会生效
    run_script("run_training.py")

    # --- Step 2: Baseline Training ---
    run_script("run_baseline_training.py")

    # --- Step 3: Evaluation ---
    model_path = PROJECT_ROOT / "models" / "high_level_trained_cmdp_sac"

    # 检查模型文件是否存在（带 .zip 后缀）
    if not (model_path.with_suffix(".zip")).exists():
        print(f"\n⚠️  Warning: The expected model file at {model_path}.zip was NOT found.")
        print("    Evaluation script might fail. Please check if run_training.py exported the model correctly.")

    # 调用评估脚本
    run_script("run_evaluation.py", args=[
        "--model_path", str(model_path),
        "--episodes", "10",
        "--output_dir", "./eval_results_auto"
    ])

    print("\n🎉🎉🎉 ALL TASKS COMPLETED SUCCESSFULLY! 🎉🎉🎉")
    print(f"Check results in: {PROJECT_ROOT}/eval_results_auto")


if __name__ == "__main__":
    main()
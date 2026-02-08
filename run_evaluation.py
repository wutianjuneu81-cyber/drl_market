import argparse
import os
import sys
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC

# 确保项目根目录在 PYTHONPATH 中
# 这样可以在任何目录下运行此脚本
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from DRL_market.env.high_level import HighLevelEnv
from DRL_market.common.config import load_config
from DRL_market.common.logging_utils import get_logger

# 默认模型路径 (与 run_training.py 导出的位置一致)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "high_level_trained_cmdp_sac"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation for DRL Market Agent")

    # [关键修改] 设置默认模型路径为标准导出路径
    # 这样 run_evaluation.py 既可以被 run_full_experiment.py 调用，也可以独立运行
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"Path to the trained model (default: {DEFAULT_MODEL_PATH})")

    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"),
                        help="Path to the configuration YAML file.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results.")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for evaluation environment.")

    return parser.parse_args()


def load_artifacts(model_path_base: str, env: HighLevelEnv, logger):
    """
    [关键逻辑] 加载辅助状态 (Normalizer)
    如果不加载这个，Agent 看到的观测值分布将是错误的，导致性能崩塌。
    """
    # 1. 尝试加载 Normalizer
    # Trainer 保存时格式为: {name}_normalizer.pkl
    norm_path = f"{model_path_base}_normalizer.pkl"

    if os.path.exists(norm_path):
        try:
            with open(norm_path, "rb") as f:
                saved_norm = pickle.load(f)

            # 恢复统计量到当前环境的 Normalizer
            # 注意：我们要把 saved_norm 的内部状态复制过去
            env.reward_normalizer.rms.mean = saved_norm.rms.mean
            env.reward_normalizer.rms.var = saved_norm.rms.var
            env.reward_normalizer.rms.count = saved_norm.rms.count

            # [#34] 强制冻结！防止评估数据污染统计量
            env.reward_normalizer.training = False

            logger.info(f"✅ Successfully loaded and FROZE RewardNormalizer from {norm_path}")

            # [Fix] 修复 IndexError: RewardNormalizer 的 mean 是标量，不能切片 [:2]
            # 直接打印标量值即可
            logger.info(f"   Normalizer Mean Preview: {env.reward_normalizer.rms.mean:.4f}")

        except Exception as e:
            logger.error(f"❌ Failed to load normalizer: {e}")
            # [可选] 打印详细堆栈以便调试
            # import traceback
            # logger.error(traceback.format_exc())
    else:
        logger.warning(f"⚠️  Normalizer file not found at {norm_path}!")
        logger.warning(
            "    If you used normalization during training, results will be WRONG (garbage in, garbage out).")


def main():
    args = parse_args()

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("Eval", os.path.join(args.output_dir, "eval.log"))

    # 处理模型路径 (去掉 .zip 后缀以便拼接其他文件名)
    # 这样无论是传 "model.zip" 还是 "model" 都能正常工作
    model_path_base = args.model_path
    if model_path_base.endswith(".zip"):
        model_path_base = model_path_base[:-4]

    logger.info("=" * 60)
    logger.info("Starting Evaluation Pipeline")
    logger.info(f"Model Base Path: {model_path_base}")
    logger.info(f"Config File:     {args.config}")
    logger.info("=" * 60)

    # 检查模型文件是否存在
    if not os.path.exists(f"{model_path_base}.zip"):
        logger.error(f"❌ Model file not found: {model_path_base}.zip")
        logger.error("   Please run training first or check the path.")
        sys.exit(1)

    try:
        # --- 1. 初始化评估环境 ---
        # 必须设置 is_eval=True，以触发 Data Loader 的噪声注入 [#22]
        logger.info("Initializing Environment (Mode: Evaluation)...")
        # 使用 absolute path 加载 config
        env = HighLevelEnv(args.config, is_eval=True)
        env.reset(seed=args.seed)

        # --- 2. 加载模型 ---
        logger.info("Loading SAC Model...")
        # 显式传递 env 以避免 SB3 重新创建环境
        model = SAC.load(f"{model_path_base}.zip", env=env)

        # --- 3. 加载 Artifacts (Normalizer) ---
        load_artifacts(model_path_base, env, logger)

        # --- 4. 执行评估循环 ---
        logger.info(f"Running {args.episodes} episodes...")

        # 用于存储所有 Episode 的汇总数据
        all_metrics = []

        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False

            # 单个 Episode 的累积指标
            ep_metrics = {
                "profit_net": 0.0,
                "revenue_energy": 0.0,
                "revenue_reg": 0.0,
                "cost_aging": 0.0,
                "cost_penalty": 0.0,
                "cost_aux": 0.0,
                "final_soc": 0.0
            }

            steps = 0
            while not done:
                # 评估时使用确定性策略 (deterministic=True)
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc

                # 累加 High-Level 返回的详细 Info [#47]
                # 注意：Info 中的值是单步 (15min) 的绝对值
                ep_metrics["profit_net"] += info.get('profit/net', 0.0)
                ep_metrics["revenue_energy"] += info.get('profit/revenue_energy', 0.0)
                ep_metrics["revenue_reg"] += info.get('profit/revenue_reg', 0.0)
                ep_metrics["cost_aging"] += info.get('cost/aging_real', 0.0)
                ep_metrics["cost_penalty"] += info.get('cost/penalty', 0.0)
                ep_metrics["cost_aux"] += info.get('cost/aux', 0.0)

                # 记录最后一步的 SoC
                if done:
                    ep_metrics["final_soc"] = info.get('state/mean_soc', 0.0)

                steps += 1

            # 打印进度
            logger.info(f"Episode {ep + 1}/{args.episodes}: Net Profit = {ep_metrics['profit_net']:.2f} CNY")
            all_metrics.append(ep_metrics)

        # --- 5. 生成报告 ---
        df = pd.DataFrame(all_metrics)

        # 保存原始数据
        csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
        df.to_csv(csv_path, index_label="Episode")

        # 计算统计量
        summary = df.describe().T
        summary_path = os.path.join(args.output_dir, "evaluation_summary.csv")
        summary.to_csv(summary_path)

        logger.info("-" * 60)
        logger.info("Evaluation Complete. Summary:")
        logger.info("-" * 60)
        logger.info(f"Mean Net Profit:   {df['profit_net'].mean():10.2f} +/- {df['profit_net'].std():.2f}")
        logger.info(f"Mean Rev Energy:   {df['revenue_energy'].mean():10.2f}")
        logger.info(f"Mean Rev Reg:      {df['revenue_reg'].mean():10.2f}")
        logger.info(f"Mean Cost Aging:   {df['cost_aging'].mean():10.2f}")
        logger.info(f"Mean Cost Penalty: {df['cost_penalty'].mean():10.2f}")
        logger.info("-" * 60)
        logger.info(f"Full report saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Critical Failure in Evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
import argparse
import os
import sys
import shutil
import traceback
import multiprocessing  # [新增] 用于设置启动方式
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 PYTHONPATH 中
# 假设 run_training.py 位于 DRL_market/ 根目录下
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from DRL_market.core.trainer import SACTrainer
from DRL_market.common.logging_utils import get_logger

# --- 定义标准模型保存路径 (供后续实验脚本对接) ---
# 这是所有 scripts/exp_*.py 默认读取的路径
TARGET_MODEL_DIR = PROJECT_ROOT / "models"
TARGET_MODEL_NAME = "high_level_trained_cmdp_sac"  # 不带后缀


def parse_args():
    parser = argparse.ArgumentParser(description="Run DRL Training for Battery Storage Market")

    # 默认 config
    default_config_path = PROJECT_ROOT / "config.yaml"

    # 基础配置
    parser.add_argument("--config", type=str, default=str(default_config_path),
                        help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", type=str, default="./logs/experiment_default",
                        help="Directory to save logs and models (training artifacts).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config if set).")

    # [关键功能] 强制重新训练
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if the target model already exists.")

    return parser.parse_args()


def check_model_exists(logger):
    """
    检查标准模型路径下是否已经存在模型文件。
    如果存在，返回 True。
    """
    zip_path = TARGET_MODEL_DIR / f"{TARGET_MODEL_NAME}.zip"
    norm_path = TARGET_MODEL_DIR / f"{TARGET_MODEL_NAME}_normalizer.pkl"

    if zip_path.exists() and norm_path.exists():
        logger.info(f"✅ Found existing trained model at: {zip_path}")
        return True
    return False


def export_model(source_dir, logger):
    """
    训练完成后，将最佳模型从 logs 目录复制到 models 目录，
    并重命名为标准名称，以便后续实验脚本自动读取。
    """
    # 确保 models 目录存在
    TARGET_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 寻找最佳模型 (best_rolling_model)
    # Trainer 默认保存名为 "best_rolling_model"
    src_zip = Path(source_dir) / "best_rolling_model.zip"
    src_norm = Path(source_dir) / "best_rolling_model_normalizer.pkl"
    src_cmdp = Path(source_dir) / "best_rolling_model_cmdp.pkl"

    # 如果没跑出 best (例如训练步数太少)，尝试用 final
    if not src_zip.exists():
        logger.warning("best_rolling_model not found, trying final_model...")
        src_zip = Path(source_dir) / "final_model.zip"
        src_norm = Path(source_dir) / "final_model_normalizer.pkl"
        src_cmdp = Path(source_dir) / "final_model_cmdp.pkl"

    if not src_zip.exists():
        logger.error("❌ Critical: No model file found to export!")
        return

    # 2. 目标路径
    dst_zip = TARGET_MODEL_DIR / f"{TARGET_MODEL_NAME}.zip"
    dst_norm = TARGET_MODEL_DIR / f"{TARGET_MODEL_NAME}_normalizer.pkl"
    dst_cmdp = TARGET_MODEL_DIR / f"{TARGET_MODEL_NAME}_cmdp.pkl"

    # 3. 执行复制
    try:
        shutil.copy2(src_zip, dst_zip)
        logger.info(f"➡️  Model exported to: {dst_zip}")

        if src_norm.exists():
            shutil.copy2(src_norm, dst_norm)
            logger.info(f"➡️  Normalizer exported to: {dst_norm}")

        if src_cmdp.exists():
            shutil.copy2(src_cmdp, dst_cmdp)
            logger.info(f"➡️  CMDP State exported to: {dst_cmdp}")

        logger.info("🎉 Model export complete. Evaluation scripts can now use this model.")

    except Exception as e:
        logger.error(f"❌ Failed to export model: {e}")


def main():
    args = parse_args()

    # 生成带时间戳的输出子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{os.path.basename(args.output_dir)}_{timestamp}"
    full_output_dir = os.path.join(os.path.dirname(args.output_dir), run_name)

    # 创建目录
    os.makedirs(full_output_dir, exist_ok=True)

    # 初始化日志
    logger = get_logger("Main", os.path.join(full_output_dir, "run.log"))
    logger.info("=" * 60)
    logger.info(f"🚀 Training Pipeline: {run_name}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 60)

    # --- [Step 1] 检查是否需要跳过训练 ---
    if check_model_exists(logger):
        if not args.force:
            logger.info("⏭️  Model already exists and --force flag is NOT set.")
            logger.info("⏭️  SKIPPING TRAINING to save time.")
            logger.info(f"    You can run: python run_training.py --force to overwrite.")
            return  # 直接退出，不再训练
        else:
            logger.warning("⚠️  Model exists but --force flag is set. Overwriting...")

    try:
        # --- [Step 2] 实例化 Trainer ---
        # SACTrainer 内部会根据 config 中的 n_envs 设置并行环境
        trainer = SACTrainer(
            config_path=args.config,
            output_dir=full_output_dir
        )

        # 简单的 Seed 记录 (Trainer 内部最好通过 config 设 seed)
        if args.seed is not None:
            logger.info(f"Note: Command line seed {args.seed} received.")

        # --- [Step 3] 开始训练 ---
        trainer.train()

        # --- [Step 4] 自动导出模型 ---
        logger.info("-" * 60)
        logger.info("Training Finished. Exporting artifacts...")
        export_model(full_output_dir, logger)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Saving emergency checkpoint...")
        if 'trainer' in locals():
            trainer.save_checkpoint("emergency_save_interrupt")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # [关键修复] 设置多进程启动方式
    # 'spawn' 是最安全的方式，兼容 Windows/Mac/Linux，且能避免 PyTorch 多线程死锁
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经被设置过，忽略

    main()
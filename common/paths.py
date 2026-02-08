import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


class RunPaths:
    """
    统一的路径管理工具 (Refactored for Campaign 3: Automated Workflow)
    负责生成、管理和检索所有实验产生的 Artifacts 路径。
    """

    def __init__(self, project_root: str, seed: int = None, strategy: str = "default", category: str = "train"):
        self.root = Path(project_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 基础目录结构
        self.logs_root = self.root / "logs"
        self.models_root = self.root / "models"
        self.results_root = self.root / "results"

        # 本次运行的唯一标识
        self.run_name = f"{category}_{strategy}_{self.timestamp}"
        if seed is not None:
            self.run_name += f"_s{seed}"

        # 本次运行的专属输出目录
        self.run_dir = self.logs_root / self.run_name

    def ensure(self):
        """创建必要的目录结构"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "tb_logs").mkdir(exist_ok=True)
        return self

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def models_dir(self) -> Path:
        """本次运行的模型保存目录"""
        return self.run_dir / "checkpoints"

    def low_level_model_path(self, tag: str = "final") -> Path:
        """获取本次运行生成的 Low-Level 模型路径"""
        return self.models_dir / f"low_level_{tag}.zip"

    def global_low_level_model_path(self) -> Path:
        """
        [关键] 获取全局共享的 Low-Level 预训练模型路径。
        High-Level Trainer 会从这里加载预训练好的底层智能体。
        """
        # 标准化名称，供所有实验复用
        return self.models_root / "low_level_pretrained.zip"

    def high_level_model_path(self, tag: str = "final") -> Path:
        """获取本次运行生成的 High-Level 模型路径"""
        return self.models_dir / f"high_level_{tag}.zip"

    def tensorboard_dir(self) -> str:
        return str(self.run_dir / "tb_logs")

    def log_file_path(self) -> str:
        return str(self.run_dir / "run.log")

    def save_config(self, config_dict: dict, filename="config.yaml"):
        """备份配置"""
        import yaml
        with open(self.run_dir / filename, 'w') as f:
            yaml.dump(config_dict, f)

    @staticmethod
    def clean_logs(project_root: str):
        """清理空的日志目录 (Utility)"""
        logs = Path(project_root) / "logs"
        if not logs.exists(): return

        for p in logs.iterdir():
            if p.is_dir() and not any(p.iterdir()):
                print(f"Removing empty log dir: {p}")
                shutil.rmtree(p)
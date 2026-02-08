import logging
import sys
import os
from pathlib import Path
from typing import Optional

# 全局日志格式
_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    获取一个配置好的 Logger。
    支持同时输出到控制台和指定文件。

    Args:
        name: Logger 名称 (e.g. 'Trainer', 'Main')
        log_file: (可选) 日志文件路径。如果提供，日志会追加写入该文件。
        level: 日志级别 (INFO, DEBUG, WARNING, ERROR)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 Handler (防止日志重复打印)
    if logger.handlers:
        # 如果 Logger 已经有 Handler 了，我们只检查是否需要新增 FileHandler
        if log_file:
            _ensure_file_handler(logger, log_file)
        return logger

    # --- 1. Console Handler (标准输出) ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(ch)

    # --- 2. File Handler (文件输出) ---
    if log_file:
        _ensure_file_handler(logger, log_file)

    # 防止日志向上传播到 Root Logger 导致双重打印
    logger.propagate = False

    return logger


def _ensure_file_handler(logger: logging.Logger, log_file: str):
    """
    辅助函数：确保 Logger 拥有指向特定文件的 Handler。
    如果已经有了，就不重复添加。
    """
    # 规范化路径以便比较
    abs_path = os.path.abspath(log_file)

    # 检查现有的 Handlers
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(h.baseFilename) == abs_path:
                return  # 已经有了，直接返回

    # 创建新的 FileHandler
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    except Exception as e:
        # 即使文件创建失败，也不要让程序崩溃，打印个警告即可
        print(f"[Logging Warning] Failed to create log file at {log_file}: {e}")


# ==========================================
# [LEGACY SUPPORT]
# 保留此函数以兼容可能还在使用旧接口的代码
# ==========================================
def init_root_logger(level="INFO", console=True, file_path=None):
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT)

    # 清理旧 handlers
    if root.handlers:
        for h in root.handlers:
            root.removeHandler(h)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root
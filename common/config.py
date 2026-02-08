import yaml
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


# ==========================================
# [CORE ARCHITECTURE]
# 混合态配置类：同时支持 .attribute 和 ['key'] 访问
# ==========================================

class ConfigNode:
    """
    配置节点：递归地将字典转换为既支持属性访问也支持下标访问的对象。
    这解决了 LowLevelEnv 使用 cfg['key'] 访问报错的问题。
    """

    def __init__(self, data):
        self._data = {}  # 内部保留一份原始字典
        for k, v in data.items():
            if isinstance(v, dict):
                # 递归转换子节点
                wrapper = ConfigNode(v)
                setattr(self, k, wrapper)
                self._data[k] = wrapper
            else:
                setattr(self, k, v)
                self._data[k] = v

    # --- 关键：字典行为支持 (Dict Protocol) ---
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value
        setattr(self, key, value)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_dict(self):
        """递归还原为纯字典，供不支持对象的库使用"""
        d = {}
        for k, v in self._data.items():
            if isinstance(v, ConfigNode):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def __repr__(self):
        return f"ConfigNode({list(self._data.keys())})"


class FinalRLBMSConfig:
    """
    [Root Config Class]
    整个项目的单一事实来源 (Single Source of Truth)。
    修复了 'env' vs 'environment' 的分裂问题，并完全兼容字典操作。
    """

    def __init__(self, config_path="config_simplified.yml"):
        self.config_path = config_path

        # 1. 路径解析与加载
        if not os.path.exists(config_path):
            # 尝试在项目根目录查找
            base_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(base_dir)  # DRL_market/
            alt_path = os.path.join(root_dir, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                # 最后的尝试：如果是相对路径，可能是在运行目录下
                if "/" not in config_path and "\\" not in config_path:
                    alt_path_2 = os.path.join(root_dir, config_path)
                    if os.path.exists(alt_path_2):
                        config_path = alt_path_2

        if not os.path.exists(config_path):
            print(f"❌ Error: Config file '{config_path}' not found.")
            # 创建一个空的根节点防止崩溃
            self._root = ConfigNode({})
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML config: {e}")

        # 2. 建立混合节点
        self._root = ConfigNode(raw_data)

        # 3. 将根节点的属性映射到 self (代理模式)
        # 这样 cfg.env 就能工作
        for k, v in self._root.items():
            setattr(self, k, v)

        # 4. [架构修复] 强制统一 'env' 和 'environment'
        # 无论 YAML 里写的是什么，代码里都可以用 cfg.env 或 cfg.environment 访问
        if hasattr(self, 'env') and not hasattr(self, 'environment'):
            setattr(self, 'environment', getattr(self, 'env'))
            self._root['environment'] = self._root['env']  # 字典层也要同步

        elif hasattr(self, 'environment') and not hasattr(self, 'env'):
            setattr(self, 'env', getattr(self, 'environment'))
            self._root['env'] = self._root['environment']

    # --- 代理字典操作到内部 _root ---
    # 这样 cfg['env'] 就能工作
    def __getitem__(self, key):
        return self._root[key]

    def get(self, key, default=None):
        return self._root.get(key, default)

    def __contains__(self, key):
        return key in self._root

    def to_dict(self):
        return self._root.to_dict()

    # --- 兼容性接口 (Legacy Support) ---
    def get_env_kwargs(self):
        """提取用于 gym 环境初始化的参数字典"""
        # 此时可以用 .env 也可以用 ['env']，非常安全
        env_node = self.get('env', self.get('environment', {}))

        # 转换为纯字典，防止下游库不支持 ConfigNode
        if isinstance(env_node, ConfigNode):
            env_dict = env_node.to_dict()
        else:
            env_dict = env_node if env_dict else {}

        # 处理外部文件路径
        profile_path = env_dict.get("external_power_profile_path")
        external_profile = None
        if profile_path:
            p = Path(profile_path)
            if p.exists() and p.suffix == ".npy":
                try:
                    external_profile = np.load(p)
                except:
                    pass

        # 构造 kwargs
        kwargs = {
            "num_cells": env_dict.get("n_clusters", 24),
            "time_passed_by_step": env_dict.get("low_level_step_seconds", 1.0),
            "randomization": env_dict.get("randomization", True),
            "external_power_profile": external_profile,
            # 将 config 本身传回去，用于 Wrapper
            # 注意：现在传回去的是 self (FinalRLBMSConfig)，它支持字典操作，所以 Wrapper 不会报错
            "config": self
        }
        return kwargs


# --- 辅助函数 ---
def load_config(config_path: str) -> Dict[str, Any]:
    """
    兼容旧代码的加载函数。
    如果调用者期望纯字典，这里会返回纯字典；
    但建议直接使用 FinalRLBMSConfig 以获得属性访问能力。
    """
    cfg = FinalRLBMSConfig(config_path)
    return cfg.to_dict()
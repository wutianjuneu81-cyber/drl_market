from typing import Dict, Any, List


class CurriculumScheduler:
    """
    CurriculumScheduler
    [Full Profit Refactoring]
    Manages economic strictness over time.
    Instead of adjusting reward weights, it adjusts:
    1. penalty_scale: How strictly market penalties are applied (0.1 -> 1.0)
    2. tracking_slack: How much physical error is tolerated (0.10 -> 0.03)
    3. budget_scale: Controls aging budget strictness if used.
    """

    def __init__(self, phases: List[Dict[str, Any]]):
        # phases must be sorted by start frame
        # phases 是从 config.yaml 中读取的列表，包含 start, penalty_scale 等
        self.phases = sorted(phases, key=lambda x: x["start"])
        self.current = self.phases[0] if self.phases else {}

        # 默认参数 (如果 config 为空时的兜底值)
        self.default_params = {
            "penalty_scale": 1.0,  # 默认全额罚款
            "tracking_slack": 0.03,  # 默认 3% 死区
            "budget_scale": 1.0,  # 默认不缩放预算
            "mismatch_scale": 1.0  # 默认 HIRO 匹配阈值不缩放
        }

    def get_phase(self, window_index: int) -> Dict[str, Any]:
        """
        根据当前的 window_index 查找对应的课程阶段配置
        """
        active = self.current
        # 简单向后查找，假设 phases 按 start 排序
        for ph in self.phases:
            if window_index >= ph["start"]:
                active = ph
            else:
                break
        self.current = active
        return active

    def adapt_high_level(self, window_index: int) -> Dict[str, Any]:
        """
        返回当前训练阶段的环境/经济学参数。
        该字典会被 Trainer 获取并注入到 Environment 中。
        """
        ph = self.get_phase(window_index)

        # 提取经济学参数，如果阶段配置中没有，则使用 default_params 中的默认值
        env_params = {
            # 罚款敏感度：早期低 (如 0.1)，鼓励 Agent 探索；后期高 (1.0)，模拟真实市场残酷性
            "penalty_scale": ph.get("penalty_scale", self.default_params["penalty_scale"]),

            # 物理死区：早期宽 (如 0.10)，允许较大误差；后期窄 (0.03)，对齐市场规则
            "tracking_slack": ph.get("tracking_slack", self.default_params["tracking_slack"]),

            # 老化预算缩放 (可选，如果使用 budget 约束的话)
            "budget_scale": ph.get("budget_scale", 1.0),

            # HIRO 匹配阈值缩放 (保留以兼容 GoalBuffer 逻辑)
            # 随着训练进行，可能要求 High/Low Level 的对齐更严格
            "mismatch_scale": ph.get("mismatch_scale", 1.0),

            # 兼容性保留 (PER 参数，如果有的话)
            "prio_alpha": ph.get("prio_alpha", 0.6),
            "beta_frames": ph.get("beta_frames", 80000)
        }
        return env_params
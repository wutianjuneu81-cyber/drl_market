from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class AgingScheduleGoal:
    """
    [全利润叙事重构]
    定义高层传给底层的经济学参数:
      - shadow_price_vector: 长度=24 (簇数)，表示各簇的寿命边际成本 (CNY/kWh 或 CNY/Ah)
                             High Level根据当前电池健康度和市场预期，动态定价。
      - penalty_scale: 罚款敏感度系数 (默认为1.0，可视作对风险的厌恶程度)
      - tracking_slack: 物理层的死区/容忍度 (与市场规则一致)
      - proxy_goal: (可选) 代理目标，用于 HIRO 等分层算法
    """
    shadow_price_vector: List[float]
    penalty_scale: float
    tracking_slack: float
    proxy_goal: Optional[Dict[str, float]] = None


class GoalInterface:
    def __init__(self, economics_cfg: Optional[Dict[str, Any]] = None):
        """
        初始化目标接口，并计算影子价格基准 (Base Price)。

        Args:
            economics_cfg: 包含电池成本和寿命预期的配置字典。
                           应包含:
                           - battery_cost_per_kwh (float): 电池单价 (CNY/kWh)
                           - system_capacity_kwh (float): 系统总容量
                           - expected_cycle_life (int): 预期循环寿命 (次)
        """
        self.latest: AgingScheduleGoal | None = None

        # --- [改进点 1] 动态计算影子价格基准 (Base Price) ---
        # 默认值 0.2 元/kWh (保守估计)
        self.base_price_cny = 0.2

        if economics_cfg:
            try:
                # 获取参数，提供默认值防止 KeyMissing
                cost_per_kwh = float(economics_cfg.get("battery_cost_per_kwh", 1500.0))
                capacity = float(economics_cfg.get("system_capacity_kwh", 24.0))  # 默认单柜
                cycle_life = float(economics_cfg.get("expected_cycle_life", 6000.0))

                # 总投资成本 (CNY)
                total_investment = cost_per_kwh * capacity

                # 全生命周期预期总吞吐量 (kWh)
                # 估算公式: 容量 * 循环次数 * 2 (一充一放) * 实际可用深度系数(0.9)
                # 这里使用简化物理定义: 容量 * 循环 * 2
                total_throughput = capacity * cycle_life * 2.0

                if total_throughput > 1e-6:
                    self.base_price_cny = total_investment / total_throughput

                # 打印日志 (实际工程中应使用 logger)
                print(f"[GoalInterface] Initialized Dynamic Base Price: {self.base_price_cny:.4f} CNY/kWh "
                      f"(Cost={cost_per_kwh}, Cycles={cycle_life})")
            except Exception as e:
                print(f"[GoalInterface] Warning: Failed to calc dynamic price, using default 0.2. Error: {e}")
        else:
            print(
                f"[GoalInterface] No economics_cfg provided, using default Base Price: {self.base_price_cny:.2f} CNY/kWh")

    def set_goal(self, action_vec, proxy_goal: Optional[Dict[str, float]] = None):
        """
        解析高层动作向量 -> 经济学目标
        Vector Layout: [p1...p24 (Shadow Prices), penalty_scale, tracking_slack]
        """
        # 兼容 Tensor 或 Numpy
        if hasattr(action_vec, 'detach'):
            action_vec = action_vec.detach().cpu().numpy()
        if hasattr(action_vec, 'flatten'):
            action_vec = action_vec.flatten()

        # 补齐维度，防止传入向量长度不足
        expected_dim = 26
        if len(action_vec) < expected_dim:
            padding = np.zeros(expected_dim - len(action_vec))
            action_vec = np.concatenate([action_vec, padding])

        # 解析动作
        # [改进点 1] 使用动态计算的 base_price_cny
        # High Level 输出倍率 (Scale)，通常在 [0.0, 2.0] 之间
        # 0.0x 表示当下完全不在乎寿命 (Free to use)
        # 2.0x 表示极度惜售 (例如预期未来电价极高，现在不舍得用)
        raw_price_scales = action_vec[:24]

        # 计算绝对影子价格: Base * Scale
        shadow_prices = [float(scale * self.base_price_cny) for scale in raw_price_scales]

        # 解析罚款敏感度 (通常由 Curriculum 控制，但 Agent 也可以微调)
        penalty_scale = float(action_vec[24])

        # 解析物理死区
        tracking_slack = float(action_vec[25])

        self.latest = AgingScheduleGoal(
            shadow_price_vector=shadow_prices,
            penalty_scale=penalty_scale,
            tracking_slack=tracking_slack,
            proxy_goal=proxy_goal
        )

    def adapt_tracking_slack(self, window_index: int):
        """
        课程学习逻辑：随着训练进行，逐渐收紧物理容忍度。
        通常由 Trainer 在每个 Window 开始时调用。
        """
        if not self.latest: return

        # 阶段性收紧死区
        if window_index < 10000:
            slack = 0.10  # 早期非常宽松 (100kW per unit MW)
        elif window_index < 30000:
            slack = 0.06
        else:
            slack = 0.03  # 最终对齐市场死区 3%

        self.latest.tracking_slack = slack

    def get_goal(self) -> Optional[AgingScheduleGoal]:
        return self.latest

    def as_dict(self) -> Dict[str, Any]:
        """
        将目标转换为字典格式，供 LowLevelEnv 消费
        """
        if not self.latest: return {}
        g = self.latest
        out = {
            'shadow_prices': g.shadow_price_vector,
            'penalty_scale': g.penalty_scale,
            'tracking_slack': g.tracking_slack
        }
        if g.proxy_goal: out['proxy_goal'] = g.proxy_goal
        return out
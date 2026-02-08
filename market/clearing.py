from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import numpy as np

try:
    from DRL_market.market.mechanism import MarketConstraints
except ImportError:
    from market.mechanism import MarketConstraints


# =========================
# 数据结构
# =========================

@dataclass
class FrequencyBid:
    unit_id: str
    time_period: int
    unit_type: str  # "storage", "thermal", "hydro"
    area: str
    capacity_mw: float  # 申报调频容量 (MW)
    mileage_price: float  # 里程报价 (元/MW)
    k_value: float  # 综合排序性能指标 k
    is_independent_storage: bool = False
    rated_power_mw: Optional[float] = None
    rated_capacity_mwh: Optional[float] = None


@dataclass
class EnergyMarketResult:
    """电能量市场结果（外部输入）"""
    time_period: int
    base_power_mw: float  # 电能量市场基点功率 (MW，充电负)
    energy_price: float = 0.0  # 外部统一结算点价格 (元/MWh)


@dataclass
class ClearedUnit:
    unit_id: str
    cleared_capacity_mw: float
    sorting_price: float
    P_i: float
    F_m: float
    internal_price: float
    ranking_position: int
    is_marginal_unit: bool = False


@dataclass
class ClearingResult:
    time_period: int
    marginal_price: float
    unified_price: float
    total_cleared_capacity: float
    k_max: float
    cleared_units: List[ClearedUnit] = field(default_factory=list)
    calc_time_ms: float = 0.0


class FrequencyCapacityCalculator:
    """
    计算调频可用容量（研究假设）

    assumption_mode:
      - "simplified": rated_power - |base_power|
      - "energy_limited": 加入 SOC 约束 (能量受限)
      - "policy_strict": 加入持续响应时间约束 (1h)
    """

    def __init__(self, assumption_mode: str = "simplified", duration_req_hours: float = 1.0):
        self.assumption_mode = assumption_mode
        self.duration_req_hours = duration_req_hours

    def calculate_available_capacity(
            self,
            rated_power_mw: float,
            base_power_mw: float,
            soc: Optional[float],
            rated_capacity_mwh: Optional[float],
    ) -> float:
        base_available = max(0.0, rated_power_mw - abs(base_power_mw))

        if self.assumption_mode == "simplified":
            return base_available

        if soc is None or rated_capacity_mwh is None:
            return base_available

        energy_available = max(0.0, soc * rated_capacity_mwh)
        if self.assumption_mode == "energy_limited":
            return min(base_available, energy_available)

        if self.assumption_mode == "policy_strict":
            energy_required = base_available * self.duration_req_hours
            if energy_required <= 0:
                return 0.0
            if energy_required > energy_available:
                return max(0.0, energy_available / self.duration_req_hours)
            return base_available

        return base_available


class FrequencyMarketClearing:
    """
    调频市场出清引擎（单站研究视角）

    ✅ 研究假设（需在论文中明确声明）：
    1) 电能量市场不出清，基点功率由外部提供
    2) 调频容量剩余 = 额定功率 - |基点功率|
    3) SOC 持续响应约束可配置为简化/严格模式
    4) 分布区下限约束可配置，但无强制重分配
    """

    def __init__(
            self,
            market_cfg: Dict[str, Any],
            assumption_mode: str = "simplified",
            duration_req_hours: float = 1.0,
    ):
        self.cfg = market_cfg
        self.constraints = MarketConstraints(market_cfg)
        self.capacity_calc = FrequencyCapacityCalculator(
            assumption_mode=assumption_mode,
            duration_req_hours=duration_req_hours
        )

        # ✅ [FIX P0-1] 预读取全局额定功率作为 fallback
        self.global_rated_power_mw = 0.0
        if "rated_power_mw" in self.cfg:
            self.global_rated_power_mw = float(self.cfg["rated_power_mw"])
        elif "rated_power_kw" in self.cfg:
            self.global_rated_power_mw = float(self.cfg["rated_power_kw"]) / 1000.0

    def clear_frequency_market(
            self,
            bids: List[FrequencyBid],
            system_demand_mw: float,
            energy_results: Dict[str, EnergyMarketResult],
            soc_snapshot: Dict[str, float],
            area_demands: Optional[Dict[str, float]] = None,
            area_min_ratios: Optional[Dict[str, float]] = None,
    ) -> ClearingResult:
        """
        调频市场出清核心算法（政策符合性 + 研究简化）

        算法流程：
        1. 计算可用容量（物理限制 + 申报限制）
        2. 归一化性能指标 P_i = k / k_max
        3. 计算独立储能边际替代率系数 F_m
        4. 计算排序价格 = 里程报价 / (P_i × F_m)
        5. 按排序价格出清，考虑分布区下限
        """
        t0 = time.time()

        # 边界条件检查
        if system_demand_mw <= 0 or not bids:
            return ClearingResult(
                time_period=bids[0].time_period if bids else 0,
                marginal_price=0.0,
                unified_price=0.0,
                total_cleared_capacity=0.0,
                k_max=0.0,
                cleared_units=[],
                calc_time_ms=0.0
            )

        # --- Step 1: 可用容量计算（物理 + 政策申报限制）
        available_caps: Dict[str, float] = {}

        for bid in bids:
            energy_res = energy_results.get(bid.unit_id)
            base_power = energy_res.base_power_mw if energy_res else 0.0

            # ✅ [FIX P0-1] 严格 rated_power_mw 回退逻辑
            rated_power = bid.rated_power_mw
            if rated_power is None or rated_power <= 0:
                rated_power = self.global_rated_power_mw
                if rated_power <= 0:
                    raise ValueError(
                        f"FrequencyBid {bid.unit_id} missing rated_power_mw and config has no rated_power"
                    )

            rated_energy = bid.rated_capacity_mwh
            soc = soc_snapshot.get(bid.unit_id, None)

            # 物理可用容量（研究假设）
            physical_cap = self.capacity_calc.calculate_available_capacity(
                rated_power_mw=rated_power,
                base_power_mw=base_power,
                soc=soc,
                rated_capacity_mwh=rated_energy
            )

            # 政策申报限制（细则第44条）
            bid_limit = self.constraints.check_capacity_bid(
                bid_mw=bid.capacity_mw,
                rated_mw=rated_power,
                area_demand_mw=area_demands.get(bid.area) if area_demands else None,
                unit_type=bid.unit_type
            )

            # 最终可用容量 = min(物理可用, 申报限制)
            available_caps[bid.unit_id] = max(0.0, min(physical_cap, bid_limit))

        # --- Step 2: P_i 归一化（k / k_max）
        k_max_raw = max((b.k_value for b in bids), default=0.0)
        use_perf = k_max_raw >= 0.1
        k_max = max(k_max_raw, 1e-9)

        # --- Step 3: 计算独立储能 F_m（附录5）
        storage_items = []
        for bid in bids:
            if bid.is_independent_storage:
                if use_perf:
                    P_i = max(bid.k_value / k_max, 1e-2)
                else:
                    P_i = 1.0
                internal_price = bid.mileage_price / P_i
                storage_items.append((bid, P_i, internal_price))

        # 按内部价格排序（细则附录5）
        storage_items.sort(key=lambda x: x[2])

        storage_fm: Dict[str, float] = {}
        cumulative_ratio = 0.0
        for bid, P_i, internal_price in storage_items:
            cap = available_caps.get(bid.unit_id, 0.0)
            cumulative_ratio += cap / system_demand_mw
            F_m = self.constraints.calculate_marginal_substitution_factor(cumulative_ratio)
            storage_fm[bid.unit_id] = F_m

        # --- Step 4: 排序价格计算（细则第46-47条）
        sorting_pool = []
        for bid in bids:
            if use_perf:
                P_i = max(bid.k_value / k_max, 1e-2)
            else:
                P_i = 1.0
            F_m = storage_fm.get(bid.unit_id, 1.0)
            internal_price = bid.mileage_price / P_i
            sorting_price = bid.mileage_price / (P_i * F_m)
            sorting_pool.append((bid, P_i, F_m, internal_price, sorting_price))

        # 排序（同价时按 P_i、k 值优先 - 细则第49条）
        sorting_pool.sort(key=lambda x: (x[4], -x[1], -x[0].k_value))

        # --- Step 5: 出清（考虑分布区下限）
        cleared_units: List[ClearedUnit] = []
        remaining_total = system_demand_mw
        remaining_by_area = {}
        if area_demands and area_min_ratios:
            for area, demand in area_demands.items():
                min_ratio = area_min_ratios.get(area, 0.0)
                # ✅ 检查合理性：分布区下限不能超过区域需求
                remaining_by_area[area] = min(demand * min_ratio, demand)

        marginal_price = 0.0
        rank = 0

        for bid, P_i, F_m, internal_price, sorting_price in sorting_pool:
            # ✅ [FIX P0-3] 僵尸循环防护
            if remaining_total <= 0:
                break

            cap = available_caps.get(bid.unit_id, 0.0)
            if cap <= 0:
                continue

            # 初始可中标容量
            to_clear = min(cap, remaining_total)

            # 区域下限：保底逻辑（保持原有结构）
            if bid.area in remaining_by_area and remaining_by_area[bid.area] > 0:
                area_needed = remaining_by_area[bid.area]
                to_clear = min(cap, area_needed, remaining_total)

            if to_clear <= 0:
                continue

            rank += 1
            cleared_units.append(
                ClearedUnit(
                    unit_id=bid.unit_id,
                    cleared_capacity_mw=to_clear,
                    sorting_price=sorting_price,
                    P_i=P_i,
                    F_m=F_m,
                    internal_price=internal_price,
                    ranking_position=rank
                )
            )

            # 更新剩余需求
            remaining_total -= to_clear
            if bid.area in remaining_by_area:
                remaining_by_area[bid.area] -= to_clear

            marginal_price = sorting_price

        # 统一出清价
        if not cleared_units:
            marginal_price = 0.0
            unified_price = 0.0
        else:
            cleared_units[-1].is_marginal_unit = True
            unified_price = min(
                max(marginal_price, self.constraints.price_floor),
                self.constraints.price_cap
            )

        return ClearingResult(
            time_period=bids[0].time_period if bids else 0,
            marginal_price=marginal_price,
            unified_price=unified_price,
            total_cleared_capacity=system_demand_mw - remaining_total,
            k_max=k_max,
            cleared_units=cleared_units,
            calc_time_ms=(time.time() - t0) * 1000.0
        )
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Union


# 喵~ Final Fixed Version
# 包含：0申报允许、机会成本实体化、API兼容性保护、偏差罚款修正、负电价支持

class MarketConstraints:
    """
    [市场约束与边界条件管理器]
    依据：细则第44、45、47条及附录5
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.price_cap = cfg.get("price_cap", 15.0)
        # ✅ [FIX] 默认支持负电价，但需配合 config 使用
        self.price_floor = cfg.get("price_floor", -1000.0)

        self.cap_ratio_limits = {
            "thermal": {"min": 0.03, "max": 0.075},
            "hydro": {"min": 0.05, "max": 0.10},
            "storage": {
                "min_mw": cfg.get("storage_min_mw", 5.0),
                "min_ratio": cfg.get("storage_min_ratio", 0.20),
                "max_ratio": cfg.get("storage_max_ratio", 1.00),
            },
        }

        self.marginal_sub_params = {
            "U_y": cfg.get("marginal_U_y", 2.5),
            "U_x": cfg.get("marginal_U_x", 0.6),
        }
        self.marginal_curve = cfg.get("marginal_sub_curve", None)

    def check_capacity_bid(
            self,
            bid_mw: float,
            rated_mw: float,
            area_demand_mw: Optional[float] = None,
            unit_type: str = "storage",
    ) -> float:
        if unit_type == "storage":
            # ✅ [FIX] 允许 0 申报 (离场机制)
            limits = self.cap_ratio_limits["storage"]
            min_by_rated = rated_mw * limits.get("min_ratio", 0.20)
            min_by_lb = limits.get("min_mw", 5.0)
            min_req = max(min_by_rated, min_by_lb)
            max_by_rated = rated_mw * limits.get("max_ratio", 1.0)

            # 如果申报量极小，视为不参与，直接返回 0.0
            if bid_mw < min_req - 1e-6:
                return 0.0

            if area_demand_mw is not None and area_demand_mw > 0:
                min_by_demand = area_demand_mw * 0.15
                max_by_demand = area_demand_mw * 0.20
                max_req = min(max_by_rated, max_by_demand)
                if max_req < min_req:
                    return 0.0
            else:
                max_req = max_by_rated

            return float(np.clip(bid_mw, min_req, max_req))

        # 非储能机组同理
        limits = self.cap_ratio_limits.get(unit_type, {})
        min_mw = rated_mw * limits.get("min", 0.0)
        max_mw = rated_mw * limits.get("max", 1.0)
        if bid_mw < min_mw:
            return 0.0
        return float(np.clip(bid_mw, min_mw, max_mw))

    def calculate_marginal_substitution_factor(self, ratio: float) -> float:
        method = self.cfg.get("marginal_sub_method", "linear")
        if method == "curve":
            return self._calculate_with_curve(ratio)
        if method == "smooth":
            return self._calculate_smooth(ratio)
        return self._calculate_linear(ratio)

    def _calculate_linear(self, ratio: float) -> float:
        U_y = self.marginal_sub_params["U_y"]
        U_x = self.marginal_sub_params["U_x"]
        if ratio <= 0:
            return U_y
        if ratio >= U_x:
            return 1.0
        return float(U_y - (U_y - 1.0) * (ratio / U_x))

    def _calculate_with_curve(self, ratio: float) -> float:
        if not self.marginal_curve:
            return self._calculate_linear(ratio)
        xs = [pt[0] for pt in self.marginal_curve]
        ys = [pt[1] for pt in self.marginal_curve]
        return float(np.interp(ratio, xs, ys))

    def _calculate_smooth(self, ratio: float) -> float:
        U_y = self.marginal_sub_params["U_y"]
        U_x = self.marginal_sub_params["U_x"]
        if ratio <= 0:
            return U_y
        if ratio >= U_x:
            return 1.0
        k = -np.log(0.01 / (U_y - 1.0)) / U_x
        return float(1.0 + (U_y - 1.0) * np.exp(-k * ratio))


class AllocationMechanism:
    def __init__(self, cfg: Dict):
        self.user_allocation_ratio = cfg.get("user_allocation_ratio", 0.5)

    def calculate_allocation_complete(
            self,
            total_reg_cost: float,
            is_continuous_spot: bool,
            my_gen_mwh: float,
            total_gen_mwh: float,
            my_load_mwh: float,
            total_load_mwh: float,
            province_id: str = "GD",
    ) -> float:
        if total_reg_cost <= 0:
            return 0.0

        if not is_continuous_spot:
            if total_gen_mwh <= 0:
                return 0.0
            return total_reg_cost * (my_gen_mwh / total_gen_mwh)

        ratio_user = self.user_allocation_ratio
        ratio_gen = 1.0 - ratio_user

        cost_gen = 0.0
        cost_user = 0.0
        if total_gen_mwh > 0:
            cost_gen = total_reg_cost * ratio_gen * (my_gen_mwh / total_gen_mwh)
        if total_load_mwh > 0:
            cost_user = total_reg_cost * ratio_user * (my_load_mwh / total_load_mwh)
        return cost_gen + cost_user


class GuangdongMarketEngine:
    """
    [广东/南方区域电力市场结算引擎]
    Fixed Version by 🐱 & Expert Review
    """

    def __init__(self, full_cfg: Dict):
        self.cfg = full_cfg.get("market", {})
        self.TIME_STEP_HOURS = float(self.cfg.get("time_step_hours", 0.25))

        if "rated_power_mw" in self.cfg:
            self.rated_power_mw = float(self.cfg["rated_power_mw"])
        elif "rated_power_kw" in self.cfg:
            self.rated_power_mw = float(self.cfg["rated_power_kw"]) / 1000.0
        else:
            self.rated_power_mw = 2.58

        self.best_coal_speed_mw_per_min = float(self.cfg.get("best_coal_speed_mw_per_min", 30.0))
        self.avg_standard_speed_mw_per_min = float(self.cfg.get("avg_standard_speed_mw_per_min", 20.0))

        self.allowed_error_mw = 0.01 * self.rated_power_mw
        self.deadband_mw = self.cfg.get("deadband_mw", 0.01 * self.rated_power_mw)

        self.k_weights = {"rate": 0.5, "delay": 0.25, "accuracy": 0.25}
        self.m_weights = {"rate": 0.16, "delay": 0.42, "accuracy": 0.42}

        self.k_i_max = self.cfg.get("k_I_max", None)
        self.m_i_max = self.cfg.get("m_I_max", None)

        self.constraints = MarketConstraints(self.cfg)
        self.allocation = AllocationMechanism(self.cfg)

        self.sampling_interval = float(self.cfg.get("sampling_interval", 1.0))
        self.default_tolerance = float(self.cfg.get("default_tolerance", 0.05))
        self.penalty_ratio = float(self.cfg.get("penalty_ratio", 2.0))
        self.energy_price_scaler = float(self.cfg.get("energy_price_scaler", 1.0))

        self.cap_cfg = self.cfg.get("capacity_mechanism", {})
        self.capacity_enabled = bool(self.cap_cfg.get("enabled", False))

        if self.capacity_enabled:
            T_s = self.cap_cfg.get("storage_duration_hours", 2.0)
            T_peak = self.cap_cfg.get("max_net_load_peak_hours", 4.0)
            self.capacity_ratio_k = min(1.0, T_s / max(T_peak, 1e-6))

            # ✅ 优先读取 storage_capacity_price
            if "storage_capacity_price_cny_per_kw_year" in self.cap_cfg:
                P_base_kw = self.cap_cfg["storage_capacity_price_cny_per_kw_year"]
            else:
                P_base_kw = self.cap_cfg.get("coal_capacity_price_cny_per_kw_year", 100.0)

            P_storage_mw = P_base_kw * 1000.0 * self.capacity_ratio_k
            steps_per_year = (365.0 * 24.0) / max(self.TIME_STEP_HOURS, 1e-9)
            self.capacity_price_per_step_mw = P_storage_mw / steps_per_year
        else:
            self.capacity_price_per_step_mw = 0.0
            self.capacity_ratio_k = 0.0

    def update_tolerance(self, tol: float):
        self.default_tolerance = float(max(0.0, tol))

    def _estimate_response_delay(self, target_series: np.ndarray, actual_series: np.ndarray) -> float:
        if len(target_series) < 10:
            return 0.0
        t_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-6)
        a_norm = (actual_series - np.mean(actual_series)) / (np.std(actual_series) + 1e-6)
        correlation = np.correlate(a_norm, t_norm, mode="full")
        lags = np.arange(-len(a_norm) + 1, len(a_norm))
        lag = lags[np.argmax(correlation)]
        return max(0.0, float(lag) * self.sampling_interval)

    def calculate_metrics_from_series(
            self, actual_p: np.ndarray, instruct_p: np.ndarray
    ) -> Dict[str, float]:
        if len(actual_p) != len(instruct_p) or len(actual_p) < 2:
            return {"speed": 0.0, "delay": 60.0, "error": 0.0}

        abs_error_mw = np.abs(actual_p - instruct_p)
        avg_error_mw = float(np.mean(abs_error_mw))

        delta_ins = np.diff(instruct_p)
        delta_act = np.diff(actual_p)
        mask = np.abs(delta_ins) > 1e-6
        if np.any(mask):
            speed_mw_min = np.mean(np.abs(delta_act[mask])) / max(
                self.sampling_interval / 60.0, 1e-6
            )
        else:
            total_movement = np.sum(np.abs(delta_act))
            total_time_min = (len(actual_p) * self.sampling_interval) / 60.0
            speed_mw_min = total_movement / max(total_time_min, 1e-6)

        delay_sec = self._estimate_response_delay(instruct_p, actual_p)
        return {"speed": float(speed_mw_min), "error": avg_error_mw, "delay": delay_sec}

    def calculate_sorting_performance_k(self, speed: float, delay_sec: float, error_mw: float) -> float:
        k_i = speed / (self.avg_standard_speed_mw_per_min + 1e-9)
        k_ii = max(0.0, 1.0 - (delay_sec / 300.0))
        k_iii = max(0.0, 1.0 - (error_mw / (0.015 * self.rated_power_mw + 1e-9)))

        k_raw = (
                self.k_weights["rate"] * k_i
                + self.k_weights["delay"] * k_ii
                + self.k_weights["accuracy"] * k_iii
        )

        if self.k_i_max is not None:
            k_raw = float(np.clip(k_raw, 0.0, self.k_i_max))
        return float(k_raw)

    def calculate_performance_m(self, speed: float, delay_sec: float, error_mw: float) -> float:
        m_i = speed / (self.best_coal_speed_mw_per_min + 1e-9)
        m_ii = max(0.0, 1.0 - (delay_sec / 60.0))
        m_iii = max(0.0, 1.0 - (error_mw / (self.allowed_error_mw + 1e-9)))

        m_raw = (
                self.m_weights["rate"] * m_i
                + self.m_weights["delay"] * m_ii
                + self.m_weights["accuracy"] * m_iii
        )

        if self.m_i_max is not None:
            m_raw = float(np.clip(m_raw, 0.0, self.m_i_max))
        return float(m_raw)

    def normalize_k_to_pi(self, k_value: float, k_max: float) -> float:
        if k_max <= 0:
            return 0.0
        return float(np.clip(k_value / k_max, 0.0, 1.0))

    def calculate_regulation_mileage(self, actual_p: np.ndarray) -> float:
        if len(actual_p) < 2:
            return 0.0
        delta = np.abs(np.diff(actual_p))
        noise_thr = float(self.cfg.get("mileage_noise_mw", 0.0))
        if noise_thr > 0:
            delta = delta[delta >= noise_thr]
        return float(np.sum(delta))

    def calculate_capacity_revenue_split(self, installed_cap_mw, availability):
        if not self.capacity_enabled:
            return 0.0
        return float(installed_cap_mw * self.capacity_price_per_step_mw * availability)

    def calculate_regulation_penalty(self, revenue_reg: float, violation_flags: Optional[Dict[str, bool]] = None):
        if not violation_flags:
            return 0.0
        penalty = 0.0
        if violation_flags.get("agc_exit", False):
            penalty += revenue_reg * 1.0
        if violation_flags.get("capacity_shortfall", False):
            penalty += revenue_reg * 0.5
        if violation_flags.get("param_violation", False):
            penalty += revenue_reg * 1.0
        return min(penalty, revenue_reg)

    def _validate_clearing_price(self, price: float) -> float:
        return min(max(price, self.constraints.price_floor), self.constraints.price_cap)

    def calculate_net_regulation_revenue(
            self,
            cap_bid_mw: float,
            mil_mw: float,
            p_mil: float,
            m_coeff: float,
            p_energy: float,
            installed_cap_mw: Optional[float] = None,
            availability: float = 1.0,
            violation_flags: Optional[Dict[str, bool]] = None,
            safety_check_passed: bool = True,
            total_reg_cost_system: Optional[float] = None,
            total_gen_mwh: float = 1000.0,
            total_load_mwh: float = 1000.0,
            my_gen_mwh: float = 0.0,
            my_load_mwh: float = 0.0,
    ) -> Dict:
        """
        调频结算核心逻辑 (Internal)
        """
        if installed_cap_mw is None:
            installed_cap_mw = self.rated_power_mw

        if not safety_check_passed and self.cfg.get("force_zero_on_safety_fail", False):
            gross_reg = 0.0
            reg_penalty = 0.0
        else:
            p_mil_eff = self._validate_clearing_price(p_mil)
            gross_reg = mil_mw * p_mil_eff * m_coeff
            reg_penalty = self.calculate_regulation_penalty(gross_reg, violation_flags)

        # ✅ [FIX] 机会成本实体化
        opp_cost = 0.0
        if self.cfg.get("enable_opportunity_cost", True):
            # 策略：即使负电价，机会成本也暂不为负（保守策略）
            opp_cost = cap_bid_mw * self.TIME_STEP_HOURS * max(0.0, p_energy)

        rev_capacity = self.calculate_capacity_revenue_split(installed_cap_mw, availability)

        allocation_cost = 0.0
        if total_reg_cost_system is not None and total_reg_cost_system > 0:
            allocation_cost = self.allocation.calculate_allocation_complete(
                total_reg_cost_system, True, my_gen_mwh, total_gen_mwh, my_load_mwh, total_load_mwh
            )

        net_profit = (gross_reg - reg_penalty) + rev_capacity - opp_cost - allocation_cost

        return {
            "net_profit": net_profit,
            "gross_regulation": gross_reg,
            "regulation_penalty": reg_penalty,
            "opportunity_cost": opp_cost,
            "capacity_revenue": rev_capacity,
            "allocation_cost": allocation_cost,
            "safety_passed": safety_check_passed,
        }

    def calculate_energy_settlement(
            self,
            target_power_mw: float,
            actual_power_mw: float,
            price: float,
            duration_hours: float,
            tolerance: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        电能量结算 (Fixed: 额定容量基准)
        """
        revenue = actual_power_mw * duration_hours * price

        # ✅ [FIX] 偏差罚款基准修正为额定容量 (归一化)
        base = max(self.rated_power_mw, 1e-6)
        dev_rate = abs(target_power_mw - actual_power_mw) / base

        tol = self.default_tolerance if tolerance is None else tolerance

        penalty = 0.0
        if dev_rate > tol:
            excess_rate = dev_rate - tol
            penalty = excess_rate * base * duration_hours * abs(price) * self.penalty_ratio

        return revenue, penalty


@dataclass
class MarketConfig:
    price_cap: float = 15.0
    price_floor: float = 3.5
    default_tolerance: float = 0.05
    penalty_ratio: float = 2.0
    energy_price_scaler: float = 1.0
    user_allocation_ratio: float = 0.5


class MarketEngine(GuangdongMarketEngine):
    """
    兼容旧接口的封装 (Fixed & Compatible)
    """

    def calculate_mileage(self, actual_curve_kw: np.ndarray) -> float:
        actual_mw = np.array(actual_curve_kw, dtype=float) / 1000.0
        return self.calculate_regulation_mileage(actual_mw)

    def calculate_performance_index(
            self, target_curve_kw: np.ndarray, actual_curve_kw: np.ndarray, k_max: float
    ) -> float:
        if k_max <= 0:
            raise ValueError("k_max must be positive for normalization")

        target_mw = np.array(target_curve_kw, dtype=float) / 1000.0
        actual_mw = np.array(actual_curve_kw, dtype=float) / 1000.0

        metrics = self.calculate_metrics_from_series(actual_mw, target_mw)
        k_raw = self.calculate_sorting_performance_k(metrics["speed"], metrics["delay"], metrics["error"])
        return self.normalize_k_to_pi(k_raw, k_max)

    def calculate_regulation_revenue(
            self,
            capacity_kw: float,
            mileage_kw: float,
            price_mileage: float,
            m_coeff: float,
            price_energy: float = 0.0,
            return_detail: bool = False,
            return_net: bool = True,  # ✅ 新增：默认返回净收益
    ) -> Union[float, Dict[str, float]]:

        if m_coeff is None:
            raise ValueError("m_coeff must be provided for regulation settlement")

        m_coeff = float(np.clip(m_coeff, 0.0, 2.0))
        cap_mw = capacity_kw / 1000.0
        mil_mw = mileage_kw / 1000.0
        p_mil = self._validate_clearing_price(price_mileage)

        res = self.calculate_net_regulation_revenue(
            cap_bid_mw=cap_mw,
            mil_mw=mil_mw,
            p_mil=p_mil,
            m_coeff=m_coeff,
            p_energy=price_energy,
            installed_cap_mw=self.rated_power_mw,
            availability=1.0,
        )

        if return_detail:
            return res

        # ✅ 默认返回净收益，避免机会成本丢失
        return float(res["net_profit"] if return_net else res["gross_regulation"])

    def calculate_settlement(
            self,
            target_power_kw: float,
            actual_power_kw: float,
            price: float,
            is_reg_mode: bool,
            reg_capacity_kw: float,
            accumulated_mileage_kw: float,
            price_reg_mileage: float,
            duration_hours: float,
            m_coeff: float,
            capacity_availability: float = 1.0,
            installed_capacity_mw: Optional[float] = None,
    ) -> Dict[str, float]:
        if m_coeff is None:
            m_coeff = 1.0

        target_mw = target_power_kw / 1000.0
        actual_mw = actual_power_kw / 1000.0

        energy_rev, penalty = self.calculate_energy_settlement(
            target_power_mw=target_mw,
            actual_power_mw=actual_mw,
            price=price,
            duration_hours=duration_hours,
        )

        # ✅ [FIX] 自动检测调频 (阈值 1瓦)
        effective_reg_mode = is_reg_mode or (reg_capacity_kw > 1e-3)

        reg_rev = 0.0
        opp_cost = 0.0

        if effective_reg_mode:
            # ✅ [FIX] 内部调用请求详情，以获取 opp_cost
            reg_res = self.calculate_regulation_revenue(
                capacity_kw=reg_capacity_kw,
                mileage_kw=accumulated_mileage_kw,
                price_mileage=price_reg_mileage,
                m_coeff=m_coeff,
                price_energy=price,
                return_detail=True
            )
            reg_rev = reg_res["gross_regulation"]
            opp_cost = reg_res["opportunity_cost"]

        if installed_capacity_mw is None:
            installed_capacity_mw = self.rated_power_mw
        cap_rev = self.calculate_capacity_revenue_split(installed_capacity_mw, capacity_availability)

        # Base Reward = 能量 + 调频(Gross) + 容量 - 罚款 - 机会成本
        base_reward = energy_rev + reg_rev + cap_rev - penalty - opp_cost
        scaled_reward = base_reward * self.energy_price_scaler

        return {
            "revenue_energy": energy_rev,
            "revenue_regulation": reg_rev,
            "revenue_capacity": cap_rev,
            "penalty": penalty,
            "opportunity_cost": opp_cost,
            "base_reward": base_reward,
            "scaled_reward": scaled_reward,
        }
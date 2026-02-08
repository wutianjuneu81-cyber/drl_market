import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SchedulerConfig:
    rated_power_kw: float = 24.0
    rated_capacity_kwh: float = 24.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    efficiency: float = 0.95
    reg_capacity_reserve_hours: float = 1.0
    horizon: int = 96
    dt_hours: float = 0.25
    reg_power_ratio_limit: float = 0.15
    marginal_aging_cost_default: float = 0.05


class DayAheadScheduler:
    """
    Day-Ahead Scheduler (Robust SCUC with Relaxation)
    [Scientific Rigor]:
    1. Uses Efficiency (eta) in SOC dynamics.
    2. IRO Fallback: Prevents crashes if solver fails.
    3. Parameterized constraints.
    """

    def __init__(self, config: SchedulerConfig):
        self.cfg = config
        self.last_price_energy_cache = None
        self.last_price_reg_cache = None

    def solve(self,
              price_energy: np.ndarray,
              price_reg: np.ndarray,
              agc_intensity: np.ndarray = None,
              current_soh: float = 1.0,
              current_rint_factor: float = 1.0,
              marginal_aging_cost: Optional[float] = None,
              uncertainty_ratio: float = 0.1,
              initial_soc: float = 0.5
              ) -> Dict[str, np.ndarray]:

        if marginal_aging_cost is None:
            marginal_aging_cost = self.cfg.marginal_aging_cost_default

        self.last_price_energy_cache = price_energy
        self.last_price_reg_cache = price_reg

        T = self.cfg.horizon
        dt = self.cfg.dt_hours
        eff = self.cfg.efficiency

        if agc_intensity is None:
            agc_intensity = np.zeros(T)

        # Variables
        p_base = cp.Variable(T, name="p_base")
        c_reg = cp.Variable(T, name="c_reg", nonneg=True)
        soc = cp.Variable(T + 1, name="soc")
        p_abs = cp.Variable(T, nonneg=True)

        # Slack Variables for Soft Constraints
        soc_slack = cp.Variable(T + 1, nonneg=True)
        BIG_M = 2000.0

        constraints_map = {
            "soc_dynamics": [], "power_limits": [], "soc_bounds": [], "auxiliary": []
        }

        # 1. Init SOC
        constraints_map["soc_dynamics"].append(soc[0] == initial_soc)

        # Safety Margin: Planning using 95% of rated power
        safe_rated_power = self.cfg.rated_power_kw * 0.95

        for t in range(T):
            # Power Limits
            con_p_min = p_base[t] - c_reg[t] >= -safe_rated_power
            con_p_max = p_base[t] + c_reg[t] <= safe_rated_power
            con_reg_limit = c_reg[t] <= safe_rated_power * self.cfg.reg_power_ratio_limit
            constraints_map["power_limits"].extend([con_p_min, con_p_max, con_reg_limit])

            # SOC Dynamics (with efficiency approximation)
            constraints_map["soc_dynamics"].append(
                soc[t + 1] == soc[t] - (p_base[t] * dt / self.cfg.rated_capacity_kwh) * (1.0 / eff)
            )

            # Capacity Reservation (Relaxed)
            reserve_ratio = (c_reg[t] * self.cfg.reg_capacity_reserve_hours) / self.cfg.rated_capacity_kwh
            con_s_min = soc[t + 1] >= self.cfg.soc_min + reserve_ratio - soc_slack[t + 1]
            con_s_max = soc[t + 1] <= self.cfg.soc_max - reserve_ratio + soc_slack[t + 1]
            constraints_map["soc_bounds"].extend([con_s_min, con_s_max])

            # Auxiliary for absolute value
            constraints_map["auxiliary"].append(p_abs[t] >= p_base[t])
            constraints_map["auxiliary"].append(p_abs[t] >= -p_base[t])

        all_constraints = [c for group in constraints_map.values() for c in group]

        # --- Objective ---
        dynamic_aging_cost_coeff = marginal_aging_cost * max(1.0, current_rint_factor)
        soh_penalty = 1.0 + 5.0 * (0.8 - current_soh) if current_soh < 0.8 else 1.0
        final_cost_coeff = dynamic_aging_cost_coeff * soh_penalty

        # Linear Aging Cost
        throughput_est = cp.sum(p_abs + c_reg * 0.3) * dt
        cost_aging_linear = throughput_est * final_cost_coeff

        # Quadratic term: Penalize heavy usage per step squared
        cost_aging_quad = cp.sum_squares(p_abs) * dt * (final_cost_coeff * 0.001)

        total_aging_cost = cost_aging_linear + cost_aging_quad

        revenue_nominal = cp.sum(cp.multiply(p_base, price_energy)) * dt + \
                          cp.sum(cp.multiply(c_reg, price_reg)) * dt

        robust_penalty = cp.sum(cp.multiply(p_abs, price_energy * uncertainty_ratio)) * dt
        slack_penalty = BIG_M * cp.sum(soc_slack)

        prob = cp.Problem(cp.Maximize(revenue_nominal - robust_penalty - total_aging_cost - slack_penalty),
                          all_constraints)

        status = 'solver_error'
        # Try multiple solvers for robustness
        solvers_to_try = [
            (cp.CLARABEL, {}),
            (cp.ECOS, {}),
            (cp.OSQP, {'eps_abs': 1e-4, 'eps_rel': 1e-4, 'max_iter': 10000}),
            (cp.SCS, {'eps': 1e-3, 'max_iters': 5000})
        ]

        for solver, kwargs in solvers_to_try:
            try:
                prob.solve(solver=solver, **kwargs)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    status = 'optimal'
                    break
            except Exception:
                continue

        if status != 'optimal':
            return self._get_fallback_plan(T, initial_soc)

        # Extract Duals (Shadow Prices)
        lambda_risk = np.zeros(T)
        lambda_aging = np.zeros(T)
        power_con_list = constraints_map["power_limits"]
        soc_con_list = constraints_map["soc_bounds"]

        for t in range(T):
            dual_p_min = power_con_list[3 * t].dual_value
            dual_p_max = power_con_list[3 * t + 1].dual_value
            lambda_risk[t] = abs(dual_p_min) + abs(dual_p_max) if dual_p_min else 0.0

            dual_s_min = soc_con_list[2 * t].dual_value
            dual_s_max = soc_con_list[2 * t + 1].dual_value
            lambda_aging[t] = abs(dual_s_min) + abs(dual_s_max) if dual_s_min else 0.0

        return {
            "p_base": p_base.value,
            "c_reg": c_reg.value,
            "soc_plan": soc.value[:-1],
            "lambda_risk": lambda_risk,
            "lambda_aging": lambda_aging,
            "status": "optimal"
        }

    def solve_rolling(self, current_step, current_soc, original_plan, current_soh=1.0, current_rint_factor=1.0,
                      marginal_aging_cost=0.05):
        """
        Intra-Day Rolling Optimization (IRO)
        Includes Emergency Fallback if optimization fails.
        """
        T_full = self.cfg.horizon
        dt = self.cfg.dt_hours
        eff = self.cfg.efficiency
        steps_left_real = T_full - current_step
        if steps_left_real <= 0: return {}

        MIN_HORIZON = 48
        planning_horizon = max(steps_left_real, MIN_HORIZON)

        def _extend_array(arr, length):
            if len(arr) >= length: return arr[:length]
            needed = length - len(arr)
            pad = arr[:needed] if needed <= len(arr) else np.resize(arr, needed)
            return np.concatenate([arr, pad])

        c_reg_orig = _extend_array(original_plan['c_reg'][current_step:], planning_horizon)
        p_base_orig = _extend_array(original_plan['p_base'][current_step:], planning_horizon)

        # Variables
        p_base_new = cp.Variable(planning_horizon, name="p_base_new")
        c_reg_new = cp.Variable(planning_horizon, name="c_reg_new", nonneg=True)
        soc = cp.Variable(planning_horizon + 1, name="soc_rolling")

        constraints = []
        constraints.append(soc[0] == current_soc)
        constraints.append(c_reg_new <= c_reg_orig + 1e-5)  # Can't increase reg commitment

        safe_rated_power = self.cfg.rated_power_kw * 0.95

        for i in range(planning_horizon):
            # Physical Limits
            constraints.append(p_base_new[i] - c_reg_new[i] >= -safe_rated_power)
            constraints.append(p_base_new[i] + c_reg_new[i] <= safe_rated_power)

            # SOC Dynamics (with efficiency)
            constraints.append(soc[i + 1] == soc[i] - (p_base_new[i] * dt / self.cfg.rated_capacity_kwh) * (1.0 / eff))

            # Reserve constraints
            reserve_ratio = (c_reg_new[i] * self.cfg.reg_capacity_reserve_hours) / self.cfg.rated_capacity_kwh
            constraints.append(soc[i + 1] >= self.cfg.soc_min + reserve_ratio)
            constraints.append(soc[i + 1] <= self.cfg.soc_max - reserve_ratio)

        # Objectives
        deviation_loss = cp.sum_squares(p_base_new - p_base_orig) * 1.0
        aging_loss = cp.sum(cp.abs(p_base_new)) * dt * marginal_aging_cost * 10.0
        soc_loss = cp.sum_squares(soc - 0.5) * 0.01
        reg_reduction_loss = 100.0 * cp.sum(c_reg_orig - c_reg_new)

        prob = cp.Problem(cp.Minimize(deviation_loss + aging_loss + soc_loss + reg_reduction_loss), constraints)

        solve_success = False
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status == cp.OPTIMAL:
                solve_success = True
        except:
            pass

        if solve_success:
            return {
                "p_base": p_base_new.value[:steps_left_real],
                "soc_plan": soc.value[:steps_left_real],
                "c_reg": c_reg_new.value[:steps_left_real],
                "status": "optimal"
            }

        # [NEW] Emergency Fallback Mode
        # Rule: If SOC < min, Force Charge. If SOC > max, Force Discharge. Else, Standby (0 Power).
        # Cut Regulation to 0.

        emergency_p = np.zeros(steps_left_real)
        emergency_reg = np.zeros(steps_left_real)  # Drop all regulation obligations

        sim_soc = current_soc
        for i in range(steps_left_real):
            # Simple P-Controller for SOC recovery
            if sim_soc < self.cfg.soc_min:
                cmd = -safe_rated_power * 0.5  # Charge at 0.5C
            elif sim_soc > self.cfg.soc_max:
                cmd = safe_rated_power * 0.5  # Discharge at 0.5C
            else:
                cmd = 0.0  # Safety Idle

            emergency_p[i] = cmd
            # Update sim_soc for next step estimation
            sim_soc -= (cmd * dt / self.cfg.rated_capacity_kwh) * (1.0 / eff)

        return {
            "p_base": emergency_p,
            "c_reg": emergency_reg,
            "status": "emergency_fallback"
        }

    def update_aging_parameters(self, actual_aging_cost_daily: float, daily_throughput: float):
        if daily_throughput < 1.0: return
        realized_cost_per_kwh = actual_aging_cost_daily / daily_throughput
        alpha = 0.2
        current_param = self.cfg.marginal_aging_cost_default
        new_param = (1 - alpha) * current_param + alpha * realized_cost_per_kwh
        new_param = np.clip(new_param, 0.01, 0.5)
        self.cfg.marginal_aging_cost_default = new_param

    def _get_fallback_plan(self, T, current_soc):
        p_base = np.zeros(T)
        c_reg = np.zeros(T)
        soc_plan = np.zeros(T)

        sim_soc = float(current_soc)
        cap_kwh = self.cfg.rated_capacity_kwh
        dt = self.cfg.dt_hours
        max_p = self.cfg.rated_power_kw * 0.95
        eff = self.cfg.efficiency

        prices = self.last_price_energy_cache if self.last_price_energy_cache is not None else np.ones(T)
        median_p = np.median(prices)

        for t in range(T):
            intent_power = 0.0
            # Simple Arbitrage
            if prices[t] < median_p * 0.8:
                intent_power = -max_p * 0.5  # Conservative charge
            elif prices[t] > median_p * 1.2:
                intent_power = max_p * 0.5  # Conservative discharge

            # Safety check
            if sim_soc < self.cfg.soc_min and intent_power > 0: intent_power = 0.0
            if sim_soc > self.cfg.soc_max and intent_power < 0: intent_power = 0.0

            p_base[t] = intent_power
            sim_soc -= (intent_power * dt / cap_kwh) * (1.0 / eff)
            soc_plan[t] = sim_soc

        return {
            "p_base": p_base, "c_reg": c_reg, "soc_plan": soc_plan,
            "lambda_risk": np.ones(T) * 10.0,  # High risk signal
            "lambda_aging": np.ones(T) * 10.0,
            "status": "fallback_conservative"
        }
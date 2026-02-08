import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from stable_baselines3 import SAC

from DRL_market.common.config import load_config
from DRL_market.simulation.data_loader import MarketDataLoader
from DRL_market.market.scheduler import DayAheadScheduler, SchedulerConfig
from DRL_market.market.clearing import FrequencyMarketClearing, FrequencyBid, EnergyMarketResult
from DRL_market.market.mechanism import MarketEngine
from DRL_market.env.high_level import HighLevelEnv
from DRL_market.env.low_level import LowLevelEnv
from DRL_market.simulation.battery import BatteryPack
from baselines.base_direct_controller import BaseDirectController
from baselines.direct_rule import DirectRuleController
from baselines.direct_mpc import DirectMPCController


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def calculate_aux_load_kw(actual_power_kw: float) -> float:
    """
    对齐 LowLevelEnv 的简化 HVAC 模型:
    aux_load_kw = 0.5 + 0.01 * |P|
    """
    return 0.5 + 0.01 * abs(actual_power_kw)


def calculate_aging_cost_cny(
    loss_ah: float,
    battery_capex_cny: float,
    rated_capacity_kwh: float,
    nominal_voltage_v: float
) -> float:
    """
    老化成本按 capex / 总Ah 计价
    total_capacity_ah = rated_capacity_kwh*1000 / nominal_voltage_v
    cost_per_ah = capex / total_capacity_ah
    """
    total_capacity_ah = (rated_capacity_kwh * 1000.0) / max(nominal_voltage_v, 1e-6)
    cost_per_ah = battery_capex_cny / max(total_capacity_ah, 1e-6)
    return float(loss_ah * cost_per_ah)


@dataclass
class DirectAgentState:
    name: str
    battery: BatteryPack
    controller: Optional[BaseDirectController] = None
    sac_model: Optional[SAC] = None
    last_error_norm: float = 0.0
    prev_actual_power_kw: float = 0.0
    hourly_actual: List[float] = field(default_factory=list)
    hourly_target: List[float] = field(default_factory=list)
    hourly_mileage_kw: float = 0.0
    hourly_aging_ah: float = 0.0
    hourly_aux_cost_cny: float = 0.0
    hourly_max_temp: float = -999.0

    def reset_hourly(self):
        self.hourly_actual = []
        self.hourly_target = []
        self.hourly_mileage_kw = 0.0
        self.hourly_aging_ah = 0.0
        self.hourly_aux_cost_cny = 0.0
        self.hourly_max_temp = -999.0


@dataclass
class MHCState:
    name: str
    high_model: SAC
    low_model: Optional[SAC]
    high_env: HighLevelEnv
    low_env: LowLevelEnv
    prev_actual_power_kw: float = 0.0
    hourly_actual: List[float] = field(default_factory=list)
    hourly_target: List[float] = field(default_factory=list)
    hourly_mileage_kw: float = 0.0
    hourly_aging_ah: float = 0.0
    hourly_aux_cost_cny: float = 0.0
    hourly_max_temp: float = -999.0

    def reset_hourly(self):
        self.hourly_actual = []
        self.hourly_target = []
        self.hourly_mileage_kw = 0.0
        self.hourly_aging_ah = 0.0
        self.hourly_aux_cost_cny = 0.0
        self.hourly_max_temp = -999.0


def build_direct_obs(
    target_power_kw: float,
    battery: BatteryPack,
    rated_power_kw: float,
    last_error_norm: float
) -> np.ndarray:
    current_soc = battery.get_mean_soc()
    max_temp = float(np.max(battery.temps))
    mean_soh = float(np.mean(battery.sohs))

    obs_target = target_power_kw / max(rated_power_kw, 1e-6)
    obs_soc = (current_soc - 0.5) * 2.0
    obs_temp = np.clip((max_temp - 25.0) / 20.0, 0.0, 2.0)
    obs_soh = (mean_soh - 0.9) * 10.0
    obs_err = float(np.clip(last_error_norm, -1.0, 1.0))

    return np.array([obs_target, obs_soc, obs_temp, obs_soh, obs_err], dtype=np.float32)


def step_direct_agent(
    agent: DirectAgentState,
    grid_target_kw: float,
    rated_power_kw: float,
    max_power_kw: float,
    dt: float,
    max_current_a: float,
    dcdc_eff: float,
    current_price: float
) -> Tuple[float, float]:
    current_soc = agent.battery.get_mean_soc()

    # 评估曲线记录的是原始指令
    agent.hourly_target.append(grid_target_kw)

    # 安全熔断只作用于控制器输入
    cmd_target_kw = grid_target_kw
    if current_soc > 0.98:
        cmd_target_kw = max(0.0, cmd_target_kw)
    elif current_soc < 0.02:
        cmd_target_kw = min(0.0, cmd_target_kw)

    # 控制器决策
    if agent.sac_model is not None:
        obs = build_direct_obs(
            target_power_kw=cmd_target_kw,
            battery=agent.battery,
            rated_power_kw=rated_power_kw,
            last_error_norm=agent.last_error_norm
        )
        action, _ = agent.sac_model.predict(obs, deterministic=True)
        p_cmd_kw = float(np.clip(action[0], -1.0, 1.0)) * max_power_kw
    else:
        max_temp = float(np.max(agent.battery.temps))
        p_cmd_kw = agent.controller.get_action(
            current_soc=current_soc,
            current_max_temp=max_temp,
            target_power_kw=cmd_target_kw
        )

    num_clusters = agent.battery.n_clusters
    voltages = agent.battery.get_voltages()

    p_req_per_cluster = p_cmd_kw / num_clusters
    currents = []
    actual_cluster_powers = []

    for i in range(num_clusters):
        v = max(0.1, float(voltages[i]))

        if v < 2.5 * agent.battery.cells_per_cluster and p_req_per_cluster > 0:
            p_req = 0.0
        else:
            p_req = p_req_per_cluster

        if p_req > 0:
            p_cell_side = p_req / dcdc_eff
        else:
            p_cell_side = p_req * dcdc_eff

        curr = (p_cell_side * 1000.0) / v
        curr = float(np.clip(curr, -max_current_a, max_current_a))
        currents.append(curr)

        p_cell = (curr * v) / 1000.0
        if p_cell >= 0:
            p_actual = p_cell * dcdc_eff
        else:
            p_actual = p_cell / max(dcdc_eff, 1e-6)
        actual_cluster_powers.append(p_actual)

    phys_info = agent.battery.step(np.array(currents), dt)
    actual_power_kw = float(np.sum(actual_cluster_powers))

    # Mileage
    step_mileage = abs(actual_power_kw - agent.prev_actual_power_kw)
    agent.prev_actual_power_kw = actual_power_kw
    agent.hourly_mileage_kw += step_mileage

    # Aging (Ah)
    loss_ah_vec = phys_info.get("total_loss_ah", np.zeros(num_clusters))
    agent.hourly_aging_ah += float(np.sum(loss_ah_vec))

    # Aux cost (CNY)
    aux_kw = calculate_aux_load_kw(actual_power_kw)
    agent.hourly_aux_cost_cny += aux_kw * (dt / 3600.0) * current_price

    # Temp
    agent.hourly_max_temp = max(agent.hourly_max_temp, float(np.max(agent.battery.temps)))

    # Error: 用原始电网指令
    err_kw = abs(actual_power_kw - grid_target_kw)
    agent.last_error_norm = err_kw / max(rated_power_kw, 1e-6)

    agent.hourly_actual.append(actual_power_kw)

    return actual_power_kw, step_mileage


def parse_args():
    parser = argparse.ArgumentParser(description="Market Competition Experiment (MHC vs Direct Baselines)")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--demand_ratio", type=float, default=0.8)
    parser.add_argument("--floor_price", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    cfg = load_config(args.config)

    # Paths
    model_dir = PROJECT_ROOT / "models"
    high_model_path = model_dir / "high_level_trained_cmdp_sac.zip"
    low_model_path = model_dir / "low_level_pretrained.zip"
    direct_sac_path = model_dir / "baseline_direct_sac.zip"

    if not high_model_path.exists():
        raise FileNotFoundError(f"MHC high-level model missing: {high_model_path}")
    if not low_model_path.exists():
        raise FileNotFoundError(f"MHC low-level model missing: {low_model_path}")
    if not direct_sac_path.exists():
        raise FileNotFoundError(f"Direct SAC baseline missing: {direct_sac_path}")

    # Load models
    high_model = SAC.load(high_model_path, device="cpu")
    low_model = SAC.load(low_model_path, device="cpu")
    direct_sac = SAC.load(direct_sac_path, device="cpu")

    # Config shortcuts
    market_cfg = cfg.get("market", {})
    env_cfg = cfg.get("env") or cfg.get("environment")
    rated_power_kw = float(market_cfg.get("rated_power_kw", 2580.0))
    rated_capacity_kwh = float(market_cfg.get("rated_capacity_kwh", 5160.0))
    battery_capex_cny = float(market_cfg.get("battery_capex_cny", 3.6e6))
    nominal_voltage_v = float(env_cfg.get("nominal_voltage_v", 768.0))

    dt = float(env_cfg.get("low_level_step_seconds", 1.0))
    max_current_a = float(env_cfg.get("max_current_a", 500.0))
    dcdc_eff = float(env_cfg.get("dcdc_efficiency", 0.97))

    power_max = float(env_cfg.get("limits", {}).get("power_max", 1.2))
    max_power_kw = rated_power_kw * power_max

    interval_s = int(env_cfg.get("high_level_step_minutes", 15) * 60)
    steps_per_hour = int(3600 / interval_s)
    steps_per_day = int(24 * 3600 / interval_s)

    floor_price = args.floor_price
    if floor_price is None:
        floor_price = float(market_cfg.get("price_floor", 3.5))

    # Market components
    data_loader = MarketDataLoader(cfg)
    sched_cfg = SchedulerConfig(
        rated_power_kw=rated_power_kw,
        rated_capacity_kwh=rated_capacity_kwh,
        dt_hours=interval_s / 3600.0,
        horizon=steps_per_day
    )
    scheduler = DayAheadScheduler(sched_cfg)
    clearing_engine = FrequencyMarketClearing(market_cfg=market_cfg)
    market_engine = MarketEngine(cfg)

    # MHC setup
    mhc_env = HighLevelEnv(cfg, is_eval=True, low_model=low_model, training_mode=False)
    mhc_state = MHCState(
        name="MHC",
        high_model=high_model,
        low_model=low_model,
        high_env=mhc_env,
        low_env=mhc_env.low_env
    )

    # Direct baselines
    direct_agents: Dict[str, DirectAgentState] = {
        "SAC": DirectAgentState(name="SAC", battery=BatteryPack(cfg), sac_model=direct_sac),
        "MPC": DirectAgentState(name="MPC", battery=BatteryPack(cfg), controller=DirectMPCController(cfg, dt_seconds=dt)),
        "Rule": DirectAgentState(name="Rule", battery=BatteryPack(cfg), controller=DirectRuleController(cfg)),
    }

    results = []

    for day in range(args.days):
        price_series, agc_series = data_loader.sample_day_data()
        forecast_prices = data_loader.get_forecast_with_noise(price_series, noise_level=0.05)
        forecast_reg_prices = np.ones(steps_per_day, dtype=np.float32) * float(
            market_cfg.get("reg_mileage_bid_price", 6.0)
        )

        # Reset physics
        mhc_state.low_env.reset(options={"reset_physics": True})
        for agent in direct_agents.values():
            agent.battery.reset()
            agent.last_error_norm = 0.0
            agent.prev_actual_power_kw = 0.0
            agent.hourly_actual = []
            agent.hourly_target = []

        for agent in direct_agents.values():
            batt = agent.battery
            batt.voltages[:] = batt.nominal_voltage_v
            batt.currents[:] = 0.0

        target_init_soc = 0.5
        target_init_temp = 25.0
        target_init_soh = 1.0

        mhc_batt = mhc_state.low_env.battery
        mhc_batt.voltages[:] = mhc_batt.nominal_voltage_v
        mhc_batt.currents[:] = 0.0
        mhc_batt.socs[:] = target_init_soc
        mhc_batt.temps[:] = target_init_temp
        mhc_batt.sohs[:] = target_init_soh
        for cell in mhc_batt.cells:
            cell.reset(initial_soc=target_init_soc)

        for agent in direct_agents.values():
            batt = agent.battery
            batt.socs[:] = target_init_soc
            batt.temps[:] = target_init_temp
            batt.sohs[:] = target_init_soh
            for cell in batt.cells:
                cell.reset(initial_soc=target_init_soc)

        init_soc = target_init_soc

        plan = scheduler.solve(
            price_energy=forecast_prices,
            price_reg=forecast_reg_prices,
            initial_soc=init_soc
        )

        mhc_state.high_env.current_da_plan = plan
        mhc_state.high_env.current_price_series = price_series
        mhc_state.high_env.current_agc_series = agc_series
        current_unified_price = 0.0

        k_history = {"MHC": 1.0, "SAC": 1.0, "MPC": 1.0, "Rule": 1.0}

        for step_idx in range(steps_per_day):
            hour_idx = step_idx // steps_per_hour

            # Hourly clearing
            if step_idx % steps_per_hour == 0:
                for agent in direct_agents.values():
                    agent.reset_hourly()
                mhc_state.reset_hourly()

                plan_p_base = float(plan["p_base"][step_idx])
                plan_reg_bid = float(plan["c_reg"][step_idx])

                bids = []
                energy_results = {}
                soc_snapshot = {}

                for name in ["MHC", "SAC", "MPC", "Rule"]:
                    bids.append(
                        FrequencyBid(
                            unit_id=name,
                            time_period=hour_idx,
                            unit_type="storage",
                            area="GD",
                            capacity_mw=plan_reg_bid / 1000.0,
                            mileage_price=floor_price,
                            k_value=k_history[name],
                            is_independent_storage=True,
                            rated_power_mw=rated_power_kw / 1000.0,
                            rated_capacity_mwh=rated_capacity_kwh / 1000.0,
                        )
                    )

                    energy_results[name] = EnergyMarketResult(
                        time_period=hour_idx,
                        base_power_mw=plan_p_base / 1000.0,
                        energy_price=0.0
                    )

                soc_snapshot["MHC"] = float(mhc_state.low_env.battery.get_mean_soc())
                for name, agent in direct_agents.items():
                    soc_snapshot[name] = float(agent.battery.get_mean_soc())

                total_cap = plan_reg_bid / 1000.0 * len(bids)
                system_demand = total_cap * args.demand_ratio

                clearing = clearing_engine.clear_frequency_market(
                    bids=bids,
                    system_demand_mw=system_demand,
                    energy_results=energy_results,
                    soc_snapshot=soc_snapshot,
                    area_demands={"GD": system_demand},
                    area_min_ratios={"GD": 0.0}
                )

                current_unified_price = clearing.unified_price
                cleared_caps = {u.unit_id: u.cleared_capacity_mw for u in clearing.cleared_units}
                for name in ["MHC", "SAC", "MPC", "Rule"]:
                    if name not in cleared_caps:
                        cleared_caps[name] = 0.0

            # Execution
            plan_p_base = float(plan["p_base"][step_idx])
            cleared_cap_kw = {k: v * 1000.0 for k, v in cleared_caps.items()}

            mhc_state.high_env.current_step = step_idx
            obs = mhc_state.high_env._get_obs()
            mhc_action, _ = mhc_state.high_model.predict(obs, deterministic=True)

            hierarchy_cfg = cfg.get("hierarchy", {})
            lambda_max = float(hierarchy_cfg.get("lambda_max", 3.0))
            n_clusters = int(env_cfg.get("n_clusters", 24))

            shadow_prices = (mhc_action[:n_clusters] + 1.0) / 2.0 * lambda_max
            p_real = float(mhc_action[n_clusters] * max_power_kw)

            start_idx = step_idx * interval_s
            end_idx = start_idx + interval_s
            agc_segment = agc_series[start_idx:end_idx]

            current_price = float(price_series[step_idx])

            for agc_val in agc_segment:
                # --- MHC ---
                if cleared_cap_kw["MHC"] > 0:
                    mean_soc = float(mhc_state.low_env.battery.get_mean_soc())
                    if mean_soc > 0.98:
                        p_real = max(0.0, p_real)
                        phys_reg_cap = 0.0
                    elif mean_soc < 0.02:
                        p_real = min(0.0, p_real)
                        phys_reg_cap = 0.0
                    else:
                        phys_reg_cap = cleared_cap_kw["MHC"]

                    target_kw = plan_p_base + agc_val * cleared_cap_kw["MHC"]
                    agent_req_kw = p_real + agc_val * phys_reg_cap

                    if mhc_state.low_model:
                        ll_obs = mhc_state.low_env._get_obs(agent_req_kw)
                        ll_action, _ = mhc_state.low_model.predict(ll_obs, deterministic=True)
                    else:
                        ll_action = np.zeros(n_clusters)

                    _, _, _, _, info_ll = mhc_state.low_env.step(
                        action=ll_action,
                        power_request_kw=agent_req_kw,
                        shadow_prices=shadow_prices
                    )

                    actual_p = float(info_ll.get("actual_power_kw", 0.0))
                    step_mileage = float(info_ll.get("step_mileage_kw", 0.0))
                    mhc_state.hourly_mileage_kw += step_mileage
                    mhc_state.hourly_aging_ah += float(info_ll.get("aging_cost_real", 0.0))
                    mhc_state.hourly_max_temp = max(
                        mhc_state.hourly_max_temp,
                        float(np.max(mhc_state.low_env.battery.temps))
                    )
                    mhc_state.hourly_actual.append(actual_p)
                    mhc_state.hourly_target.append(target_kw)

                    aux_kw = float(info_ll.get("aux_load_kw", calculate_aux_load_kw(actual_p)))
                    mhc_state.hourly_aux_cost_cny += aux_kw * (dt / 3600.0) * current_price

                # --- Direct agents ---
                for name, agent in direct_agents.items():
                    if cleared_cap_kw[name] <= 0:
                        continue
                    target_kw = plan_p_base + agc_val * cleared_cap_kw[name]
                    step_direct_agent(
                        agent=agent,
                        grid_target_kw=target_kw,
                        rated_power_kw=rated_power_kw,
                        max_power_kw=max_power_kw,
                        dt=dt,
                        max_current_a=max_current_a,
                        dcdc_eff=dcdc_eff,
                        current_price=current_price
                    )

            # Hourly settlement
            if (step_idx + 1) % steps_per_hour == 0:
                hour_start = hour_idx * steps_per_hour
                hour_end = hour_start + steps_per_hour
                avg_hour_price = float(np.mean(price_series[hour_start:hour_end]))
                hour_plan_base = float(np.mean(plan["p_base"][hour_start:hour_end]))

                capacity_penalty_ratio = float(market_cfg.get("capacity_penalty_ratio", 2.0))
                k_threshold_cap = float(market_cfg.get("k_threshold_cap", 1.8))
                temp_limit_cap = float(env_cfg.get("limits", {}).get("temp_max", 45.0))

                for agent_name, agent_obj in [("MHC", mhc_state)] + list(direct_agents.items()):
                    if cleared_cap_kw[agent_name] <= 0:
                        results.append({
                            "day": day,
                            "hour": hour_idx,
                            "agent": agent_name,
                            "cleared_cap_mw": 0.0,
                            "k_value": k_history[agent_name],
                            "m_value": 0.0,
                            "mileage_mw": 0.0,
                            "revenue_energy": 0.0,
                            "revenue_reg": 0.0,
                            "revenue_capacity": 0.0,
                            "penalty": 0.0,
                            "aging_loss_ah": 0.0,
                            "aging_cost_cny": 0.0,
                            "aux_cost_cny": 0.0,
                            "net_profit": 0.0
                        })
                        continue

                    actual_curve_kw = np.array(agent_obj.hourly_actual, dtype=float)
                    target_curve_kw = np.array(agent_obj.hourly_target, dtype=float)

                    metrics = market_engine.calculate_metrics_from_series(
                        actual_p=actual_curve_kw / 1000.0,
                        instruct_p=target_curve_kw / 1000.0
                    )
                    k_value = market_engine.calculate_sorting_performance_k(
                        metrics["speed"], metrics["delay"], metrics["error"]
                    )
                    m_value = market_engine.calculate_performance_m(
                        metrics["speed"], metrics["delay"], metrics["error"]
                    )

                    k_history[agent_name] = k_value
                    m_coeff = m_value if k_value >= market_cfg.get("k_threshold", 2.5) else 0.0

                    pass_k = k_value >= k_threshold_cap
                    pass_t = agent_obj.hourly_max_temp < temp_limit_cap
                    capacity_availability = 1.0 if (pass_k and pass_t) else -capacity_penalty_ratio

                    avg_hour_price_kwh = float(np.mean(price_series[hour_start:hour_end]))
                    avg_hour_price_mwh = avg_hour_price_kwh * 1000.0

                    settlement = market_engine.calculate_settlement(
                        target_power_kw=hour_plan_base,
                        actual_power_kw=float(np.mean(actual_curve_kw)),
                        price=avg_hour_price_mwh,
                        is_reg_mode=(cleared_cap_kw[agent_name] > 0.1),
                        reg_capacity_kw=cleared_cap_kw[agent_name],
                        accumulated_mileage_kw=agent_obj.hourly_mileage_kw,
                        price_reg_mileage=current_unified_price,  # ✅ 用缓存变量
                        duration_hours=1.0,
                        m_coeff=m_coeff,
                        capacity_availability=capacity_availability,
                        installed_capacity_mw=rated_power_kw / 1000.0
                    )

                    aging_cost_cny = calculate_aging_cost_cny(
                        loss_ah=agent_obj.hourly_aging_ah,
                        battery_capex_cny=battery_capex_cny,
                        rated_capacity_kwh=rated_capacity_kwh,
                        nominal_voltage_v=nominal_voltage_v
                    )

                    net_profit = (
                        settlement["revenue_energy"]
                        + settlement["revenue_regulation"]
                        + settlement["revenue_capacity"]
                        - settlement["penalty"]
                        - aging_cost_cny
                        - agent_obj.hourly_aux_cost_cny
                    )

                    results.append({
                        "day": day,
                        "hour": hour_idx,
                        "agent": agent_name,
                        "cleared_cap_mw": cleared_cap_kw[agent_name] / 1000.0,
                        "k_value": k_value,
                        "m_value": m_value,
                        "mileage_mw": agent_obj.hourly_mileage_kw / 1000.0,
                        "revenue_energy": settlement["revenue_energy"],
                        "revenue_reg": settlement["revenue_regulation"],
                        "revenue_capacity": settlement["revenue_capacity"],
                        "penalty": settlement["penalty"],
                        "aging_loss_ah": agent_obj.hourly_aging_ah,
                        "aging_cost_cny": aging_cost_cny,
                        "aux_cost_cny": agent_obj.hourly_aux_cost_cny,
                        "net_profit": net_profit
                    })

    # 保存结果
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "market_competition_results.csv"

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"✅ Competition finished. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import numpy as np
import torch
import sys
import argparse
from pathlib import Path
from stable_baselines3 import SAC, PPO  # <--- Added PPO

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from common.config import FinalRLBMSConfig
from market.scheduler import DayAheadScheduler, SchedulerConfig
from market.mechanism import MarketConfig
from env.physics_wrapper import MarketPhysicsKernel
from env.high_level import HighLevelEnv
from env.low_level import LowLevelEnv
from env.interface import GoalInterface
from baselines.controllers.rule_based import RuleBasedController
from baselines.controllers.mpc import MPCController

PRICE_UNIT_DIVISOR = 1.0


# Helper to save experiment data for plotting
class ExperimentDataCollector:
    def __init__(self, agent_name):
        self.data = {
            "day": [],
            "profit_daily": [],
            "profit_cumulative": [],
            "soh": [],
            "agent_type": agent_name
        }
        self.cumulative_profit = 0.0

    def add(self, day, profit, soh):
        self.cumulative_profit += profit
        self.data["day"].append(day)
        self.data["profit_daily"].append(profit)
        self.data["profit_cumulative"].append(self.cumulative_profit)
        self.data["soh"].append(soh)

    def save(self, path):
        np.savez(path, **self.data)


class BaselineWrapper:
    def __init__(self, controller, env_unwrapped, env_type=None):
        self.controller = controller
        self.env = env_unwrapped
        self.env_type = env_type

    def predict(self, obs, deterministic=True):
        # Extract info for Rule/MPC
        target = self.env.unwrapped.low_env.target_power
        socs = self.env.unwrapped.low_env.battery.get_SoC_values()
        temps = [t - 273.15 for t in self.env.unwrapped.low_env.battery.get_temp_values()]

        # We need current price. Accessing via wrapper is tricky,
        # assume passed via set_daily_plan or just mean.
        info = {
            "TargetPower": target,
            "SoC": socs,
            "Temperature": temps,
            "price_energy": 0.5  # Default if not accessible
        }
        return self.controller.predict(obs, deterministic=True, info=info)


def load_data_horizon(data_path, day_idx, horizon=96):
    try:
        data = np.load(data_path)
        total_seconds = len(data['price'])
        total_days_available = total_seconds // 86400
        mapped_day = day_idx % total_days_available
        start_sec = mapped_day * 86400
        indices = np.linspace(start_sec, start_sec + 86400, horizon, endpoint=False, dtype=int)
        indices = np.clip(indices, 0, total_seconds - 1)
        lmp = data['price'][indices]
        agc = data['agc'][indices]
        reg_cap = lmp * 10.0 + 5.0
        reg_mil = lmp * 8.0 + 2.0
        return lmp, reg_cap, reg_mil, agc
    except Exception as e:
        print(f"⚠️ Failed to load .npz: {e}")
        return np.ones(horizon) * 0.5, np.ones(horizon) * 6.0, np.ones(horizon) * 8.0, np.zeros(horizon)


def run_market_simulation(agent_name, days, config_path=None, data_path=None, silent=False):
    """
    Runs the market simulation for a specific agent and returns aggregated metrics.
    Used by test_market.py (main) and exp_radar_chart.py.
    """
    if config_path is None: config_path = PROJECT_ROOT / "config.yaml"
    if data_path is None: data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"

    cfg = FinalRLBMSConfig(str(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sched_cfg = SchedulerConfig(rated_power_kw=24.0)
    scheduler = DayAheadScheduler(sched_cfg)
    gi = GoalInterface()

    def make_low_env():
        kwargs = cfg.get_env_kwargs()
        kwargs['subset'] = 'test'
        kwargs['external_power_profile'] = None
        kwargs['randomization'] = False
        return LowLevelEnv(kwargs, gi)

    low_path = PROJECT_ROOT / "models/low_level_pretrained.zip"
    high_path = PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip"
    ppo_path = PROJECT_ROOT / "models/baseline_flat_ppo.zip"

    low_model = SAC.load(low_path, device=device) if low_path.exists() else None
    high_model = None

    # --- Model Loading Strategy ---
    if agent_name == "RL" or agent_name == "MHC":
        if high_path.exists():
            if not silent: print(f"✅ Loading MHC (SAC) from {high_path}")
            high_model = SAC.load(high_path, device=device)
        else:
            raise FileNotFoundError("MHC High-Level model not found.")

    elif agent_name == "PPO":
        if ppo_path.exists():
            if not silent: print(f"✅ Loading Baseline (Flat SAC) from {ppo_path}")
            # [Fix] The file is named PPO but contains a SAC model
            high_model = SAC.load(ppo_path, device=device)
        else:
            raise FileNotFoundError("Baseline model (PPO/SAC) not found.")

    high_env = HighLevelEnv(make_low_env, interval=900, goal_interface=gi, low_model=low_model,
                            reward_weights=cfg.reward_high.weights.to_dict(), training_mode=False)

    env = MarketPhysicsKernel(high_env, market_config=MarketConfig(), obs_mode="transparent", scheduler=scheduler)

    # --- Controller Wrapping ---
    controller = None
    if agent_name in ["RL", "MHC", "PPO"]:
        controller = high_model
    elif agent_name == "Rule":
        controller = BaselineWrapper(RuleBasedController(), env)
    elif agent_name == "MPC":
        controller = BaselineWrapper(MPCController(), env)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # --- Simulation Loop ---
    current_soh = 1.0
    current_rint = 1.0
    current_marginal_aging_cost = cfg.market.scheduler_marginal_aging_cost or 0.15

    metrics = {
        "revenue_en": 0.0, "revenue_reg": 0.0, "penalty": 0.0,
        "aging_cost": 0.0, "hvac_cost": 0.0, "net_profit": 0.0,
        "violations": 0, "final_soh": 1.0
    }

    collector = ExperimentDataCollector(agent_name)

    if not silent:
        print("\n🔄 Clean Start Protocol...")
        print(f"{'=' * 105}")
        print(
            f"{'Day':<4} | {'Rev(Eng)':<10} | {'Rev(Reg)':<10} | {'Penalty':<10} | {'Aging':<10} | {'HVAC':<10} | {'Net Profit':<12} | {'Stress':<6} | {'SOH':<6}")
        print(f"{'-' * 105}")

    env.reset(options={'reset_physics': True})

    # Warmup
    for _ in range(50): env.env.unwrapped.low_env.step(np.zeros(25, dtype=np.float32))
    try:
        env.env.unwrapped.low_env.battery._initialize_cells(manual_socs=[0.5] * 24, manual_temps=[298.15] * 24)
    except:
        pass

    for day in range(days):
        lmp_real, p_cap, p_mil, agc = load_data_horizon(data_path, day)

        # Forecast creation
        noise = np.random.normal(1.0, 0.05, size=lmp_real.shape)
        lmp_forecast = np.maximum(0.01, lmp_real * noise)

        agc_forecast_intensity = np.zeros_like(agc)
        steps_per_hour = 4
        for h in range(24):
            s = h * steps_per_hour;
            e = (h + 1) * steps_per_hour
            mean_val = np.mean(np.abs(agc[s:e]))
            agc_forecast_intensity[s:e] = mean_val

        soc_now = np.mean(env.env.unwrapped.low_env.battery.get_SoC_values())

        # Solve Scheduler
        plan = scheduler.solve(
            price_energy=lmp_forecast, price_reg=p_mil / PRICE_UNIT_DIVISOR,
            agc_intensity=agc_forecast_intensity,
            current_soh=current_soh, current_rint_factor=current_rint,
            marginal_aging_cost=current_marginal_aging_cost, initial_soc=soc_now
        )

        env.set_daily_plan(plan, agc, {'energy': lmp_real, 'reg_cap': p_cap, 'reg_mil': p_mil})

        if day > 0:
            obs, _ = env.reset(options={'reset_physics': False})
        else:
            # First day init trick
            high_env = env.env.unwrapped
            high_env.metrics.reset_window()
            dummy_summary = high_env.metrics.finalize_window()
            dummy_summary['soc_mean'] = np.mean(high_env.low_env.battery.get_SoC_values())
            aging_info = high_env.low_env.battery.get_latest_aging_info()
            obs = high_env._compose_features(dummy_summary, aging_info, None, high_env.proxy_goal_last,
                                             high_env.proxy_goal_last, cooling_active=0.0)
            obs = high_env.update_obs_with_current_lambdas(np.array(obs, dtype=np.float32))

        # Daily stats
        d_stats = {k: 0.0 for k in metrics if k != "final_soh"}
        stress_accum = 0.0
        d_throughput = 0.0

        for t in range(96):
            action, _ = controller.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)

            d_stats["revenue_en"] += info.get('revenue_energy', 0)
            d_stats["revenue_reg"] += info.get('revenue_reg', 0)
            d_stats["penalty"] += info.get('penalty', 0)
            d_stats["aging_cost"] += info.get('cost_aging_realized', 0)
            d_stats["hvac_cost"] += info.get('cost_hvac_realized', 0)
            d_stats["net_profit"] += info.get('true_net_profit', 0)

            if info.get('violation_flag', 0) > 0:
                d_stats["violations"] += 1

            stress_accum += info.get('avg_stress', 1.0)
            if 'TotalOutputPower' in info: d_throughput += abs(info['TotalOutputPower']) * 0.25

            if 'aging_info' in info:
                current_soh = info['aging_info']['soh_estimate']
                current_rint = info['aging_info']['rint_factor']

        # Update marginal cost
        if d_throughput > 1.0:
            implied_cost = d_stats["aging_cost"] / d_throughput
            alpha = 0.3
            current_marginal_aging_cost = (1 - alpha) * current_marginal_aging_cost + alpha * implied_cost
            current_marginal_aging_cost = np.clip(current_marginal_aging_cost, 0.01, 0.5)

        # Accumulate to global metrics
        for k in d_stats:
            metrics[k] += d_stats[k]
        metrics["final_soh"] = current_soh

        collector.add(day + 1, d_stats["net_profit"], current_soh)

        if not silent:
            avg_stress = stress_accum / 96.0
            print(f"{day + 1:<4} | {d_stats['revenue_en']:>10.2f} | {d_stats['revenue_reg']:>10.2f} | "
                  f"{d_stats['penalty']:>10.2f} | {d_stats['aging_cost']:>10.2f} | "
                  f"{d_stats['hvac_cost']:>10.2f} | {d_stats['net_profit']:>12.2f} | "
                  f"{avg_stress:>6.2f} | {current_soh:.4f}")

    # Save collector
    save_dir = PROJECT_ROOT / "results" / "experiment_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    collector.save(save_dir / f"data_{agent_name}.npz")
    env.close()

    return metrics

def main():
    parser = argparse.ArgumentParser()
    # [MODIFIED] Added PPO to choices
    parser.add_argument("--agent", type=str, default="RL", choices=["RL", "PPO", "Rule", "MPC"], help="Agent to test")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    args = parser.parse_args()

    print(f"🚀 [GD-Market] Starting Simulation for Agent: {args.agent} ...")
    config_path = PROJECT_ROOT / "config.yaml"
    cfg = FinalRLBMSConfig(str(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sched_cfg = SchedulerConfig(rated_power_kw=24.0)
    scheduler = DayAheadScheduler(sched_cfg)
    gi = GoalInterface()

    def make_low_env():
        kwargs = cfg.get_env_kwargs()
        kwargs['subset'] = 'test'
        kwargs['external_power_profile'] = None
        kwargs['randomization'] = False
        return LowLevelEnv(kwargs, gi)

    low_path = PROJECT_ROOT / "models/low_level_pretrained.zip"
    high_path = PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip"
    ppo_path = PROJECT_ROOT / "models/baseline_flat_sac.zip"  # [NEW]

    low_model = SAC.load(low_path, device=device) if low_path.exists() else None

    # Initialize High Level Model based on agent type
    high_model = None

    # [MODIFIED] Model Loading Logic
    if args.agent == "RL":
        if high_path.exists():
            print(f"✅ Loading MHC (SAC) from {high_path}")
            high_model = SAC.load(high_path, device=device)
        else:
            print("❌ RL High-Level model not found! Run training first.")
            return
    elif args.agent == "PPO":
        if ppo_path.exists():
            print(f"✅ Loading Baseline (Flat SAC) from {ppo_path}")
            # [CRITICAL FIX] 您的 Baseline 实际上是用 SAC 训练的 (run_baseline_training.py line 166)
            # 即使文件名叫 ppo.zip，内容也是 SAC。必须用 SAC.load
            try:
                high_model = SAC.load(ppo_path, device=device)
            except Exception:
                # 以此防万一您真的换成了 PPO
                print("⚠️ SAC load failed, trying PPO...")
                high_model = PPO.load(ppo_path, device=device)
        else:
            print("❌ Baseline model not found! Run run_baseline_training.py first.")
            return

    # Use HighLevelEnv as the unified container
    # Note: Even for PPO, we use HighLevelEnv. PPO was trained on the same Observation Space.
    # The 'hiro_cfg' passed here might enable HIRO internally in Env, but PPO ignores Goal parts.
    high_env = HighLevelEnv(make_low_env, interval=900, goal_interface=gi, low_model=low_model,
                            reward_weights=cfg.reward_high.weights.to_dict(), training_mode=False)

    env = MarketPhysicsKernel(high_env, market_config=MarketConfig(), obs_mode="transparent", scheduler=scheduler)

    # Wrap Controller
    controller = None
    if args.agent in ["RL", "PPO"]:  # [MODIFIED] PPO shares interface with SAC
        controller = high_model
    elif args.agent == "Rule":
        controller = BaselineWrapper(RuleBasedController(), env)
    elif args.agent == "MPC":
        controller = BaselineWrapper(MPCController(), env)

    current_soh = 1.0
    current_rint = 1.0
    current_marginal_aging_cost = cfg.market.scheduler_marginal_aging_cost or 0.05

    collector = ExperimentDataCollector(args.agent)
    data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"

    total_stats = {"revenue_en": 0.0, "revenue_reg": 0.0, "penalty": 0.0, "aging_cost": 0.0, "hvac_cost": 0.0,
                   "net_profit": 0.0}

    print("\n🔄 Clean Start Protocol...")
    env.reset(options={'reset_physics': True})
    for _ in range(50): env.env.unwrapped.low_env.step(np.zeros(25, dtype=np.float32))

    try:
        batt = env.env.unwrapped.low_env.battery
        batt._initialize_cells(manual_socs=[0.5] * 24, manual_temps=[298.15] * 24)
    except:
        pass

    print(f"\n{'=' * 105}")
    print(
        f"{'Day':<4} | {'Rev(Eng)':<10} | {'Rev(Reg)':<10} | {'Penalty':<10} | {'Aging':<10} | {'HVAC':<10} | {'Net Profit':<12} | {'Stress':<6} | {'SOH':<6}")
    print(f"{'-' * 105}")

    for day in range(args.days):
        lmp_real, p_cap, p_mil, agc = load_data_horizon(data_path, day)

        # [CRITICAL] Create Forecasts (No Peeking)
        noise = np.random.normal(1.0, 0.05, size=lmp_real.shape)
        lmp_forecast = np.maximum(0.01, lmp_real * noise)

        # AGC Forecast: Only average intensity per hour
        agc_forecast_intensity = np.zeros_like(agc)
        steps_per_hour = 4
        for h in range(24):
            s = h * steps_per_hour
            e = (h + 1) * steps_per_hour
            mean_val = np.mean(np.abs(agc[s:e]))
            agc_forecast_intensity[s:e] = mean_val

        soc_now = np.mean(env.env.unwrapped.low_env.battery.get_SoC_values())

        # Scheduler solves based on FORECAST
        plan = scheduler.solve(
            price_energy=lmp_forecast, price_reg=p_mil / PRICE_UNIT_DIVISOR,
            agc_intensity=agc_forecast_intensity,  # Use forecast
            current_soh=current_soh, current_rint_factor=current_rint,
            marginal_aging_cost=current_marginal_aging_cost, initial_soc=soc_now
        )

        # Env executes based on REAL
        env.set_daily_plan(plan, agc, {'energy': lmp_real, 'reg_cap': p_cap, 'reg_mil': p_mil})

        if day > 0:
            obs, _ = env.reset(options={'reset_physics': False})
        else:
            high_env = env.env.unwrapped
            high_env.metrics.reset_window()
            # Initial Observation construction
            dummy_summary = high_env.metrics.finalize_window()
            dummy_summary['soc_mean'] = np.mean(high_env.low_env.battery.get_SoC_values())
            aging_info = high_env.low_env.battery.get_latest_aging_info()
            obs = high_env._compose_features(dummy_summary, aging_info, None, high_env.proxy_goal_last,
                                             high_env.proxy_goal_last, cooling_active=0.0)
            obs = high_env.update_obs_with_current_lambdas(np.array(obs, dtype=np.float32))

        d_rev_en = 0.0;
        d_rev_reg = 0.0;
        d_pen = 0.0;
        d_aging = 0.0;
        d_hvac = 0.0;
        d_net = 0.0;
        d_throughput = 0.0;
        stress_accum = 0.0

        for t in range(96):
            action, _ = controller.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)

            d_rev_en += info.get('revenue_energy', 0)
            d_rev_reg += info.get('revenue_reg', 0)
            d_pen += info.get('penalty', 0)
            d_aging += info.get('cost_aging_realized', 0)
            d_hvac += info.get('cost_hvac_realized', 0)
            d_net += info.get('true_net_profit', 0)
            stress_accum += info.get('avg_stress', 1.0)

            if 'TotalOutputPower' in info: d_throughput += abs(info['TotalOutputPower']) * 0.25
            if 'aging_info' in info:
                current_soh = info['aging_info']['soh_estimate']
                current_rint = info['aging_info']['rint_factor']

        if d_throughput > 1.0:
            implied_cost = d_aging / d_throughput
            alpha = 0.3
            current_marginal_aging_cost = (1 - alpha) * current_marginal_aging_cost + alpha * implied_cost
            current_marginal_aging_cost = np.clip(current_marginal_aging_cost, 0.01, 0.5)

        avg_stress = stress_accum / 96.0

        print(
            f"{day + 1:<4} | {d_rev_en:>10.2f} | {d_rev_reg:>10.2f} | {d_pen:>10.2f} | {d_aging:>10.2f} | {d_hvac:>10.2f} | {d_net:>12.2f} | {avg_stress:>6.2f} | {current_soh:.4f}")

        total_stats["revenue_en"] += d_rev_en
        total_stats["revenue_reg"] += d_rev_reg
        total_stats["penalty"] += d_pen
        total_stats["aging_cost"] += d_aging
        total_stats["hvac_cost"] += d_hvac
        total_stats["net_profit"] += d_net

        collector.add(day + 1, d_net, current_soh)

    print(f"{'=' * 105}")
    print(f"🏆 FINAL SUMMARY ({args.agent})")
    print(f"   (+) Total Rev (Energy): {total_stats['revenue_en']:>10.2f}")
    print(f"   (+) Total Rev (Reg):    {total_stats['revenue_reg']:>10.2f}")
    print(f"   (-) Total Penalty:      {total_stats['penalty']:>10.2f}")
    print(f"   (-) Real Aging Cost:    {total_stats['aging_cost']:>10.2f}")
    print(f"   (-) HVAC Energy Cost:   {total_stats['hvac_cost']:>10.2f}")
    print(f"   (=) NET PROFIT:         {total_stats['net_profit']:>10.2f} CNY")
    print(f"{'=' * 105}")

    save_dir = PROJECT_ROOT / "results" / "experiment_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    collector.save(save_dir / f"data_{args.agent}.npz")
    env.close()


if __name__ == "__main__":
    main()
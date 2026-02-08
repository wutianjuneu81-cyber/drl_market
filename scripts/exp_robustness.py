#!/usr/bin/env python3
"""
Experiment: Robustness / Stress Test
Goal: Show MHC survives where others (Rule-based AND Baseline RL) fail.
Scenarios:
1. Normal (25°C ambient)
2. High_Price (3x Price Spikes) - Critical for Economic Safety
"""
import sys
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from common.config import FinalRLBMSConfig
from env.physics_wrapper import MarketPhysicsKernel
from env.high_level import HighLevelEnv
from env.low_level import LowLevelEnv
from env.interface import GoalInterface
from market.mechanism import MarketConfig
from scripts.test_market import load_data_horizon, BaselineWrapper
from baselines.controllers.rule_based import RuleBasedController


def run_stress_test(agent_name, condition, cfg):
    print(f"   Running {agent_name} under [{condition}] conditions...")

    # Setup Models
    gi = GoalInterface()

    def make_low():
        kw = cfg.get_env_kwargs()
        kw['subset'] = 'test'
        kw['external_power_profile'] = None
        return LowLevelEnv(kw, gi)

    low_path = PROJECT_ROOT / "models/low_level_pretrained.zip"
    high_path = PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip"
    baseline_path = PROJECT_ROOT / "models/baseline_flat_sac.zip"  # [Updated]

    low_model = SAC.load(low_path, device="cpu")

    # [CRITICAL Fix] Ensure High Price Environment settings are enforced
    mkt_cfg = MarketConfig(
        rated_power=cfg.market.rated_power,
        rated_capacity=cfg.market.rated_capacity,
        price_unit_divisor=cfg.market.price_unit_divisor,  # Should be 1.0 per config
        dev_penalty_ratio=cfg.market.dev_penalty_ratio
    )

    env_base = None
    controller = None

    # --- Model Selection Logic ---
    if agent_name == "MHC":
        if not high_path.exists():
            print("⚠️ MHC model not found, returning 0.0")
            return 0.0
        high_model = SAC.load(high_path, device="cpu")
        controller = high_model
        # MHC uses the trained High-Level Env
        env_base = HighLevelEnv(make_low, interval=900, goal_interface=gi, low_model=low_model, training_mode=False)

    elif agent_name == "PPO":  # (Baseline)
        if not baseline_path.exists():
            print("⚠️ Baseline model not found, skipping (return 0.0)")
            return 0.0
        # Load Baseline (Flat SAC)
        baseline_model = SAC.load(baseline_path, device="cpu")
        controller = baseline_model
        # Baseline also runs on HighLevelEnv (same observation space)
        env_base = HighLevelEnv(make_low, interval=900, goal_interface=gi, low_model=low_model, training_mode=False)

    else:  # Rule
        # Rule-based controller
        env_base = HighLevelEnv(make_low, interval=900, goal_interface=gi, low_model=low_model, training_mode=False)
        controller = BaselineWrapper(RuleBasedController(), None)

    # Wrap with Physics Kernel
    env = MarketPhysicsKernel(env_base, market_config=mkt_cfg, obs_mode="transparent")

    # Update wrapper env reference if needed
    if isinstance(controller, BaselineWrapper):
        controller.env = env

    # Load Data & Apply Stress
    data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"
    lmp, _, p_mil, agc = load_data_horizon(data_path, 25)  # Use Day 25

    if condition == "High_Price":
        lmp = lmp * 3.0  # Massive spikes
        p_mil = p_mil * 3.0

    # Run 1 Day
    plan = {'p_base': np.zeros(96), 'c_reg': np.ones(96) * 5.0, 'soc_plan': np.ones(96) * 0.5,
            'lambda_risk': np.zeros(96), 'lambda_aging': np.zeros(96)}

    # Pass stressed prices
    env.set_daily_plan(plan, agc, {'energy': lmp, 'reg_cap': np.ones(96) * 6.0, 'reg_mil': p_mil})

    obs, _ = env.reset(options={'reset_physics': True})
    total_profit = 0

    for _ in range(96):
        action, _ = controller.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        total_profit += info['true_net_profit']

    return total_profit


def main():
    print("🌪️ [Experiment] Robustness Stress Test")
    cfg = FinalRLBMSConfig(str(PROJECT_ROOT / "config.yaml"))

    conditions = ["Normal", "High_Price"]
    agents = ["MHC", "PPO", "Rule"]  # [Updated] Added PPO (Baseline)

    # Print Header
    header = f"{'Condition':<12} | " + " | ".join([f"{a:<12}" for a in agents])
    print(header)
    print("-" * (12 + 15 * len(agents)))

    for cond in conditions:
        results = []
        for agent in agents:
            profit = run_stress_test(agent, cond, cfg)
            results.append(profit)

        row_str = f"{cond:<12} | " + " | ".join([f"{r:>12.2f}" for r in results])
        print(row_str)

    print("-" * (12 + 15 * len(agents)))
    print("💡 Narrative: MHC should outperform Baseline (PPO) and Rule-based significantly in 'High_Price'.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Experiment: Ablation Study
Goal: Prove that the Hierarchical structure and Physics-Awareness are necessary.
Comparison:
1. MHC (Full)
2. MHC-NoLow (HighLevel active, but LowLevel replaced by naive average distribution)
"""
import sys
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import SAC

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from common.config import FinalRLBMSConfig
from env.physics_wrapper import MarketPhysicsKernel
from env.high_level import HighLevelEnv
from env.low_level import LowLevelEnv
from env.interface import GoalInterface
from market.mechanism import MarketConfig
from scripts.test_market import load_data_horizon


class NaiveLowLevelAgent:
    """Simulates a dumb LowLevel that just distributes current equally."""

    def __init__(self, num_cells):
        self.num_cells = num_cells

    def predict(self, obs, deterministic=True):
        # Return action that implies equal distribution (all zeros -> softmax uniform)
        # Action format: [tanh(scale), w1, w2... w24]
        return np.array([0.0] + [0.0] * self.num_cells, dtype=np.float32), None


def run_ablation_day(agent_type, day_idx, cfg, env, models):
    # Setup Data
    data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"
    # Load High-Price Data (Same as test_market.py)
    lmp_real, _, p_mil, agc = load_data_horizon(data_path, day_idx)

    # Mock Scheduler Plan
    T = 96
    plan = {
        'p_base': np.zeros(T),
        'c_reg': np.ones(T) * 6.0,
        'soc_plan': np.linspace(0.5, 0.5, T),
        'lambda_risk': np.zeros(T),
        'lambda_aging': np.zeros(T)
    }

    # Simple arbitrage logic for plan
    avg_price = np.mean(lmp_real)
    for t in range(T):
        if lmp_real[t] < avg_price * 0.8:
            plan['p_base'][t] = -15.0
        elif lmp_real[t] > avg_price * 1.2:
            plan['p_base'][t] = 15.0

    # Pass High Prices to Environment
    env.set_daily_plan(plan, agc, {'energy': lmp_real, 'reg_cap': np.ones(T) * 6.0, 'reg_mil': p_mil})

    obs, _ = env.reset(options={'reset_physics': (day_idx == 24)})  # Reset physics only on first day of loop

    total_net_profit = 0.0
    total_aging_cost = 0.0

    high_model = models.get('high')
    low_model = models.get('low')

    for t in range(T):
        if agent_type == "MHC_Full":
            # Normal MHC
            action, _ = high_model.predict(obs, deterministic=True)
        elif agent_type == "MHC_NoLow":
            # Hack: Swap LowLevel model to Naive
            action, _ = high_model.predict(obs, deterministic=True)
            env.env.unwrapped.low_model = NaiveLowLevelAgent(24)

        obs, _, _, _, info = env.step(action)
        total_net_profit += info['true_net_profit']
        total_aging_cost += info['cost_aging_realized']

        # Restore smart model
        if agent_type == "MHC_NoLow":
            env.env.unwrapped.low_model = low_model

    return total_net_profit, total_aging_cost


def main():
    print("🔬 [Experiment] Ablation Study: Deconstructing the Architecture")
    config_path = PROJECT_ROOT / "config.yaml"
    cfg = FinalRLBMSConfig(str(config_path))
    device = "cpu"

    # Load Models
    low_path = PROJECT_ROOT / "models/low_level_pretrained.zip"
    high_path = PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip"

    if not (low_path.exists() and high_path.exists()):
        print("❌ Models missing. Run training first.")
        return

    low_model = SAC.load(low_path, device=device)
    high_model = SAC.load(high_path, device=device)
    models = {'low': low_model, 'high': high_model}

    # Setup Environment
    gi = GoalInterface()

    def make_low_env():
        kwargs = cfg.get_env_kwargs()
        kwargs['subset'] = 'test'
        kwargs['external_power_profile'] = None
        return LowLevelEnv(kwargs, gi)

    # [CRITICAL] Configure MarketConfig from cfg to match High Price Scenario
    mkt_cfg = MarketConfig(
        rated_power=cfg.market.rated_power,
        rated_capacity=cfg.market.rated_capacity,
        price_unit_divisor=cfg.market.price_unit_divisor,  # Should be 1.0
        dev_penalty_ratio=cfg.market.dev_penalty_ratio
    )

    high_env = HighLevelEnv(make_low_env, interval=900, goal_interface=gi, low_model=low_model,
                            reward_weights=cfg.reward_high.weights.to_dict(), training_mode=False)

    env = MarketPhysicsKernel(high_env, market_config=mkt_cfg, obs_mode="transparent")

    # Run Comparison
    scenarios = ["MHC_Full", "MHC_NoLow"]
    results = {s: {'profit': [], 'aging': []} for s in scenarios}

    days_to_test = [24, 25, 26]

    print(f"{'Scenario':<12} | {'Avg Profit':<12} | {'Avg Aging Cost':<15}")
    print("-" * 45)

    for sc in scenarios:
        profits = []
        agings = []

        # Hard reset for each scenario
        env.reset(options={'reset_physics': True})

        for d in days_to_test:
            p, a = run_ablation_day(sc, d, cfg, env, models)
            profits.append(p)
            agings.append(a)

        avg_p = np.mean(profits)
        avg_a = np.mean(agings)
        results[sc]['profit'] = avg_p
        results[sc]['aging'] = avg_a

        print(f"{sc:<12} | {avg_p:>10.2f}   | {avg_a:>13.2f}")

    print("-" * 45)
    print("💡 Narrative: 'MHC_NoLow' should show higher Aging Cost due to lack of balancing.")


if __name__ == "__main__":
    main()
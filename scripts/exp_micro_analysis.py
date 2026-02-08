#!/usr/bin/env python3
"""
Experiment: Micro-Level Behavior Analysis
Goal: Visualize the 'Smooth Control' vs 'Bang-Bang' behavior in HVAC.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
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
from market.scheduler import DayAheadScheduler, SchedulerConfig
from baselines.controllers.rule_based import RuleBasedController
from scripts.test_market import BaselineWrapper, load_data_horizon

COLORS = {"MHC": "#1f77b4", "Rule": "#7f7f7f", "PPO": "#d62728"}


def run_micro_sim(agent_type, cfg):
    # Setup similar to other scripts...
    gi = GoalInterface()
    scheduler = DayAheadScheduler(SchedulerConfig(rated_power_kw=24.0))

    def make_low():
        kw = cfg.get_env_kwargs()
        kw['subset'] = 'test'
        kw['external_power_profile'] = None
        # Use a hot day condition
        env = LowLevelEnv(kw, gi)
        env.reset()
        for c in env.battery.cell: c.T_batt = 308.15  # Start at 35C
        return env

    low_model = SAC.load(PROJECT_ROOT / "models/low_level_pretrained.zip", device='cpu')
    if agent_type == "MHC":
        controller = SAC.load(PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip", device='cpu')
    elif agent_type == "Rule":
        controller = BaselineWrapper(RuleBasedController(), None)
    elif agent_type == "PPO":
        controller = BaselineWrapper(SAC.load(PROJECT_ROOT / "models/baseline_flat_sac.zip", device='cpu'), env_type="PPO")

    high_env = HighLevelEnv(make_low, interval=900, goal_interface=gi, low_model=low_model, training_mode=False)
    env = MarketPhysicsKernel(high_env, market_config=MarketConfig(), obs_mode="transparent", scheduler=scheduler)
    if isinstance(controller, BaselineWrapper): controller.env = env

    data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"
    obs, _ = env.reset(options={'reset_physics': True})

    # Load Day 15 (Mid summer scenario usually)
    lmp, _, p_mil, agc = load_data_horizon(data_path, 15)
    plan = scheduler._get_fallback_plan(96, 0.5)
    env.set_daily_plan(plan, agc, {'energy': lmp, 'reg_cap': np.ones(96) * 6.0, 'reg_mil': p_mil})

    hvac_power = []
    temp_curve = []

    # Run 1 day
    for _ in range(96):
        action, _ = controller.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)

        # Get Mean Temp
        temps = info.get('Temperature', [35.0])
        avg_t = np.mean(temps)

        # Get HVAC Power (Cost is roughly prop to power)
        # Using cost_hvac_realized from info / duration / price to est power
        # Or read from low level wrapper if exposed.
        # MarketPhysicsKernel info contains 'aux_power_kw' if passed from low level?
        # Let's check low_level.py -> it exports 'aux_power_kw' in info.
        # But MarketPhysicsKernel might not merge it.
        # However, we have 'cost_hvac_realized'.
        # Assume constant price for visualization proxy
        hvac_proxy = info.get('cost_hvac_realized', 0.0)

        hvac_power.append(hvac_proxy)
        temp_curve.append(avg_t)

    return hvac_power, temp_curve


def main():
    print("🔬 Running Micro-Level Behavior Analysis...")
    cfg = FinalRLBMSConfig(str(PROJECT_ROOT / "config.yaml"))

    agents = ["MHC", "Rule"]  # Focus on these two for contrast
    data = {}

    for a in agents:
        data[a] = run_micro_sim(a, cfg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    time_steps = range(96)

    # Plot Temperature
    for a in agents:
        ax1.plot(time_steps, data[a][1], label=a, color=COLORS[a], linewidth=2)
    ax1.set_ylabel("Battery Temp (°C)")
    ax1.set_title("Thermal Dynamics: Rule vs MHC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot HVAC "Activity" (Cost Proxy)
    for a in agents:
        ax2.plot(time_steps, data[a][0], label=a, color=COLORS[a], linewidth=2)
    ax2.set_ylabel("HVAC Cost/Activity")
    ax2.set_xlabel("Time Step (15 min)")
    ax2.grid(True, alpha=0.3)

    save_path = PROJECT_ROOT / "results/paper_figures/Fig_Micro_Behavior.svg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Micro-behavior plot saved to {save_path}")


if __name__ == "__main__":
    main()
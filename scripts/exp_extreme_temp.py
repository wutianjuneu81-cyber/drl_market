#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC, PPO

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

AGENTS = ["MHC", "Rule", "PPO"]
COLORS = {"MHC": "#1f77b4", "Rule": "#7f7f7f", "PPO": "#d62728"}


def run_temp_sim(agent_type, temp_c, cfg):
    gi = GoalInterface()

    def make_low():
        kw = cfg.get_env_kwargs()
        kw['subset'] = 'test'
        kw['external_power_profile'] = None
        env = LowLevelEnv(kw, gi)
        env.reset()
        init_temp_k = temp_c + 273.15
        for c in env.battery.cell: c.T_batt = init_temp_k
        return env

    low_model = SAC.load(PROJECT_ROOT / "models/low_level_pretrained.zip", device='cpu')

    # [Fix] Explicit Market Config
    mkt_cfg = MarketConfig(
        rated_power=cfg.market.rated_power,
        rated_capacity=cfg.market.rated_capacity,
        price_unit_divisor=cfg.market.price_unit_divisor,
        dev_penalty_ratio=cfg.market.dev_penalty_ratio
    )

    if agent_type == "MHC":
        high_model = SAC.load(PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip", device='cpu')
        controller = high_model
    elif agent_type == "Rule":
        controller = BaselineWrapper(RuleBasedController(), None)
    elif agent_type == "PPO":
        ppo_model = SAC.load(PROJECT_ROOT / "models/baseline_flat_sac.zip", device='cpu')
        controller = BaselineWrapper(ppo_model, None, env_type="PPO")

    high_env = HighLevelEnv(make_low, interval=900, goal_interface=gi, low_model=low_model, training_mode=False)
    # Use mkt_cfg
    env = MarketPhysicsKernel(high_env, market_config=mkt_cfg, obs_mode="transparent",
                              scheduler=DayAheadScheduler(SchedulerConfig(rated_power_kw=24.0)))

    if isinstance(controller, BaselineWrapper):
        controller.env = env

    obs, _ = env.reset(options={'reset_physics': True})

    if hasattr(env.env.unwrapped, 'low_env'):
        for c in env.env.unwrapped.low_env.battery.cell: c.T_batt = temp_c + 273.15

    total_hvac = 0.0
    data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"

    for _ in range(96 * 3):  # 3 Days
        lmp, _, p_mil, agc = load_data_horizon(data_path, 0)
        env.set_daily_plan(env.scheduler._get_fallback_plan(96, 0.5), agc,
                           {'energy': lmp, 'reg_cap': lmp, 'reg_mil': p_mil})
        action, _ = controller.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        total_hvac += info.get('cost_hvac_realized', 0.0)

    return total_hvac


def main():
    print("🌡️ Running Temperature Sensitivity Analysis...")
    cfg = FinalRLBMSConfig(str(PROJECT_ROOT / "config.yaml"))
    temps = [25, 30, 35, 40]
    results = {a: [] for a in AGENTS}

    for t in temps:
        print(f"   Testing at {t}°C...")
        for a in AGENTS:
            results[a].append(run_temp_sim(a, t, cfg))

    plt.figure(figsize=(8, 5))
    for a in AGENTS:
        if results[a]:
            plt.plot(temps[:len(results[a])], results[a], marker='o', linewidth=2, label=a, color=COLORS[a])

    plt.title("HVAC Cost Sensitivity")
    plt.xlabel("Ambient Temp (°C)")
    plt.ylabel("3-Day HVAC Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = PROJECT_ROOT / "results/paper_figures/Fig_Temp_Sensitivity.svg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print("✅ Done.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Experiment: Comprehensive Radar Chart
Goal: Compare agents across 5 dimensions using REAL simulation data.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# Import the reusable simulation function
from scripts.test_market import run_market_simulation

# Config
AGENTS = ["MHC", "Rule", "PPO"]
COLORS = {"MHC": "#1f77b4", "Rule": "#7f7f7f", "PPO": "#d62728"}
SIM_DAYS = 7  # Short run for metrics collection


def get_agent_metrics(agent_name):
    print(f"   > Collecting metrics for {agent_name}...")
    # Map 'MHC' to 'RL' if test_market expects 'RL' or use 'MHC' if supported
    # In our refactored test_market, we support 'MHC' alias for 'RL'.

    # Run simulation silently
    raw = run_market_simulation(agent_name, days=SIM_DAYS, silent=True)

    # Process Metrics for Radar (All should be "Higher is Better")
    # 1. Profit: Normalized Net Profit
    # 2. Safety: Inverse of Penalty
    # 3. Health: Inverse of Aging Cost
    # 4. Efficiency: Inverse of HVAC Cost
    # 5. Reliability: (Total Steps - Violations) / Total Steps

    total_steps = SIM_DAYS * 96
    reliability_score = (total_steps - raw["violations"]) / total_steps

    return {
        "Profit": raw["net_profit"],
        "Safety": -raw["penalty"],  # Less penalty is better
        "Health": -raw["aging_cost"],  # Less aging is better
        "Efficiency": -raw["hvac_cost"],  # Less HVAC is better
        "Reliability": reliability_score * 1000  # Scale up for visualization if needed
    }


def main():
    print("🕸️ Generating Comprehensive Radar Chart (Auto-Data)...")

    metrics_data = {}
    for agent in AGENTS:
        # PPO requires specific handling in test_market (already fixed there)
        metrics_data[agent] = get_agent_metrics(agent)

    # Categories
    categories = ["Profit", "Safety", "Health", "Efficiency", "Reliability"]

    # Normalize Data to [0, 1] for plotting
    data_normalized = {a: [] for a in AGENTS}

    for cat in categories:
        values = [metrics_data[a][cat] for a in AGENTS]
        min_v, max_v = min(values), max(values)

        for a in AGENTS:
            val = metrics_data[a][cat]
            if max_v - min_v == 0:
                norm = 1.0
            else:
                norm = (val - min_v) / (max_v - min_v + 1e-9)

            # Scale to 0.2 - 1.0 to make the chart look fuller
            data_normalized[a].append(0.2 + 0.8 * norm)

    # Plotting
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]  # Close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8], ["40%", "60%", "80%"], color="grey", size=7)
    plt.ylim(0, 1.05)

    for agent in AGENTS:
        values = data_normalized[agent]
        values += [values[0]]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent, color=COLORS[agent])
        ax.fill(angles, values, color=COLORS[agent], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f"Agent Capability Radar ({SIM_DAYS} Days Avg)", y=1.08)

    save_path = PROJECT_ROOT / "results/paper_figures/Fig_Radar.svg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Radar chart saved to {save_path}")


if __name__ == "__main__":
    main()
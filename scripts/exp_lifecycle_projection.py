#!/usr/bin/env python3
"""
Experiment: Lifecycle Projection
Analysis: Uses 30-day normal simulation data to extrapolate End-of-Life (EOL).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))


def main():
    print("🔮 [Projection] Calculating Life-cycle Expectancy...")
    data_dir = PROJECT_ROOT / "results/experiment_data"

    files = list(data_dir.glob("data_*.npz"))
    if not files:
        print("❌ No experiment data found. Run 'run_evaluation.py' first.")
        return

    eol_threshold = 0.80
    start_soh = 1.00

    results = {}

    plt.figure(figsize=(10, 6))

    for f in files:
        data = np.load(f)
        agent = str(data["agent_type"])
        soh_curve = data["soh"]

        # Calculate daily fade rate
        # Use robust regression or simple mean of diffs
        # Excluding first day (burn-in)
        if len(soh_curve) < 5: continue

        # Linear fit: SOH = m * day + c
        days = np.arange(len(soh_curve))
        m, c = np.polyfit(days, soh_curve, 1)

        # Fade per day = -m
        daily_fade = -m

        if daily_fade <= 0:
            print(f"⚠️ Agent {agent} has no degradation ({daily_fade}). Skipping.")
            continue

        days_to_eol = (start_soh - eol_threshold) / daily_fade
        years_to_eol = days_to_eol / 365.0

        results[agent] = years_to_eol
        print(f"   Agent {agent:<10}: Daily Fade = {daily_fade:.6f}, EOL = {years_to_eol:.2f} Years")

        # Plot Extrapolation
        proj_days = np.linspace(0, days_to_eol, 100)
        proj_soh = start_soh + m * proj_days

        label = f"{agent} (Proj: {years_to_eol:.1f} Yrs)"
        plt.plot(proj_days / 365.0, proj_soh, label=label, linewidth=2)

    plt.axhline(eol_threshold, color='r', linestyle='--', label="EOL (80%)")
    plt.title("Battery Life-cycle Projection")
    plt.xlabel("Years")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = PROJECT_ROOT / "results/paper_figures/Fig_Lifecycle_Projection.svg"

    # [FIX] 确保父目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    print(f"✅ Projection plot saved to {save_path}")


if __name__ == "__main__":
    main()
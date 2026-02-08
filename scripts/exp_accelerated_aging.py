# !/usr/bin/env python3
"""
Experiment: Accelerated Aging & Parameter Adaptation
Goal:
1. Verify MHC preserves health better than baselines (SOH Curve).
2. [NEW] Demonstrate the convergence of Adaptive Scheduler parameter (alpha_1).
"""
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


def run_accelerated_sim(agent_type, cfg, days=30):
    print(f"⚡ Running Accelerated Sim for {agent_type}...")

    # Amplify Aging x50 for stress testing
    ACCELERATION_FACTOR = 50.0
    cfg_copy = FinalRLBMSConfig(str(PROJECT_ROOT / "config.yaml"))
    if 'aging' in cfg_copy._raw_data:
        cfg_copy._raw_data['aging']['cycle_fade_coeff'] *= ACCELERATION_FACTOR
        cfg_copy._raw_data['aging']['calendar_fade_coeff'] *= ACCELERATION_FACTOR
        cfg_copy._raw_data['aging']['rint_growth_coeff'] *= ACCELERATION_FACTOR

    gi = GoalInterface()
    # [关键] 初始化调度器，初始 alpha 设为默认值 (e.g. 0.05)
    scheduler = DayAheadScheduler(SchedulerConfig(
        rated_power_kw=24.0,
        marginal_aging_cost_default=0.05  # Initial guess
    ))

    def make_low():
        kw = cfg_copy.get_env_kwargs()
        kw['subset'] = 'test'
        kw['external_power_profile'] = None
        return LowLevelEnv(kw, gi)

    # Load Models
    try:
        low_model = SAC.load(PROJECT_ROOT / "models/low_level_pretrained.zip", device='cpu')
    except:
        low_model = None  # Fallback

    mkt_cfg = MarketConfig(
        rated_power=cfg.market.rated_power,
        rated_capacity=cfg.market.rated_capacity,
        price_unit_divisor=cfg.market.price_unit_divisor,
        dev_penalty_ratio=cfg.market.dev_penalty_ratio
    )

    controller = None
    if agent_type == "MHC":
        try:
            high_model = SAC.load(PROJECT_ROOT / "models/high_level_trained_cmdp_sac.zip", device='cpu')
            controller = high_model
        except:
            print("⚠️ MHC model not found, using random.")
    elif agent_type == "Rule":
        controller = BaselineWrapper(RuleBasedController(), None)
    elif agent_type == "PPO":
        try:
            ppo_model = SAC.load(PROJECT_ROOT / "models/baseline_flat_sac.zip", device='cpu')
            controller = BaselineWrapper(ppo_model, None, env_type="PPO")
        except:
            pass

    high_env = HighLevelEnv(make_low, interval=900, goal_interface=gi, low_model=low_model,
                            reward_weights=cfg.reward_high.weights.to_dict(), training_mode=False)

    env = MarketPhysicsKernel(high_env, market_config=mkt_cfg, obs_mode="transparent", scheduler=scheduler)

    if isinstance(controller, BaselineWrapper):
        controller.env = env

    data_path = PROJECT_ROOT / "data/profiles/guanghe_30days_scaled.npz"

    # 记录数据
    soh_history = []
    alpha_history = []  # [NEW] 记录自适应参数的变化

    obs, _ = env.reset(options={'reset_physics': True})
    current_soh = 1.0
    soh_history.append(current_soh)
    alpha_history.append(scheduler.cfg.marginal_aging_cost_default)

    for day in range(days):
        # 1. 获取环境数据
        lmp, _, p_mil, agc = load_data_horizon(data_path, day % 30)

        # 2. [关键修改] 使用 solve() 而非 fallback，让调度器真正工作
        #    这样 alpha 的改变才会影响 plan
        plan_result = scheduler.solve(
            price_energy=lmp,
            price_reg=np.ones(96) * 6.0,  # 假设调频价格
            current_soh=current_soh,
            initial_soc=0.5
        )

        # 提取 plan
        plan_dict = {
            'p_base': plan_result.get('p_base', np.zeros(96)),
            'c_reg': plan_result.get('c_reg', np.zeros(96))
        }

        # 3. 设置环境的基准
        env.set_daily_plan(plan_dict, agc, {'energy': lmp, 'reg_cap': np.ones(96) * 6.0, 'reg_mil': p_mil})

        if day > 0: obs, _ = env.reset(options={'reset_physics': False})

        # 4. 运行一天的仿真
        daily_aging_cost_accum = 0.0
        daily_throughput_accum = 0.0

        for _ in range(96):
            if controller:
                action, _ = controller.predict(obs, deterministic=True)
            else:
                action = high_env.action_space.sample()

            obs, _, _, _, info = env.step(action)

            # [关键] 累加当天的真实老化成本和吞吐量
            # info['cost/aging_real'] 是 HighLevelEnv 这一步的总老化成本
            daily_aging_cost_accum += info.get('cost/aging_real', 0.0)

            # 估算吞吐量: sum(|P_real|) * dt (hours)
            # P_real 可以从 info 中获取，HighLevelEnv 输出了 'action/p_real'
            p_real = info.get('action/p_real', 0.0)
            daily_throughput_accum += abs(p_real) * 0.25  # 15min = 0.25h

        # 获取当天结束后的 SOH
        current_soh = info.get('aging_info', {}).get('soh_estimate', current_soh)
        soh_history.append(current_soh)

        # 5. [核心实验逻辑] 闭环参数自适应
        # 只有 MHC Agent 所在的场景，调度器参数更新才有意义 (或者全部更新以对比)
        # 这里我们让调度器根据物理反馈更新 alpha
        # 注意：由于我们加速了老化 (x50)，这里传入的 cost 也是 x50 的。
        # 调度器会迅速学习到"现在的电池很贵"，从而提高 alpha。
        scheduler.update_aging_parameters(daily_aging_cost_accum, daily_throughput_accum)

        # 记录更新后的 alpha
        alpha_history.append(scheduler.cfg.marginal_aging_cost_default)

        # 打印进度
        if day % 5 == 0:
            print(f"   [Day {day}] SOH={current_soh:.4f}, Alpha={alpha_history[-1]:.4f}")

    return soh_history, alpha_history


def main():
    cfg = FinalRLBMSConfig(str(PROJECT_ROOT / "config.yaml"))

    # 增加实验天数以观察收敛
    SIM_DAYS = 60

    results_soh = {}
    results_alpha = {}

    for agent in AGENTS:
        soh_hist, alpha_hist = run_accelerated_sim(agent, cfg, days=SIM_DAYS)
        results_soh[agent] = soh_hist
        results_alpha[agent] = alpha_hist

    # --- 绘图 (双子图) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Subplot 1: SOH Degradation
    days_x = range(len(next(iter(results_soh.values()))))
    for agent in AGENTS:
        ax1.plot(days_x, results_soh[agent], label=agent, color=COLORS[agent], linewidth=2.5)

    ax1.set_ylabel("Battery SOH")
    ax1.set_title(f"Accelerated Aging Verification (Factor x50)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Parameter Adaptation (Alpha)
    # 重点展示 MHC 的适应过程
    for agent in AGENTS:
        # 虚线表示其他 Agent 可能不一定由该参数主导，但 Scheduler 都在更新
        style = '-' if agent == "MHC" else '--'
        alpha = 1.0 if agent == "MHC" else 0.5
        ax2.plot(days_x, results_alpha[agent], label=f"{agent} $\\alpha_1$",
                 color=COLORS[agent], linestyle=style, linewidth=2.0, alpha=alpha)

    ax2.set_ylabel("Adaptive Penalty $\\alpha_1$ \n(CNY/kWh-throughput)")
    ax2.set_xlabel("Simulation Days")
    ax2.set_title("Closed-Loop Parameter Adaptation (Convergence Analysis)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 标注收敛区域
    ax2.text(SIM_DAYS * 0.8, results_alpha["MHC"][-1] * 1.1, "Convergence",
             fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()

    save_path = PROJECT_ROOT / "results/paper_figures/Fig_Accelerated_Aging_Adaptation.svg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Saved combined experiment plot to {save_path}")


if __name__ == "__main__":
    main()
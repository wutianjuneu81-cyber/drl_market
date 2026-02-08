import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# 尝试导入子模块
try:
    from DRL_market.simulation.cell import LithiumIronPhosphateCell
    from DRL_market.simulation.aging import LithiumIonAging
except ImportError:
    from simulation.cell import LithiumIronPhosphateCell
    from simulation.aging import LithiumIonAging


class BatteryPack:
    """ Battery Pack System Model (Final Robust Version) """

    def __init__(self, full_config: Dict):
        self.cfg = full_config
        self.env_cfg = full_config['env']
        self.market_cfg = full_config['market']

        self.n_clusters = self.env_cfg['n_clusters']
        self.cells_per_cluster = self.env_cfg.get('cells_per_cluster', 240)
        self.nominal_voltage_v = self.env_cfg.get('nominal_voltage_v', 768.0)

        total_kwh = self.market_cfg['rated_capacity_kwh']
        self.cluster_capacity_nominal_ah = (total_kwh * 1000.0) / self.nominal_voltage_v / self.n_clusters

        init_soc = self.env_cfg.get('initial_soc', 0.5)
        init_temp = self.env_cfg.get('initial_temp', 25.0)

        self.socs = np.ones(self.n_clusters) * init_soc
        self.temps = np.ones(self.n_clusters) * init_temp
        self.sohs = np.ones(self.n_clusters) * 1.0
        self.voltages = np.ones(self.n_clusters) * self.nominal_voltage_v
        self.currents = np.zeros(self.n_clusters)
        self.current_capacities_ah = np.ones(self.n_clusters) * self.cluster_capacity_nominal_ah

        cell_config = {'env': {'cell_capacity_ah': self.cluster_capacity_nominal_ah, 'initial_soc': init_soc,
                               'nominal_voltage': 3.2}}
        self.cells = [LithiumIronPhosphateCell(cell_config) for _ in range(self.n_clusters)]
        self.aging_model = LithiumIonAging(self.cfg)
        self.thermal_coupling_k = 0.05
        self.ambient_temp = 25.0

    def reset(self):
        init_soc = self.env_cfg.get('initial_soc', 0.5)
        if self.env_cfg.get('randomization', True):
            init_soc = np.clip(np.random.normal(init_soc, 0.05), 0.1, 0.9)

        self.socs = np.ones(self.n_clusters) * init_soc
        self.temps = np.ones(self.n_clusters) * 25.0
        self.sohs = np.ones(self.n_clusters) * 1.0
        self.current_capacities_ah = np.ones(self.n_clusters) * self.cluster_capacity_nominal_ah

        for cell in self.cells:
            cell.reset(initial_soc=init_soc)
        self.aging_model.reset()

    def step(self, cluster_currents: np.ndarray, dt: float) -> Dict[str, Any]:
        cluster_currents = np.nan_to_num(cluster_currents, nan=0.0)
        self.currents = cluster_currents

        q_coupling = np.zeros(self.n_clusters)
        if self.n_clusters > 1:
            q_coupling[0] += self.thermal_coupling_k * (self.temps[1] - self.temps[0])
            q_coupling[-1] += self.thermal_coupling_k * (self.temps[-2] - self.temps[-1])
            for i in range(1, self.n_clusters - 1):
                q_coupling[i] += self.thermal_coupling_k * (self.temps[i - 1] - self.temps[i])
                q_coupling[i] += self.thermal_coupling_k * (self.temps[i + 1] - self.temps[i])

        step_weighted_costs = []  # [Fix] 变量名统一
        step_real_costs = []

        for i in range(self.n_clusters):
            cell_info = self.cells[i].step(current_a=cluster_currents[i], dt=dt, soh_capacity=self.sohs[i],
                                           soh_resistance=(1.0 / (self.sohs[i] + 1e-6)))

            soc_next = cell_info['soc']
            v_term = cell_info['voltage_v']
            heat_gen = cell_info.get('heat_watts', 0.0)

            total_heat = heat_gen + q_coupling[i]
            heat_diss = (self.temps[i] - self.ambient_temp) * 5.0
            thermal_mass = 10000.0
            delta_temp = (total_heat - heat_diss) * dt / thermal_mass
            t_next = self.temps[i] + delta_temp

            self.socs[i] = np.clip(soc_next, 0.0, 1.0)
            self.temps[i] = np.clip(t_next, -40.0, 150.0)
            self.voltages[i] = max(0.1, v_term)
            self.cells[i].soc = self.socs[i]

            aging_out = self.aging_model.compute_aging_step(cluster_id=i, current_soc=self.socs[i],
                                                            temperature_c=self.temps[i], current_a=cluster_currents[i],
                                                            dt=dt)

            loss_ah = aging_out['total_loss_ah']
            weighted_loss = aging_out['stress_weighted_loss']

            if loss_ah > 0:
                self.current_capacities_ah[i] -= loss_ah
                self.current_capacities_ah[i] = max(0.1, self.current_capacities_ah[i])
                self.sohs[i] = self.current_capacities_ah[i] / self.cluster_capacity_nominal_ah

            step_weighted_costs.append(weighted_loss)
            step_real_costs.append(loss_ah)

        # [Fix] 返回一致的 key
        return {
            "aging_costs": np.array(step_weighted_costs),  # 统一接口，传给 low_level 做 Reward
            "total_loss_ah": np.array(step_real_costs),  # 传给 Info 做记录
            "avg_voltage": np.mean(self.voltages),
            "avg_temp": np.mean(self.temps)
        }

    def get_voltages(self) -> np.ndarray:
        return self.voltages

    def get_mean_soc(self) -> float:
        return float(np.mean(self.socs))

    def get_total_energy(self) -> float:
        safe_socs = np.clip(self.socs, 0.0, 1.0)
        safe_caps = np.maximum(self.current_capacities_ah, 0.0)
        energies = safe_caps * safe_socs * self.nominal_voltage_v / 1000.0
        total_energy = np.sum(energies)
        return float(total_energy) if np.isfinite(total_energy) else 0.0

    def check_safety_constraints(self) -> bool:
        limits = self.env_cfg.get('limits', {})
        max_temp = limits.get('temp_max', 50.0)
        if np.any(self.temps > max_temp): return True
        if np.any(self.voltages < 2.0) or np.any(self.voltages > 4.2 * self.cells_per_cluster): return True
        return False
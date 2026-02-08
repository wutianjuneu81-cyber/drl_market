import numpy as np
from typing import Dict

ACHIEVED_KEYS = ["soc_std", "temp_peak", "utilization", "soh_std", "delta_soh"]

# 距离归一化统计（在线均值/方差）
_distance_stats = {k: {"count":0.0, "mean":0.0, "M2":0.0} for k in ACHIEVED_KEYS}
_min_samples_normalize = 30  # 超过后使用标准化

def update_distance_stats(g1: Dict[str,float], g2: Dict[str,float]):
    for k in ACHIEVED_KEYS:
        x = g1.get(k,0.0) - g2.get(k,0.0)
        s = _distance_stats[k]
        s["count"] += 1.0
        delta = x - s["mean"]
        s["mean"] += delta / s["count"]
        delta2 = x - s["mean"]
        s["M2"] += delta * delta2

def _zscore(k: str, x: float) -> float:
    s = _distance_stats[k]
    if s["count"] < _min_samples_normalize:
        return x
    var = s["M2"] / max(s["count"] - 1.0, 1.0)
    std = np.sqrt(max(var, 1e-9))
    return (x - s["mean"]) / (std + 1e-9)

def goal_distance(g1: Dict[str, float], g2: Dict[str, float], weights: Dict[str, float],
                  metric="l2", normalize=False) -> float:
    """计算两个 Goal 之间的加权距离"""
    diffs = []
    for k in ACHIEVED_KEYS:
        w = weights.get(k, 1.0)
        raw = g1.get(k, 0.0) - g2.get(k, 0.0)
        val = _zscore(k, raw) if normalize else raw
        diffs.append(w * val)
    d = np.array(diffs, dtype=float)
    if metric == "l1":
        return float(np.sum(np.abs(d)))
    return float(np.sqrt(np.sum(d * d) + 1e-12))

def extract_achieved_goal(window_summary: Dict[str, float]) -> Dict[str, float]:
    return {k: window_summary.get(k, 0.0) for k in ACHIEVED_KEYS}

def interpolate_goals(g_start: Dict[str, float], g_end: Dict[str, float], n_splits: int):
    outs = []
    for i in range(1, n_splits + 1):
        ratio = i / (n_splits + 1)
        outs.append({k: g_start[k] + ratio * (g_end[k] - g_start[k]) for k in g_start})
    return outs

def get_distance_stats():
    out = {}
    for k, s in _distance_stats.items():
        count = s["count"]
        if count < 1:
            out[k] = {"mean":0.0,"std":0.0,"samples":0}
        else:
            var = s["M2"] / max(count - 1.0, 1.0)
            out[k] = {"mean":float(s["mean"]), "std":float(np.sqrt(max(var,0.0))), "samples": int(count)}
    return out
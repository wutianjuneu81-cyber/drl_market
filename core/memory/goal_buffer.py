import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional
from core.math.hiro import extract_achieved_goal, interpolate_goals


class GoalTransition:
    def __init__(
            self,
            state,
            action,
            reward,
            next_state,
            goal_dict,
            window_summary,
            achieved_goal,
            mismatch_distance: float,
            improvement_score: float,
            proxy_slice=None,
            diff_slice=None
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.goal_dict = goal_dict
        self.window_summary = window_summary
        self.achieved_goal = achieved_goal
        self.proxy_goal = goal_dict.get("proxy_goal", achieved_goal)
        self.mismatch_distance = mismatch_distance
        self.improvement_score = improvement_score
        self.relabelled = False
        self.relabelled_action: Optional[np.ndarray] = None
        self.relabel_probability_used: float = 0.0
        self.proxy_slice = proxy_slice
        self.diff_slice = diff_slice
        self.original_action_l2 = 0.0
        self.distance_rank = None


class GoalBuffer:
    """
    [FIXED] 改进版动态重标策略 (HIRO Relabeling Buffer)
    修复了 _maybe_relabel 中 Observation 修改的逻辑错误。
    """

    def __init__(self, capacity: int, mismatch_threshold: float, hiro_cfg: dict | None = None):
        self.capacity = capacity
        self.mismatch_threshold = mismatch_threshold
        self.buffer: List[GoalTransition] = []
        self.ptr = 0
        self.hiro_cfg = hiro_cfg or {}

        # 统计数据容器
        self.stats_data = {
            "attempts": 0,
            "applied": 0,
            "avg_distance": 0.0,
            "component_diff_mean": {k: deque(maxlen=4000) for k in
                                    ["soc_std", "temp_peak", "utilization", "soh_std", "delta_soh"]},
            "subsplits_total": 0,
            "mismatch_trigger_count": 0,
            "distances": deque(maxlen=4000),
            "relabel_probs_used": deque(maxlen=4000),
            "dist_relabel": deque(maxlen=4000),
            "dist_non_relabel": deque(maxlen=4000),
            "action_l2": deque(maxlen=4000),
            "rank_distribution": deque(maxlen=4000),
            "improvement_scores": deque(maxlen=4000),
        }

        # 参数配置
        self.dynamic_relabel_prob = self.hiro_cfg.get("relabel_probability", 0.3)
        self.min_relabel_prob = self.hiro_cfg.get("min_relabel_probability", 0.15)
        self.max_relabel_ratio = self.hiro_cfg.get("max_relabel_ratio", 0.60)
        self.min_relabel_ratio = self.hiro_cfg.get("min_relabel_ratio", 0.12)
        self.force_random = self.hiro_cfg.get("force_random", False)

        self.distance_decay_window = self.hiro_cfg.get("distance_decay_window", 50)
        self._recent_distances = deque(maxlen=self.distance_decay_window)

        self.rank_alpha = self.hiro_cfg.get("rank_alpha", 0.3)
        self.relabel_reward_kappa = self.hiro_cfg.get("relabel_reward_kappa", 0.02)
        self.diversity_l2_clip = self.hiro_cfg.get("diversity_l2_clip", 3.2)
        self.aging_action_adjust_scale = self.hiro_cfg.get("aging_action_adjust_scale", 0.08)
        self.util_boost_scale = self.hiro_cfg.get("util_boost_scale", 0.06)

        self.distance_threshold_cfg = self.hiro_cfg.get("distance_threshold", 0.34)
        self.keep_prob_if_mismatch_high = self.hiro_cfg.get("keep_prob_if_mismatch_high", True)
        self.mismatch_high_threshold = self.hiro_cfg.get("mismatch_high_threshold", 1.6)
        self.probability_decay = self.hiro_cfg.get("probability_decay", 0.97)
        self.probability_increase = self.hiro_cfg.get("probability_increase", 1.05)

    def _make_relabel_action(self, original_action: np.ndarray, achieved: Dict[str, float],
                             proxy: Dict[str, float]) -> np.ndarray:
        new_act = original_action.copy()
        aw = new_act[:4]
        soc_diff = achieved.get("soc_std", 0.0) - proxy.get("soc_std", 0.0)
        temp_diff = achieved.get("temp_peak", 0.0) - proxy.get("temp_peak", 0.0)
        util_diff = proxy.get("utilization", 0.0) - achieved.get("utilization", 0.0)

        adjust = -self.aging_action_adjust_scale * soc_diff - (
                self.aging_action_adjust_scale * 0.55) * temp_diff + self.util_boost_scale * util_diff
        aw_new = np.clip(aw * (1.0 + adjust), 0.0, 2.0)
        new_act[:4] = aw_new
        return new_act

    def _update_dynamic_probability(self):
        attempts = self.stats_data["attempts"]
        applied = self.stats_data["applied"]
        if attempts < 200:
            return

        ratio = applied / max(attempts, 1)
        if ratio > self.max_relabel_ratio:
            self.dynamic_relabel_prob = max(self.min_relabel_prob, self.dynamic_relabel_prob * self.probability_decay)
        elif ratio < self.min_relabel_ratio:
            self.dynamic_relabel_prob = min(0.95, self.dynamic_relabel_prob * self.probability_increase)

        if len(self._recent_distances) == self._recent_distances.maxlen:
            avg_last = np.mean(self._recent_distances)
            if self.keep_prob_if_mismatch_high and avg_last > self.mismatch_high_threshold:
                self.dynamic_relabel_prob = max(self.dynamic_relabel_prob, self.min_relabel_prob * 1.05)

    def add(self, state, action, reward, next_state, goal_dict, window_summary,
            mismatch_distance, improvement_score,
            replay_buffer=None, rb_index: int = -1,
            proxy_slice=None, diff_slice=None):
        achieved = extract_achieved_goal(window_summary)
        tr = GoalTransition(state, action.copy(), reward, next_state, goal_dict.copy(),
                            window_summary, achieved, mismatch_distance, improvement_score,
                            proxy_slice=proxy_slice, diff_slice=diff_slice)

        if len(self.buffer) < self.capacity:
            self.buffer.append(tr)
        else:
            self.buffer[self.ptr] = tr
        self.ptr = (self.ptr + 1) % self.capacity

        self._assign_rank(tr)
        self._maybe_relabel(tr, replay_buffer, rb_index)

    def _assign_rank(self, tr: GoalTransition):
        dist = tr.mismatch_distance
        self.stats_data["distances"].append(dist)
        all_dist = list(self.stats_data["distances"])
        if len(all_dist) < 5:
            tr.distance_rank = len(all_dist)
            return
        sorted_vals = sorted(all_dist)
        try:
            tr.distance_rank = sorted_vals.index(dist) + 1
        except ValueError:
            tr.distance_rank = int(len(all_dist) / 2)
        self.stats_data["rank_distribution"].append(tr.distance_rank)

    def _maybe_relabel(self, tr: GoalTransition, replay_buffer, rb_index: int):
        if not self.hiro_cfg.get("enabled", False):
            return

        dist = tr.mismatch_distance
        self.stats_data["attempts"] += 1
        self._recent_distances.append(dist)
        self.stats_data["avg_distance"] += (dist - self.stats_data["avg_distance"]) / self.stats_data["attempts"]
        self.stats_data["improvement_scores"].append(tr.improvement_score)

        thr = self.distance_threshold_cfg
        self._update_dynamic_probability()

        if tr.distance_rank is None:
            rank_scale = 1.0
        else:
            N = max(len(self.stats_data["distances"]), tr.distance_rank)
            rank_scale = (tr.distance_rank / N) ** self.rank_alpha

        if self.force_random:
            final_prob = self.dynamic_relabel_prob
        else:
            prob_scale = 1.0
            if dist > self.mismatch_threshold:
                prob_scale = 1.0 + 0.28 * (dist / max(self.mismatch_threshold, 1e-6))
                self.stats_data["mismatch_trigger_count"] += 1
            if tr.improvement_score <= 0.0 and dist > thr:
                prob_scale *= 1.05
            final_prob = min(0.95, self.dynamic_relabel_prob * prob_scale * rank_scale)

        tr.relabel_probability_used = final_prob
        self.stats_data["relabel_probs_used"].append(final_prob)

        condition_ok = self.force_random or dist > thr
        relabel_applied = False

        if condition_ok and np.random.rand() < final_prob:
            tr.relabelled = True
            self.stats_data["applied"] += 1
            relabel_applied = True

            for k in tr.achieved_goal.keys():
                self.stats_data["component_diff_mean"][k].append(
                    tr.achieved_goal[k] - tr.proxy_goal.get(k, 0.0)
                )

            if self.hiro_cfg.get("subtrajectory", True):
                splits = min(self.hiro_cfg.get("max_subsplits", 3), 6)
                _ = interpolate_goals(tr.proxy_goal, tr.achieved_goal, splits)
                self.stats_data["subsplits_total"] += splits

            if replay_buffer is not None and rb_index >= 0:
                relabel_act = self._make_relabel_action(tr.action, tr.achieved_goal, tr.proxy_goal)
                tr.relabelled_action = relabel_act
                try:
                    orig_act = replay_buffer.actions[rb_index, 0, :].copy()
                    replay_buffer.actions[rb_index, 0, :] = relabel_act
                    if hasattr(replay_buffer, "relabel_flags"):
                        replay_buffer.relabel_flags[rb_index] = 1

                    # [CRITICAL FIX START] Observation Correction Logic
                    if tr.proxy_slice and tr.diff_slice:
                        obs_old = replay_buffer.observations[rb_index, 0, :]
                        next_old = replay_buffer.next_observations[rb_index, 0, :]

                        for i, k in enumerate(tr.proxy_goal.keys()):
                            # 逻辑修正：
                            # 1. Goal 部分 -> 替换为 Achieved Goal
                            # 2. Diff 部分 -> Achieved - Achieved = 0.0

                            achieved_val = tr.achieved_goal[k]

                            # 更新 Observation
                            obs_old[tr.proxy_slice.start + i] = achieved_val
                            obs_old[tr.diff_slice.start + i] = 0.0  # Diff is zero

                            # 更新 Next Observation
                            next_old[tr.proxy_slice.start + i] = achieved_val
                            next_old[tr.diff_slice.start + i] = 0.0  # Diff is zero

                        replay_buffer.observations[rb_index, 0, :] = obs_old
                        replay_buffer.next_observations[rb_index, 0, :] = next_old
                    # [CRITICAL FIX END]

                    l2 = float(np.linalg.norm(orig_act - relabel_act))
                    tr.original_action_l2 = min(l2, self.diversity_l2_clip)
                    self.stats_data["action_l2"].append(tr.original_action_l2)

                    kappa = self.relabel_reward_kappa
                    if kappa > 0:
                        shaping = kappa * max(0.0, dist - thr)
                        replay_buffer.rewards[rb_index, 0] += shaping
                except Exception:
                    pass

        if relabel_applied:
            self.stats_data["dist_relabel"].append(dist)
        else:
            self.stats_data["dist_non_relabel"].append(dist)

    def sample(self, batch_size: int):
        if not self.buffer:
            return []
        idxs = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idxs]

    def _compute_entropy(self) -> float:
        if not self.buffer: return 0.0
        actions = []
        for tr in self.buffer:
            act = tr.relabelled_action if tr.relabelled_action is not None else tr.action
            if act is None: continue
            aw = np.clip(act[:4], 1e-6, 2.0)
            dist = aw / np.sum(aw)
            actions.append(dist)
        if not actions: return 0.0
        arr = np.array(actions, dtype=float)
        entropy = -np.mean(np.sum(arr * np.log(arr + 1e-9), axis=1))
        return float(entropy)

    def stats(self) -> Dict[str, Any]:
        comp_means = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in
                      self.stats_data["component_diff_mean"].items()}
        attempts = self.stats_data["attempts"]
        applied = attempts and self.stats_data["applied"] or self.stats_data["applied"]
        return {
            "size": len(self.buffer),
            "attempts": attempts,
            "applied": self.stats_data["applied"],
            "relabel_ratio": (applied / attempts) if attempts else 0.0,
            "avg_distance": float(np.mean(self.stats_data["distances"])) if self.stats_data["distances"] else 0.0,
            "component_diff_mean": comp_means,
            "mismatch_trigger_count": self.stats_data["mismatch_trigger_count"],
            "avg_relabel_prob": float(np.mean(self.stats_data["relabel_probs_used"])) if self.stats_data[
                "relabel_probs_used"] else 0.0,
            "avg_distance_relabel": float(np.mean(self.stats_data["dist_relabel"])) if self.stats_data[
                "dist_relabel"] else 0.0,
            "avg_distance_non_relabel": float(np.mean(self.stats_data["dist_non_relabel"])) if self.stats_data[
                "dist_non_relabel"] else 0.0,
            "improvement_score_mean": float(np.mean(self.stats_data["improvement_scores"])) if self.stats_data[
                "improvement_scores"] else 0.0
        }
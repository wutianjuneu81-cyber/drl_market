import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

ARTIFACT_ROOT = Path("artifacts")
INDEX_DIR = ARTIFACT_ROOT / "indexes"
ARTIFACT_INDEX = INDEX_DIR / "artifact_index.json"
MODEL_REGISTRY = INDEX_DIR / "model_registry.json"
RUN_REGISTRY = INDEX_DIR / "run_registry.json"

_CACHE = {
    "artifact_index": None,
    "model_registry": None,
    "run_registry": None,
    "timestamp": 0
}
_CACHE_TTL = 4.0


def _ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _refresh_cache():
    import time
    now = time.time()
    if _CACHE["timestamp"] + _CACHE_TTL > now:
        return
    _CACHE["artifact_index"] = _read_json(ARTIFACT_INDEX, {"artifacts": []})
    _CACHE["model_registry"] = _read_json(MODEL_REGISTRY, {"models": [], "latest": {}})
    _CACHE["run_registry"] = _read_json(RUN_REGISTRY, {"runs": []})
    _CACHE["timestamp"] = now


def list_artifacts(artifact_type: Optional[str] = None,
                   strategy: Optional[str] = None,
                   seed: Optional[int] = None,
                   category: Optional[str] = None) -> List[Dict[str, Any]]:
    _ensure_dirs();
    _refresh_cache()
    data = _CACHE["artifact_index"]
    out = []
    for rec in data.get("artifacts", []):
        if artifact_type and rec.get("artifact_type") != artifact_type: continue
        if strategy and rec.get("strategy") != strategy: continue
        if seed is not None and rec.get("seed") != seed: continue
        if category and rec.get("category") != category: continue
        out.append(rec)
    return out


def get_latest_artifact(artifact_type: str,
                        strategy: Optional[str] = None,
                        seed: Optional[int] = None,
                        category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    matches = list_artifacts(artifact_type, strategy, seed, category)
    if not matches:
        return None

    def _ts(rec):
        ts = rec.get("created_at")
        if ts: return ts
        return rec.get("run_id", "")

    matches.sort(key=_ts)
    return matches[-1]


def get_latest_model(strategy: str, seed: int, level: str = "high") -> Optional[Dict[str, Any]]:
    _ensure_dirs();
    _refresh_cache()
    reg = _CACHE["model_registry"]
    key = f"{strategy}_seed{seed}_{level}_level"
    path = reg.get("latest", {}).get(key)
    if not path:
        for m in reversed(reg.get("models", [])):
            if m.get("strategy") == strategy and m.get("seed") == seed and m.get("low_or_high") == f"{level}_level":
                return m
        return None
    for m in reg.get("models", []):
        if m.get("path") == path:
            return m
    return None


def register_run(run_meta: Dict[str, Any]):
    _ensure_dirs();
    _refresh_cache()
    reg = _read_json(RUN_REGISTRY, {"runs": []})
    reg["runs"].append(run_meta)
    _write_json(RUN_REGISTRY, reg)
    _CACHE["run_registry"] = reg


def register_model(model_meta: Dict[str, Any]):
    _ensure_dirs();
    _refresh_cache()
    reg = _read_json(MODEL_REGISTRY, {"models": [], "latest": {}})
    reg["models"].append(model_meta)
    key = f"{model_meta['strategy']}_seed{model_meta['seed']}_{model_meta['low_or_high']}"
    reg["latest"][key] = model_meta["path"]
    _write_json(MODEL_REGISTRY, reg)
    _CACHE["model_registry"] = reg


def register_artifact(record: Dict[str, Any]):
    _ensure_dirs();
    _refresh_cache()
    idx = _read_json(ARTIFACT_INDEX, {"artifacts": []})
    idx["artifacts"].append(record)
    _write_json(ARTIFACT_INDEX, idx)
    _CACHE["artifact_index"] = idx


def artifact_metadata_template(**kwargs):
    meta = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "version_tag": "v1"
    }
    meta.update(kwargs)
    return meta


def resolve_curve(strategy: str, seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
    rec = get_latest_artifact("curve", strategy=strategy, seed=seed, category="hierarchy")
    if not rec:
        return None
    path = Path(rec["path"])
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_metric_index() -> Dict[str, Any]:
    strategies = ["hiro", "norelabel", "randomrelabel", "baseline"]
    metrics = {}
    for st in strategies:
        cur = get_latest_artifact("curve", strategy=st, category="hierarchy")
        if cur:
            try:
                data = json.loads(Path(cur["path"]).read_text(encoding="utf-8"))
            except Exception:
                data = {}
            metrics[st] = {
                "early_reward_auc_first20pct": data.get("early_reward_auc_first20pct"),
                "last_25pct_tracking_error_mean": data.get("last_25pct_tracking_error_mean"),
                "mismatch_trend_slope": data.get("mismatch_trend_slope"),
                "time_to_threshold": data.get("time_to_threshold"),
                "relabel_td_contribution_last": (data.get("relabel_td_contribution_curve") or [None])[-1]
            }
    sv_mlp = get_latest_artifact("summary", strategy="hiro", category="hierarchy")
    if sv_mlp and "structured_vs_mlp" in sv_mlp.get("run_id", ""):
        try:
            comp = json.loads(Path(sv_mlp["path"]).read_text(encoding="utf-8"))
            metrics["structured_vs_mlp"] = comp.get("delta", {})
            metrics["structured_vs_mlp"]["structured_auc"] = comp.get("structured", {}).get("early_auc") or comp.get(
                "structured", {}).get("early_auc_mean")
            metrics["structured_vs_mlp"]["mlp_auc"] = comp.get("mlp", {}).get("early_auc") or comp.get("mlp", {}).get(
                "early_auc_mean")
        except Exception:
            pass
    prog = get_latest_artifact("progress", strategy="baseline", seed=None, category="baseline")
    if prog:
        try:
            pjson = json.loads(Path(prog["path"]).read_text(encoding="utf-8"))
            if isinstance(pjson, list) and pjson:
                last = pjson[-1]
                metrics["baseline"] = {
                    "tracking_error_last": last.get("power_tracking_error_rel_mean_avg"),
                    "seeds_count": last.get("seeds_count")
                }
        except Exception:
            pass
    return {"metrics": metrics}


def get_pipeline_status() -> Dict[str, Any]:
    status = {}
    baseline_progress = get_latest_artifact("progress", "baseline", None, "baseline")
    status["baseline_batch"] = "DONE" if baseline_progress else "MISSING"

    outlier_report = None
    for rpt in list_artifacts("report", "baseline", None, "baseline"):
        if "baseline_outlier_report" in rpt.get("path", "") or "outlier_report" in rpt.get("run_id", ""):
            outlier_report = rpt;
            break
    status["baseline_outlier_filter"] = "DONE" if outlier_report else "WARN"

    hiro_curve = get_latest_artifact("curve", "hiro", None, "hierarchy")
    status["hierarchy_main"] = "DONE" if hiro_curve else "MISSING"

    hiro_seeds = list_artifacts("curve", "hiro", None, "hierarchy")
    norel_seeds = list_artifacts("curve", "norelabel", None, "hierarchy")
    rand_seeds = list_artifacts("curve", "randomrelabel", None, "hierarchy")
    status["multi_seed_hiro"] = "DONE" if len(hiro_seeds) >= 3 else "WARN"
    status["multi_seed_norelabel"] = "DONE" if len(norel_seeds) >= 3 else "WARN"
    status["multi_seed_randomrelabel"] = "DONE" if len(rand_seeds) >= 3 else "WARN"

    signif = None
    for s in list_artifacts("summary", "hiro", None, "hierarchy"):
        if "significance" in s.get("run_id", ""): signif = s; break
    status["significance_tests"] = "DONE" if signif else "WARN"

    structured_summary = None
    for s in list_artifacts("summary", "hiro", None, "hierarchy"):
        if "structured_vs_mlp" in s.get("run_id", ""): structured_summary = s; break
    status["structured_vs_mlp"] = "DONE" if structured_summary else "WARN"

    timeseries = list_artifacts("timeseries", "hiro", None, "hierarchy")
    status["timeseries_compare"] = "DONE" if timeseries else "WARN"

    metrics_all = get_metric_index()["metrics"]
    flags = {}
    try:
        h_auc = metrics_all.get("hiro", {}).get("early_reward_auc_first20pct")
        n_auc = metrics_all.get("norelabel", {}).get("early_reward_auc_first20pct")
        if h_auc and n_auc and n_auc > 0:
            flags["early_auc_improve_vs_norelabel_pass"] = (h_auc - n_auc) / n_auc * 100.0 >= 5.0
        else:
            flags["early_auc_improve_vs_norelabel_pass"] = False
    except Exception:
        flags["early_auc_improve_vs_norelabel_pass"] = False

    locations = {}
    if hiro_curve: locations["hiro_curve"] = hiro_curve["path"]
    if baseline_progress: locations["baseline_progress"] = baseline_progress["path"]

    return {"steps": status, "quality_flags": flags, "locations": locations}
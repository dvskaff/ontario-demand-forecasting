"""Production monitoring: score every past forecast against what actually happened, and check input drift.

Live accuracy uses the forecast that was really issued (with the weather forecast available at the time), so it is
the honest number; the backtest uses observed temperatures and is a best case.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DRIFT_COLUMNS = ["temp", "temp_day_mean", "lag24", "lag168", "prev_day_mean", "demand_mw"]


def load_log(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def save_forecast(path: Path, forecast: pd.DataFrame) -> pd.DataFrame:
    """Append the new forecast; a re-run for the same target day replaces the earlier one."""
    old = load_log(path)
    if len(old):
        old = old[old["target_day"] != forecast["target_day"].iloc[0]]
    out = pd.concat([old, forecast], ignore_index=True).sort_values("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def score(log_df: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Forecast log joined to actuals, one row per hour that has both."""
    if log_df.empty:
        return log_df
    j = log_df.merge(
        demand["demand_mw"].rename("actual"), left_on="ts", right_index=True, how="inner"
    ).dropna(subset=["actual"])
    full_days = j.groupby("target_day")["ts"].transform("count") == 24
    j = j[full_days].copy()
    j["ape"] = (j["p50"] - j["actual"]).abs() / j["actual"] * 100
    j["inside"] = (j["actual"] >= j["p10"]) & (j["actual"] <= j["p90"])
    return j


def daily_scores(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["target_day", "mape", "coverage_80", "peak_ape", "model_version"])
    g = scored.groupby("target_day")
    out = g.agg(
        mape=("ape", "mean"), coverage_80=("inside", "mean"), model_version=("model_version", "first")
    )
    peaks = g.apply(
        lambda d: abs(d["p50"].max() - d["actual"].max()) / d["actual"].max() * 100, include_groups=False
    )
    out["peak_ape"] = peaks
    out["coverage_80"] *= 100
    return out.reset_index()


def live_summary(daily: pd.DataFrame, days: int) -> dict:
    d = daily.sort_values("target_day").tail(days)
    if d.empty:
        return {"days": 0}
    return {"days": int(len(d)), "mape": float(d["mape"].mean()), "coverage_80": float(d["coverage_80"].mean()),
            "peak_ape": float(d["peak_ape"].mean()), "last7_mape": float(d.tail(7)["mape"].mean())}  # fmt: skip


def drift(features: pd.DataFrame, end: pd.Timestamp, days: int, html_path: Path | None) -> dict:
    """Evidently data-drift report: the last `days` days against the same calendar window a year earlier
    (a seasonal reference, so normal winter-vs-summer change is not flagged as drift)."""
    cur = features[(features.index > end - pd.Timedelta(days=days)) & (features.index <= end)]
    ref = features[
        (features.index > end - pd.Timedelta(days=days + 365))
        & (features.index <= end - pd.Timedelta(days=365))
    ]
    cur, ref = cur[DRIFT_COLUMNS].dropna(), ref[DRIFT_COLUMNS].dropna()
    if len(cur) < 24 * 7 or len(ref) < 24 * 7:
        return {"available": False, "reason": "not enough history for a seasonal reference"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snap = Report([DataDriftPreset()]).run(cur.reset_index(drop=True), ref.reset_index(drop=True))
        if html_path:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            snap.save_html(str(html_path))
        result = snap.dict()
    columns, share = [], None
    for m in result["metrics"]:
        name, value = m.get("metric_name", ""), m.get("value")
        if name.startswith("DriftedColumnsCount"):
            share = float(value["share"])
        elif name.startswith("ValueDrift"):
            col = name.split("column=")[1].split(",")[0]
            method = name.split("method=")[1].split(",")[0]
            threshold = float(name.split("threshold=")[1].rstrip(")"))
            score = float(value)
            # p-value style methods drift when the value is below the threshold; distance-style ones when above
            drifted = score < threshold if "p_value" in method else score > threshold
            columns.append(
                {"column": col, "method": method, "score": score, "threshold": threshold, "drifted": drifted}
            )
    return {"available": True, "drift_share": share, "columns": columns,
            "window_end": str(end.date()),
            "reference_means": ref.mean().round(1).to_dict(), "current_means": cur.mean().round(1).to_dict(),
            "drifted": int(sum(c["drifted"] for c in columns)), "n_columns": len(columns)}  # fmt: skip


def safe_float(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

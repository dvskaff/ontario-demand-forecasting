"""The daily production run.

1. Ingest: IESO demand and Open-Meteo temperatures, with sanity checks.
2. Monitor: score every past forecast against actuals; check drift with Evidently.
3. Retrain (weekly, when there is no champion, or when live accuracy degrades): train a challenger, backtest it,
   compare it with the champion on the latest week neither has seen, export to ONNX, verify parity, log and register
   it in MLflow, and promote it only if it passes the gate.
4. Forecast the next day with the champion and log the forecast.
5. Publish: the API's model bundle, the dashboard, the README results.

python -m demand.pipeline              # real data
python -m demand.pipeline --fixture    # offline run on synthetic data (tests and CI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from . import monitor, registry
from .config import Config
from .features import FEATURES, HISTORY_HOURS, TARGET, build_features, inference_features, training_rows
from .ingest import check_demand, get_source, weighted_temperature

log = logging.getLogger("demand")
MAX_PARITY_MW = 1.0  # exported ONNX models must match LightGBM to within 1 MW


def last_complete_day(demand: pd.DataFrame, before) -> pd.Timestamp:
    counts = demand["demand_mw"].dropna().groupby(demand["demand_mw"].dropna().index.normalize()).count()
    full = counts[(counts == 24) & (counts.index < pd.Timestamp(before))]
    if full.empty:
        raise RuntimeError("No complete day of demand data")
    return pd.Timestamp(full.index.max())


# --------------------------------------------------------------------------- retraining
def retrain(cfg: Config, rows: pd.DataFrame, data_end: pd.Timestamp, champ: dict | None, reason: str) -> dict:
    # Gate window: the latest days (up to gate_days) that the champion was NOT trained on, so neither model has
    # seen them. If the champion is very recent (early retrain), at least 3 days are used.
    gate_start = data_end - pd.Timedelta(days=cfg.gate_days - 1)
    if champ and champ["tags"].get("data_end"):
        unseen = pd.Timestamp(champ["tags"]["data_end"]) + pd.Timedelta(days=1)
        gate_start = min(max(gate_start, unseen), data_end - pd.Timedelta(days=2))
    bt, oos = M.backtest(rows, cfg.backtest_folds)
    adjust = M.conformal_adjust(oos["actual"], oos) if len(oos) else 0.0  # conformal widening of the 80% band
    gate = rows[rows.index >= gate_start]
    earlier = rows[rows.index < gate_start]
    challenger_eval = M.fit(earlier)
    gate_scores = {"challenger": M.metrics(gate[TARGET], M.predict(challenger_eval, gate, adjust)),
                   "seasonal_naive": M.metrics(gate[TARGET], M.baseline(gate, "Same hour last week"))}  # fmt: skip
    if champ:
        gate_scores["champion"] = M.metrics(
            gate[TARGET], M.OnnxModels.from_dir(champ["onnx_dir"]).predict(gate)
        )

    overall = bt.groupby("model")[["mape", "mae", "peak_ape", "coverage_80"]].mean()
    beats_naive = overall.loc["LightGBM (quantile)", "mape"] < overall.loc["Same hour last week", "mape"]
    if champ is None:
        promote, why = (
            beats_naive,
            "first model" if beats_naive else "does not beat the seasonal-naive baseline",
        )
    else:
        limit = gate_scores["champion"]["mape"] * cfg.gate_tolerance
        ok = gate_scores["challenger"]["mape"] <= limit
        promote = bool(beats_naive and ok)
        why = (f"MAPE {gate_scores['challenger']['mape']:.2f}% vs champion "
               f"{gate_scores['champion']['mape']:.2f}% (limit {limit:.2f}%) on {gate_start.date()} to {data_end.date()}")  # fmt: skip
        if not beats_naive:
            why = "does not beat the seasonal-naive baseline in the backtest"

    final = M.fit(rows)  # the version that would ship is trained on all data, including the gate week
    tmp = Path(tempfile.mkdtemp())
    onnx_dir = tmp / "onnx"
    onnx_dir.mkdir()
    for q, blob in M.to_onnx(final).items():
        (onnx_dir / f"{M.QCOLS[q]}.onnx").write_bytes(blob)
    registry.write_json(onnx_dir / "model.json", {**M.SPEC, "interval_adjust_mw": adjust})
    diff = M.parity(final, M.OnnxModels.from_dir(onnx_dir), rows.tail(24 * 30))
    if diff > MAX_PARITY_MW:
        raise RuntimeError(f"ONNX export does not match LightGBM (max difference {diff:.3f} MW)")
    imp = M.importance(final)
    bt.to_csv(tmp / "backtest.csv", index=False)
    imp.to_csv(tmp / "importance.csv", index=False)
    meta = {"features": FEATURES, "quantiles": list(M.QCOLS.values()), "train_start": str(rows.index.min()),
            "train_end": str(rows.index.max()), "rows": len(rows), "onnx_parity_mw": diff}  # fmt: skip
    registry.write_json(tmp / "metadata.json", meta)
    metrics = {f"backtest_{k}": float(v) for k, v in overall.loc["LightGBM (quantile)"].items()}
    metrics |= {f"backtest_naive_{k}": float(v) for k, v in overall.loc["Same hour last week"].items()}
    for who, s in gate_scores.items():
        metrics |= {f"gate_{who}_mape": s["mape"], f"gate_{who}_mae": s["mae"]}
    metrics["onnx_parity_mw"] = diff
    metrics["interval_adjust_mw"] = adjust
    params = {**{k: v for k, v in M.PARAMS.items() if k not in ("verbose", "num_threads")}, "rounds": M.ROUNDS,
              "quantiles": ",".join(map(str, cfg.quantiles)), "features": len(FEATURES)}  # fmt: skip
    tags = {"status": "champion" if promote else "rejected", "decision": why, "trigger": reason,
            "gate_window": f"{gate_start.date()} to {data_end.date()}",
            "data_end": str(data_end.date()), "train_rows": str(len(rows))}  # fmt: skip
    version = registry.log_version(cfg, onnx_dir, params, metrics, tags,
                                   {"reports": tmp / "backtest.csv", "reports/importance": tmp / "importance.csv",
                                    "reports/metadata": tmp / "metadata.json"})  # fmt: skip
    if promote:
        registry.promote(cfg, version)
    pruned = registry.prune(cfg)
    log.info("Retrain (%s): version %s %s: %s", reason, version, "PROMOTED" if promote else "rejected", why)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"version": version, "promoted": bool(promote), "decision": why, "trigger": reason, "gate": gate_scores,
            "backtest": bt.to_dict(orient="records"), "parity_mw": diff, "pruned": pruned}  # fmt: skip


def retrain_reason(cfg: Config, champ: dict | None, live: dict, today) -> str | None:
    if os.environ.get("DEMAND_FORCE_RETRAIN") == "1":
        return "manual"
    if champ is None:
        return "no champion yet"
    if pd.Timestamp(today).dayofweek == cfg.retrain_weekday:
        return "weekly schedule"
    bt = champ["metrics"].get("backtest_mape")
    if live.get("days", 0) >= 7 and bt and live["last7_mape"] > cfg.live_alert_ratio * bt:
        return f"live error alert ({live['last7_mape']:.2f}% vs backtest {bt:.2f}%)"
    return None


# --------------------------------------------------------------------------- forecasting & serving bundle
def forecast(cfg: Config, frame: pd.DataFrame, target: pd.Timestamp, champ: dict) -> pd.DataFrame:
    hours = pd.date_range(target, periods=24, freq="h")
    temps = frame["temp_c"].reindex(hours)
    if temps.isna().any():
        raise RuntimeError(f"No temperature forecast for {target.date()}")
    history = frame[frame.index < target]
    feats = inference_features(history, target.date(), temps.tolist())
    pred = M.OnnxModels.from_dir(champ["onnx_dir"]).predict(feats)
    out = pred.reset_index(names="ts")
    out.insert(0, "target_day", str(target.date()))
    out["hour_ending"] = out["ts"].dt.hour + 1
    out["temp_forecast"] = temps.to_numpy()
    out["model_version"] = champ["version"]
    out["issued_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return out


def write_bundle(cfg: Config, champ: dict, frame: pd.DataFrame, target: pd.Timestamp) -> bool:
    """Everything the API image needs: the champion's ONNX models, metadata, and an example request.
    Returns True when the model changed (so CI knows to build and push a new image)."""
    b = cfg.bundle_dir
    b.mkdir(parents=True, exist_ok=True)
    meta_path = b / "metadata.json"
    before = json.loads(meta_path.read_text()).get("version") if meta_path.exists() else None
    for f in [*Path(champ["onnx_dir"]).glob("*.onnx"), *Path(champ["onnx_dir"]).glob("model.json")]:
        shutil.copy(f, b / f.name)
    spec_file = Path(champ["onnx_dir"]) / "model.json"
    spec = json.loads(spec_file.read_text()) if spec_file.exists() else {"target": "absolute"}
    hist = frame[(frame.index < target) & (frame.index >= target - pd.Timedelta(hours=HISTORY_HOURS))]
    example = {"date": str(target.date()),
               "history": [{"ts": ts.isoformat(), "demand_mw": float(r.demand_mw), "temp_c": round(float(r.temp_c), 2)}
                           for ts, r in hist.iterrows()],
               "temperature_c": [round(float(v), 2) for v in frame["temp_c"].reindex(pd.date_range(target, periods=24, freq="h"))]}  # fmt: skip
    registry.write_json(b / "example_request.json", example)
    m = champ["metrics"]
    registry.write_json(meta_path, {"name": cfg.model_name, "version": champ["version"], "features": FEATURES,
                                    "quantiles": ["p10", "p50", "p90"], "data_end": champ["tags"].get("data_end"),
                                    "backtest_mape": m.get("backtest_mape"), "backtest_coverage_80": m.get("backtest_coverage_80"),
                                    "timezone": "EST (UTC-5, no daylight saving)",
                                    "target": spec.get("target", "absolute"),
                                    "interval_adjust_mw": spec.get("interval_adjust_mw", 0.0)})  # fmt: skip
    return before != champ["version"]


# --------------------------------------------------------------------------- main
def run(cfg: Config) -> dict:
    t0 = time.time()
    src = get_source(cfg)
    today = src.today()
    demand = src.demand()
    dq = check_demand(demand)
    temp = weighted_temperature(src.weather())
    frame = demand.join(temp, how="outer").sort_index()
    data_end_day = last_complete_day(demand, today)
    frame.loc[frame.index >= data_end_day + pd.Timedelta(days=1), "demand_mw"] = (
        np.nan
    )  # never peek at partial days
    feats = build_features(frame)
    rows = training_rows(feats[feats.index < data_end_day + pd.Timedelta(days=1)])
    log.info(
        "Demand through %s (%s training rows); data checks %s", data_end_day.date(), f"{len(rows):,}", dq
    )

    log_df = monitor.load_log(cfg.forecast_log)
    scored = monitor.score(log_df, demand)
    daily = monitor.daily_scores(scored)
    live = monitor.live_summary(daily, 28)

    champ = registry.champion(cfg)
    reason = retrain_reason(cfg, champ, live, today)
    retrained = retrain(cfg, rows, data_end_day, champ, reason) if reason else None
    champ = registry.champion(cfg)
    if champ is None:
        raise RuntimeError("No champion model: the first model did not pass the gate")

    target = data_end_day + pd.Timedelta(days=1)
    fc = forecast(cfg, frame, target, champ)
    log_df = monitor.save_forecast(cfg.forecast_log, fc)
    drift = monitor.drift(
        feats, data_end_day + pd.Timedelta(hours=23), cfg.drift_days, cfg.site_dir / "drift.html"
    )
    model_changed = write_bundle(cfg, champ, frame, target)

    run_info = {"today": str(today), "data_end": str(data_end_day.date()), "target_day": str(target.date()),
                "data_checks": dq, "live": live, "retrain": retrained, "drift": drift, "model_changed": model_changed,
                "champion": {k: champ[k] for k in ("version", "tags", "metrics")}, "seconds": round(time.time() - t0)}  # fmt: skip
    from .readme import update_readme
    from .report import build_site

    summary = build_site(cfg, frame, fc, scored, daily, run_info)
    # Weather what-if: the champion's ONNX models run in the visitor's browser on shifted temperatures
    from plotly.offline import get_plotlyjs_version

    from . import whatif

    scenario = whatif.export(cfg, frame, target, champ, fc)
    runtime = whatif.copy_runtime(cfg.vendor_dir, cfg.site_dir)
    whatif.write_page(cfg.site_dir, scenario, runtime, cfg.repo_url, cfg.fixture, get_plotlyjs_version())
    if not cfg.fixture:
        update_readme(cfg.readme_path, summary)
    registry.write_json(cfg.state_dir / "last_run.json", run_info)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(
                f"model_changed={'true' if model_changed else 'false'}\nmodel_version={champ['version']}\n"
            )
    log.info(
        "Done in %ss: forecast for %s with model v%s", run_info["seconds"], target.date(), champ["version"]
    )
    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", action="store_true", help="offline run on synthetic data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for name in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.fixture:
        os.environ.setdefault("DEMAND_FIXTURE", "1")
        os.environ.setdefault("DEMAND_BACKTEST_FOLDS", "3")
        base = Path(os.environ.get("DEMAND_FIXTURE_DIR", "/tmp/demand-fixture"))
        for k, sub in (("DEMAND_STATE_DIR", "state"), ("DEMAND_SITE_DIR", "site")):
            os.environ.setdefault(k, str(base / sub))
    run_info = run(Config())
    print(
        json.dumps({k: run_info[k] for k in ("data_end", "target_day", "model_changed", "live")}, default=str)
    )


if __name__ == "__main__":
    main()

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd

from demand import pipeline, registry
from demand.readme import update_readme

ROOT = Path(__file__).resolve().parents[1]


def test_daily_runs_retrain_gate_and_monitoring(cfg, monkeypatch):
    monkeypatch.setenv("DEMAND_FIXTURE_TODAY", "2024-04-11")  # Thursday: no champion yet -> first model
    first = pipeline.run(cfg)
    assert first["retrain"]["promoted"] and first["model_changed"]
    assert first["target_day"] == "2024-04-11" and first["live"] == {"days": 0}

    monkeypatch.setenv(
        "DEMAND_FIXTURE_TODAY", "2024-04-12"
    )  # Friday: no retrain, yesterday's forecast is scored
    second = pipeline.run(cfg)
    assert second["retrain"] is None and not second["model_changed"]
    assert second["live"]["days"] == 1 and second["live"]["mape"] < 5

    strict = replace(cfg, gate_tolerance=0.5)  # Monday retrain with an impossible gate: must be rejected
    monkeypatch.setenv("DEMAND_FIXTURE_TODAY", "2024-04-15")
    third = pipeline.run(strict)
    assert third["retrain"]["trigger"] == "weekly schedule" and not third["retrain"]["promoted"]
    assert registry.champion(cfg)["version"] == "1" and not third["model_changed"]

    monkeypatch.setenv("DEMAND_FIXTURE_TODAY", "2024-04-16")
    monkeypatch.setenv("DEMAND_FORCE_RETRAIN", "1")
    fourth = pipeline.run(replace(cfg, gate_tolerance=10.0))  # a gate that passes
    assert (
        fourth["retrain"]["promoted"] and registry.champion(cfg)["version"] == "3" and fourth["model_changed"]
    )
    statuses = {v["version"]: v.get("status") for v in registry.versions(cfg)}
    assert statuses == {1: "retired", 2: "rejected", 3: "champion"}

    log = pd.read_parquet(cfg.forecast_log)
    assert sorted(log["target_day"].unique()) == ["2024-04-11", "2024-04-12", "2024-04-15", "2024-04-16"]
    assert (log.groupby("target_day").size() == 24).all()
    for f in ("p10.onnx", "p50.onnx", "p90.onnx", "metadata.json", "example_request.json"):
        assert (cfg.bundle_dir / f).exists()
    assert json.loads((cfg.bundle_dir / "metadata.json").read_text())["version"] == "3"
    page = (cfg.site_dir / "index.html").read_text()
    assert "SYNTHETIC TEST DATA" in page and (cfg.site_dir / "drift.html").exists()
    assert (cfg.site_dir / "openapi.json").exists()

    summary = json.loads((cfg.site_dir / "results.json").read_text())
    shutil.copy(ROOT / "README.md", cfg.readme_path)
    update_readme(cfg.readme_path, summary)
    text = cfg.readme_path.read_text()
    assert "Next-day forecast" in text and text.count("<!-- RESULTS:START -->") == 1

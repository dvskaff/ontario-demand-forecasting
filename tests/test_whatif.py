"""Weather what-if page: the shifted features must equal what the real feature code builds from warmer weather,
and the export must carry everything the page needs."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from demand import model as M
from demand import whatif
from demand.features import FEATURES, inference_features


@pytest.mark.parametrize("delta", [-10.0, -2.5, 0.0, 4.0, 10.0])
def test_shift_matches_the_feature_code(frame, delta):
    day = pd.Timestamp("2024-02-14")  # inside the fixture's history, with full weather
    hours = pd.date_range(day, periods=24, freq="h")
    temps = frame["temp_c"].reindex(hours).tolist()
    history = frame[frame.index < day]
    base = inference_features(history, day.date(), temps)
    prev23 = (
        history["temp_c"].reindex(pd.date_range(day - pd.Timedelta(hours=23), periods=23, freq="h")).tolist()
    )
    mine = whatif.shift(base, prev23, temps, delta)
    truth = inference_features(history, day.date(), [t + delta for t in temps])
    pd.testing.assert_frame_equal(mine[FEATURES], truth[FEATURES], check_dtype=False, atol=1e-9)
    # only temperature columns move; demand history and calendar stay put
    moved = [c for c in FEATURES if not np.allclose(mine[c], base[c])]
    assert set(moved) <= set(whatif.TEMP_COLUMNS)


def test_export_and_page(frame, rows, tmp_path):
    models = M.fit(rows.tail(24 * 200))
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    for q, blob in M.to_onnx(models).items():
        (onnx_dir / f"{M.QCOLS[q]}.onnx").write_bytes(blob)
    (onnx_dir / "model.json").write_text(json.dumps({**M.SPEC, "interval_adjust_mw": 50.0}))
    target = pd.Timestamp("2024-04-10")
    hours = pd.date_range(target, periods=24, freq="h")
    feats = inference_features(
        frame[frame.index < target], target.date(), frame["temp_c"].reindex(hours).tolist()
    )
    fc = M.OnnxModels.from_dir(onnx_dir).predict(feats).reset_index(drop=True)
    fc["hour_ending"] = range(1, 25)
    cfg = SimpleNamespace(site_dir=tmp_path / "site")
    data = whatif.export(cfg, frame, target, {"version": "7", "onnx_dir": str(onnx_dir)}, fc)
    assert len(data["rows"]) == 24 and len(data["rows"][0]) == len(FEATURES) and len(data["prev23"]) == 23
    assert data["spec"]["interval_adjust_mw"] == 50.0 and len(data["forecast"]["p50"]) == 24
    assert all((cfg.site_dir / "model" / f"{q}.onnx").exists() for q in ("p10", "p50", "p90"))
    lo, hi = data["trained_temp_range"]
    assert lo < min(data["temps"]) and hi > max(data["temps"]) - 15
    # warmer days in a cold month need less heating: the model should forecast less demand
    sessions = M.OnnxModels.from_dir(onnx_dir)
    cold = sessions.predict(whatif.shift(feats, data["prev23"], data["temps"], -8))["p50"].sum()
    warm = sessions.predict(whatif.shift(feats, data["prev23"], data["temps"], +8))["p50"].sum()
    assert cold > warm
    assert whatif.copy_runtime(tmp_path / "missing", cfg.site_dir) is False
    whatif.write_page(cfg.site_dir, data, False, "https://example.org", True, "2.35.2")
    page = (cfg.site_dir / "whatif.html").read_text()
    assert "SYNTHETIC TEST DATA" in page and "data/whatif.json" in page and "model/${q}.onnx" in page

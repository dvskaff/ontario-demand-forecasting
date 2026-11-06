import json

import numpy as np
import pytest

from demand import model as M
from demand.features import TARGET


@pytest.fixture(scope="module")
def fitted(rows):
    train, test = rows[rows.index < "2024-03-01"], rows[rows.index >= "2024-03-01"]
    return M.fit(train), test


def test_model_beats_naive_and_intervals_are_sane(fitted):
    models, test = fitted
    pred = M.predict(models, test)
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    m = M.metrics(test[TARGET], pred)
    naive = M.metrics(test[TARGET], M.baseline(test, "Same hour last week"))
    assert m["mape"] < naive["mape"]
    assert 50 < m["coverage_80"] < 97


def test_onnx_export_matches_lightgbm(fitted, tmp_path):
    models, test = fitted
    for q, blob in M.to_onnx(models).items():
        (tmp_path / f"{M.QCOLS[q]}.onnx").write_bytes(blob)
    (tmp_path / "model.json").write_text(json.dumps({**M.SPEC, "interval_adjust_mw": 40.0}))
    onnx = M.OnnxModels.from_dir(tmp_path)
    assert M.parity(models, onnx, test) < 1.0
    (tmp_path / "model.json").unlink()  # a version saved before model.json existed predicted absolute MW
    assert M.OnnxModels.from_dir(tmp_path).spec == {"target": "absolute"}


def test_backtest_and_conformal_calibration(rows):
    bt, oos = M.backtest(rows, folds=2)
    assert set(bt["model"]) == {"LightGBM (quantile)", *M.BASELINES}
    assert bt["month"].nunique() == 2 and np.isfinite(bt["mape"]).all()
    adj = M.conformal_adjust(oos["actual"], oos)
    widened = oos.assign(p10=oos["p10"] - adj, p90=oos["p90"] + adj)
    inside = ((oos["actual"] >= widened["p10"]) & (oos["actual"] <= widened["p90"])).mean()
    assert 0.79 <= inside <= 0.83  # calibrated to 80% on these errors


def test_ratio_target_follows_a_new_demand_level(frame):
    """Tree models cannot extrapolate a level they never saw; predicting relative to the recent level can."""
    from demand.features import build_features, training_rows

    shifted = frame.copy()
    shifted.loc[shifted.index >= "2024-01-01", "demand_mw"] *= (
        1.08  # demand jumps 8% to a level never seen before
    )
    rows = training_rows(build_features(shifted))
    models = M.fit(rows[rows.index < "2024-02-01"])
    test = rows[(rows.index >= "2024-02-01") & (rows.index < "2024-03-01")]
    bias = M.metrics(test[TARGET], M.predict(models, test))["bias"]
    assert abs(bias) / test[TARGET].mean() < 0.01

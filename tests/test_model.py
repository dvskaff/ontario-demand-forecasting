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
    assert M.parity(models, M.OnnxModels.from_dir(tmp_path), test) < 1.0


def test_backtest_shape(rows):
    bt = M.backtest(rows, folds=2)
    assert set(bt["model"]) == {"LightGBM (quantile)", *M.BASELINES}
    assert bt["month"].nunique() == 2 and np.isfinite(bt["mape"]).all()

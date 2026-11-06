"""Quantile LightGBM models (10th, 50th and 90th percentiles), backtesting, baselines and ONNX export."""

from __future__ import annotations

import logging
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, TARGET, to_matrix

log = logging.getLogger(__name__)
PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, deterministic=True, force_row_wise=True, num_threads=4)  # fmt: skip
ROUNDS = 600
QCOLS = {0.1: "p10", 0.5: "p50", 0.9: "p90"}


def fit(rows: pd.DataFrame, quantiles=(0.1, 0.5, 0.9), seed: int = 42) -> dict[float, lgb.Booster]:
    X, y = to_matrix(rows), rows[TARGET].to_numpy()
    return {
        q: lgb.train({**PARAMS, "objective": "quantile", "alpha": q, "seed": seed}, lgb.Dataset(X, y), ROUNDS)
        for q in quantiles
    }


def _frame(preds: dict[float, np.ndarray], index) -> pd.DataFrame:
    p = np.sort(np.column_stack([preds[q] for q in sorted(preds)]), axis=1)  # no crossing quantiles
    return pd.DataFrame(p, index=index, columns=[QCOLS[q] for q in sorted(preds)])


def predict(models: dict[float, lgb.Booster], feats: pd.DataFrame) -> pd.DataFrame:
    X = to_matrix(feats)
    return _frame({q: m.predict(X) for q, m in models.items()}, feats.index)


# --------------------------------------------------------------------------- metrics
def metrics(y: pd.Series, pred: pd.DataFrame) -> dict:
    """Errors in MW and %; daily-peak error; how often the truth fell inside the 10-90% band; pinball loss."""
    y = y.loc[pred.index].to_numpy(dtype=float)
    p50 = pred["p50"].to_numpy()
    out = {"hours": int(len(y)), "mape": float(np.mean(np.abs(p50 - y) / y) * 100), "mae": float(np.mean(np.abs(p50 - y))),
           "bias": float(np.mean(p50 - y))}  # fmt: skip
    day = pd.Index(pred.index).normalize()
    peaks = pd.DataFrame({"y": y, "p": p50}, index=day).groupby(level=0).max()
    out["peak_ape"] = float(np.mean(np.abs(peaks["p"] - peaks["y"]) / peaks["y"]) * 100)
    if {"p10", "p90"} <= set(pred.columns):
        out["coverage_80"] = float(
            np.mean((y >= pred["p10"].to_numpy()) & (y <= pred["p90"].to_numpy())) * 100
        )
        pin = 0.0
        for q, c in QCOLS.items():
            e = y - pred[c].to_numpy()
            pin += np.mean(np.maximum(q * e, (q - 1) * e))
        out["pinball"] = float(pin / 3)
    return out


BASELINES = {"Same hour last week": "lag168", "Same hour yesterday": "lag24"}


def baseline(feats: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame({"p50": feats[BASELINES[name]].to_numpy()}, index=feats.index)


def backtest(rows: pd.DataFrame, folds: int, seed: int = 42, min_train_days: int = 365) -> pd.DataFrame:
    """Rolling-origin evaluation: for each of the last `folds` complete months, train on everything before the month
    and forecast every day of it (with observed temperatures, so this is a best case for the weather input)."""
    months = pd.Index(rows.index.to_period("M").unique()).sort_values()
    last_complete = months[-2] if rows.index.max().day < 28 else months[-1]
    out = []
    for m in [p for p in months if p <= last_complete][-folds:]:
        start, end = m.start_time, m.end_time
        train, test = rows[rows.index < start], rows[(rows.index >= start) & (rows.index <= end)]
        if (train.index.max() - train.index.min()).days < min_train_days or test.empty:
            continue
        models = fit(train, seed=seed)
        out.append(
            {"month": str(m), "model": "LightGBM (quantile)", **metrics(test[TARGET], predict(models, test))}
        )
        for name in BASELINES:
            out.append({"month": str(m), "model": name, **metrics(test[TARGET], baseline(test, name))})
        log.info("Backtest %s: MAPE %.2f%%", m, out[-3]["mape"])
    return pd.DataFrame(out)


def importance(models: dict[float, lgb.Booster]) -> pd.DataFrame:
    gain = models[0.5].feature_importance("gain")
    return pd.DataFrame({"feature": FEATURES, "gain_share": gain / gain.sum()}).sort_values(
        "gain_share", ascending=False
    )


# --------------------------------------------------------------------------- ONNX
def to_onnx(models: dict[float, lgb.Booster]) -> dict[float, bytes]:
    """Convert each booster to ONNX so the API can serve it with onnxruntime alone (no LightGBM in the image)."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q, m in models.items():
            onx = convert_lightgbm(m, initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
                                   target_opset=15)  # fmt: skip
            out[q] = onx.SerializeToString()
    return out


class OnnxModels:
    """The three quantile models as onnxruntime sessions (what the API and the champion checks use)."""

    def __init__(self, blobs: dict[float, bytes]):
        import onnxruntime as ort

        self.sessions = {
            q: ort.InferenceSession(b, providers=["CPUExecutionProvider"]) for q, b in blobs.items()
        }

    @classmethod
    def from_dir(cls, path) -> OnnxModels:
        from pathlib import Path

        return cls({q: (Path(path) / f"{c}.onnx").read_bytes() for q, c in QCOLS.items()})

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = to_matrix(feats)
        return _frame(
            {q: s.run(None, {"features": X})[0].ravel() for q, s in self.sessions.items()}, feats.index
        )


def parity(models: dict[float, lgb.Booster], onnx: OnnxModels, feats: pd.DataFrame) -> float:
    """Largest difference (MW) between LightGBM and the exported ONNX models on the same rows."""
    return float((predict(models, feats) - onnx.predict(feats)).abs().to_numpy().max())

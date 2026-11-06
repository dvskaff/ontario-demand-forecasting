import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demand import model as M
from demand.features import FEATURES, HISTORY_HOURS, inference_features


@pytest.fixture(scope="module")
def client(rows, frame, tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle")
    models = M.fit(rows[rows.index < "2024-03-01"])
    for q, blob in M.to_onnx(models).items():
        (d / f"{M.QCOLS[q]}.onnx").write_bytes(blob)
    (d / "metadata.json").write_text(json.dumps({"version": "7", "features": FEATURES}))
    import app.main as api

    api.MODEL_DIR = d
    with TestClient(api.app) as c:  # startup loads the bundle
        yield c, M.OnnxModels.from_dir(d)


def request_for(frame, day: str) -> dict:
    t = pd.Timestamp(day)
    hist = frame[(frame.index < t) & (frame.index >= t - pd.Timedelta(hours=HISTORY_HOURS))]
    return {"date": day, "temperature_c": frame.loc[pd.date_range(t, periods=24, freq="h"), "temp_c"].tolist(),
            "history": [{"ts": ts.isoformat(), "demand_mw": r.demand_mw, "temp_c": r.temp_c} for ts, r in hist.iterrows()]}  # fmt: skip


def test_health_and_model(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "model_version": "7"}
    assert c.get("/model").json()["features"] == FEATURES


def test_forecast_matches_offline_model(client, frame):
    c, onnx = client
    body = request_for(frame, "2024-03-12")
    r = c.post("/forecast", json=body)
    assert r.status_code == 200, r.text
    hours = r.json()["hours"]
    assert len(hours) == 24 and hours[0]["hour_ending"] == 1
    assert all(h["p10"] <= h["p50"] <= h["p90"] for h in hours)
    t = pd.Timestamp("2024-03-12")
    hist = frame[(frame.index < t) & (frame.index >= t - pd.Timedelta(hours=HISTORY_HOURS))]
    offline = onnx.predict(inference_features(hist, t.date(), body["temperature_c"]))
    assert abs(offline["p50"].to_numpy() - [h["p50"] for h in hours]).max() < 0.1


def test_forecast_validation_errors(client, frame):
    c, _ = client
    body = request_for(frame, "2024-03-12")
    assert c.post("/forecast", json={**body, "temperature_c": [1.0] * 23}).status_code == 422
    assert c.post("/forecast", json={**body, "history": body["history"][:50]}).status_code == 422
    gap = {**body, "history": body["history"][:100] + body["history"][101:] + body["history"][:1]}
    r = c.post("/forecast", json=gap)
    assert r.status_code == 422 and "missing" in r.json()["detail"]

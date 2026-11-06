"""Forecast API: serves the champion model (ONNX, via onnxruntime) behind FastAPI.

Send the last 7 days of hourly demand and temperature plus the next day's 24 hourly temperatures; get back the
day's hourly demand forecast with an 80% prediction interval. Features are built by the same code used in training
(demand.features), so the model sees exactly what it was trained on.

    uvicorn app.main:app --port 8000          (from the serving/ folder, with src/ on PYTHONPATH)
    docker run -p 8000:8000 ghcr.io/dvskaff/ontario-demand-api
"""

from __future__ import annotations

import datetime as dt
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from demand.features import FEATURES, HISTORY_HOURS, inference_features, to_matrix

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "model"))
STATE: dict = {}


class HourObs(BaseModel):
    ts: dt.datetime = Field(
        description="Hour-beginning timestamp, Eastern Standard Time (UTC-5, no daylight saving)"
    )
    demand_mw: float = Field(gt=0, description="Ontario demand for that hour, MW")
    temp_c: float = Field(description="Temperature for that hour, °C")


class ForecastRequest(BaseModel):
    date: dt.date = Field(description="Day to forecast (EST)")
    history: list[HourObs] = Field(
        min_length=HISTORY_HOURS, description="At least the 168 hours before `date`"
    )
    temperature_c: list[float] = Field(
        min_length=24, max_length=24, description="Forecast temperature, 24 hours"
    )


class ForecastHour(BaseModel):
    ts: dt.datetime
    hour_ending: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    date: dt.date
    model_version: str
    unit: str = "MW"
    interval: str = "p10-p90 (80% prediction interval)"
    hours: list[ForecastHour]


def load(model_dir: Path | None = None) -> None:
    model_dir = model_dir or MODEL_DIR
    meta = json.loads((model_dir / "metadata.json").read_text())
    if meta["features"] != FEATURES:
        raise RuntimeError("Model was trained with a different feature list than this API builds")
    STATE["meta"] = meta
    STATE["sessions"] = {q: ort.InferenceSession(str(model_dir / f"{q}.onnx"), providers=["CPUExecutionProvider"])
                         for q in ("p10", "p50", "p90")}  # fmt: skip
    example = model_dir / "example_request.json"
    STATE["example"] = json.loads(example.read_text()) if example.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    yield


app = FastAPI(title="Ontario Electricity Demand Forecast API", version="1.0", lifespan=lifespan,
              description="Next-day hourly Ontario electricity demand with an 80% prediction interval.")  # fmt: skip


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": STATE["meta"]["version"]}


@app.get("/model")
def model_info() -> dict:
    return STATE["meta"]


@app.get("/forecast/example")
def example_request() -> dict:
    """A ready-to-send request built from real recent data (pipe it into POST /forecast)."""
    if not STATE.get("example"):
        raise HTTPException(404, "No example bundled with this model")
    return STATE["example"]


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    hist = pd.DataFrame([h.model_dump() for h in req.history])
    hist["ts"] = pd.to_datetime(hist["ts"]).dt.tz_localize(None)
    hist = hist.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    try:
        feats = inference_features(hist, req.date, req.temperature_c)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    X = to_matrix(feats)
    preds = np.sort(np.column_stack([STATE["sessions"][q].run(None, {"features": X})[0].ravel()
                                     for q in ("p10", "p50", "p90")]), axis=1)  # fmt: skip
    hours = [ForecastHour(ts=ts, hour_ending=ts.hour + 1, p10=round(float(a), 1), p50=round(float(b), 1),
                          p90=round(float(c), 1)) for ts, (a, b, c) in zip(feats.index, preds, strict=True)]  # fmt: skip
    return ForecastResponse(date=req.date, model_version=STATE["meta"]["version"], hours=hours)

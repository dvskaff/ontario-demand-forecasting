"""Features for next-day hourly demand. Used by BOTH training and the API, so a model always sees features built
the same way (no training/serving skew).

Timestamps are hour-beginning, Eastern Standard Time all year (IESO does not observe daylight saving).
Forecasting rule: to predict day D we may use demand up to the end of day D-1 and the temperature forecast for D.
Every demand feature below respects that rule.
"""

from __future__ import annotations

from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd

FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_holiday", "doy_sin", "doy_cos",
    "temp", "heating_degrees", "cooling_degrees", "temp_roll24", "temp_day_mean", "temp_day_max", "temp_prev_day_mean",
    "lag24", "lag168", "same_hour_mean_7d", "prev_day_mean", "prev_day_max", "prev_day_min",
]  # fmt: skip
TARGET = "demand_mw"
HISTORY_HOURS = 7 * 24  # hours of history the API needs before the target day


def ontario_holidays(years) -> set[date]:
    return set(holidays.country_holidays("CA", subdiv="ON", years=list(years)).keys())


def _daily(series: pd.Series, how: str) -> pd.Series:
    """Daily aggregate that is NaN unless the day has all 24 hours."""
    g = series.groupby(series.index.normalize())
    out = g.agg(how)
    return out.where(g.count() == 24)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """frame: hourly index, columns demand_mw and temp_c (demand may be NaN for the day being forecast)."""
    f = frame.sort_index()
    f = f[~f.index.duplicated(keep="last")].asfreq("h")
    idx = f.index
    d, t = f[TARGET], f["temp_c"]
    day = idx.normalize()
    hol = ontario_holidays(range(idx.year.min(), idx.year.max() + 1))

    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = np.isin(idx.date, list(hol)).astype(int)
    doy = 2 * np.pi * idx.dayofyear / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(doy), np.cos(doy)

    out["temp"] = t
    out["heating_degrees"] = (15.5 - t).clip(lower=0)
    out["cooling_degrees"] = (t - 18).clip(lower=0)
    out["temp_roll24"] = t.rolling(24, min_periods=24).mean()
    out["temp_day_mean"] = t.groupby(day).transform("mean").where(t.groupby(day).transform("count") == 24)
    out["temp_day_max"] = t.groupby(day).transform("max").where(t.groupby(day).transform("count") == 24)
    prev_temp = _daily(t, "mean").shift(1, freq="D")
    out["temp_prev_day_mean"] = prev_temp.reindex(day).to_numpy()

    out["lag24"] = d.shift(24)
    out["lag168"] = d.shift(168)
    out["same_hour_mean_7d"] = pd.concat([d.shift(24 * k) for k in range(1, 8)], axis=1).mean(
        axis=1, skipna=False
    )
    for how in ("mean", "max", "min"):
        out[f"prev_day_{how}"] = _daily(d, how).shift(1, freq="D").reindex(day).to_numpy()
    out[TARGET] = d
    return out


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    return features.dropna(subset=[*FEATURES, TARGET])


def inference_features(history: pd.DataFrame, target_day: date, temps: list[float]) -> pd.DataFrame:
    """The 24 feature rows for target_day, from the 7 days of history before it and that day's temperatures.

    history: hourly index with demand_mw and temp_c. Raises ValueError with a clear message if anything is missing."""
    if len(temps) != 24:
        raise ValueError(f"temperatures must have 24 hourly values for {target_day}, got {len(temps)}")
    start = pd.Timestamp(target_day)
    need = pd.date_range(start - timedelta(hours=HISTORY_HOURS), start - timedelta(hours=1), freq="h")
    hist = history.reindex(need)
    missing = hist[hist[[TARGET, "temp_c"]].isna().any(axis=1)].index
    if len(missing):
        raise ValueError(f"history is missing {len(missing)} of the {HISTORY_HOURS} hours before {target_day} "
                         f"(first missing: {missing[0]})")  # fmt: skip
    day = pd.DataFrame({TARGET: np.nan, "temp_c": temps}, index=pd.date_range(start, periods=24, freq="h"))
    feats = build_features(pd.concat([hist, day])).loc[day.index, FEATURES]
    if feats.isna().any().any():
        raise ValueError(
            f"could not build features for {target_day}: {feats.columns[feats.isna().any()].tolist()}"
        )
    return feats


def to_matrix(features: pd.DataFrame) -> np.ndarray:
    return features[FEATURES].to_numpy(dtype=np.float32)

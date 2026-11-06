"""SYNTHETIC demand and weather in the IESO / Open-Meteo shapes, used only by tests and the offline CI run.

Demand is generated from temperature (heating and cooling load), the daily and weekly cycle and holidays, plus noise,
so the tests can check the pipeline recovers those relationships. Pages built from it carry a red banner.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import CITIES, Config
from .features import ontario_holidays

START, TODAY, END = date(2022, 1, 1), date(2024, 4, 10), date(2024, 7, 1)


def synthetic(today: date = TODAY, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # One fixed series (same numbers whatever "today" is), then cut: demand up to yesterday, weather 3 days ahead.
    idx = pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq="h", inclusive="left")
    doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()
    season = 8 - 14 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    diurnal = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather_noise = pd.Series(rng.normal(0, 1, len(idx))).rolling(48, min_periods=1).mean().to_numpy() * 6
    cities = pd.DataFrame({c: season + diurnal + weather_noise + rng.normal(0, 0.7, len(idx)) + off
                           for c, off in zip(CITIES, (0.0, -2.5, 0.5), strict=True)}, index=idx)  # fmt: skip
    t = sum(cities[c] * CITIES[c][2] for c in CITIES) / sum(v[2] for v in CITIES.values())
    profile = 1500 * np.sin(np.pi * np.clip(hour - 5, 0, 17) / 17) - 600 * (hour < 5)
    weekend = np.where(idx.dayofweek >= 5, -1100, 0)
    hol = np.isin(idx.date, list(ontario_holidays(range(START.year, today.year + 1)))) * -1300
    demand = (14_500 + 110 * np.clip(15.5 - t, 0, None) + 260 * np.clip(t - 18, 0, None) + profile + weekend + hol
              + rng.normal(0, 180, len(idx)))  # fmt: skip
    d = pd.DataFrame({"demand_mw": demand.round(0)}, index=idx)
    d.index.name = "ts"
    horizon = pd.Timestamp(today + timedelta(days=3))
    return d[d.index < pd.Timestamp(today)], cities[cities.index < horizon]


class FixtureSource:
    def __init__(self, cfg: Config, today: date | None = None):
        import os

        today = today or date.fromisoformat(os.environ.get("DEMAND_FIXTURE_TODAY", TODAY.isoformat()))
        self._today = today
        self._d, self._w = synthetic(today)

    def today(self) -> date:
        return self._today

    def demand(self) -> pd.DataFrame:
        return self._d.copy()

    def weather(self) -> pd.DataFrame:
        return self._w.copy()


def ieso_csv(df: pd.DataFrame, year: int) -> str:
    """The same rows written in the IESO PUB_Demand layout (for parser tests)."""
    d = df[df.index.year == year]
    lines = ["\\Hourly Demand Report,,,", "\\Created at 2024-04-10 00:44:16,,,", f"\\For {year},,,",
             "Date,Hour,Market Demand,Ontario Demand"]  # fmt: skip
    for ts, v in d["demand_mw"].items():
        lines.append(f"{ts:%Y-%m-%d},{ts.hour + 1},{int(v) + 1500},{int(v)}")
    return "\n".join(lines) + "\n"

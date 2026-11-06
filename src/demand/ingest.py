"""Data sources: hourly Ontario demand from the IESO, and temperatures from Open-Meteo (both public, no key).

All timestamps are converted to hour-beginning Eastern Standard Time (UTC-5 all year), the IESO's convention.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CITIES, Config

log = logging.getLogger(__name__)
EST_OFFSET = timedelta(hours=-5)


class SourceError(RuntimeError):
    pass


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "ontario-demand-forecasting (github.com/dvskaff/ontario-demand-forecasting)"
    return s


def now_est() -> datetime:
    return (datetime.now(UTC) + EST_OFFSET).replace(tzinfo=None)


# --------------------------------------------------------------------------- IESO
def parse_ieso(text: str) -> pd.DataFrame:
    """PUB_Demand_YYYY.csv: a few comment lines, then Date, Hour (1-24, hour ending), Market Demand, Ontario Demand."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("date,hour"))
    except StopIteration as exc:
        raise SourceError(f"IESO file has no 'Date,Hour' header; it starts with: {lines[:5]}") from exc
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip() for c in df.columns]
    missing = {"Date", "Hour", "Ontario Demand"} - set(df.columns)
    if missing:
        raise SourceError(f"IESO file is missing columns {missing}; it has {list(df.columns)}")
    ts = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"].astype(int) - 1, unit="h")
    out = pd.DataFrame(
        {"demand_mw": pd.to_numeric(df["Ontario Demand"], errors="coerce").to_numpy()}, index=ts
    )
    out.index.name = "ts"
    return out


def check_demand(df: pd.DataFrame) -> dict:
    """Basic sanity checks; anything outside the physical range is set to missing (and counted)."""
    bad = ~df["demand_mw"].between(5_000, 30_000)
    stats = {"rows": len(df), "out_of_range": int(bad.sum()), "duplicates": int(df.index.duplicated().sum())}
    df.loc[bad, "demand_mw"] = float("nan")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    stats["missing_hours"] = int(len(full) - df.index.nunique())
    return stats


class LiveSource:
    def __init__(self, cfg: Config):
        self.cfg, self.http = cfg, session()

    def today(self) -> date:
        return now_est().date()

    def demand(self) -> pd.DataFrame:
        frames = []
        for year in range(self.cfg.first_year, self.today().year + 1):
            r = self.http.get(self.cfg.ieso_url.format(year=year), timeout=120)
            if r.status_code == 404 and year == self.today().year:
                log.info("IESO has no file for %s yet", year)  # first hours of a new year
                continue
            r.raise_for_status()
            frames.append(parse_ieso(r.text))
        df = pd.concat(frames).sort_index()
        return df[~df.index.duplicated(keep="last")]

    def _weather_call(self, url: str, params: dict) -> pd.DataFrame:
        names = list(CITIES)
        r = self.http.get(url, params={
            "latitude": ",".join(str(CITIES[c][0]) for c in names),
            "longitude": ",".join(str(CITIES[c][1]) for c in names),
            "hourly": "temperature_2m", "timezone": "GMT", **params}, timeout=300)  # fmt: skip
        r.raise_for_status()
        body = r.json()
        body = body if isinstance(body, list) else [body]
        cols = {}
        for name, loc in zip(names, body, strict=True):
            h = loc["hourly"]
            cols[name] = pd.Series(
                h["temperature_2m"], index=pd.to_datetime(h["time"]) + EST_OFFSET, dtype=float
            )
        return pd.DataFrame(cols)

    def weather(self) -> pd.DataFrame:
        """Hourly temperature by city: reanalysis history, then recent analysis and the forecast for the next days."""
        end = self.today() - timedelta(days=2)
        hist = self._weather_call(self.cfg.weather_archive_url,
                                  {"start_date": f"{self.cfg.first_year - 1}-12-24", "end_date": end.isoformat()})  # fmt: skip
        recent = self._weather_call(self.cfg.weather_forecast_url, {"past_days": 14, "forecast_days": 3})
        return hist.combine_first(recent) if not hist.empty else recent  # history wins where both exist


def weighted_temperature(cities: pd.DataFrame) -> pd.Series:
    w = pd.Series({c: CITIES[c][2] for c in cities.columns})
    t = (cities * w).sum(axis=1, min_count=len(w)) / w.sum()
    return t.rename("temp_c")


def get_source(cfg: Config):
    if cfg.fixture:
        from .fixture import FixtureSource

        return FixtureSource(cfg)
    return LiveSource(cfg)

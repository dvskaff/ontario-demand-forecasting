"""Project settings. Paths can be overridden with environment variables (tests and CI)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ[name]).resolve() if os.environ.get(name) else default


# Ontario demand is concentrated in a few regions; a population-weighted average of three cities is a simple,
# transparent temperature proxy for the province.
CITIES = {
    "Toronto": (43.65, -79.38, 0.60),
    "Ottawa": (45.42, -75.70, 0.25),
    "London": (42.98, -81.25, 0.15),
}


@dataclass
class Config:
    ieso_url: str = "https://reports-public.ieso.ca/public/Demand/PUB_Demand_{year}.csv"
    weather_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    weather_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    first_year: int = 2019
    fixture: bool = field(default_factory=lambda: os.environ.get("DEMAND_FIXTURE") == "1")

    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    backtest_folds: int = field(default_factory=lambda: int(os.environ.get("DEMAND_BACKTEST_FOLDS", "12")))
    gate_days: int = 7  # recent days neither model was trained on, used for the promotion decision
    gate_tolerance: float = (
        1.02  # promote if the new model's error is within 2% of the champion's (or better)
    )
    retrain_weekday: int = 0  # Monday (EST)
    drift_days: int = 28
    live_alert_ratio: float = 1.5  # retrain early if live error runs 1.5x the backtest error

    state_dir: Path = field(default_factory=lambda: _env_path("DEMAND_STATE_DIR", ROOT / "state"))
    site_dir: Path = field(default_factory=lambda: _env_path("DEMAND_SITE_DIR", ROOT / "site"))
    bundle_dir: Path = field(
        default_factory=lambda: _env_path("DEMAND_BUNDLE_DIR", ROOT / "serving" / "model")
    )
    readme_path: Path = field(default_factory=lambda: _env_path("DEMAND_README", ROOT / "README.md"))
    # ONNX Runtime Web for the weather what-if page (fetched from npm by scripts/fetch_web_runtime.sh)
    vendor_dir: Path = field(default_factory=lambda: _env_path("DEMAND_VENDOR_DIR", ROOT / "web" / "vendor"))

    model_name: str = "ontario-demand"
    repo_url: str = "https://github.com/dvskaff/ontario-demand-forecasting"
    site_url: str = "https://dvskaff.github.io/ontario-demand-forecasting/"
    image: str = "ghcr.io/dvskaff/ontario-demand-api"

    @property
    def forecast_log(self) -> Path:
        return self.state_dir / "forecast_log.parquet"

    @property
    def mlflow_uri(self) -> str:
        return f"sqlite:///{self.state_dir / 'mlflow.db'}"

    @property
    def artifact_root(self) -> Path:
        return self.state_dir / "mlartifacts"

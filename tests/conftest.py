import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))  # the API package (`app`)

from demand.config import Config  # noqa: E402
from demand.features import build_features, training_rows  # noqa: E402
from demand.fixture import synthetic  # noqa: E402
from demand.ingest import weighted_temperature  # noqa: E402


@pytest.fixture(scope="session")
def frame():
    demand, cities = synthetic()
    return demand.join(weighted_temperature(cities), how="outer").sort_index()


@pytest.fixture(scope="session")
def rows(frame):
    return training_rows(build_features(frame))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMAND_FIXTURE", "1")
    return replace(Config(), state_dir=tmp_path / "state", site_dir=tmp_path / "site", bundle_dir=tmp_path / "bundle",
                   readme_path=tmp_path / "README.md", fixture=True, backtest_folds=2)  # fmt: skip

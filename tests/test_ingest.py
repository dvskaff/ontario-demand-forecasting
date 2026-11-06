import pandas as pd
import pytest

from demand.fixture import ieso_csv, synthetic
from demand.ingest import SourceError, check_demand, parse_ieso, weighted_temperature


def test_parse_ieso_layout():
    demand, _ = synthetic()
    df = parse_ieso(ieso_csv(demand, 2023))
    assert df.index[0] == pd.Timestamp("2023-01-01 00:00")  # hour ending 1 = 00:00-01:00 EST
    assert len(df) == 8760 and df["demand_mw"].between(5000, 30000).all()
    with pytest.raises(SourceError, match="Date,Hour"):
        parse_ieso("garbage\n1,2,3\n")


def test_check_demand_removes_impossible_values():
    df = pd.DataFrame({"demand_mw": [15000.0, 99999.0, 14000.0]},
                      index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]))  # fmt: skip
    stats = check_demand(df)
    assert stats == {"rows": 3, "out_of_range": 1, "duplicates": 0, "missing_hours": 1}
    assert df["demand_mw"].isna().sum() == 1


def test_weighted_temperature_needs_every_city():
    _, cities = synthetic()
    t = weighted_temperature(cities.head(3))
    assert t.notna().all()
    broken = cities.head(3).copy()
    broken.iloc[0, 1] = None
    assert weighted_temperature(broken).isna().sum() == 1

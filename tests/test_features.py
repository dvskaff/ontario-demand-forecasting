from datetime import date

import numpy as np
import pandas as pd
import pytest

from demand.features import FEATURES, HISTORY_HOURS, build_features, inference_features, ontario_holidays


def test_no_peeking_at_the_forecast_day(frame):
    """Changing a day's demand must not change that day's features (they may only use the previous days)."""
    day = pd.Timestamp("2024-03-12")
    hours = pd.date_range(day, periods=24, freq="h")
    before = build_features(frame).loc[hours, FEATURES]
    changed = frame.copy()
    changed.loc[hours, "demand_mw"] *= 1.5
    after = build_features(changed).loc[hours, FEATURES]
    pd.testing.assert_frame_equal(before, after)


def test_serving_features_equal_training_features(frame):
    """The API path (history + temperatures) produces exactly the training features."""
    day = pd.Timestamp("2024-03-12")
    hours = pd.date_range(day, periods=24, freq="h")
    train = build_features(frame).loc[hours, FEATURES]
    history = frame[(frame.index < day) & (frame.index >= day - pd.Timedelta(hours=HISTORY_HOURS))]
    serve = inference_features(history, day.date(), frame.loc[hours, "temp_c"].tolist())
    pd.testing.assert_frame_equal(train, serve, check_freq=False)


def test_inference_rejects_incomplete_history(frame):
    day = pd.Timestamp("2024-03-12")
    history = frame[(frame.index < day) & (frame.index >= day - pd.Timedelta(hours=100))]
    with pytest.raises(ValueError, match="missing"):
        inference_features(history, day.date(), [5.0] * 24)
    with pytest.raises(ValueError, match="24 hourly"):
        inference_features(history, day.date(), [5.0] * 23)


def test_calendar_and_degree_features(frame):
    f = build_features(frame)
    assert date(2023, 7, 1) in ontario_holidays([2023])  # Canada Day
    assert f.loc["2023-07-01 12:00", "is_holiday"] == 1 and f.loc["2023-07-04 12:00", "is_holiday"] == 0
    assert (f["heating_degrees"] >= 0).all() and (f["cooling_degrees"] >= 0).all()
    assert np.allclose(f.loc["2023-07-05 12:00", "lag24"], frame.loc["2023-07-04 12:00", "demand_mw"])

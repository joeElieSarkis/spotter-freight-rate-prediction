import numpy as np
import pandas as pd
import pytest

from freight.data import regression_metrics, temporal_split, validate_inputs


def loads():
    return pd.DataFrame({
        "load_id": ["a", "b", "c"], "pickup": ["A"] * 3, "delivery": ["B"] * 3,
        "distance": [100., 200., 300.], "equipment": ["Dry Van"] * 3,
        "weight": [32000., np.nan, -100.],
        "date": pd.to_datetime(["2025-04-30", "2025-05-01", "2025-07-01"]),
        "posted_rate": [200., 400., 600.],
    })


def test_temporal_split_keeps_boundary_days_separate():
    train, test = temporal_split(loads(), "2025-05-01", "2025-07-01")
    assert train.load_id.tolist() == ["a"]
    assert test.load_id.tolist() == ["b"]


@pytest.mark.parametrize("column,value", [("distance", 0), ("distance", np.inf), ("posted_rate", -1), ("date", "bad")])
def test_invalid_required_values_fail(column, value):
    frame = loads().astype({"date": "object"})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_inputs(frame, labeled=True)


def test_duplicate_ids_fail():
    frame = loads()
    frame.loc[1, "load_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        validate_inputs(frame)


def test_metrics_use_all_rows_including_large_errors():
    metrics = regression_metrics([100, 200, 10000], [110, 190, 1000])
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(9020 / 3)
    assert metrics["wape_pct"] == pytest.approx(100 * 9020 / 10300)

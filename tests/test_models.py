import numpy as np
import pandas as pd
import pytest

from freight.models import FreightModel, RateBaseline, load_model


def synthetic_loads():
    size = 80
    miles = np.linspace(100, 2000, size)
    return pd.DataFrame({
        "pickup": ["A"] * size, "delivery": ["B"] * size,
        "distance": miles, "equipment": ["Dry Van", "Reefer"] * (size // 2),
        "weight": [32000.] * size, "date": pd.date_range("2025-01-01", periods=size),
        "posted_rate": 150 + 2 * miles,
    })


@pytest.mark.parametrize("kind", ["baseline", "core"])
def test_saved_model_round_trip_and_unseen_city(tmp_path, kind):
    train = synthetic_loads()
    model = RateBaseline() if kind == "baseline" else FreightModel("core", iterations=8, threads=1)
    model.fit(train)
    future = train.iloc[:4].copy()
    future.loc[:, "pickup"] = "Unseen city"
    expected = model.predict(future)
    assert np.isfinite(expected).all() and (expected > 0).all()
    model.save(tmp_path)
    np.testing.assert_allclose(load_model(tmp_path).predict(future), expected, rtol=0, atol=1e-10)

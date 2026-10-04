import numpy as np
import pandas as pd

from freight.benchmark import RidgeRateModel


def loads():
    count = 80
    distance = np.linspace(100, 2000, count)
    return pd.DataFrame({
        "load_id": [f"load-{i}" for i in range(count)],
        "pickup": ["A", "B"] * (count // 2), "delivery": ["C"] * count,
        "equipment": ["Dry Van", "Reefer"] * (count // 2),
        "distance": distance, "weight": [32000., np.nan] * (count // 2),
        "market_index": [1., np.nan] * (count // 2),
        "date": pd.date_range("2025-01-01", periods=count),
        "posted_rate": 150 + 2 * distance,
    })


def test_unseen_categories_and_missing_values_do_not_refit_preprocessing():
    train = loads()
    model = RidgeRateModel().fit(train)
    numeric = model.preprocessor.named_transformers_["numeric"]
    medians = numeric.named_steps["impute"].statistics_.copy()
    centers = numeric.named_steps["scale"].mean_.copy()
    future = train.iloc[:3].copy()
    future["pickup"] = "Unseen city"
    future["equipment"] = "Unseen equipment"
    future["weight"] = [np.nan, -500, 1e8]
    future["market_index"] = [np.nan, 1e4, 1.1]
    predictions = model.predict(future)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    np.testing.assert_array_equal(medians, numeric.named_steps["impute"].statistics_)
    np.testing.assert_array_equal(centers, numeric.named_steps["scale"].mean_)
    assert "Unseen city" not in model.features.city_coordinates


def test_predictions_do_not_use_target_or_identifier_and_preserve_inputs():
    train = loads()
    model = RidgeRateModel().fit(train)
    before = train.copy(deep=True)
    expected = model.predict(train)
    changed = train.copy()
    changed["posted_rate"] = 1e9
    changed["load_id"] = [f"changed-{i}" for i in range(len(changed))]
    np.testing.assert_allclose(model.predict(changed), expected, rtol=0, atol=1e-10)
    pd.testing.assert_frame_equal(train, before)

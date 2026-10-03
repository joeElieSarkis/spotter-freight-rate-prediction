import numpy as np
import pandas as pd
import pytest

from freight.features import FeatureBuilder


def frame():
    return pd.DataFrame({
        "load_id": ["a", "b"], "pickup": ["A", "B"], "delivery": ["B", "A"],
        "equipment": ["Dry Van", "Reefer"], "distance": [100., 200.],
        "weight": [-100., 30000.], "date": ["2025-01-01", "2025-01-02"],
        "pickup_lat": [30., 40.], "pickup_lon": [-80., -90.],
        "delivery_lat": [40., 30.], "delivery_lon": [-90., -80.],
        "market_index": [1., np.nan], "quote_signal": [2., 2.5],
        "posted_rate": [200., 500.],
    })


def test_features_exclude_ids_and_labels_and_do_not_mutate_inputs():
    source = frame()
    original = source.copy(deep=True)
    output = FeatureBuilder("full").fit(source).transform(source)
    assert "load_id" not in output and "posted_rate" not in output
    assert np.isnan(output.loc[0, "weight"])
    assert output.loc[0, "weight_missing"] == 1
    pd.testing.assert_frame_equal(source, original)


def test_core_reproduces_december_input_availability():
    source = frame()
    builder = FeatureBuilder("core").fit(source)
    reduced = source[["pickup", "delivery", "distance", "equipment", "weight", "date"]]
    pd.testing.assert_frame_equal(builder.transform(source), builder.transform(reduced))


def test_city_lookup_never_learns_from_inference_rows():
    source = frame()
    builder = FeatureBuilder("core").fit(source)
    future = source.copy()
    future.loc[0, "pickup"] = "Unseen city"
    future.loc[0, "pickup_lat"] = 12.
    result = builder.transform(future)
    assert np.isnan(result.loc[0, "pickup_lat"])
    assert "Unseen city" not in builder.city_coordinates


def test_quote_ablation_cannot_use_changed_quote_values():
    source = frame()
    builder = FeatureBuilder("no_quote").fit(source)
    changed = source.copy()
    changed["quote_signal"] = 100000.
    pd.testing.assert_frame_equal(builder.transform(source), builder.transform(changed))


def test_signal_model_rejects_missing_columns():
    source = frame()
    with pytest.raises(ValueError, match="requires market_index"):
        FeatureBuilder("no_quote").fit(source).transform(source.drop(columns="market_index"))

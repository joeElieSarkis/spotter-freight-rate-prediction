import numpy as np
import pandas as pd
import pytest

from freight.predict import align_predictions


def test_predictions_join_by_id_in_template_order():
    loads = pd.DataFrame({"load_id": ["c", "a", "b"]})
    template = pd.DataFrame({"load_id": ["a", "b", "c"], "predicted_rate": [np.nan] * 3})
    output = align_predictions(loads, [30, 10, 20], template)
    assert output.predicted_rate.tolist() == [10, 20, 30]
    assert template.predicted_rate.isna().all()


@pytest.mark.parametrize("predictions", [[1, np.nan], [1, np.inf], [1, 0], [1], [1, -1]])
def test_invalid_predictions_fail(predictions):
    loads = pd.DataFrame({"load_id": ["a", "b"]})
    template = loads.assign(predicted_rate=np.nan)
    with pytest.raises(ValueError):
        align_predictions(loads, predictions, template)


def test_mismatched_template_fails():
    loads = pd.DataFrame({"load_id": ["a", "b"]})
    template = pd.DataFrame({"load_id": ["a", "c"], "predicted_rate": [np.nan] * 2})
    with pytest.raises(ValueError, match="do not match"):
        align_predictions(loads, [10, 20], template)

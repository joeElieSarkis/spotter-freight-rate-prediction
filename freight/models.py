"""A distance-aware baseline and robust boosted rate-per-mile models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from freight.data import TARGET, validate_inputs
from freight.features import CATEGORICAL, FeatureBuilder


class RateBaseline:
    """Median rate per mile by equipment and distance band."""

    @staticmethod
    def keys(frame):
        bands = np.digitize(frame.distance, [250, 500, 1000, 2000])
        return frame.equipment.astype(str) + ":" + pd.Series(bands, index=frame.index).astype(str)

    def fit(self, frame):
        validate_inputs(frame, labeled=True)
        rpm = frame[TARGET] / frame.distance
        self.overall = float(rpm.median())
        self.rates = rpm.groupby(self.keys(frame)).median().to_dict()
        return self

    def predict(self, frame):
        validate_inputs(frame)
        rpm = self.keys(frame).map(self.rates).fillna(self.overall)
        return rpm.to_numpy() * frame.distance.to_numpy()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps({"kind": "baseline", "overall": self.overall, "rates": self.rates}, indent=2), encoding="utf-8")


class FreightModel:
    def __init__(self, feature_set="no_quote", *, iterations=700, depth=6, seed=42, threads=4):
        self.features = FeatureBuilder(feature_set)
        self.parameters = {
            "iterations": iterations, "depth": depth, "learning_rate": .05,
            "loss_function": "MAE", "l2_leaf_reg": 5,
            "random_seed": seed, "thread_count": threads,
            "allow_writing_files": False, "verbose": False,
        }
        self.model = CatBoostRegressor(**self.parameters)

    def fit(self, frame):
        validate_inputs(frame, labeled=True)
        features = self.features.fit(frame).transform(frame)
        rpm = frame[TARGET] / frame.distance
        # distance * absolute RPM error == absolute total-dollar error.
        # Scaling targets this way also lets rate predictions scale with mileage.
        self.model.fit(features, rpm, cat_features=CATEGORICAL, sample_weight=frame.distance)
        return self

    def predict(self, frame):
        rpm = self.model.predict(self.features.transform(frame))
        return np.maximum(rpm * frame.distance.to_numpy(), .01)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "model.cbm"))
        metadata = {"kind": "catboost", "features": self.features.to_dict(), "parameters": self.parameters}
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(path: Path):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["kind"] == "baseline":
        model = RateBaseline()
        model.overall, model.rates = metadata["overall"], metadata["rates"]
        return model
    model = FreightModel(metadata["features"]["feature_set"])
    model.features = FeatureBuilder.from_dict(metadata["features"])
    model.parameters = metadata["parameters"]
    model.model.load_model(str(path / "model.cbm"))
    return model

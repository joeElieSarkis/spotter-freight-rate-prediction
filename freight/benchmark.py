"""Run a fixed Ridge comparison on the original model-selection windows.

This follow-up does not change the frozen CatBoost selection or submission.
September-October results were already inspected before this benchmark was added.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from freight.data import SELECTION_WINDOWS, TARGET, read_loads, regression_metrics, temporal_split, validate_inputs
from freight.features import CATEGORICAL, FeatureBuilder


class RidgeRateModel:
    """Linear rate-per-mile benchmark with preprocessing fitted on training only."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.features = FeatureBuilder("no_quote")
        self.regressor = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=10000)

    def fit(self, frame):
        validate_inputs(frame, labeled=True)
        features = self.features.fit(frame).transform(frame)
        numeric = [column for column in features if column not in CATEGORICAL]
        self.preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        matrix = self.preprocessor.fit_transform(features)
        distance = frame.distance.to_numpy(dtype=float)
        target = frame[TARGET].to_numpy(dtype=float) / distance
        # d^2 * (predicted RPM - actual RPM)^2 is squared total-dollar error.
        # Normalize weights to mean 1 so alpha has a stable interpretation.
        weights = distance**2
        weights /= weights.mean()
        self.regressor.fit(matrix, target, sample_weight=weights)
        return self

    def predict(self, frame):
        matrix = self.preprocessor.transform(self.features.transform(frame))
        rpm = self.regressor.predict(matrix)
        return np.maximum(rpm * frame.distance.to_numpy(dtype=float), .01)


def run(data_dir: Path, output_dir: Path):
    data_path = data_dir / "train_test.csv"
    data = read_loads(data_path, labeled=True)
    rows = []
    for start, end in SELECTION_WINDOWS:
        train, test = temporal_split(data, start, end)
        model = RidgeRateModel().fit(train)
        metrics = regression_metrics(test[TARGET], model.predict(test))
        rows.append({"model": "ridge_no_quote", "test_start": start,
                     "test_end_exclusive": end, "train_n": len(train), **metrics})
        print(f"{start} Ridge MAE=${metrics['mae']:.2f} RMSE=${metrics['rmse']:.2f}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "ridge_benchmark.csv", index=False)
    design = {
        "purpose": "Follow-up algorithm comparison; does not alter the original selection or predictions",
        "timing": "Added after the initial September-October holdout results were inspected",
        "evaluation": "Original May-June and July-August windows only; no holdout score is claimed",
        "feature_set": "no_quote, the same inputs as the selected CatBoost model",
        "preprocessing": "Training-only city lookup, numeric median imputation and standardization, categorical one-hot encoding with unknown categories ignored",
        "alpha": 10.0, "solver": "lsqr", "tol": 1e-6, "max_iter": 10000,
        "target": "posted_rate / distance",
        "sample_weight": "distance squared, normalized to mean 1",
        "objective": "Squared total-dollar error plus L2 coefficient penalty; unlike CatBoost's MAE objective",
        "target_handling": "All evaluation targets retained; no target clipping or outlier removal",
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "training_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (output_dir / "ridge_benchmark.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

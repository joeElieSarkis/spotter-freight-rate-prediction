"""Input contracts and date-based splits. No fitted statistics live here."""

from pathlib import Path

import numpy as np
import pandas as pd

CORE_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
SIGNAL_COLUMNS = ["market_index", "quote_signal"]
COORD_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
TARGET = "posted_rate"
SELECTION_WINDOWS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01")]
HOLDOUT_WINDOW = ("2025-09-01", "2025-11-01")


def validate_inputs(frame: pd.DataFrame, *, labeled: bool = False) -> None:
    required = CORE_COLUMNS + ([TARGET] if labeled else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input contains no loads")
    for column in ["pickup", "delivery", "equipment"]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {column}")
    if pd.to_datetime(frame["date"], format="ISO8601", errors="coerce").isna().any():
        raise ValueError("Invalid date")
    for column in ["distance"] + ([TARGET] if labeled else []):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must be finite and positive")
    if "load_id" in frame:
        if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
            raise ValueError("Missing or duplicate load_id")


def read_loads(path: Path, *, labeled: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_inputs(frame, labeled=labeled)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def temporal_split(frame: pd.DataFrame, start: str, end: str):
    dates = pd.to_datetime(frame["date"])
    train = frame.loc[dates < pd.Timestamp(start)].copy()
    test = frame.loc[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty temporal split: {start} to {end}")
    if train["date"].max() >= test["date"].min():
        raise ValueError("Training dates overlap evaluation dates")
    return train, test


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match the target shape")
    error = predicted - actual
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "n": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_pct": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(actual))),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else None,
        "bias": float(np.mean(error)),
        "median_ae": float(np.median(np.abs(error))),
    }

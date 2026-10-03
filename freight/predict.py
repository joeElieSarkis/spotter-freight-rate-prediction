"""Produce submission CSVs by ID and preserve the fixed scenario inputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from freight.data import read_loads, validate_inputs
from freight.models import load_model


def align_predictions(loads, predicted, template):
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Template must have exactly load_id,predicted_rate")
    if loads.load_id.isna().any() or loads.load_id.duplicated().any():
        raise ValueError("Load IDs must be unique and nonmissing")
    if template.load_id.isna().any() or template.load_id.duplicated().any():
        raise ValueError("Template IDs must be unique and nonmissing")
    if set(loads.load_id) != set(template.load_id):
        raise ValueError("Template IDs do not match input loads")
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != (len(loads),) or not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Predictions must be finite, positive, and match the input length")
    by_id = pd.Series(predicted, index=loads.load_id)
    output = template.copy()
    output["predicted_rate"] = output.load_id.map(by_id)
    return output


def write_predictions(data_dir: Path, model_dir: Path, output_dir: Path):
    loads = read_loads(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    model = load_model(model_dir / "main")
    predictions = align_predictions(loads, model.predict(loads), template)
    december = pd.read_csv(data_dir / "december_chart_inputs.csv")
    validate_inputs(december)
    december_model = load_model(model_dir / "december")
    december["predicted_rate"] = december_model.predict(december)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False, float_format="%.2f")
    december.to_csv(output_dir / "december_predictions.csv", index=False, float_format="%.2f")
    return predictions, december


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    write_predictions(args.data_dir, args.model_dir, args.output_dir)
    print("Created validation_predictions.csv and december_predictions.csv")


if __name__ == "__main__":
    main()

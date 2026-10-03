"""Summarize supplied inputs without modifying them."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freight.data import TARGET, read_loads


def describe(frame: pd.DataFrame) -> dict:
    result = {
        "rows": len(frame),
        "date_min": str(frame.date.min().date()),
        "date_max": str(frame.date.max().date()),
        "missing": {key: int(value) for key, value in frame.isna().sum().items()},
        "nonpositive_weight": int((frame.weight <= 0).sum()),
        "duplicate_ids": int(frame.load_id.duplicated().sum()),
        "duplicate_loads_excluding_id": int(frame.drop(columns="load_id").duplicated().sum()),
        "equipment_counts": {key: int(value) for key, value in frame.equipment.value_counts().items()},
    }
    if TARGET in frame:
        rpm = frame[TARGET] / frame.distance
        result["rate_per_mile_quantiles"] = {
            str(q): float(rpm.quantile(q)) for q in [0, .01, .25, .5, .75, .99, 1]
        }
        # Diagnostic flags only: these rows remain in training and evaluation.
        result["rate_per_mile_below_1_or_above_5"] = int(((rpm < 1) | (rpm > 5)).sum())
        monthly = []
        for month, group in frame.groupby(frame.date.dt.strftime("%Y-%m")):
            group_rpm = group[TARGET] / group.distance
            monthly.append({
                "month": month,
                "rows": len(group),
                "median_rate": float(group[TARGET].median()),
                "median_rate_per_mile": float(group_rpm.median()),
                "quote_rpm_correlation": float(group.quote_signal.corr(group_rpm)),
                "quote_total_mae": float(np.abs(group.quote_signal * group.distance - group[TARGET]).mean()),
            })
        result["monthly"] = monthly
    return result


def run(data_dir: Path, output_dir: Path) -> dict:
    train = read_loads(data_dir / "train_test.csv", labeled=True)
    future = read_loads(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    if template.load_id.duplicated().any() or set(template.load_id) != set(future.load_id):
        raise ValueError("Template IDs differ from validation IDs")
    train_cities = set(train.pickup) | set(train.delivery)
    future_cities = set(future.pickup) | set(future.delivery)
    known_lanes = set(zip(train.pickup, train.delivery))
    future_lanes = list(zip(future.pickup, future.delivery))
    report = {
        "development": describe(train),
        "final_inputs": describe(future),
        "id_overlap": len(set(train.load_id) & set(future.load_id)),
        "new_cities": sorted(future_cities - train_cities),
        "final_loads_on_unseen_lanes": sum(lane not in known_lanes for lane in future_lanes),
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(data_dir.glob("*.csv"))},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(report["development"]["monthly"]).to_csv(output_dir / "monthly_audit.csv", index=False)
    print(json.dumps({key: report[key] for key in ["new_cities", "final_loads_on_unseen_lanes", "id_overlap"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

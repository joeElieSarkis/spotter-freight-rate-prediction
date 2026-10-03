"""Select models using two forward windows; reserve September-October."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from freight.data import HOLDOUT_WINDOW, SELECTION_WINDOWS, TARGET, read_loads, regression_metrics, temporal_split
from freight.models import FreightModel, RateBaseline

CANDIDATES = ["baseline", "core", "no_quote", "full"]


def make_model(name, iterations=700, threads=4):
    return RateBaseline() if name == "baseline" else FreightModel(name, iterations=iterations, threads=threads)


def select(data: pd.DataFrame, output_dir: Path, *, iterations=700, threads=4):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start, end in SELECTION_WINDOWS:
        train, test = temporal_split(data, start, end)
        for name in CANDIDATES:
            started = time.perf_counter()
            model = make_model(name, iterations, threads).fit(train)
            prediction = model.predict(test)
            result = {"model": name, "test_start": start, "test_end_exclusive": end,
                      "train_n": len(train), **regression_metrics(test[TARGET], prediction)}
            rows.append(result)
            pd.DataFrame(rows).to_csv(output_dir / "selection_metrics.csv", index=False)
            print(f"{start} {name:9s} MAE=${result['mae']:.2f} RMSE=${result['rmse']:.2f} ({time.perf_counter()-started:.1f}s)", flush=True)
    scores = pd.DataFrame(rows).groupby("model").mae.mean().sort_values()
    chosen = {
        "main_model": str(scores.index[0]),
        "december_model": str(scores.loc[["baseline", "core"]].idxmin()),
        "criterion": "Mean total-dollar MAE across the May-June and July-August windows",
        "iterations": iterations, "seed": 42, "threads": threads,
        "selection_mean_mae": scores.to_dict(),
        "holdout_start": HOLDOUT_WINDOW[0], "holdout_end_exclusive": HOLDOUT_WINDOW[1],
    }
    (output_dir / "selection.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(chosen, indent=2), flush=True)
    return chosen


def holdout_slices(train, test, predicted):
    frame = test.copy()
    frame["predicted_rate"] = predicted
    known_lanes = set(zip(train.pickup, train.delivery))
    known_cities = set(train.pickup) | set(train.delivery)
    frame["lane_seen"] = ["seen" if lane in known_lanes else "unseen" for lane in zip(frame.pickup, frame.delivery)]
    frame["city_seen"] = (frame.pickup.isin(known_cities) & frame.delivery.isin(known_cities)).map({True: "seen", False: "unseen"})
    frame["weight_quality"] = ((frame.weight > 0) & frame.weight.notna()).map({True: "valid", False: "missing_or_invalid"})
    frame["month"] = frame.date.dt.strftime("%Y-%m")
    frame["distance_band"] = pd.cut(frame.distance, [0, 250, 500, 1000, 2000, float("inf")], right=False).astype(str)
    rows = []
    for column in ["month", "equipment", "lane_seen", "city_seen", "weight_quality", "distance_band"]:
        for value, group in frame.groupby(column):
            rows.append({"slice": column, "value": str(value), **regression_metrics(group[TARGET], group.predicted_rate)})
    return pd.DataFrame(rows), frame


def evaluate(data: pd.DataFrame, selection: dict, output_dir: Path):
    train, test = temporal_split(data, *HOLDOUT_WINDOW)
    rows = []
    # Selection is frozen before this function sees any holdout labels.
    for name in dict.fromkeys(["baseline", selection["main_model"], selection["december_model"]]):
        model = make_model(name, selection["iterations"], selection["threads"]).fit(train)
        predicted = model.predict(test)
        metrics = {"model": name, "train_n": len(train), **regression_metrics(test[TARGET], predicted)}
        rows.append(metrics)
        print(f"Holdout {name}: MAE=${metrics['mae']:.2f} RMSE=${metrics['rmse']:.2f}", flush=True)
        if name == selection["main_model"]:
            slices, details = holdout_slices(train, test, predicted)
            slices.to_csv(output_dir / "holdout_slices.csv", index=False)
            details[["load_id", "date", "posted_rate", "predicted_rate", "equipment", "distance", "lane_seen", "city_seen", "weight_quality"]].to_csv(output_dir / "holdout_predictions.csv", index=False, float_format="%.6f")
            if hasattr(model, "model"):
                pd.DataFrame({"feature": model.model.feature_names_, "importance": model.model.get_feature_importance()}).sort_values("importance", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "holdout_metrics.csv", index=False)
    return pd.DataFrame(rows)


def stress_unseen_cities(data: pd.DataFrame, selection: dict, output_dir: Path):
    """Remove six training cities, then predict their loads in July-August."""
    train, test = temporal_split(data, "2025-07-01", "2025-09-01")
    cities = sorted(set(train.pickup) | set(train.delivery))
    # Fixed, outcome-independent sample. No city is chosen using its error.
    withheld = cities[::11]
    train = train.loc[~train.pickup.isin(withheld) & ~train.delivery.isin(withheld)].copy()
    test = test.loc[test.pickup.isin(withheld) | test.delivery.isin(withheld)].copy()
    rows = []
    for name in dict.fromkeys(["baseline", selection["main_model"], selection["december_model"]]):
        model = make_model(name, selection["iterations"], selection["threads"]).fit(train)
        rows.append({"model": name, "train_n": len(train), **regression_metrics(test[TARGET], model.predict(test))})
    pd.DataFrame(rows).to_csv(output_dir / "unseen_city_metrics.csv", index=False)
    (output_dir / "unseen_city_design.json").write_text(json.dumps({
        "withheld_cities": withheld,
        "training_end_exclusive": "2025-07-01",
        "evaluation_start": "2025-07-01", "evaluation_end_exclusive": "2025-09-01",
        "purpose": "Diagnostic only; does not change the frozen model selection",
        "limitation": "Known-data city holdout approximates, but cannot verify, performance on the eight actual new cities.",
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "evaluate", "stress"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    data = read_loads(args.data_dir / "train_test.csv", labeled=True)
    if args.stage == "select":
        select(data, args.output_dir, iterations=args.iterations, threads=args.threads)
    else:
        selection = json.loads((args.output_dir / "selection.json").read_text(encoding="utf-8"))
        if args.stage == "evaluate":
            evaluate(data, selection, args.output_dir)
        else:
            stress_unseen_cities(data, selection, args.output_dir)


if __name__ == "__main__":
    main()

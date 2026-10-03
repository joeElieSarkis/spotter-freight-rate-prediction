"""Fit frozen selections on all labeled loads and record provenance."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from freight.data import read_loads
from freight.experiment import make_model


def train(data_dir: Path, artifact_dir: Path, model_dir: Path):
    selection_path = artifact_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    data_path = data_dir / "train_test.csv"
    data = read_loads(data_path, labeled=True)
    fitted = {}
    for role, name in [("main", selection["main_model"]), ("december", selection["december_model"])]:
        if name not in fitted:
            print(f"Fitting {name} on {len(data):,} loads", flush=True)
            fitted[name] = make_model(name, selection["iterations"], selection["threads"]).fit(data)
        fitted[name].save(model_dir / role)
    manifest = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "catboost", "matplotlib"]},
        "training_rows": len(data), "training_date_min": str(data.date.min().date()),
        "training_date_max": str(data.date.max().date()),
        "training_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selection": selection,
    }
    (artifact_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    train(args.data_dir, args.artifact_dir, args.model_dir)


if __name__ == "__main__":
    main()

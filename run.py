"""Rebuild the audit, experiments, predictions, and supplied December chart."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = Path(__file__).resolve().parent
    commands = [
        ["-m", "pytest", "-q"],
        ["-m", "freight.audit"],
        ["-m", "freight.experiment", "select", "--threads", str(args.threads)],
        ["-m", "freight.experiment", "evaluate"],
        ["-m", "freight.experiment", "stress"],
        ["-m", "freight.benchmark"],
        ["-m", "freight.train"],
        ["-m", "freight.predict"],
        ["score.py", "--predictions", "validation_predictions.csv", "--december-predictions", "december_predictions.csv"],
    ]
    for command in commands:
        print(f"\nRunning: python {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=root, check=True)


if __name__ == "__main__":
    main()


from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sder import sder_sjf, sder_srtf

RESULTS = ROOT / "results" / "real_workload"
PREDICTIONS = RESULTS / "predictions.csv"


def unpack_sder(value):
    """Accept either (errors, eligible) or a dictionary result."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return int(value[0]), int(value[1])

    if isinstance(value, dict):
        errors = value.get("incorrect_pairs")
        eligible = value.get("eligible_pairs")
        if errors is not None and eligible is not None:
            return int(errors), int(eligible)

    raise TypeError(
        f"Unexpected SDER return format: {type(value).__name__}. "
        "Inspect the return statements in src/sder.py."
    )


def main():
    df = pd.read_csv(PREDICTIONS)

    methods = [
        ("actual_burst", "Oracle"),
        ("pred_exp", "Exponential Averaging"),
        ("pred_lr", "Linear Regression")
    ]

    rows = []

    print("\nMEASURED-WORKLOAD SDER RESULTS")
    print("=" * 65)

    for scheduler, function in [
        ("SJF", sder_sjf),
        ("SRTF", sder_srtf)
    ]:
        for column, label in methods:
            incorrect, eligible = unpack_sder(
                function(df, column)
            )

            percentage = (
                100 * incorrect / eligible
                if eligible > 0 else float("nan")
            )

            rows.append({
                "scheduler": scheduler,
                "method": label,
                "incorrect_pairs": incorrect,
                "eligible_pairs": eligible,
                "sder_percent": percentage
            })

            print(
                f"{scheduler:4} | {label:23} | "
                f"{incorrect}/{eligible} | "
                f"SDER {percentage:.3f}%"
            )

    pd.DataFrame(rows).to_csv(
        RESULTS / "sder_results.csv",
        index=False
    )

    print("\nSaved results/real_workload/sder_results.csv")


if __name__ == "__main__":
    main()

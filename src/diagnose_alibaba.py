
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjf import simulate_sjf
from srtf import simulate_srtf
from sder import sder_sjf, sder_srtf

PATH = ROOT / "results" / "extended" / "alibaba_predictions.csv"


def main():
    df = pd.read_csv(PATH)
    df = df.sort_values(
        ["arrival_time", "process_id"]
    ).reset_index(drop=True)

    print("\nALIBABA WORKLOAD DIAGNOSTICS")
    print("=" * 65)

    print("Processes:", len(df))
    print("Unique arrival times:", df["arrival_time"].nunique())
    print("Unique actual bursts:", df["actual_burst"].nunique())

    print("\nArrival-time statistics:")
    print(df["arrival_time"].describe().to_string())

    print("\nBurst-time statistics:")
    print(df["actual_burst"].describe().to_string())

    print("\nPrediction variation:")
    for col in ["pred_exp", "pred_lr", "pred_rf"]:
        print(
            f"{col:12} "
            f"unique={df[col].nunique():4} "
            f"std={df[col].std():.6f} "
            f"min={df[col].min():.6f} "
            f"max={df[col].max():.6f}"
        )

    print("\nSDER raw counts:")
    for col in [
        "actual_burst",
        "pred_exp",
        "pred_lr",
        "pred_rf"
    ]:
        for scheduler, func in [
            ("SJF", sder_sjf),
            ("SRTF", sder_srtf)
        ]:
            result = func(df, col)

            if isinstance(result, dict):
                wrong = result["incorrect_pairs"]
                eligible = result["eligible_pairs"]
            else:
                wrong, eligible = result[:2]

            print(
                f"{col:14} {scheduler:4} "
                f"incorrect={wrong} "
                f"eligible={eligible} "
                f"SDER={100 * wrong / eligible if eligible else float('nan'):.3f}%"
            )

    print("\nScheduling differences:")

    for col in [
        "actual_burst",
        "pred_exp",
        "pred_lr",
        "pred_rf"
    ]:
        sjf = simulate_sjf(df, col)
        srtf = simulate_srtf(df, col)

        if isinstance(sjf, tuple):
            sjf = sjf[0]
        if isinstance(srtf, tuple):
            srtf = srtf[0]

        comparison = sjf[
            ["process_id", "completion_time"]
        ].merge(
            srtf[
                ["process_id", "completion_time"]
            ],
            on="process_id",
            suffixes=("_sjf", "_srtf")
        )

        differences = ~np.isclose(
            comparison["completion_time_sjf"],
            comparison["completion_time_srtf"],
            rtol=0,
            atol=1e-7
        )

        print(
            f"{col:14} "
            f"Different completion times: {differences.sum()}"
        )


if __name__ == "__main__":
    main()

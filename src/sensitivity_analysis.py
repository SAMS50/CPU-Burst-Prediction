
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjf import simulate_sjf
from srtf import simulate_srtf
from sder import sder_sjf, sder_srtf

SOURCE = ROOT / "results" / "extended"
OUT = ROOT / "results" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "synthetic": SOURCE / "synthetic_predictions.csv",
    "measured": SOURCE / "measured_predictions.csv",
}

NOISE_LEVELS = [0, 0.1, 0.2, 0.3, 0.5]
REPEATS = 6


def sder_percentage(result):
    if isinstance(result, dict):
        wrong = result["incorrect_pairs"]
        total = result["eligible_pairs"]
    else:
        wrong, total = result[:2]
    return 100 * wrong / total if total else np.nan


def run_experiments():
    rows = []

    for dataset_name, path in DATASETS.items():
        df = pd.read_csv(path)
        actual = df["actual_burst"].to_numpy(dtype=float)
        baseline = df["pred_rf"].to_numpy(dtype=float)

        oracle = {}
        for scheduler, simulator in [
            ("SJF", simulate_sjf),
            ("SRTF", simulate_srtf),
        ]:
            result = simulator(df, "actual_burst")
            if isinstance(result, tuple):
                result = result[0]
            oracle[scheduler] = result["waiting_time"].mean()

        for noise in NOISE_LEVELS:
            for repeat in range(REPEATS):
                rng = np.random.default_rng(
                    1000 + repeat + int(noise * 1000)
                )

                # Noise relative to each process's baseline prediction.
                disturbed = baseline * (
                    1 + rng.normal(0, noise, len(df))
                )
                disturbed = np.clip(disturbed, 0.001, None)

                experiment = df.copy()
                experiment["pred_noise"] = disturbed

                mae = np.mean(np.abs(actual - disturbed))
                rmse = np.sqrt(
                    np.mean((actual - disturbed) ** 2)
                )

                for scheduler, simulator, sder_fn in [
                    ("SJF", simulate_sjf, sder_sjf),
                    ("SRTF", simulate_srtf, sder_srtf),
                ]:
                    result = simulator(experiment, "pred_noise")
                    if isinstance(result, tuple):
                        result = result[0]

                    waiting = result["waiting_time"].mean()
                    turnaround = result["turnaround_time"].mean()
                    sder = sder_percentage(
                        sder_fn(experiment, "pred_noise")
                    )

                    rows.append({
                        "dataset": dataset_name,
                        "scheduler": scheduler,
                        "noise_level": noise,
                        "repeat": repeat,
                        "mae": mae,
                        "rmse": rmse,
                        "sder_percent": sder,
                        "avg_waiting": waiting,
                        "avg_turnaround": turnaround,
                        "waiting_degradation": (
                            waiting - oracle[scheduler]
                        ),
                    })

            print(
                f"Completed {dataset_name}: "
                f"{noise * 100:.0f}% noise"
            )

    return pd.DataFrame(rows)


def analyze(results):
    correlations = []

    for (dataset, scheduler), group in results.groupby(
        ["dataset", "scheduler"]
    ):
        for metric in ["mae", "rmse", "sder_percent"]:
            valid = group[
                [metric, "waiting_degradation"]
            ].dropna()

            # Correlation across perturbation runs, not
            # across individual processes.
            pearson = valid[metric].corr(
                valid["waiting_degradation"],
                method="pearson"
            )
            spearman = valid[metric].corr(
                valid["waiting_degradation"],
                method="spearman"
            )

            correlations.append({
                "dataset": dataset,
                "scheduler": scheduler,
                "metric": metric,
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "runs": len(valid),
            })

        fig, ax = plt.subplots(figsize=(8, 5))

        for noise, subset in group.groupby("noise_level"):
            ax.scatter(
                subset["sder_percent"],
                subset["waiting_degradation"],
                label=f"{noise * 100:.0f}% noise",
                alpha=0.7,
            )

        ax.set_xlabel("SDER (%)")
        ax.set_ylabel("Waiting time above oracle")
        ax.set_title(f"{dataset.title()} — {scheduler}")
        ax.legend(title="Prediction noise")
        fig.tight_layout()
        fig.savefig(
            OUT / f"{dataset}_{scheduler.lower()}_correlation.png",
            dpi=200
        )
        plt.close(fig)

        means = group.groupby("noise_level")[
            ["mae", "rmse", "sder_percent", "avg_waiting"]
        ].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            means.index * 100,
            means["sder_percent"],
            marker="o"
        )
        ax.set_xlabel("Noise standard deviation (%)")
        ax.set_ylabel("Mean SDER (%)")
        ax.set_title(f"SDER sensitivity — {dataset}, {scheduler}")
        fig.tight_layout()
        fig.savefig(
            OUT / f"{dataset}_{scheduler.lower()}_sensitivity.png",
            dpi=200
        )
        plt.close(fig)

    return pd.DataFrame(correlations)


def main():
    results = run_experiments()
    results.to_csv(OUT / "all_experiments.csv", index=False)

    correlations = analyze(results)
    correlations.to_csv(
        OUT / "correlations.csv", index=False
    )

    print("\nCORRELATION WITH WAITING-TIME DEGRADATION")
    print("=" * 75)
    print(correlations.round(4).to_string(index=False))
    print(f"\nCompleted {len(results)} scheduling experiments.")
    print("Saved CSV results and four graphs in results/sensitivity/")


if __name__ == "__main__":
    main()

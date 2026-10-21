
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "extended"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PRED = pd.read_csv(SOURCE / "prediction_metrics.csv")
SCHED = pd.read_csv(SOURCE / "scheduling_metrics.csv")

METHODS = [
    "Exponential-style baseline",
    "Linear Regression",
    "Random Forest",
]

DATASETS = [
    "synthetic",
    "measured",
    "alibaba",
]


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUT / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {OUT / filename}")


def prediction_plots():
    for dataset in DATASETS:
        subset = PRED.loc[
            PRED["dataset"] == dataset
        ].set_index("method").reindex(METHODS)

        x = np.arange(len(METHODS))
        width = 0.35

        plt.figure(figsize=(9, 5))
        plt.bar(
            x - width / 2,
            subset["mae"],
            width,
            label="MAE",
        )
        plt.bar(
            x + width / 2,
            subset["rmse"],
            width,
            label="RMSE",
        )

        plt.xticks(
            x,
            ["Exponential", "Linear Regression", "Random Forest"],
        )
        plt.ylabel("Prediction error")
        plt.title(
            f"Prediction Accuracy — {dataset.title()}"
        )
        plt.legend()

        save_plot(f"{dataset}_prediction_accuracy.png")


def waiting_plots():
    for dataset in DATASETS:
        subset = SCHED.loc[
            SCHED["dataset"] == dataset
        ]

        methods = [
            "Oracle",
            *METHODS,
        ]

        x = np.arange(len(methods))
        width = 0.35

        sjf = subset.loc[
            subset["scheduler"] == "SJF"
        ].set_index("method").reindex(methods)

        srtf = subset.loc[
            subset["scheduler"] == "SRTF"
        ].set_index("method").reindex(methods)

        plt.figure(figsize=(10, 5))
        plt.bar(
            x - width / 2,
            sjf["avg_waiting"],
            width,
            label="SJF",
        )
        plt.bar(
            x + width / 2,
            srtf["avg_waiting"],
            width,
            label="SRTF",
        )

        plt.xticks(
            x,
            ["Oracle", "Exponential", "Linear Regression", "Random Forest"],
        )
        plt.ylabel("Average waiting time")
        plt.title(
            f"Scheduling Performance — {dataset.title()}"
        )
        plt.legend()

        save_plot(f"{dataset}_waiting_time.png")


def sder_plots():
    for dataset in DATASETS:
        subset = SCHED.loc[
            (SCHED["dataset"] == dataset)
            & (SCHED["method"] != "Oracle")
        ]

        x = np.arange(len(METHODS))
        width = 0.35

        sjf = subset.loc[
            subset["scheduler"] == "SJF"
        ].set_index("method").reindex(METHODS)

        srtf = subset.loc[
            subset["scheduler"] == "SRTF"
        ].set_index("method").reindex(METHODS)

        plt.figure(figsize=(9, 5))
        plt.bar(
            x - width / 2,
            sjf["sder_percent"],
            width,
            label="SJF",
        )
        plt.bar(
            x + width / 2,
            srtf["sder_percent"],
            width,
            label="SRTF",
        )

        plt.xticks(
            x,
            ["Exponential", "Linear Regression", "Random Forest"],
        )
        plt.ylabel("Tie-aware SDER (%)")
        plt.title(
            f"Scheduling Decision Error — {dataset.title()}"
        )
        plt.legend()

        save_plot(f"{dataset}_sder.png")


def main():
    prediction_plots()
    waiting_plots()
    sder_plots()

    print("\nGenerated 9 comparison graphs.")
    print(f"Location: {OUT}")


if __name__ == "__main__":
    main()

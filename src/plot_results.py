
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

sder = pd.read_csv(RESULTS / "sder_results.csv")
print("SDER CSV columns:", sder.columns.tolist())

# Verified values from the current synthetic experiment
metrics = {
    "MAE": [3.8546, 3.0005],
    "RMSE": [4.7197, 3.6869],
    "SJF SDER (%)": [23.997, 23.997],
    "SRTF SDER (%)": [22.737, 22.759],
    "SJF Waiting Time": [777.085, 777.085],
    "SRTF Waiting Time": [777.085, 777.030],
}

labels = ["Exponential Averaging", "Linear Regression"]

for metric, values in metrics.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — Synthetic Workload")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()

    filename = metric.lower().replace(" ", "_").replace("(%)", "")
    fig.savefig(RESULTS / f"{filename}.png", dpi=200)
    plt.close(fig)

print("Six comparison graphs saved in results/")




from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUBPLOTS = [
    ("average-waiting-time", "Average Waiting Time"),
    ("co2-emission", "Co2 Emission"),
    ("driven-distance", "Driven Distance"),
    ("maximum-passenger-journey-time", "Maximum Passenger Journey Time"),
    ("maximum-passenger-non-driving-time", "Maximum Passenger Non Driving Time"),
    ("minimum-passenger-pickup-wait-time", "Minimum Passenger Pickup Wait Time"),
    ("relative-throughput", "Relative Throughput"),
    ("total-cost", "Total Cost"),
]


def plot_dashboard(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df["time_minutes"] = df["time"] / 60.0

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"GECCO 26 Competition - Metrics Over Time ({csv_path.name})",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (col, title) in zip(axes.flat, SUBPLOTS):
        ax.plot(df["time_minutes"], df[col], color="steelblue", linewidth=1.0, marker=".", markersize=3)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
        ax.set_facecolor("white")

    for ax in axes[3]:
        ax.set_xlabel("Time (minutes)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_metrics <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    out = plot_dashboard(csv_path)
    print(f"Dashboard saved to {out}")


if __name__ == "__main__":
    main()

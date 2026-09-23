from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy_by_layer(
    results_by_pooling: dict[str, pd.DataFrame],
    output_path: str,
    metric: str = "test_auroc",
) -> None:
    """
    One figure, both pooling strategies overlaid, so the README can make a
    direct claim about whether pooling choice matters.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for pooling, df in results_by_pooling.items():
        ax.plot(df["layer"], df[metric], marker="o", label=pooling)
        if "cv_std_accuracy" in df.columns and metric == "test_auroc":
            pass 

    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Linear probe performance by layer")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.legend()
    ax.grid(alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_head_heatmap(
    head_results: pd.DataFrame,
    output_path: str,
    metric: str = "test_auroc",
) -> None:
    """Layer x head grid, colored by probe performance."""
    pivot = head_results.pivot(index="layer", columns="head", values=metric)

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.4), max(3, pivot.shape[0] * 0.6)))
    sns.heatmap(pivot, annot=False, cmap="viridis", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f"Per-head probe {metric.replace('_', ' ')} (top layers)")
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

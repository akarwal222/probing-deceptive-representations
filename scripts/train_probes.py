"""
train layer-wise probes on cached activations and drill into the best layers with per-head probes
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml

from src.activations import load_cached_activations
from src.data import load_liars_bench
from src.head_probes import extract_head_activations, train_head_probes
from src.model_utils import load_model
from src.probes import best_layer, train_probes_across_layers
from src.visualize import plot_accuracy_by_layer, plot_head_heatmap

N_TOP_LAYERS_FOR_HEAD_PROBE = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tag = config["dataset"]["subset"]
    activations_dir = config["output"]["activations_dir"]

    # Layer-wise sweep, both poolings
    results_by_pooling: dict[str, pd.DataFrame] = {}
    for pooling in ("last_token", "mean"):
        path = f"{activations_dir}/{tag}_{pooling}.pt"
        activations, labels = load_cached_activations(path)
        print(f"[{pooling}] activations shape: {tuple(activations.shape)}")

        df = train_probes_across_layers(
            activations,
            labels,
            pooling=pooling,
            test_size=config["probing"]["test_size"],
            cv_folds=config["probing"]["cv_folds"],
            C=config["probing"]["C"],
            max_iter=config["probing"]["max_iter"],
            seed=config["probing"]["seed"],
        )
        results_by_pooling[pooling] = df
        print(df[["layer", "cv_mean_accuracy", "test_accuracy", "test_auroc"]])

    all_results = pd.concat(results_by_pooling.values(), ignore_index=True)
    all_results.to_csv(config["output"]["metrics_path"], index=False)
    print(f"Saved layer-wise metrics -> {config['output']['metrics_path']}")

    plot_accuracy_by_layer(
        results_by_pooling,
        output_path=f"{config['output']['figures_dir']}/accuracy_by_layer.png",
    )

    # Head-level probes on the top layers with last_token pooling
    primary_df = results_by_pooling["last_token"]
    top_layers_hidden_states_idx = (
        primary_df.sort_values("test_auroc", ascending=False)
        .head(N_TOP_LAYERS_FOR_HEAD_PROBE)["layer"]
        .tolist()
    )
    
    top_block_indices = [idx - 1 for idx in top_layers_hidden_states_idx if idx > 0]
    print(f"Top layers for head-level probing (block indices): {top_block_indices}")

    print("Reloading model + dataset for head-level extraction...")
    loaded = load_model(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
    )
    examples = load_liars_bench(
        hf_path=config["dataset"]["hf_path"],
        subset=config["dataset"]["subset"],
        split=config["dataset"]["split"],
        max_examples_per_class=config["dataset"]["max_examples_per_class"],
        seed=config["dataset"]["seed"],
    )
    texts = [ex.text for ex in examples]
    labels_tensor = load_cached_activations(f"{activations_dir}/{tag}_last_token.pt")[1]

    head_activations = extract_head_activations(
        loaded, texts, layer_indices=top_block_indices, pooling="last_token",
        max_seq_len=config["model"]["max_seq_len"], batch_size=16,
    )
    head_results = train_head_probes(
        head_activations, labels_tensor, seed=config["probing"]["seed"]
    )
    head_results.to_csv(
        config["output"]["metrics_path"].replace(".csv", "_heads.csv"), index=False
    )
    plot_head_heatmap(
        head_results, output_path=f"{config['output']['figures_dir']}/head_heatmap.png"
    )

    summary = {
        "best_layer_last_token": best_layer(results_by_pooling["last_token"]),
        "best_layer_mean": best_layer(results_by_pooling["mean"]),
        "top_block_indices_for_head_probe": top_block_indices,
        "best_head_result": head_results.sort_values("test_auroc", ascending=False).iloc[0].to_dict(),
    }
    with open(f"{config['output']['figures_dir']}/../summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

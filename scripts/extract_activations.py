"""
extract and cache activations.
"""

from __future__ import annotations

import argparse

import yaml

from src.activations import build_and_cache_activations
from src.data import load_liars_bench
from src.model_utils import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loading model: {config['model']['name']}")
    loaded = load_model(
        model_name=config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=config["model"]["device"],
    )
    print(
        f"Loaded {config['model']['name']} "
        f"({loaded.num_layers} layers, hidden size {loaded.hidden_size}) on {loaded.device}"
    )

    print(f"Loading dataset: {config['dataset']['hf_path']} / {config['dataset']['subset']}")
    examples = load_liars_bench(
        hf_path=config["dataset"]["hf_path"],
        subset=config["dataset"]["subset"],
        split=config["dataset"]["split"],
        max_examples_per_class=config["dataset"]["max_examples_per_class"],
        seed=config["dataset"]["seed"],
    )
    print(f"Loaded {len(examples)} balanced examples")

    saved_paths = build_and_cache_activations(
        loaded=loaded,
        examples=examples,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=16,
        output_dir=config["output"]["activations_dir"],
        tag=config["dataset"]["subset"],
    )

    for pooling, path in saved_paths.items():
        print(f"Saved {pooling} activations -> {path}")


if __name__ == "__main__":
    main()

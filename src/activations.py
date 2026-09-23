"""
Ties model_utils + data together: run the dataset through the model once,
cache pooled per-layer activations for BOTH pooling strategies
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data import ProbingExample
from src.model_utils import LoadedModel, extract_hidden_states


def build_and_cache_activations(
    loaded: LoadedModel,
    examples: list[ProbingExample],
    max_seq_len: int,
    batch_size: int,
    output_dir: str,
    tag: str,
) -> dict[str, Path]:
    """
    Extracts activations for both "last_token" and "mean" pooling and saves
    each as a .pt file containing {"activations": Tensor[L+1, N, H], "labels": Tensor[N]}
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [ex.text for ex in examples]
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.long)

    saved_paths: dict[str, Path] = {}
    for pooling in ("last_token", "mean"):
        activations = extract_hidden_states(
            loaded,
            texts,
            pooling=pooling,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
        ) 

        save_path = out_dir / f"{tag}_{pooling}.pt"
        torch.save({"activations": activations, "labels": labels}, save_path)
        saved_paths[pooling] = save_path

    with open(out_dir / f"{tag}_texts.json", "w") as f:
        json.dump(texts, f)

    return saved_paths


def load_cached_activations(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path)
    return payload["activations"], payload["labels"]

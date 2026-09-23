"""
Extract and analyze the activations of individual attention heads in a pre-trained language model.
"""

from __future__ import annotations

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ProbingExample
from src.model_utils import LoadedModel


@torch.no_grad()
def extract_head_activations(
    loaded: LoadedModel,
    texts: list[str],
    layer_indices: list[int],
    pooling: str = "last_token",
    max_seq_len: int = 128,
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """
    Returns {layer_idx: Tensor[N, num_heads, head_dim]}, pooled the same way
    as the main residual-stream extraction so results are comparable.
    """
    num_heads = getattr(loaded.model.config, "num_attention_heads")
    head_dim = getattr(
        loaded.model.config, "head_dim", loaded.hidden_size // num_heads
    )

    captured: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args):
            captured[layer_idx].append(args[0].detach())
        return hook

    try:
        decoder_layers = loaded.model.model.layers  # Llama/Qwen-style architecture
    except AttributeError as e:
        raise AttributeError(
            "Could not find `model.model.layers` -- this model's module "
            "structure differs from the Llama/Qwen convention this function "
            "assumes. Inspect `loaded.model` and adjust the path."
        ) from e

    for layer_idx in layer_indices:
        o_proj = decoder_layers[layer_idx].self_attn.o_proj
        hooks.append(o_proj.register_forward_pre_hook(make_hook(layer_idx)))

    per_layer_pooled: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}

    try:
        for start in range(0, len(texts), batch_size):
            for layer_idx in layer_indices:
                captured[layer_idx].clear()

            batch_texts = texts[start : start + batch_size]
            encoded = loaded.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(loaded.device)
            attention_mask = encoded["attention_mask"]

            loaded.model(**encoded)  

            for layer_idx in layer_indices:
                o_proj_input = captured[layer_idx][0]  # (batch, seq, num_heads * head_dim)
                per_head = o_proj_input.view(
                    o_proj_input.size(0), o_proj_input.size(1), num_heads, head_dim
                )

                if pooling == "last_token":
                    last_real_token = attention_mask.sum(dim=1) - 1
                    pooled = per_head[torch.arange(per_head.size(0)), last_real_token]
                elif pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).unsqueeze(-1).to(per_head.dtype)
                    pooled = (per_head * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")

                per_layer_pooled[layer_idx].append(pooled.float().cpu())
    finally:
        for hook in hooks:
            hook.remove()

    return {
        layer_idx: torch.cat(chunks, dim=0)  # (N, num_heads, head_dim)
        for layer_idx, chunks in per_layer_pooled.items()
    }


def train_head_probes(
    head_activations: dict[int, torch.Tensor],
    labels: torch.Tensor,
    test_size: float = 0.2,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fits one logistic regression per (layer, head) pair on that head's
    pooled activation vector alone. Returns a long-format DataFrame ready
    for a layer x head heatmap.
    """
    y = labels.numpy()
    rows = []

    for layer_idx, acts in head_activations.items():
        num_heads = acts.shape[1]
        for head_idx in range(num_heads):
            X = acts[:, head_idx, :].numpy()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )

            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
            )
            probe.fit(X_train, y_train)
            test_proba = probe.predict_proba(X_test)[:, 1]

            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "test_accuracy": accuracy_score(y_test, probe.predict(X_test)),
                    "test_auroc": roc_auc_score(y_test, test_proba),
                }
            )

    return pd.DataFrame(rows)

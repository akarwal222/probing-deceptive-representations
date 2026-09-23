"""
Load a pretrained causal LM and extract per-layer residual-stream activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    num_layers: int         
    hidden_size: int
    device: torch.device


def load_model(model_name: str, dtype: str = "float16", device: str = "cuda") -> LoadedModel:
    """
    Load a HF causal LM + tokenizer, falling back to CPU/float32 if CUDA
    isn't available so the same config works locally and on Colab.
    """
    torch_dtype = getattr(torch, dtype) if torch.cuda.is_available() else torch.float32
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    ).to(resolved_device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        device=resolved_device,
    )


@torch.no_grad()
def extract_hidden_states(
    loaded: LoadedModel,
    texts: list[str],
    pooling: Literal["last_token", "mean"] = "last_token",
    max_seq_len: int = 128,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Pooling:
      - "last_token": the representation at the final token.
        Standard choice for decoder-only models, by the last position the
        model has "seen" the whole input via causal attention.
      - "mean": mean over non-padding tokens. Included as a robustness check
    """
    all_layer_vectors: list[list[torch.Tensor]] = [[] for _ in range(loaded.num_layers + 1)]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = loaded.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(loaded.device)

        outputs = loaded.model(**encoded)
        hidden_states = outputs.hidden_states  # tuple: (num_layers + 1) x (batch, seq, hidden)
        attention_mask = encoded["attention_mask"]  # (batch, seq)

        for layer_idx, layer_hidden in enumerate(hidden_states):
            if pooling == "last_token":
                last_real_token = attention_mask.sum(dim=1) - 1  # (batch,)
                pooled = layer_hidden[
                    torch.arange(layer_hidden.size(0)), last_real_token
                ]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).to(layer_hidden.dtype)
                pooled = (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            all_layer_vectors[layer_idx].append(pooled.float().cpu())

    # (num_layers + 1, num_texts, hidden_size)
    return torch.stack([torch.cat(layer_chunks, dim=0) for layer_chunks in all_layer_vectors])

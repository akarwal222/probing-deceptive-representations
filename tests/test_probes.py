"""
Sanity tests
"""

from __future__ import annotations

import numpy as np
import torch

from src.probes import best_layer, train_probes_across_layers


def _make_synthetic_activations(num_layers_plus_one: int, n: int, hidden: int, signal_layer: int, seed: int = 0):
    """
    Builds fake per-layer activations where only `signal_layer` actually
    encodes the label; every other layer is pure noise. A correct probing
    pipeline should recover `signal_layer` as the best layer.
    """
    rng = np.random.RandomState(seed)
    labels = torch.tensor(rng.randint(0, 2, size=n), dtype=torch.long)

    layers = []
    for layer in range(num_layers_plus_one):
        if layer == signal_layer:
            # class-conditional mean shift means linearly separable
            base = rng.randn(n, hidden)
            base[labels.numpy() == 1] += 3.0
            layers.append(torch.tensor(base, dtype=torch.float32))
        else:
            layers.append(torch.tensor(rng.randn(n, hidden), dtype=torch.float32))

    return torch.stack(layers), labels


def test_probe_recovers_signal_layer():
    activations, labels = _make_synthetic_activations(
        num_layers_plus_one=6, n=200, hidden=16, signal_layer=3
    )

    results = train_probes_across_layers(
        activations, labels, pooling="last_token", cv_folds=3, seed=0
    )

    assert best_layer(results, metric="test_auroc") == 3
    signal_row = results[results["layer"] == 3].iloc[0]
    assert signal_row["test_accuracy"] > 0.9

    noise_rows = results[results["layer"] != 3]
    assert (noise_rows["test_accuracy"] < 0.75).all()


def test_probe_output_shape_matches_num_layers():
    activations, labels = _make_synthetic_activations(
        num_layers_plus_one=4, n=100, hidden=8, signal_layer=1
    )
    results = train_probes_across_layers(activations, labels, pooling="mean", cv_folds=3, seed=0)
    assert len(results) == 4
    assert set(results["layer"]) == {0, 1, 2, 3}

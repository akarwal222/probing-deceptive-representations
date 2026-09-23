"""
Layer-wise linear probing: for each layer's pooled activations, fit a
standardized logistic regression to predict lie (1) vs honest (0), and
report both cross-validated train accuracy and held-out test performance.

(Alain & Bengio, 2016; Belinkov, 2022) 
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LayerProbeResult:
    layer: int
    pooling: str
    cv_mean_accuracy: float
    cv_std_accuracy: float
    test_accuracy: float
    test_auroc: float
    n_train: int
    n_test: int


def train_probes_across_layers(
    activations: torch.Tensor,  # (num_layers + 1, N, hidden_size)
    labels: torch.Tensor,       # (N,)
    pooling: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    y = labels.numpy()
    num_layers_plus_one = activations.shape[0]

    results: list[LayerProbeResult] = []
    for layer in range(num_layers_plus_one):
        X = activations[layer].numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(probe, X_train, y_train, cv=cv, scoring="accuracy")

        probe.fit(X_train, y_train)
        test_pred = probe.predict(X_test)
        test_proba = probe.predict_proba(X_test)[:, 1]

        results.append(
            LayerProbeResult(
                layer=layer,
                pooling=pooling,
                cv_mean_accuracy=float(cv_scores.mean()),
                cv_std_accuracy=float(cv_scores.std()),
                test_accuracy=float(accuracy_score(y_test, test_pred)),
                test_auroc=float(roc_auc_score(y_test, test_proba)),
                n_train=len(y_train),
                n_test=len(y_test),
            )
        )

    return pd.DataFrame([r.__dict__ for r in results])


def best_layer(results_df: pd.DataFrame, metric: str = "test_auroc") -> int:
    """Convenience accessor used by scripts/train_probes.py and analysis.py."""
    return int(results_df.loc[results_df[metric].idxmax(), "layer"])

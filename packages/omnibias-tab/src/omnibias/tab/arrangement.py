# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Hyperplane-arrangement view of tabular classification (05-02 G1/G2).

An arrangement with ``H`` hyperplanes applies **all** gates to every input and
forms soft cell memberships over the ``2**H`` sign patterns -- the same product
as :func:`omnibias.partition.partition_weights` with ``depth = H``. For the
Wave-0 falsifier ``H = 2`` this is exact on the constructed oblique XOR and
axis AND datasets; larger ``H`` reuses the same primitive without enumerating
cells at prediction time beyond the soft sum.

Terminology: gate hardening ``beta -> inf`` is temperature collapse, distinct
from the founding bias collapse ``delta -> 0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omnibias.partition import (
    PartitionConfig,
    PartitionParams,
    hard_assignment,
    init_params,
    partition_weights,
)
from omnibias.partition.certify import PartitionGapCertificate, certify_partition_gap
from omnibias.partition.registry import combine_outputs

FloatArray = np.ndarray


def make_oblique_xor(
    *,
    n_samples: int = 10_000,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Constructed oblique dataset: ``y = 1[w1.x > 0] XOR 1[w2.x > 0]``."""
    rng = np.random.default_rng(int(seed))
    X = rng.normal(size=(int(n_samples), int(n_features)))
    w1 = rng.normal(size=int(n_features))
    w2 = rng.normal(size=int(n_features))
    y = ((X @ w1 > 0.0) ^ (X @ w2 > 0.0)).astype(np.float64)
    meta = {
        "family": "oblique_xor",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "seed": int(seed),
        "w1": w1,
        "w2": w2,
        "bayes_rate": 1.0,
    }
    return X, y, meta


def make_axis_rule(
    *,
    n_samples: int = 10_000,
    n_features: int = 10,
    seed: int = 0,
    feature_a: int = 2,
    feature_b: int = 6,
    threshold_a: float = 0.5,
    threshold_b: float = 0.2,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Constructed axis dataset: ``y = 1[x_a > t_a] AND 1[x_b < t_b]``."""
    rng = np.random.default_rng(int(seed))
    d = int(n_features)
    fa, fb = int(feature_a), int(feature_b)
    if not (0 <= fa < d and 0 <= fb < d and fa != fb):
        raise ValueError(f"feature indices must be distinct in [0, {d})")
    X = rng.uniform(size=(int(n_samples), d))
    y = ((X[:, fa] > float(threshold_a)) & (X[:, fb] < float(threshold_b))).astype(
        np.float64
    )
    majority = float(max(y.mean(), 1.0 - y.mean()))
    meta = {
        "family": "axis_and",
        "n_samples": int(n_samples),
        "n_features": d,
        "seed": int(seed),
        "feature_a": fa,
        "feature_b": fb,
        "threshold_a": float(threshold_a),
        "threshold_b": float(threshold_b),
        "majority_class_rate": majority,
        "bayes_rate": 1.0,
    }
    return X, y, meta


def obliqueness_diagnostic(X: FloatArray, y: FloatArray) -> float:
    """Dense-linear-probe accuracy over best single-feature probe accuracy.

    Detects *linear* oblique structure (a dense probe beats axis probes). It
    does **not** discriminate parity / XOR structure from axis rules: XOR is
    not linearly separable, so the ratio stays near 1 on both families.
    Frozen for Wave-0 (reported, not gated -- G4 is out of scope); do not
    retune against the constructed G1/G2 datasets.
    """
    Xv = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xv.ndim != 2 or Xv.shape[0] != yv.shape[0]:
        raise ValueError("X and y shape mismatch")
    # Dense linear probe via least squares on a bias-augmented design.
    xb = np.concatenate([Xv, np.ones((Xv.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(xb, yv, rcond=None)
    dense_scores = xb @ coef
    # Choose the better of threshold-0.5 and sign polarity.
    dense_acc = max(
        float(((dense_scores >= 0.5).astype(np.float64) == yv).mean()),
        float(((dense_scores < 0.5).astype(np.float64) == yv).mean()),
        float(((dense_scores >= 0.0).astype(np.float64) == yv).mean()),
        float(((dense_scores < 0.0).astype(np.float64) == yv).mean()),
    )
    best_axis = 0.0
    for j in range(Xv.shape[1]):
        qs = np.unique(np.quantile(Xv[:, j], np.linspace(0.05, 0.95, 19)))
        for thr in qs:
            for side in (1.0, -1.0):
                pred = ((side * (Xv[:, j] - thr)) > 0.0).astype(np.float64)
                acc = float((pred == yv).mean())
                acc_flip = float((1.0 - pred == yv).mean())
                best_axis = max(best_axis, acc, acc_flip)
    best_axis = max(best_axis, 1e-6)
    return float(dense_acc / best_axis)


def arrangement_params(
    W: FloatArray,
    t: FloatArray,
    *,
    n_features: int | None = None,
    beta_final: float = 32.0,
) -> PartitionParams:
    """Wrap hyperplane normals / offsets as a :class:`PartitionParams`."""
    Ww = np.asarray(W, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    d = int(n_features) if n_features is not None else int(Ww.shape[1])
    cfg = PartitionConfig(
        n_features=d,
        depth=int(Ww.shape[0]),
        split_kind="oblique",
        beta_final=float(beta_final),
    )
    return PartitionParams(cfg, Ww, tt)


def certify_arrangement_gap(
    W: FloatArray,
    t: FloatArray,
    X: FloatArray,
    *,
    beta: float,
    n_features: int | None = None,
) -> PartitionGapCertificate:
    """Sound soft->hard membership gap via :func:`certify_partition_gap`."""
    params = arrangement_params(W, t, n_features=n_features, beta_final=float(beta))
    return certify_partition_gap(params, X, beta=float(beta))


def arrangement_weights(
    W: FloatArray,
    t: FloatArray,
    X: FloatArray,
    beta: float,
) -> FloatArray:
    """Soft cell memberships ``(n, 2**H)`` via the partition POU primitive."""
    params = arrangement_params(W, t)
    return partition_weights(params, X, float(beta))


def predict_proba_np(
    W: FloatArray,
    t: FloatArray,
    cell_logits: FloatArray,
    X: FloatArray,
    beta: float,
) -> FloatArray:
    """Soft arrangement probabilities ``sigmoid(sum_cell w_cell * logit_cell)``."""
    weights = arrangement_weights(W, t, X, beta)
    logits = combine_outputs(
        weights, np.broadcast_to(np.asarray(cell_logits, dtype=np.float64), weights.shape)
    )
    # Numerically stable sigmoid.
    out = np.empty_like(logits)
    pos = logits >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    ez = np.exp(logits[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def hard_predict_np(
    W: FloatArray,
    t: FloatArray,
    cell_logits: FloatArray,
    X: FloatArray,
) -> FloatArray:
    """Crisp ``beta -> inf`` prediction from hard cell assignment."""
    params = arrangement_params(W, t)
    idx = hard_assignment(params, X)
    logits = np.asarray(cell_logits, dtype=np.float64).reshape(-1)
    return (logits[idx] > 0.0).astype(np.float64)


def init_arrangement(
    n_features: int,
    n_hyperplanes: int = 2,
    *,
    seed: int = 0,
    split_kind: str = "oblique",
) -> tuple[PartitionParams, FloatArray]:
    """Random hyperplanes + zero cell logits."""
    cfg = PartitionConfig(
        n_features=int(n_features),
        depth=int(n_hyperplanes),
        split_kind=split_kind,
        seed=int(seed),
    )
    params = init_params(cfg, seed)
    logits = np.zeros(cfg.n_regions, dtype=np.float64)
    return params, logits


__all__ = [
    "arrangement_params",
    "arrangement_weights",
    "certify_arrangement_gap",
    "hard_predict_np",
    "init_arrangement",
    "make_axis_rule",
    "make_oblique_xor",
    "obliqueness_diagnostic",
    "predict_proba_np",
]

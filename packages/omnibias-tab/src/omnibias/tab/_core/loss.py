# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form losses, score-space curvature, and metrics (numpy).

The **score-space gradient and Hessian** are closed form thanks to the Riccati structure
of the sigmoid tower -- ``d/dF sigmoid(F) = p (1 - p)`` -- and are exactly what the
stagewise **Newton-boosting** driver (:mod:`omnibias.tab.torch.boosting`, the GBM-mirror)
fits each weak learner to, and what a Gauss-Newton / Fisher metric provider uses:

* binary logistic:  ``g = p - y``,        ``h = p (1 - p)``   with ``p = sigmoid(F)``;
* multiclass CE:    ``g = softmax(F) - Y``, ``h = p (1 - p)``  (diagonal);
* regression MSE:   ``g = 2 (F - y)``,     ``h = 2``.

``per_sample=False`` returns the mean-loss reduction (the ``1/n`` folded in).
"""

from __future__ import annotations

import numpy as np
from omnibias.tab._core.forward import sigmoid_np, softmax_np
from omnibias.tab._core.params import FloatArray

_EPS = 1e-12


def _one_hot(y: FloatArray, k: int) -> FloatArray:
    idx = np.asarray(y, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], k), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def loss_value(F: FloatArray, y: FloatArray, task: str) -> float:
    r"""Mean loss for scores ``F`` (``(n, k)``) and targets ``y``."""
    Fv = np.asarray(F, dtype=np.float64)
    if task == "binary":
        p = np.clip(sigmoid_np(Fv[:, 0]), _EPS, 1.0 - _EPS)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        return float(-np.mean(yv * np.log(p) + (1.0 - yv) * np.log(1.0 - p)))
    if task == "multiclass":
        pm = np.clip(softmax_np(Fv), _EPS, 1.0)
        Y = _one_hot(y, Fv.shape[1])
        return float(-np.mean(np.sum(Y * np.log(pm), axis=-1)))
    yr = np.asarray(y, dtype=np.float64).reshape(Fv.shape)
    return float(np.mean((Fv - yr) ** 2))


def score_grad_hess(F: FloatArray, y: FloatArray, task: str) -> tuple[FloatArray, FloatArray]:
    r"""Per-sample score-space gradient ``g`` and (diagonal) Hessian ``h``, both ``(n, k)``.

    These are the closed-form derivatives ``dL_i / dF_i`` and ``d^2 L_i / dF_i^2`` of the
    per-sample loss (no ``1/n``); the boosting driver forms the Newton leaf target
    ``-sum g / (sum h + lambda)`` from them.
    """
    Fv = np.asarray(F, dtype=np.float64)
    if task == "binary":
        p = sigmoid_np(Fv[:, 0])
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        g = (p - yv)[:, None]
        h = np.clip(p * (1.0 - p), _EPS, None)[:, None]
        return g, h
    if task == "multiclass":
        pm = softmax_np(Fv)
        Y = _one_hot(y, Fv.shape[1])
        gm = pm - Y
        hm = np.clip(pm * (1.0 - pm), _EPS, None)
        return gm, hm
    yr = np.asarray(y, dtype=np.float64).reshape(Fv.shape)
    gr = 2.0 * (Fv - yr)
    hr = np.full_like(Fv, 2.0)
    return gr, hr


def accuracy(F: FloatArray, y: FloatArray, task: str) -> float:
    r"""Classification accuracy (binary / multiclass)."""
    Fv = np.asarray(F, dtype=np.float64)
    if task == "binary":
        pred = (Fv[:, 0] > 0.0).astype(np.float64)
        return float(np.mean(pred == np.asarray(y, dtype=np.float64).reshape(-1)))
    if task == "multiclass":
        pred = np.argmax(Fv, axis=-1)
        return float(np.mean(pred == np.asarray(y, dtype=np.int64).reshape(-1)))
    raise ValueError("accuracy is only defined for classification tasks")


def rmse(F: FloatArray, y: FloatArray) -> float:
    r"""Root-mean-square error for a regression model."""
    Fv = np.asarray(F, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(Fv.shape)
    return float(np.sqrt(np.mean((Fv - yv) ** 2)))


def metric(F: FloatArray, y: FloatArray, task: str) -> float:
    r"""The task's headline held-out metric (accuracy for clf, negative RMSE for reg).

    Returned so that **higher is better** for every task (the benchmark's ``>=`` gate)."""
    if task == "regression":
        return -rmse(F, y)
    return accuracy(F, y, task)


__all__ = [
    "accuracy",
    "loss_value",
    "metric",
    "rmse",
    "score_grad_hess",
]

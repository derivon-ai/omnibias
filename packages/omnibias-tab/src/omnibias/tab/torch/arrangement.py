# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Trainable hyperplane-arrangement classifier (torch).

Builds on :func:`omnibias.partition.torch.partition_weights_arrays`: ``H``
hyperplanes yield soft memberships over ``2**H`` cells, blended by per-cell
logits. Training anneals ``beta -> beta_final`` (temperature collapse) and
supports an ``L1`` penalty on the normals plus a sparse feature-pair warm-start
used by the axis-structured falsifier (05-02 G2).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omnibias.partition.torch.weights import combine, partition_weights_arrays
from torch import Tensor, nn

torch.set_default_dtype(torch.float64)


class ArrangementClassifier(nn.Module):
    """``H`` hyperplanes, soft cell membership, per-cell logits."""

    def __init__(
        self,
        n_features: int,
        n_hyperplanes: int = 2,
        *,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if n_hyperplanes < 1:
            raise ValueError("n_hyperplanes must be >= 1")
        self.n_features = int(n_features)
        self.n_hyperplanes = int(n_hyperplanes)
        self.n_cells = 1 << self.n_hyperplanes
        self.beta = float(beta)
        scale = 1.0 / (self.n_features**0.5)
        self.W = nn.Parameter(torch.randn(self.n_hyperplanes, self.n_features) * scale)
        self.t = nn.Parameter(torch.zeros(self.n_hyperplanes))
        self.cell_logits = nn.Parameter(torch.zeros(self.n_cells))

    def forward(self, X: Tensor) -> Tensor:
        weights = partition_weights_arrays(
            self.W, self.t, X, self.beta, self.n_hyperplanes
        )
        logits = self.cell_logits.expand(X.shape[0], -1)
        return combine(weights, logits)

    @torch.no_grad()
    def predict(self, X: Tensor | np.ndarray) -> np.ndarray:
        xv = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
        return (self.forward(xv).detach().numpy() > 0.0).astype(np.float64)

    @torch.no_grad()
    def predict_proba(self, X: Tensor | np.ndarray) -> np.ndarray:
        xv = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
        logits = self.forward(xv).detach().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    def numpy_state(self) -> dict[str, np.ndarray]:
        return {
            "W": self.W.detach().cpu().numpy().copy(),
            "t": self.t.detach().cpu().numpy().copy(),
            "cell_logits": self.cell_logits.detach().cpu().numpy().copy(),
        }


@dataclass
class FitResult:
    model: ArrangementClassifier
    train_acc: float
    val_acc: float
    beta_final: float
    l1: float
    n_restarts: int
    best_restart: int
    sparse_warmstart: bool


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred.reshape(-1) == np.asarray(y, dtype=np.float64).reshape(-1)).mean())


def _hard_cell_logits(W: np.ndarray, t: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    G = X @ W.T > t[None, :]
    H = W.shape[0]
    logits = np.zeros(1 << H, dtype=np.float64)
    for r in range(1 << H):
        mask = np.ones(X.shape[0], dtype=bool)
        for j in range(H):
            bit = (r >> j) & 1
            mask &= G[:, j] == bool(bit)
        rate = float(y[mask].mean()) if mask.any() else 0.5
        rate = float(np.clip(rate, 1e-3, 1.0 - 1e-3))
        logits[r] = float(np.log(rate / (1.0 - rate)))
    return logits


def _sparse_warmstarts(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_hyperplanes: int = 2,
    n_quantiles: int = 19,
    top_k: int = 8,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Feature-pair + threshold grid warm-starts for axis-structured data."""
    if n_hyperplanes != 2:
        return []
    n, d = X.shape
    qs = np.linspace(0.05, 0.95, int(n_quantiles))
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for f0, f1 in itertools.combinations(range(d), 2):
        q0 = np.unique(np.quantile(X[:, f0], qs))
        q1 = np.unique(np.quantile(X[:, f1], qs))
        best_local = -1.0
        best_pack: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for s0, s1 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            for thr0 in q0:
                for thr1 in q1:
                    W = np.zeros((2, d), dtype=np.float64)
                    t = np.zeros(2, dtype=np.float64)
                    W[0, f0] = s0
                    t[0] = s0 * float(thr0)
                    W[1, f1] = s1
                    t[1] = s1 * float(thr1)
                    logits = _hard_cell_logits(W, t, X, y)
                    G = X @ W.T > t[None, :]
                    idx = G[:, 0].astype(np.int64) + 2 * G[:, 1].astype(np.int64)
                    pred = (logits[idx] > 0.0).astype(np.float64)
                    acc = float((pred == y).mean())
                    if acc > best_local:
                        best_local = acc
                        best_pack = (W, t, logits)
        if best_pack is not None:
            scored.append((best_local, *best_pack))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [
        (W, t, logits, float(acc))
        for acc, W, t, logits in scored[: int(top_k)]
    ]


def _train_restart(
    model: ArrangementClassifier,
    Xtr: Tensor,
    ytr: Tensor,
    *,
    steps: int,
    beta_init: float,
    beta_final: float,
    l1: float,
    lr: float,
    freeze_W: bool = False,
) -> None:
    params = []
    if freeze_W:
        model.W.requires_grad_(False)
        params.append({"params": [model.t, model.cell_logits], "lr": lr})
    else:
        model.W.requires_grad_(True)
        params.append({"params": [model.W], "lr": lr * 0.5})
        params.append({"params": [model.t, model.cell_logits], "lr": lr})
    opt = torch.optim.Adam(params)
    for step in range(int(steps)):
        frac = step / max(int(steps) - 1, 1)
        model.beta = float(beta_init * (beta_final / beta_init) ** frac)
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ytr)
        if l1 > 0.0:
            loss = loss + float(l1) * model.W.abs().sum()
        loss.backward()
        opt.step()
    model.beta = float(beta_final)
    model.W.requires_grad_(True)


def fit_arrangement(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_hyperplanes: int = 2,
    l1: float = 0.0,
    restarts: int = 8,
    steps: int = 600,
    beta_init: float = 1.0,
    beta_final: float = 64.0,
    lr: float = 0.05,
    seed: int = 0,
    val_fraction: float = 0.2,
    sparse_warmstart: bool | None = None,
) -> FitResult:
    """Fit an :class:`ArrangementClassifier` with beta anneal and restarts.

    When ``sparse_warmstart`` is true (default if ``l1 > 0``), also try
    feature-pair threshold warm-starts -- the path that recovers axis-aligned
    rules for 05-02 G2.
    """
    Xv = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xv.ndim != 2 or Xv.shape[0] != yv.shape[0]:
        raise ValueError("X/y shape mismatch")
    n, d = Xv.shape
    n_val = max(1, int(round(n * float(val_fraction))))
    n_tr = n - n_val
    if n_tr < 1:
        raise ValueError("not enough samples for a train/val split")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
    Xtr_np, ytr_np = Xv[tr_idx], yv[tr_idx]
    Xva_np, yva_np = Xv[va_idx], yv[va_idx]
    Xtr = torch.as_tensor(Xtr_np)
    ytr = torch.as_tensor(ytr_np)
    use_sparse = bool(l1 > 0.0) if sparse_warmstart is None else bool(sparse_warmstart)

    candidates: list[tuple[str, dict[str, Any]]] = []
    for r in range(int(restarts)):
        candidates.append(("dense", {"restart": r}))
    if use_sparse:
        for pack in _sparse_warmstarts(Xtr_np, ytr_np, n_hyperplanes=n_hyperplanes):
            W, t, logits, acc = pack
            candidates.append(
                ("sparse", {"W": W, "t": t, "logits": logits, "hard_acc": acc})
            )

    best: FitResult | None = None
    for i, (kind, cfg) in enumerate(candidates):
        torch.manual_seed(int(seed) * 1009 + i)
        model = ArrangementClassifier(d, n_hyperplanes, beta=float(beta_init))
        freeze_W = False
        if kind == "sparse":
            with torch.no_grad():
                model.W.copy_(torch.as_tensor(cfg["W"]))
                model.t.copy_(torch.as_tensor(cfg["t"]))
                model.cell_logits.copy_(torch.as_tensor(cfg["logits"]))
            freeze_W = True
            local_steps = max(200, int(steps) // 2)
            local_l1 = float(l1)
            local_beta_init = max(float(beta_init), 8.0)
        else:
            local_steps = int(steps)
            local_l1 = float(l1)
            local_beta_init = float(beta_init)
        _train_restart(
            model,
            Xtr,
            ytr,
            steps=local_steps,
            beta_init=local_beta_init,
            beta_final=float(beta_final),
            l1=local_l1,
            lr=float(lr),
            freeze_W=freeze_W,
        )
        # Optional unfreeze polish for sparse warm-starts.
        if kind == "sparse":
            _train_restart(
                model,
                Xtr,
                ytr,
                steps=max(100, int(steps) // 4),
                beta_init=float(beta_final),
                beta_final=float(beta_final),
                l1=float(l1),
                lr=float(lr) * 0.5,
                freeze_W=False,
            )
        tr_acc = _accuracy(model.predict(Xtr_np), ytr_np)
        va_acc = _accuracy(model.predict(Xva_np), yva_np)
        result = FitResult(
            model=model,
            train_acc=tr_acc,
            val_acc=va_acc,
            beta_final=float(beta_final),
            l1=float(l1),
            n_restarts=len(candidates),
            best_restart=i,
            sparse_warmstart=use_sparse,
        )
        if best is None or va_acc > best.val_acc:
            best = result
    assert best is not None
    return best


__all__ = ["ArrangementClassifier", "FitResult", "fit_arrangement"]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Trainable hyperplane-arrangement classifier (torch).

Builds on :func:`omnibias.partition.torch.partition_weights_arrays`: ``H``
hyperplanes yield soft memberships over ``2**H`` cells, blended by per-cell
logits. Training anneals ``beta -> beta_final`` (temperature collapse) and
supports an ``L1`` penalty on the normals plus a sparse feature-pair warm-start
used by the axis-structured falsifier (05-02 G2).

Training contract (LightGBM-matched): train on ``X``, early-stop + restore best
on a held-out validation set (optional ``X_val`` / ``y_val``, else an inner
``val_fraction`` split). ``steps`` is a max cap; stop metric is val BCE.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omnibias.partition._core.params import region_code_matrix
from omnibias.partition.torch.weights import combine, partition_weights_arrays
from torch import Tensor, nn

_DTYPE = torch.float64
_TASKS = ("binary", "multiclass", "regression")


def _feature_rows(X: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Flatten ``X`` of shape ``(..., d)`` to ``(-1, d)``; return leading dims."""
    if X.ndim < 2:
        raise ValueError("X must have shape (..., n_features)")
    leading = tuple(int(s) for s in X.shape[:-1])
    return X.reshape(-1, X.shape[-1]), leading


def _cell_2d(cell: Tensor) -> Tensor:
    return cell.unsqueeze(-1) if cell.ndim == 1 else cell


def _squeeze_binary_np(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[-1] == 1:
        return arr.reshape(arr.shape[:-1])
    return arr


def _batched_arrangement_logits(
    W: Tensor,
    t: Tensor,
    cell_logits: Tensor,
    X: Tensor,
    beta: Tensor,
) -> Tensor:
    """Soft arrangement logits ``(n, M, k)`` from stacked members.

    ``W`` is ``(M, H, d)``, ``t`` is ``(M, H)``, ``cell_logits`` is
    ``(M, 2**H, k)``, ``X`` is ``(n, d)``, ``beta`` is ``()`` or ``(M,)``.
    """
    codes = torch.as_tensor(
        region_code_matrix(int(W.shape[1])), dtype=X.dtype, device=X.device
    )
    z = torch.einsum("nd,mhd->nmh", X, W) - t.unsqueeze(0)
    if beta.ndim == 0:
        g = torch.sigmoid(beta * z)
    else:
        g = torch.sigmoid(beta.reshape(1, -1, 1) * z)
    gexp = g.unsqueeze(2)
    bexp = codes.view(1, 1, codes.shape[0], codes.shape[1])
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)
    weights = factors.prod(dim=-1)
    return torch.einsum("nml,mlk->nmk", weights, cell_logits)


def _copy_cell_logits(param: Tensor, values: Tensor | np.ndarray) -> None:
    src = torch.as_tensor(values, dtype=param.dtype, device=param.device)
    if src.ndim == 1:
        src = src.unsqueeze(-1)
    param.copy_(src)


class ArrangementClassifier(nn.Module):
    """``H`` hyperplanes, soft cell membership, per-cell logits.

    ``forward`` is a tensor-in / tensor-out layer: ``X`` of shape ``(..., d)``
    yields logits ``(..., k)``. Plug in after any encoder that emits a feature
    vector; constructors stay float64 CPU (certify / G3), so call
    ``.to(device=z.device, dtype=z.dtype)`` (or :func:`omnibias.tab.torch.plugin.as_head`)
    before composing with a host net. Numpy ``predict`` / ``predict_proba`` squeeze
    ``k == 1`` so G3 benches stay 1-D.
    """

    def __init__(
        self,
        n_features: int,
        n_hyperplanes: int = 2,
        *,
        beta: float = 1.0,
        n_outputs: int = 1,
        task: str = "binary",
        learnable_beta: bool = False,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if n_hyperplanes < 1:
            raise ValueError("n_hyperplanes must be >= 1")
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        if task == "binary" and int(n_outputs) != 1:
            raise ValueError("binary task requires n_outputs == 1")
        if task == "multiclass" and int(n_outputs) < 2:
            raise ValueError("multiclass task requires n_outputs >= 2")
        if int(n_outputs) < 1:
            raise ValueError("n_outputs must be >= 1")
        self.n_features = int(n_features)
        self.n_hyperplanes = int(n_hyperplanes)
        self.n_cells = 1 << self.n_hyperplanes
        self.n_outputs = int(n_outputs)
        self.task = str(task)
        scale = 1.0 / (self.n_features**0.5)
        self.W = nn.Parameter(
            torch.randn(self.n_hyperplanes, self.n_features, dtype=_DTYPE) * scale
        )
        self.t = nn.Parameter(torch.zeros(self.n_hyperplanes, dtype=_DTYPE))
        self.cell_logits = nn.Parameter(
            torch.zeros(self.n_cells, self.n_outputs, dtype=_DTYPE)
        )
        beta0 = torch.tensor(float(beta), dtype=_DTYPE)
        if learnable_beta:
            self._beta = nn.Parameter(beta0)
        else:
            self.register_buffer("_beta", beta0, persistent=True)

    @property
    def beta(self) -> float:
        return float(self._beta.detach().cpu().item())

    @beta.setter
    def beta(self, value: float) -> None:
        with torch.no_grad():
            self._beta.fill_(float(value))

    def forward(self, X: Tensor) -> Tensor:
        rows, leading = _feature_rows(X)
        weights = partition_weights_arrays(
            self.W, self.t, rows, self._beta, self.n_hyperplanes
        )
        cell = _cell_2d(self.cell_logits)
        logits = cell.unsqueeze(0).expand(rows.shape[0], -1, -1)
        out = combine(weights, logits)
        return out.reshape(*leading, out.shape[-1])

    def _to_tensor(self, X: Tensor | np.ndarray) -> Tensor:
        if isinstance(X, Tensor):
            return X
        return torch.as_tensor(
            np.asarray(X, dtype=np.float64), dtype=self.W.dtype, device=self.W.device
        )

    @torch.no_grad()
    def predict(self, X: Tensor | np.ndarray) -> np.ndarray:
        logits = _squeeze_binary_np(
            self.forward(self._to_tensor(X)).detach().cpu().numpy()
        )
        if self.task == "multiclass":
            return np.argmax(logits, axis=-1).astype(np.float64)
        if self.task == "regression":
            return logits
        return (logits > 0.0).astype(np.float64)

    @torch.no_grad()
    def predict_proba(self, X: Tensor | np.ndarray) -> np.ndarray:
        logits = self.forward(self._to_tensor(X)).detach().cpu().numpy()
        if self.task == "multiclass":
            z = logits - logits.max(axis=-1, keepdims=True)
            e = np.exp(np.clip(z, -40.0, 40.0))
            return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)
        squeezed = _squeeze_binary_np(logits)
        return 1.0 / (1.0 + np.exp(-squeezed))

    def numpy_state(self) -> dict[str, np.ndarray]:
        cell = self.cell_logits.detach().cpu().numpy().copy()
        if self.n_outputs == 1:
            cell = cell.reshape(-1)
        return {
            "W": self.W.detach().cpu().numpy().copy(),
            "t": self.t.detach().cpu().numpy().copy(),
            "cell_logits": cell,
        }


_OPTIMIZERS = ("adam", "trust_region", "cubic")


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
    steps_run: int = 0
    best_step: int = 0
    stopped_early: bool = False
    at_step_cap: bool = False
    train_bce: float = float("nan")
    val_bce: float = float("nan")
    train_val_gap: float = float("nan")
    optimizer: str = "adam"


class ArrangementBoosted(nn.Module):
    """Additive ensemble of :class:`ArrangementClassifier` weak learners.

    ``forward`` is ``base + lr * sum_m member_m(X)`` with a batched matmul/sum
    (autograd through every member). Tabular ``fit_arrangement_boosted`` still
    pretrains stagewise; after that the ensemble is a normal ``nn.Module`` head.
    """

    def __init__(
        self,
        members: list[ArrangementClassifier] | None = None,
        *,
        learning_rate: float = 0.3,
        base: float = 0.0,
    ) -> None:
        super().__init__()
        packed = list(members or [])
        self.members = nn.ModuleList(packed)
        n_out = int(packed[0].n_outputs) if packed else 1
        self.n_outputs = n_out
        self.register_buffer(
            "_learning_rate",
            torch.tensor(float(learning_rate), dtype=_DTYPE),
            persistent=True,
        )
        self.register_buffer(
            "_base", torch.tensor(float(base), dtype=_DTYPE), persistent=True
        )

    @property
    def learning_rate(self) -> float:
        return float(self._learning_rate.detach().cpu().item())

    @property
    def base(self) -> float:
        return float(self._base.detach().cpu().item())

    def forward(self, X: Tensor) -> Tensor:
        rows, leading = _feature_rows(X)
        k = int(self.n_outputs)
        base = self._base.to(dtype=X.dtype, device=X.device)
        if len(self.members) == 0:
            return (rows.new_zeros(rows.shape[0], k) + base).reshape(*leading, k)
        W = torch.stack([m.W for m in self.members], dim=0)
        t = torch.stack([m.t for m in self.members], dim=0)
        cell = torch.stack([_cell_2d(m.cell_logits) for m in self.members], dim=0)
        betas = torch.stack(
            [m._beta.reshape(()) for m in self.members], dim=0
        )
        contrib = _batched_arrangement_logits(W, t, cell, rows, betas)
        lr = self._learning_rate.to(dtype=X.dtype, device=X.device)
        out = base + lr * contrib.sum(dim=1)
        return out.reshape(*leading, out.shape[-1])

    def _to_tensor(self, X: Tensor | np.ndarray) -> Tensor:
        if isinstance(X, Tensor):
            return X
        dtype = self._base.dtype
        device = self._base.device
        if len(self.members) > 0:
            first = self.members[0]
            assert isinstance(first, ArrangementClassifier)
            dtype = first.W.dtype
            device = first.W.device
        return torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=dtype, device=device)

    @torch.no_grad()
    def logits(self, X: np.ndarray | Tensor) -> np.ndarray:
        return _squeeze_binary_np(
            self.forward(self._to_tensor(X)).detach().cpu().numpy()
        )

    def predict_proba(self, X: np.ndarray | Tensor) -> np.ndarray:
        z = np.clip(self.logits(X), -40.0, 40.0)
        if z.ndim > 1 and z.shape[-1] > 1:
            e = np.exp(z - z.max(axis=-1, keepdims=True))
            return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X: np.ndarray | Tensor) -> np.ndarray:
        z = self.logits(X)
        if z.ndim > 1 and z.shape[-1] > 1:
            return np.argmax(z, axis=-1).astype(np.float64)
        return (z > 0.0).astype(np.float64)


@dataclass
class BoostedFitResult:
    model: ArrangementBoosted
    train_acc: float
    val_acc: float
    train_bce: float
    val_bce: float
    train_val_gap: float
    n_stages: int
    best_stage: int
    stopped_early: bool
    at_stage_cap: bool
    learning_rate: float
    history: list[float]


@dataclass
class _TrainStats:
    steps_run: int
    best_step: int
    stopped_early: bool
    at_step_cap: bool
    best_val_bce: float


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred.reshape(-1) == np.asarray(y, dtype=np.float64).reshape(-1)).mean())


_ENCODER_STAGEWISE_MSG = (
    "encoder= is not supported on the stagewise GBM-mirror trainers; "
    "use fit_joint or fit_second_order"
)


def _snapshot(model: ArrangementClassifier, encoder: nn.Module | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "W": model.W.detach().clone(),
        "t": model.t.detach().clone(),
        "cell_logits": model.cell_logits.detach().clone(),
        "beta": float(model.beta),
    }
    if encoder is not None:
        snap["encoder"] = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    return snap


def _restore(
    model: ArrangementClassifier, snap: dict[str, Any], encoder: nn.Module | None = None
) -> None:
    with torch.no_grad():
        model.W.copy_(snap["W"])
        model.t.copy_(snap["t"])
        model.cell_logits.copy_(snap["cell_logits"])
    model.beta = float(snap["beta"])
    if encoder is not None and "encoder" in snap:
        encoder.load_state_dict(snap["encoder"])


def _task_loss_t(
    model: ArrangementClassifier,
    X: Tensor,
    y: Tensor,
    *,
    graph: nn.Module | None = None,
) -> Tensor:
    F = graph(X) if graph is not None else model(X)
    task = model.task
    if task == "binary":
        if F.shape[-1] == 1:
            F = F.reshape(-1)
        return nn.functional.binary_cross_entropy_with_logits(F, y)
    if task == "multiclass":
        return nn.functional.cross_entropy(F, y.long())
    return nn.functional.mse_loss(F, y.reshape(F.shape))


@torch.no_grad()
def _bce_logits(
    model: ArrangementClassifier,
    X: Tensor,
    y: Tensor,
    *,
    graph: nn.Module | None = None,
) -> float:
    return float(_task_loss_t(model, X, y, graph=graph).item())


def _encoder_out_dim(encoder: nn.Module, n_in: int) -> int:
    p = next(encoder.parameters())
    dummy = torch.zeros(1, n_in, dtype=p.dtype, device=p.device)
    with torch.no_grad():
        z = encoder(dummy)
    if z.ndim < 2:
        raise ValueError("encoder must map (n, d) -> (n, h)")
    return int(z.shape[-1])


def _predict_np(
    model: ArrangementClassifier,
    X: np.ndarray,
    *,
    graph: nn.Module | None = None,
) -> np.ndarray:
    if graph is None:
        return model.predict(X)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=model.W.dtype, device=model.W.device)
    with torch.no_grad():
        logits = graph(Xt).detach().cpu().numpy()
    squeezed = _squeeze_binary_np(logits)
    if model.task == "multiclass":
        return np.argmax(squeezed, axis=-1).astype(np.float64)
    if model.task == "regression":
        return squeezed
    return (squeezed > 0.0).astype(np.float64)


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
    """Feature-pair + threshold grid warm-starts for axis-structured data.

    Vectorized over threshold pairs for each feature pair and sign pattern.
    """
    if n_hyperplanes != 2:
        return []
    n, d = X.shape
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    qs = np.linspace(0.05, 0.95, int(n_quantiles))
    feat_thrs = [np.unique(np.quantile(X[:, j], qs)) for j in range(d)]
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for f0, f1 in itertools.combinations(range(d), 2):
        q0 = feat_thrs[f0]
        q1 = feat_thrs[f1]
        x0 = X[:, f0]
        x1 = X[:, f1]
        best_local = -1.0
        best_pack: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for s0, s1 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            bits0 = (s0 * x0[:, None] > s0 * q0[None, :]).astype(np.int8)
            bits1 = (s1 * x1[:, None] > s1 * q1[None, :]).astype(np.int8)
            t0, t1 = q0.size, q1.size
            n_pairs = t0 * t1
            idx = (
                bits0[:, :, None] + 2 * bits1[:, None, :]
            ).reshape(n, n_pairs)
            counts = np.zeros((4, n_pairs), dtype=np.float64)
            pos = np.zeros((4, n_pairs), dtype=np.float64)
            for c in range(4):
                mask = idx == c
                counts[c] = mask.sum(axis=0)
                pos[c] = (mask * yv[:, None]).sum(axis=0)
            rate = np.where(counts > 0.0, pos / np.maximum(counts, 1.0), 0.5)
            rate = np.clip(rate, 1e-3, 1.0 - 1e-3)
            logits_all = np.log(rate / (1.0 - rate))  # (4, n_pairs)
            pred_cell = (rate > 0.5).astype(np.float64)
            pair_ids = np.arange(n_pairs)[None, :]
            pred = pred_cell[idx, pair_ids]
            accs = (pred == yv[:, None]).mean(axis=0)
            p_best = int(np.argmax(accs))
            acc = float(accs[p_best])
            if acc > best_local:
                best_local = acc
                i0, i1 = divmod(p_best, t1)
                W = np.zeros((2, d), dtype=np.float64)
                t = np.zeros(2, dtype=np.float64)
                W[0, f0] = s0
                t[0] = s0 * float(q0[i0])
                W[1, f1] = s1
                t[1] = s1 * float(q1[i1])
                best_pack = (W, t, logits_all[:, p_best].copy())
        if best_pack is not None:
            scored.append((best_local, *best_pack))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [
        (W, t, logits, float(acc))
        for acc, W, t, logits in scored[: int(top_k)]
    ]


def _make_optimizer(
    model: ArrangementClassifier,
    *,
    optimizer: str,
    lr: float,
    freeze_W: bool,
    extra_params: list[Tensor] | None = None,
) -> Any:
    extra = list(extra_params) if extra_params else []
    if freeze_W:
        model.W.requires_grad_(False)
        train_params = [model.t, model.cell_logits]
    else:
        model.W.requires_grad_(True)
        train_params = list(model.parameters())
    name = str(optimizer)
    if name == "adam":
        if freeze_W:
            return torch.optim.Adam([{"params": extra + train_params, "lr": lr}])
        groups = [
            {"params": [model.W], "lr": lr * 0.5},
            {"params": [model.t, model.cell_logits], "lr": lr},
        ]
        if extra:
            groups.append({"params": extra, "lr": lr})
        return torch.optim.Adam(groups)
    from omnibias.torch.optim import CubicNewton, TrustRegionNewtonCG

    if name == "trust_region":
        return TrustRegionNewtonCG(extra + train_params)
    if name == "cubic":
        return CubicNewton(extra + train_params)
    raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {optimizer!r}")


def _train_restart(
    model: ArrangementClassifier,
    Xtr: Tensor,
    ytr: Tensor,
    Xva: Tensor,
    yva: Tensor,
    *,
    steps: int,
    beta_init: float,
    beta_final: float,
    beta_anneal_steps: int,
    l1: float,
    lr: float,
    patience: int,
    eval_every: int,
    min_delta: float,
    freeze_W: bool = False,
    optimizer: str = "adam",
    graph: nn.Module | None = None,
    encoder: nn.Module | None = None,
) -> _TrainStats:
    """Train with beta anneal, val-BCE early stop, and best-checkpoint restore."""
    extra = list(encoder.parameters()) if encoder is not None else None
    opt = _make_optimizer(
        model, optimizer=optimizer, lr=lr, freeze_W=freeze_W, extra_params=extra
    )
    max_steps = max(1, int(steps))
    anneal = max(1, int(beta_anneal_steps))
    every = max(1, int(eval_every))
    pat = max(1, int(patience))
    delta = float(min_delta)
    use_adam = str(optimizer) == "adam"

    best_snap = _snapshot(model, encoder)
    best_val = _bce_logits(model, Xva, yva, graph=graph)
    best_step = 0
    stall = 0
    stopped_early = False
    steps_run = 0

    def _loss() -> Tensor:
        loss = _task_loss_t(model, Xtr, ytr, graph=graph)
        if l1 > 0.0:
            loss = loss + float(l1) * model.W.abs().sum()
        return loss

    for step in range(max_steps):
        frac = min(1.0, step / max(anneal - 1, 1))
        model.beta = float(beta_init * (beta_final / beta_init) ** frac)
        if use_adam:
            opt.zero_grad(set_to_none=True)
            loss = _loss()
            loss.backward()
            opt.step()
        else:
            opt.step(_loss)  # type: ignore[arg-type]
        steps_run = step + 1

        if steps_run % every != 0 and steps_run != max_steps:
            continue
        val_bce = _bce_logits(model, Xva, yva, graph=graph)
        if val_bce < best_val - delta:
            best_val = val_bce
            best_snap = _snapshot(model, encoder)
            best_step = steps_run
            stall = 0
        else:
            stall += every
            if stall >= pat:
                stopped_early = True
                break

    _restore(model, best_snap, encoder)
    model.W.requires_grad_(True)
    return _TrainStats(
        steps_run=steps_run,
        best_step=best_step,
        stopped_early=stopped_early,
        at_step_cap=not stopped_early,
        best_val_bce=best_val,
    )


def _merge_stats(a: _TrainStats, b: _TrainStats) -> _TrainStats:
    """Combine two sequential training phases (sparse freeze + unfreeze)."""
    return _TrainStats(
        steps_run=a.steps_run + b.steps_run,
        best_step=a.steps_run + b.best_step if b.best_val_bce <= a.best_val_bce else a.best_step,
        stopped_early=a.stopped_early or b.stopped_early,
        at_step_cap=a.at_step_cap and b.at_step_cap,
        best_val_bce=min(a.best_val_bce, b.best_val_bce),
    )


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
    beta_anneal_steps: int | None = None,
    lr: float = 0.05,
    seed: int = 0,
    val_fraction: float = 0.2,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    patience: int = 50,
    eval_every: int = 5,
    min_delta: float = 1e-4,
    sparse_warmstart: bool | None = None,
    optimizer: str = "adam",
    n_outputs: int = 1,
    task: str = "binary",
    encoder: nn.Module | None = None,
) -> FitResult:
    """Fit an :class:`ArrangementClassifier` with beta anneal and restarts.

    When ``sparse_warmstart`` is true (default if ``l1 > 0``), also try
    feature-pair threshold warm-starts -- the path that recovers axis-aligned
    rules for 05-02 G2.

    ``steps`` is a maximum step cap. Training early-stops on validation
    BCE (LightGBM ``binary_logloss`` analog) with patience/restore of the best
    checkpoint. ``optimizer`` is ``adam`` (default), ``trust_region``, or
    ``cubic``. Pass ``X_val`` / ``y_val`` for an outer held-out split; otherwise
    an inner ``val_fraction`` split of ``X`` / ``y`` is used.

    Optional ``encoder`` jointly trains ``head(encoder(X))`` (``X`` is encoder
    input; head width is the encoder output). Sparse warm-start is skipped.
    Default ``None`` is the tabular G3 path.
    """
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {optimizer!r}")
    task_s = str(task)
    k_out = int(n_outputs)
    Xv = np.asarray(X, dtype=np.float64)
    if task_s == "multiclass":
        yv = np.asarray(y).reshape(-1)
    elif task_s == "regression":
        yv = np.asarray(y, dtype=np.float64)
        if yv.ndim == 1:
            yv = yv.reshape(-1, 1)
    else:
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xv.ndim != 2 or Xv.shape[0] != yv.shape[0]:
        raise ValueError("X/y shape mismatch")
    n, d = Xv.shape
    y_dtype = torch.long if task_s == "multiclass" else _DTYPE
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided together")
        Xtr_np = Xv
        ytr_np = yv
        Xva_np = np.asarray(X_val, dtype=np.float64)
        if task_s == "multiclass":
            yva_np = np.asarray(y_val).reshape(-1)
        elif task_s == "regression":
            yva_np = np.asarray(y_val, dtype=np.float64)
            if yva_np.ndim == 1:
                yva_np = yva_np.reshape(-1, 1)
        else:
            yva_np = np.asarray(y_val, dtype=np.float64).reshape(-1)
        if Xva_np.ndim != 2 or Xva_np.shape[0] != yva_np.shape[0]:
            raise ValueError("X_val/y_val shape mismatch")
        if Xva_np.shape[1] != d:
            raise ValueError("X_val feature dim does not match X")
        if Xva_np.shape[0] < 1 or Xtr_np.shape[0] < 1:
            raise ValueError("not enough samples for train/val")
    else:
        n_val = max(1, int(round(n * float(val_fraction))))
        n_tr = n - n_val
        if n_tr < 1:
            raise ValueError("not enough samples for a train/val split")
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
        Xtr_np, ytr_np = Xv[tr_idx], yv[tr_idx]
        Xva_np, yva_np = Xv[va_idx], yv[va_idx]

    Xtr = torch.as_tensor(Xtr_np, dtype=_DTYPE)
    ytr = torch.as_tensor(ytr_np, dtype=y_dtype)
    Xva = torch.as_tensor(Xva_np, dtype=_DTYPE)
    yva = torch.as_tensor(yva_np, dtype=y_dtype)
    head_d = d
    enc_init: dict[str, Tensor] | None = None
    if encoder is not None:
        from omnibias.tab.torch.train import _EncoderHead

        encoder.to(dtype=_DTYPE)
        head_d = _encoder_out_dim(encoder, d)
        enc_init = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    use_sparse = (
        encoder is None
        and (bool(l1 > 0.0) if sparse_warmstart is None else bool(sparse_warmstart))
        and task_s == "binary"
    )
    anneal_default = min(int(steps), 200)
    anneal_steps = int(anneal_default if beta_anneal_steps is None else beta_anneal_steps)

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
    best_encoder_state: dict[str, Tensor] | None = None
    for i, (kind, cfg) in enumerate(candidates):
        torch.manual_seed(int(seed) * 1009 + i)
        if enc_init is not None:
            encoder.load_state_dict(enc_init)
        model = ArrangementClassifier(
            head_d, n_hyperplanes, beta=float(beta_init), n_outputs=k_out, task=task_s
        )
        graph = _EncoderHead(encoder, model) if encoder is not None else None
        train_kw: dict[str, Any] = {"graph": graph, "encoder": encoder}
        if kind == "sparse":
            with torch.no_grad():
                model.W.copy_(torch.as_tensor(cfg["W"], dtype=_DTYPE))
                model.t.copy_(torch.as_tensor(cfg["t"], dtype=_DTYPE))
                _copy_cell_logits(model.cell_logits, cfg["logits"])
            local_steps = max(200, int(steps) // 2)
            local_l1 = float(l1)
            local_beta_init = max(float(beta_init), 8.0)
            stats = _train_restart(
                model,
                Xtr,
                ytr,
                Xva,
                yva,
                steps=local_steps,
                beta_init=local_beta_init,
                beta_final=float(beta_final),
                beta_anneal_steps=min(anneal_steps, local_steps),
                l1=local_l1,
                lr=float(lr),
                patience=int(patience),
                eval_every=int(eval_every),
                min_delta=float(min_delta),
                freeze_W=True,
                optimizer=optimizer,
                **train_kw,
            )
            polish = _train_restart(
                model,
                Xtr,
                ytr,
                Xva,
                yva,
                steps=max(100, int(steps) // 4),
                beta_init=float(beta_final),
                beta_final=float(beta_final),
                beta_anneal_steps=1,
                l1=float(l1),
                lr=float(lr) * 0.5,
                patience=int(patience),
                eval_every=int(eval_every),
                min_delta=float(min_delta),
                freeze_W=False,
                optimizer=optimizer,
                **train_kw,
            )
            stats = _merge_stats(stats, polish)
        else:
            stats = _train_restart(
                model,
                Xtr,
                ytr,
                Xva,
                yva,
                steps=int(steps),
                beta_init=float(beta_init),
                beta_final=float(beta_final),
                beta_anneal_steps=anneal_steps,
                l1=float(l1),
                lr=float(lr),
                patience=int(patience),
                eval_every=int(eval_every),
                min_delta=float(min_delta),
                freeze_W=False,
                optimizer=optimizer,
                **train_kw,
            )

        tr_acc = _accuracy(_predict_np(model, Xtr_np, graph=graph), ytr_np)
        va_acc = _accuracy(_predict_np(model, Xva_np, graph=graph), yva_np)
        train_bce = _bce_logits(model, Xtr, ytr, graph=graph)
        val_bce = _bce_logits(model, Xva, yva, graph=graph)
        result = FitResult(
            model=model,
            train_acc=tr_acc,
            val_acc=va_acc,
            beta_final=float(model.beta),
            l1=float(l1),
            n_restarts=len(candidates),
            best_restart=i,
            sparse_warmstart=use_sparse,
            steps_run=stats.steps_run,
            best_step=stats.best_step,
            stopped_early=stats.stopped_early,
            at_step_cap=stats.at_step_cap,
            train_bce=train_bce,
            val_bce=val_bce,
            train_val_gap=float(tr_acc - va_acc),
            optimizer=str(optimizer),
        )
        if best is None:
            best = result
            if encoder is not None:
                best_encoder_state = {
                    k: v.detach().clone() for k, v in encoder.state_dict().items()
                }
        elif val_bce < best.val_bce - 1e-12:
            best = result
            if encoder is not None:
                best_encoder_state = {
                    k: v.detach().clone() for k, v in encoder.state_dict().items()
                }
        elif abs(val_bce - best.val_bce) <= 1e-12 and va_acc > best.val_acc:
            best = result
            if encoder is not None:
                best_encoder_state = {
                    k: v.detach().clone() for k, v in encoder.state_dict().items()
                }
    assert best is not None
    if encoder is not None and best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
    return best


def _bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    z = np.clip(np.asarray(logits, dtype=np.float64).reshape(-1), -40.0, 40.0)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p)))


def _train_weak_mse(
    model: ArrangementClassifier,
    Xtr: Tensor,
    rtr: Tensor,
    wtr: Tensor,
    Xva: Tensor,
    rva: Tensor,
    wva: Tensor,
    *,
    steps: int,
    lr: float,
    patience: int,
    eval_every: int,
    min_delta: float,
    beta_init: float,
    beta_final: float,
) -> None:
    """Weighted-MSE fit of an arrangement to a Newton residual target."""
    opt = torch.optim.Adam(
        [
            {"params": [model.W], "lr": lr * 0.5},
            {"params": [model.t, model.cell_logits], "lr": lr},
        ]
    )
    best_snap = _snapshot(model)
    with torch.no_grad():
        pred_va = model(Xva)
        if pred_va.shape[-1] == 1:
            pred_va = pred_va.reshape(-1)
        best_val = float((wva * (pred_va - rva) ** 2).mean().item())
    stall = 0
    every = max(1, int(eval_every))
    pat = max(1, int(patience))
    max_steps = max(1, int(steps))
    anneal = max(1, min(max_steps, 40))
    for step in range(max_steps):
        frac = min(1.0, step / max(anneal - 1, 1))
        model.beta = float(beta_init * (beta_final / beta_init) ** frac)
        opt.zero_grad(set_to_none=True)
        pred = model(Xtr)
        if pred.shape[-1] == 1:
            pred = pred.reshape(-1)
        loss = (wtr * (pred - rtr) ** 2).mean()
        loss.backward()
        opt.step()
        steps_run = step + 1
        if steps_run % every != 0 and steps_run != max_steps:
            continue
        with torch.no_grad():
            pred_va = model(Xva)
            if pred_va.shape[-1] == 1:
                pred_va = pred_va.reshape(-1)
            val_mse = float((wva * (pred_va - rva) ** 2).mean().item())
        if val_mse < best_val - float(min_delta):
            best_val = val_mse
            best_snap = _snapshot(model)
            stall = 0
        else:
            stall += every
            if stall >= pat:
                break
    _restore(model, best_snap)
    model.beta = float(beta_final)


def fit_arrangement_boosted(
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_hyperplanes: int = 2,
    n_stages_max: int = 40,
    learning_rate: float = 0.3,
    stage_patience: int = 5,
    weak_restarts: int = 2,
    weak_steps: int = 200,
    weak_patience: int = 20,
    beta_final: float = 64.0,
    seed: int = 0,
    min_delta: float = 1e-4,
    n_outputs: int = 1,
    task: str = "binary",
    encoder: nn.Module | None = None,
) -> BoostedFitResult:
    """Newton-boost an additive ensemble of H-hyperplane arrangements.

    Stage ``m`` fits a fresh :class:`ArrangementClassifier` to the closed-form
    Newton residual ``r = -g / h`` (binary logistic) under Hessian weights, then
    adds it with shrinkage. Stages stop on validation BCE (LightGBM analog of
    ``n_estimators`` early stopping).

    ``encoder`` is rejected: stagewise numpy boosting does not jointly train an
    encoder. Use :func:`~omnibias.tab.torch.train.fit_joint` or
    :func:`~omnibias.tab.torch.train.fit_second_order`.
    """
    if encoder is not None:
        raise TypeError(_ENCODER_STAGEWISE_MSG)
    from omnibias.tab._core.loss import loss_value, score_grad_hess

    task_s = str(task)
    k_out = int(n_outputs)
    Xtr_np = np.asarray(X, dtype=np.float64)
    Xva_np = np.asarray(X_val, dtype=np.float64)
    if task_s == "multiclass":
        ytr_np = np.asarray(y).reshape(-1)
        yva_np = np.asarray(y_val).reshape(-1)
    elif task_s == "regression":
        ytr_np = np.asarray(y, dtype=np.float64)
        yva_np = np.asarray(y_val, dtype=np.float64)
        if ytr_np.ndim == 1:
            ytr_np = ytr_np.reshape(-1, 1)
        if yva_np.ndim == 1:
            yva_np = yva_np.reshape(-1, 1)
    else:
        ytr_np = np.asarray(y, dtype=np.float64).reshape(-1)
        yva_np = np.asarray(y_val, dtype=np.float64).reshape(-1)
    if Xtr_np.ndim != 2 or Xtr_np.shape[0] != ytr_np.shape[0]:
        raise ValueError("X/y shape mismatch")
    if Xva_np.ndim != 2 or Xva_np.shape[0] != yva_np.shape[0]:
        raise ValueError("X_val/y_val shape mismatch")
    n, d = Xtr_np.shape
    if task_s == "binary":
        p = float(np.clip(ytr_np.mean(), 1e-6, 1.0 - 1e-6))
        base = float(np.log(p / (1.0 - p)))
        Ftr = np.full(n, base, dtype=np.float64)
        Fva = np.full(Xva_np.shape[0], base, dtype=np.float64)
    else:
        base = 0.0
        Ftr = np.zeros((n, k_out), dtype=np.float64)
        Fva = np.zeros((Xva_np.shape[0], k_out), dtype=np.float64)
    Xtr = torch.as_tensor(Xtr_np, dtype=_DTYPE)
    Xva = torch.as_tensor(Xva_np, dtype=_DTYPE)
    members: list[ArrangementClassifier] = []
    history: list[float] = []
    best_val = _bce_from_logits(Fva, yva_np)
    best_members: list[ArrangementClassifier] = []
    best_stage = 0
    stall = 0
    stopped_early = False
    n_max = max(1, int(n_stages_max))
    lr_s = float(learning_rate)

    for stage in range(n_max):
        Ftr_score = Ftr[:, None] if Ftr.ndim == 1 else Ftr
        Fva_score = Fva[:, None] if Fva.ndim == 1 else Fva
        g, h = score_grad_hess(Ftr_score, ytr_np, task_s)
        rtr = -g / np.clip(h, 1e-12, None)
        wtr = h
        gv, hv = score_grad_hess(Fva_score, yva_np, task_s)
        rva = -gv / np.clip(hv, 1e-12, None)
        wva = hv
        if k_out == 1:
            rtr = rtr.reshape(-1)
            wtr = wtr.reshape(-1)
            rva = rva.reshape(-1)
            wva = wva.reshape(-1)
        rtr_t = torch.as_tensor(rtr, dtype=_DTYPE)
        wtr_t = torch.as_tensor(wtr, dtype=_DTYPE)
        rva_t = torch.as_tensor(rva, dtype=_DTYPE)
        wva_t = torch.as_tensor(wva, dtype=_DTYPE)

        best_weak: ArrangementClassifier | None = None
        best_weak_mse = float("inf")
        for r in range(max(1, int(weak_restarts))):
            torch.manual_seed(int(seed) * 1009 + stage * 17 + r)
            weak = ArrangementClassifier(
                d, n_hyperplanes, beta=1.0, n_outputs=k_out, task=task_s
            )
            _train_weak_mse(
                weak,
                Xtr,
                rtr_t,
                wtr_t,
                Xva,
                rva_t,
                wva_t,
                steps=int(weak_steps),
                lr=0.05,
                patience=int(weak_patience),
                eval_every=5,
                min_delta=float(min_delta),
                beta_init=1.0,
                beta_final=float(beta_final),
            )
            with torch.no_grad():
                pred_va = weak(Xva)
                if pred_va.shape[-1] == 1:
                    pred_va = pred_va.reshape(-1)
                mse = float((wva_t * (pred_va - rva_t) ** 2).mean().item())
            if mse < best_weak_mse:
                best_weak_mse = mse
                best_weak = weak
        assert best_weak is not None
        with torch.no_grad():
            dtr = best_weak(Xtr).detach().cpu().numpy()
            dva = best_weak(Xva).detach().cpu().numpy()
            if k_out == 1:
                Ftr = Ftr + lr_s * dtr.reshape(-1)
                Fva = Fva + lr_s * dva.reshape(-1)
            else:
                Ftr = Ftr + lr_s * dtr
                Fva = Fva + lr_s * dva
        members.append(best_weak)
        if task_s == "binary":
            val_bce = _bce_from_logits(Fva, yva_np)
        else:
            val_bce = float(loss_value(Fva, yva_np, task_s))
        history.append(val_bce)
        if val_bce < best_val - float(min_delta):
            best_val = val_bce
            best_members = list(members)
            best_stage = stage + 1
            stall = 0
        else:
            stall += 1
            if stall >= max(1, int(stage_patience)):
                stopped_early = True
                break

    kept = best_members if best_members else members
    boosted = ArrangementBoosted(
        members=kept, learning_rate=lr_s, base=base
    )
    tr_logits = boosted.logits(Xtr_np)
    va_logits = boosted.logits(Xva_np)
    train_acc = _accuracy(boosted.predict(Xtr_np), ytr_np)
    val_acc = _accuracy(boosted.predict(Xva_np), yva_np)
    if task_s == "binary":
        train_bce = _bce_from_logits(tr_logits, ytr_np)
        val_bce = _bce_from_logits(va_logits, yva_np)
    else:
        train_bce = float(loss_value(np.asarray(tr_logits), ytr_np, task_s))
        val_bce = float(loss_value(np.asarray(va_logits), yva_np, task_s))
    return BoostedFitResult(
        model=boosted,
        train_acc=train_acc,
        val_acc=val_acc,
        train_bce=train_bce,
        val_bce=val_bce,
        train_val_gap=float(train_acc - val_acc),
        n_stages=len(members),
        best_stage=int(best_stage),
        stopped_early=stopped_early,
        at_stage_cap=not stopped_early,
        learning_rate=lr_s,
        history=history,
    )


__all__ = [
    "ArrangementBoosted",
    "ArrangementClassifier",
    "BoostedFitResult",
    "FitResult",
    "fit_arrangement",
    "fit_arrangement_boosted",
]

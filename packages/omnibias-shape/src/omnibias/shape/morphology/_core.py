# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable mathematical morphology (theory 03-05).

Dilation is max-plus convolution; ``logsumexp_beta`` is the homotopy.
``beta -> inf`` is **temperature collapse** (feasibility), a soft max
hardening to a hard max. Pack structuring elements come from the
founding bias collapse (``delta -> 0``). Do not conflate the two.
The gap is worst-case ``compositions * log(N) / beta`` and is tight
only when window values coincide. Soft dilation is a conservative
upper bound. Not a seventh ``OperatorBlock`` role.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Direction = Literal["upper", "lower", "two_sided"]


def _logsumexp_beta(terms: FloatArray, beta: float) -> FloatArray:
    """Shifted ``(1/beta) log sum exp(beta a)``. Matches the struct primitive."""
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    a = np.asarray(terms, dtype=np.float64)
    finite = np.isfinite(a)
    safe = np.where(finite, a, 0.0)
    peak = np.max(np.where(finite, a, -np.inf), axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    z = np.exp(float(beta) * (safe - peak))
    z = np.where(finite, z, 0.0)
    total = np.sum(z, axis=-1)
    out = peak.reshape(total.shape) + np.log(np.maximum(total, 1e-300)) / float(beta)
    empty = ~np.any(finite, axis=-1)
    return np.where(empty, -np.inf, out)


def morphology_gap_bound(*, size: int, beta: float, compositions: int = 1) -> float:
    """``compositions * log(size) / beta``. Reuses ``logsumexp_gap_bound``."""
    if int(size) < 1:
        raise ValueError("size must be >= 1")
    if int(compositions) < 1:
        raise ValueError("compositions must be >= 1")
    try:
        from omnibias.struct import logsumexp_gap_bound

        unit = float(logsumexp_gap_bound(int(size), float(beta)))
    except ImportError:
        if float(beta) <= 0.0:
            raise ValueError("beta must be > 0") from None
        unit = math.log(int(size)) / float(beta)
    return float(int(compositions) * unit)


@dataclass(frozen=True)
class StructuringElement:
    offsets: FloatArray
    values: FloatArray
    scores: FloatArray | None = None

    def __post_init__(self) -> None:
        off = np.asarray(self.offsets, dtype=np.int64).reshape(-1)
        val = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if off.size != val.size:
            raise ValueError("offsets and values must have the same length")
        scores = None if self.scores is None else np.asarray(self.scores, dtype=np.float64).reshape(-1)
        if scores is not None and scores.size != off.size:
            raise ValueError("scores must match offsets")
        object.__setattr__(self, "offsets", off)
        object.__setattr__(self, "values", val)
        object.__setattr__(self, "scores", scores)

    @property
    def size(self) -> int:
        return int(self.offsets.size)

    @classmethod
    def flat(cls, radius: int) -> StructuringElement:
        if int(radius) < 0:
            raise ValueError("radius must be >= 0")
        off = np.arange(-int(radius), int(radius) + 1, dtype=np.int64)
        return cls(off, np.zeros(off.size, dtype=np.float64))

    @classmethod
    def pack_bump(cls, radius: int, *, height: float, alpha: float) -> StructuringElement:
        """``b(y) = c sigma'(alpha y)`` from the founding bias collapse."""
        off = np.arange(-int(radius), int(radius) + 1, dtype=np.float64)
        z = float(alpha) * off
        sig = np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        return cls(off.astype(np.int64), float(height) * sig * (1.0 - sig))


@dataclass(frozen=True)
class MorphResult:
    value: FloatArray
    gap_bound: float
    direction: Direction
    compositions: int
    observed_dev: float


def honesty_payload() -> dict[str, bool]:
    return {
        "seventh_operator_role": False,
        "gap_is_worst_case": True,
        "temperature_collapse": True,
        "founding_bias_collapse_builds_pack_se": True,
    }


def soft_support(scores: object, *, k: int, temperature: float) -> FloatArray:
    """Delegate to ``omnibias.graph.soft_top_k``. Temperature collapse."""
    from omnibias.graph.torch.ops.relaxation import soft_top_k

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("soft_support needs torch + omnibias-graph") from exc
    t = torch.as_tensor(np.asarray(scores, dtype=np.float64), dtype=torch.float64)
    return soft_top_k(t, int(k), temperature=float(temperature)).detach().cpu().numpy()


def _windows(f: FloatArray, se: StructuringElement, *, dilate: bool) -> FloatArray:
    """Dilation pads missing sites with ``-inf``; erosion pads with ``+inf``."""
    x = np.asarray(f, dtype=np.float64).reshape(-1)
    n = x.size
    n_off = se.size
    fill = -np.inf if dilate else np.inf
    out = np.full((n, n_off), fill, dtype=np.float64)
    for i, (off, val) in enumerate(zip(se.offsets, se.values, strict=True)):
        src = np.arange(n, dtype=np.int64) - int(off) if dilate else np.arange(n, dtype=np.int64) + int(off)
        ok = (src >= 0) & (src < n)
        gathered = np.where(ok, x[np.clip(src, 0, n - 1)], fill)
        out[:, i] = gathered + (float(val) if dilate else -float(val))
    return out


def hard_dilate(f: object, se: StructuringElement) -> FloatArray:
    return np.max(_windows(np.asarray(f, dtype=np.float64), se, dilate=True), axis=-1)


def hard_erode(f: object, se: StructuringElement) -> FloatArray:
    return -np.max(-_windows(np.asarray(f, dtype=np.float64), se, dilate=False), axis=-1)


def dilate(f: object, se: StructuringElement, *, beta: float) -> MorphResult:
    win = _windows(np.asarray(f, dtype=np.float64), se, dilate=True)
    value = _logsumexp_beta(win, beta)
    hard = np.max(win, axis=-1)
    return MorphResult(
        value=value,
        gap_bound=morphology_gap_bound(size=se.size, beta=beta, compositions=1),
        direction="upper",
        compositions=1,
        observed_dev=float(np.max(value - hard)),
    )


def erode(f: object, se: StructuringElement, *, beta: float) -> MorphResult:
    win = _windows(np.asarray(f, dtype=np.float64), se, dilate=False)
    value = -_logsumexp_beta(-win, beta)
    hard = -np.max(-win, axis=-1)
    return MorphResult(
        value=value,
        gap_bound=morphology_gap_bound(size=se.size, beta=beta, compositions=1),
        direction="lower",
        compositions=1,
        observed_dev=float(np.max(hard - value)),
    )


def opening(f: object, se: StructuringElement, *, beta: float) -> MorphResult:
    inner = erode(f, se, beta=beta)
    outer = dilate(inner.value, se, beta=beta)
    hard = hard_dilate(hard_erode(f, se), se)
    bound = morphology_gap_bound(size=se.size, beta=beta, compositions=2)
    return MorphResult(
        value=outer.value,
        gap_bound=bound,
        direction="two_sided",
        compositions=2,
        observed_dev=float(np.max(np.abs(outer.value - hard))),
    )


def closing(f: object, se: StructuringElement, *, beta: float) -> MorphResult:
    inner = dilate(f, se, beta=beta)
    outer = erode(inner.value, se, beta=beta)
    hard = hard_erode(hard_dilate(f, se), se)
    bound = morphology_gap_bound(size=se.size, beta=beta, compositions=2)
    return MorphResult(
        value=outer.value,
        gap_bound=bound,
        direction="two_sided",
        compositions=2,
        observed_dev=float(np.max(np.abs(outer.value - hard))),
    )


def top_hat(f: object, se: StructuringElement, *, beta: float, dual: bool = False) -> MorphResult:
    x = np.asarray(f, dtype=np.float64).reshape(-1)
    if dual:
        cl = closing(x, se, beta=beta)
        value = cl.value - x
        return MorphResult(value, cl.gap_bound, "two_sided", 2, float(np.max(np.abs(value))))
    op = opening(x, se, beta=beta)
    value = x - op.value
    return MorphResult(value, op.gap_bound, "two_sided", 2, float(np.max(np.abs(value))))


def morphological_gradient(f: object, se: StructuringElement, *, beta: float) -> MorphResult:
    d = dilate(f, se, beta=beta)
    e = erode(f, se, beta=beta)
    value = d.value - e.value
    hard = hard_dilate(f, se) - hard_erode(f, se)
    bound = morphology_gap_bound(size=se.size, beta=beta, compositions=2)
    return MorphResult(
        value=value,
        gap_bound=bound,
        direction="two_sided",
        compositions=2,
        observed_dev=float(np.max(np.abs(value - hard))),
    )


def named_worked_signal() -> FloatArray:
    return np.array([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float64)


def named_worked_soft_center(*, beta: float = 2.0) -> float:
    """Spec §5: soft dilation at position 2, ``beta=2``."""
    f = named_worked_signal()
    se = StructuringElement.flat(1)
    return float(dilate(f, se, beta=beta).value[2])


def soft_max_pool(f: object, *, kernel: int, stride: int = 1, beta: float) -> MorphResult:
    """Flat-SE dilation on sliding windows. Not a new OperatorBlock role."""
    x = np.asarray(f, dtype=np.float64).reshape(-1)
    k = int(kernel)
    if k < 1:
        raise ValueError("kernel must be >= 1")
    s = int(stride)
    if s < 1:
        raise ValueError("stride must be >= 1")
    n_out = 1 + (x.size - k) // s if x.size >= k else 0
    win = np.full((n_out, k), -np.inf, dtype=np.float64)
    for i in range(n_out):
        sl = x[i * s : i * s + k]
        win[i, : sl.size] = sl
    value = _logsumexp_beta(win, beta)
    hard = np.max(win, axis=-1)
    return MorphResult(
        value=value,
        gap_bound=morphology_gap_bound(size=k, beta=beta, compositions=1),
        direction="upper",
        compositions=1,
        observed_dev=float(np.max(value - hard)) if n_out else 0.0,
    )


def exact_distance_transform(occupancy: object) -> FloatArray:
    """Exact 1-D Euclidean distance to the nearest occupied site."""
    occ = np.asarray(occupancy, dtype=np.float64).reshape(-1) > 0.5
    idx = np.flatnonzero(occ)
    if idx.size == 0:
        raise ValueError("occupancy must contain at least one occupied site")
    xs = np.arange(occ.size, dtype=np.float64)
    return np.min(np.abs(xs[:, None] - idx[None, :]), axis=1)


def soft_distance_transform(
    occupancy: object,
    *,
    beta: float,
    metric: str = "euclidean",
) -> MorphResult:
    """Softmin of distances to occupied sites. Upper bound, gap ``log(N_obj)/beta``."""
    if metric != "euclidean":
        raise ValueError("only metric='euclidean' is implemented")
    occ = np.asarray(occupancy, dtype=np.float64).reshape(-1) > 0.5
    idx = np.flatnonzero(occ)
    if idx.size == 0:
        raise ValueError("occupancy must contain at least one occupied site")
    xs = np.arange(occ.size, dtype=np.float64)
    dist = np.abs(xs[:, None] - idx[None, :].astype(np.float64))
    value = -_logsumexp_beta(-dist, beta)
    hard = np.min(dist, axis=1)
    bound = morphology_gap_bound(size=int(idx.size), beta=beta, compositions=1)
    return MorphResult(
        value=value,
        gap_bound=bound,
        direction="lower",
        compositions=1,
        observed_dev=float(np.max(hard - value)),
    )


__all__ = [
    "MorphResult",
    "StructuringElement",
    "closing",
    "dilate",
    "erode",
    "exact_distance_transform",
    "hard_dilate",
    "hard_erode",
    "honesty_payload",
    "morphological_gradient",
    "morphology_gap_bound",
    "named_worked_signal",
    "named_worked_soft_center",
    "opening",
    "soft_distance_transform",
    "soft_max_pool",
    "soft_support",
    "top_hat",
]

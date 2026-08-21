# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable topology surrogates (theory 03-09).

No differentiable function equals a Betti number. Soft counts and
persistence values are surrogates. ``beta -> inf`` is temperature
collapse (feasibility). The founding bias collapse
(``delta -> 0``) appears only in the underlying field. Do not conflate
the two. Persistence is differentiable almost everywhere
(pair swaps). Certified integer counts require a separating
spectral enclosure; otherwise the answer is :class:`Inconclusive`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from omnibias.core.verified.eig_operator import count_eigenvalues_below

FloatArray = NDArray[np.float64]


def honesty_payload() -> dict[str, object]:
    return {
        "equals_betti": False,
        "surrogate": True,
        "persistence_almost_everywhere": True,
        "certified_requires_gap": True,
        "temperature_collapse": True,
        "founding_bias_collapse": False,
    }


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class SoftCount:
    """Soft integer surrogate. ``certified`` is True only after a separating enclosure."""

    value: float
    gap_bound: float
    rounded: int
    certified: bool
    n_faces: int = 0

    def __post_init__(self) -> None:
        if self.gap_bound < 0.0:
            raise ValueError("gap_bound must be >= 0")


@dataclass(frozen=True)
class Inconclusive:
    """Certified route refused. Never a guessed integer."""

    reason: str


@dataclass(frozen=True)
class PersistencePair:
    birth: float
    death: float
    dimension: int = 0

    @property
    def persistence(self) -> float:
        return float(self.death - self.birth)


def soft_component_count(
    eigenvalues: Sequence[float],
    *,
    epsilon: float,
    beta: float,
) -> SoftCount:
    """``sum_i sigmoid(beta (epsilon - lambda_i))``. Exact as ``beta -> inf``."""
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    ev = [float(v) for v in eigenvalues]
    value = 0.0
    gap = 0.0
    true = 0
    for lam in ev:
        s = _sigmoid(float(beta) * (float(epsilon) - lam))
        value += s
        indicator = 1.0 if lam < float(epsilon) else 0.0
        true += int(indicator)
        gap += abs(s - indicator)
    return SoftCount(value=float(value), gap_bound=float(gap), rounded=int(round(value)), certified=False)


def certified_component_count(
    laplacian: Sequence[Sequence[float]],
    *,
    epsilon: float,
) -> int | Inconclusive:
    """Inertia count of Laplacian eigenvalues below ``epsilon``. Reuses ``count_eigenvalues_below``."""
    a = np.asarray(laplacian, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("laplacian must be square")
    ident = np.eye(a.shape[0], dtype=np.float64)
    got = count_eigenvalues_below(a.tolist(), ident.tolist(), float(epsilon))
    if got is None:
        return Inconclusive("spectral_gap_does_not_separate")
    return int(got)


def soft_euler_characteristic(
    face_masses: Sequence[float],
    face_dims: Sequence[int],
) -> SoftCount:
    """Alternating sum of face masses. Bound is a sum over faces (loose when large)."""
    if len(face_masses) != len(face_dims):
        raise ValueError("face_masses and face_dims must have the same length")
    chi = 0.0
    hard = 0.0
    gap = 0.0
    for mass, dim in zip(face_masses, face_dims, strict=True):
        m = float(mass)
        sign = -1.0 if int(dim) % 2 else 1.0
        chi += sign * m
        bit = 1.0 if m >= 0.5 else 0.0
        hard += sign * bit
        gap += abs(m - bit)
    return SoftCount(
        value=float(chi),
        gap_bound=float(gap),
        rounded=int(round(chi)),
        certified=False,
        n_faces=len(face_masses),
    )


@dataclass(frozen=True)
class IndexedPair:
    birth_index: int
    death_index: int
    birth: float
    death: float


def _h0_indexed(values: Sequence[float] | FloatArray) -> tuple[IndexedPair, ...]:
    """Exact 1-D sublevel H0 pairing with critical indices."""
    vals = np.asarray(values, dtype=np.float64)
    n = int(vals.size)
    parent = list(range(n))
    birth_idx = list(range(n))
    birth_val = vals.copy()
    active = np.zeros(n, dtype=bool)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs: list[IndexedPair] = []
    for i in np.argsort(vals, kind="stable"):
        idx = int(i)
        active[idx] = True
        neigh = [find(j) for j in (idx - 1, idx + 1) if 0 <= j < n and active[j]]
        if not neigh:
            parent[idx] = idx
            birth_idx[idx] = idx
            birth_val[idx] = vals[idx]
            continue
        if len(neigh) == 1:
            parent[idx] = neigh[0]
            continue
        older, younger = neigh
        if birth_val[older] > birth_val[younger]:
            older, younger = younger, older
        pairs.append(
            IndexedPair(
                birth_index=int(birth_idx[younger]),
                death_index=idx,
                birth=float(birth_val[younger]),
                death=float(vals[idx]),
            )
        )
        parent[younger] = older
        parent[idx] = older
    return tuple(pairs)


def _h0_pairs(values: Sequence[float] | FloatArray) -> tuple[PersistencePair, ...]:
    """Exact 1-D sublevel H0 pairing (union-find). The 1-D Morse oracle."""
    return tuple(PersistencePair(p.birth, p.death, 0) for p in _h0_indexed(values))


def persistence_loss_grad(
    field: Sequence[float],
    *,
    threshold: float,
    mode: Literal["suppress", "encourage"] = "suppress",
    longest_only: bool = False,
) -> FloatArray:
    """Sparse gradient of :func:`persistence_loss` (1-D H0). Differentiable a.e."""
    vals = np.asarray(field, dtype=np.float64)
    pairs = list(_h0_indexed(vals))
    if longest_only and pairs:
        pairs = [max(pairs, key=lambda p: p.death - p.birth)]
    grad = np.zeros_like(vals)
    for pair in pairs:
        pers = pair.death - pair.birth
        if mode == "suppress" and pers < float(threshold):
            grad[pair.death_index] -= 1.0
            grad[pair.birth_index] += 1.0
        elif mode == "encourage" and pers > float(threshold):
            grad[pair.death_index] += 1.0
            grad[pair.birth_index] -= 1.0
    return grad


def soft_persistence(field: Sequence[float], *, beta: float = 1.0, max_dim: int = 0) -> tuple[PersistencePair, ...]:
    """1-D sublevel pairs of a sampled field. Differentiable almost everywhere."""
    if int(max_dim) != 0:
        raise ValueError("only H0 (max_dim=0) is implemented on a 1-D complex")
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    # beta-smoothed occupancy as a filtration; pairing is the 1-D Morse oracle.
    vals = 1.0 / (1.0 + np.exp(-float(beta) * np.asarray(field, dtype=np.float64)))
    return _h0_pairs(vals)


def exact_h0_persistence(values: Sequence[float]) -> tuple[PersistencePair, ...]:
    """Unsmoothed 1-D Morse pairing. The comparison oracle for G4."""
    return _h0_pairs(values)


def bottleneck_distance(
    left: Sequence[PersistencePair],
    right: Sequence[PersistencePair],
) -> float:
    """Bottleneck distance of finite H0 diagrams (diagonal matching allowed)."""
    a = sorted((p.birth, p.death) for p in left if math.isfinite(p.death))
    b = sorted((p.birth, p.death) for p in right if math.isfinite(p.death))
    if not a and not b:
        return 0.0
    # For |A|,|B| <= 8 the assignment plus diagonal is cheap.
    from itertools import permutations

    best = math.inf
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    extras = len(longer) - len(shorter)
    # pad shorter with diagonal matches
    padded = list(shorter) + [None] * extras
    for perm in set(permutations(padded)):
        dist = 0.0
        for p, q in zip(longer, perm, strict=True):
            if q is None:
                dist = max(dist, 0.5 * abs(p[1] - p[0]))
            else:
                dist = max(dist, max(abs(p[0] - q[0]), abs(p[1] - q[1])))
        best = min(best, dist)
    return float(best)


def persistence_loss(
    pairs: Sequence[PersistencePair],
    *,
    threshold: float,
    mode: Literal["suppress", "encourage"] = "suppress",
) -> float:
    """Piecewise-linear in birth/death. Differentiable almost everywhere."""
    acc = 0.0
    for pair in pairs:
        pers = pair.persistence
        if not math.isfinite(pers):
            continue
        if mode == "suppress":
            acc += max(0.0, float(threshold) - pers)
        elif mode == "encourage":
            acc += max(0.0, pers - float(threshold))
        else:
            raise ValueError("mode must be 'suppress' or 'encourage'")
    return float(acc)


def bimodal_saddle_index(values: Sequence[float]) -> int:
    """Index of the min between the left-half and right-half maxima (1-D Morse saddle)."""
    u = np.asarray(values, dtype=np.float64)
    n = int(u.size)
    if n < 3:
        raise ValueError("need at least 3 samples")
    left = int(np.argmax(u[: n // 2]))
    right = n // 2 + int(np.argmax(u[n // 2 :]))
    return left + int(np.argmin(u[left : right + 1]))


def connected_components_1d(occupancy: Sequence[float], *, threshold: float = 0.5) -> int:
    bits = np.asarray(occupancy, dtype=np.float64) >= float(threshold)
    if bits.size == 0:
        return 0
    return int(np.sum(bits & np.concatenate(([True], ~bits[:-1]))))


def cluster_laplacian(n_clusters: int, n_per: int = 2, *, gap: float = 0.4) -> FloatArray:
    """Block-diagonal path Laplacians: ``n_clusters`` zeros, then a gap."""
    k = int(n_per)
    n = int(n_clusters) * k
    lap = np.zeros((n, n), dtype=np.float64)
    g = float(gap)
    for c in range(int(n_clusters)):
        base = c * k
        for i in range(k - 1):
            a, b = base + i, base + i + 1
            lap[a, a] += g
            lap[b, b] += g
            lap[a, b] -= g
            lap[b, a] -= g
    return lap


__all__ = [
    "Inconclusive",
    "PersistencePair",
    "SoftCount",
    "bimodal_saddle_index",
    "bottleneck_distance",
    "certified_component_count",
    "cluster_laplacian",
    "connected_components_1d",
    "exact_h0_persistence",
    "honesty_payload",
    "persistence_loss",
    "persistence_loss_grad",
    "soft_component_count",
    "soft_euler_characteristic",
    "soft_persistence",
]

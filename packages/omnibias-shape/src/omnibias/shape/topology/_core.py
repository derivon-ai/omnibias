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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from omnibias.core.verified.eig_operator import count_eigenvalues_below

FloatArray = NDArray[np.float64]
FieldFn = Callable[[FloatArray, FloatArray], FloatArray]


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

    def as_pair(self) -> tuple[float, float]:
        """The only unpack: ``(value, gap_bound)``. Never a bare float."""
        return (float(self.value), float(self.gap_bound))

    def __iter__(self):  # noqa: ANN204 -- pair iterator, not a collection
        yield float(self.value)
        yield float(self.gap_bound)


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


def euler_value_bound(count: SoftCount) -> tuple[float, float]:
    """Public unpack of a soft Euler report. Never a bare float."""
    return count.as_pair()


def cubical_faces_2d(occupancy: np.ndarray) -> tuple[list[float], list[int]]:
    """Gray cubical faces of a 2-D occupancy grid (product edge / vertex masses)."""
    u = np.asarray(occupancy, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("occupancy must be a 2-D grid")
    if u.size == 0:
        raise ValueError("occupancy must be non-empty")
    masses: list[float] = [float(v) for v in u.ravel()]
    dims: list[int] = [2] * int(u.size)
    if u.shape[1] > 1:
        eh = u[:, :-1] * u[:, 1:]
        masses.extend(float(v) for v in eh.ravel())
        dims.extend([1] * int(eh.size))
    if u.shape[0] > 1:
        ev = u[:-1, :] * u[1:, :]
        masses.extend(float(v) for v in ev.ravel())
        dims.extend([1] * int(ev.size))
    if u.shape[0] > 1 and u.shape[1] > 1:
        vv = u[:-1, :-1] * u[:-1, 1:] * u[1:, :-1] * u[1:, 1:]
        masses.extend(float(v) for v in vv.ravel())
        dims.extend([0] * int(vv.size))
    return masses, dims


def cubical_euler_value(occupancy: np.ndarray) -> float:
    """Product-Gray Euler of a 2-D occupancy grid (smooth in the masses)."""
    u = np.asarray(occupancy, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("occupancy must be a 2-D grid")
    chi = float(u.sum())
    if u.shape[1] > 1:
        chi -= float((u[:, :-1] * u[:, 1:]).sum())
    if u.shape[0] > 1:
        chi -= float((u[:-1, :] * u[1:, :]).sum())
    if u.shape[0] > 1 and u.shape[1] > 1:
        chi += float((u[:-1, :-1] * u[:-1, 1:] * u[1:, :-1] * u[1:, 1:]).sum())
    return chi


def cubical_euler_grad(occupancy: np.ndarray) -> FloatArray:
    """``d chi / d u`` for :func:`cubical_euler_value`."""
    u = np.asarray(occupancy, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("occupancy must be a 2-D grid")
    grad = np.ones_like(u, dtype=np.float64)
    if u.shape[1] > 1:
        grad[:, :-1] -= u[:, 1:]
        grad[:, 1:] -= u[:, :-1]
    if u.shape[0] > 1:
        grad[:-1, :] -= u[1:, :]
        grad[1:, :] -= u[:-1, :]
    if u.shape[0] > 1 and u.shape[1] > 1:
        a = u[:-1, :-1]
        b = u[:-1, 1:]
        c = u[1:, :-1]
        d = u[1:, 1:]
        grad[:-1, :-1] += b * c * d
        grad[:-1, 1:] += a * c * d
        grad[1:, :-1] += a * b * d
        grad[1:, 1:] += a * b * c
    return grad


def occupancy_euler(occupancy: np.ndarray) -> SoftCount:
    """Soft Euler of a ``[0, 1]`` occupancy grid. Value and bound travel together."""
    masses, dims = cubical_faces_2d(occupancy)
    return soft_euler_characteristic(masses, dims)


def _grid_axes(
    grid: np.ndarray | Sequence[float] | tuple[np.ndarray, np.ndarray],
) -> tuple[FloatArray, FloatArray]:
    if isinstance(grid, tuple) and len(grid) == 2:
        xs = np.asarray(grid[0], dtype=np.float64).reshape(-1)
        ys = np.asarray(grid[1], dtype=np.float64).reshape(-1)
        return xs, ys
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim == 1:
        return arr, arr
    raise ValueError("grid must be a 1-D axis or a (xs, ys) pair")


def _sigmoid_array(z: np.ndarray) -> FloatArray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def field_to_occupancy(
    field: FieldFn | np.ndarray,
    *,
    beta: float,
    grid: np.ndarray | Sequence[float] | tuple[np.ndarray, np.ndarray],
) -> FloatArray:
    """``sigmoid(beta * field)`` on ``grid``. ``beta -> inf`` is temperature collapse."""
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    xs, ys = _grid_axes(grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    if callable(field):
        raw = np.asarray(field(xx, yy), dtype=np.float64)
    else:
        raw = np.asarray(field, dtype=np.float64)
    if raw.shape != xx.shape:
        raise ValueError(f"field shape {raw.shape} does not match grid {xx.shape}")
    return _sigmoid_array(float(beta) * raw)


def field_euler_characteristic(
    field: FieldFn | np.ndarray,
    *,
    beta: float,
    grid: np.ndarray | Sequence[float] | tuple[np.ndarray, np.ndarray],
) -> SoftCount:
    """Soft Euler of a sampled implicit field. Returns value and bound together.

    Occupancy is ``sigmoid(beta * field(x, y))``. The gap bound is the
    03-09 face-sum and contains the integer Euler of the independently
    thresholded cubical complex. This is not a Betti number.
    """
    occ = field_to_occupancy(field, beta=beta, grid=grid)
    return occupancy_euler(occ)


def field_euler_pair(
    field: FieldFn | np.ndarray,
    *,
    beta: float,
    grid: np.ndarray | Sequence[float] | tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """Spec 05-02 unpack: ``(value, bound_on_gap_to_integer)``."""
    return euler_value_bound(field_euler_characteristic(field, beta=beta, grid=grid))


def euler_regularizer(occupancy: np.ndarray, *, target: float) -> float:
    """``(soft_chi - target)^2``. Temperature-smoothed; not an integer Betti number."""
    chi = cubical_euler_value(occupancy)
    return float((chi - float(target)) ** 2)


def digital_components_2d(bits: np.ndarray) -> int:
    """4-connected component count of a boolean image."""
    mask = np.asarray(bits, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("bits must be a 2-D image")
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or seen[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
    return int(count)


def digital_genus(occupancy: np.ndarray, *, threshold: float = 0.5) -> int:
    """Planar holes ``C - chi`` of a padded thresholded occupancy.

    A disk has genus 0; an annulus has genus 1. Not a 3-D surface genus.
    """
    bits = np.asarray(occupancy, dtype=np.float64) >= float(threshold)
    pad = np.pad(bits, 1, constant_values=False)
    chi = int(round(cubical_euler_value(pad.astype(np.float64))))
    comps = digital_components_2d(pad)
    return int(comps - chi)


def _discrete_laplacian(u: np.ndarray) -> FloatArray:
    pad = np.pad(u, 1, mode="edge")
    return (
        4.0 * u
        - pad[1:-1, 2:]
        - pad[1:-1, :-2]
        - pad[2:, 1:-1]
        - pad[:-2, 1:-1]
    )


def regularize_occupancy(
    occupancy: np.ndarray,
    *,
    pos_mask: np.ndarray,
    exterior_mask: np.ndarray,
    target_chi: float,
    steps: int = 36,
    lr: float = 0.25,
    lam_euler: float = 0.35,
    lam_data: float = 2.5,
    lam_bg: float = 2.5,
    lam_smooth: float = 0.08,
) -> FloatArray:
    """Open an unconstrained hole by stepping occupancy toward ``target_chi``.

    Positives and the far exterior are pinned by the data term. The Euler
    gradient is applied only on the leftover interior so the regularizer
    cannot swiss-cheese the material. An unregularized fill stays filled.
    """
    u = np.clip(np.asarray(occupancy, dtype=np.float64), 0.0, 1.0)
    pos = np.asarray(pos_mask, dtype=bool)
    ext = np.asarray(exterior_mask, dtype=bool)
    if u.shape != pos.shape or u.shape != ext.shape:
        raise ValueError("occupancy and masks must share a shape")
    free = ~(pos | ext)
    for _ in range(int(steps)):
        grad = np.zeros_like(u)
        if np.any(pos):
            grad[pos] += 2.0 * float(lam_data) * (u[pos] - 1.0)
        if np.any(ext):
            grad[ext] += 2.0 * float(lam_bg) * u[ext]
        chi = cubical_euler_value(u)
        euler_g = 2.0 * (chi - float(target_chi)) * cubical_euler_grad(u)
        masked = np.zeros_like(u)
        masked[free] = euler_g[free]
        grad += float(lam_euler) * masked
        grad += float(lam_smooth) * _discrete_laplacian(u)
        u = np.clip(u - float(lr) * grad, 0.0, 1.0)
    return u


__all__ = [
    "Inconclusive",
    "PersistencePair",
    "SoftCount",
    "bimodal_saddle_index",
    "bottleneck_distance",
    "certified_component_count",
    "cluster_laplacian",
    "connected_components_1d",
    "cubical_euler_grad",
    "cubical_euler_value",
    "cubical_faces_2d",
    "digital_components_2d",
    "digital_genus",
    "euler_regularizer",
    "euler_value_bound",
    "exact_h0_persistence",
    "field_euler_characteristic",
    "field_euler_pair",
    "field_to_occupancy",
    "honesty_payload",
    "occupancy_euler",
    "persistence_loss",
    "persistence_loss_grad",
    "regularize_occupancy",
    "soft_component_count",
    "soft_euler_characteristic",
    "soft_persistence",
]

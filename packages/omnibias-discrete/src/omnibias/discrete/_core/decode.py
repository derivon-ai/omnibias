# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy) rounding, 1-flip local search, and exact oracle.

The decoder rounds a (predicted or relaxed) soft assignment ``x in (0, 1)^n`` to a
**binary point** and refines it by greedy single-bit-flip local search. It works for any
:class:`~omnibias.discrete._core.problem.DiscreteProblem`: a problem may expose a
closed-form ``flip_deltas(x)`` fast path (e.g. QUBO's one-matvec formula), otherwise the
generic fallback reads the flip energies off a single batched ``energy`` evaluation of
the ``n`` single-flip neighbours.

:func:`brute_force_min` is the exact ``O(2^n)`` optimum -- the ground-truth used to
self-check the certificate sandwich on small ``n`` (keep ``n <= ~20``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.discrete._core.problem import DiscreteProblem

FloatArray = NDArray[np.float64]

_MAX_EXACT_N = 20


def energy(problem: DiscreteProblem, x: object) -> float | FloatArray:
    """The energy of ``x`` under ``problem`` (delegates to ``problem.energy``)."""
    result: float | FloatArray = problem.energy(x)
    return result


def is_binary(x: object, *, tol: float = 1e-9) -> bool:
    """Whether every entry of ``x`` is within ``tol`` of ``0`` or ``1``."""
    xv = np.asarray(x, dtype=float)
    return bool(np.all((np.abs(xv) < tol) | (np.abs(xv - 1.0) < tol)))


def round_relaxed(x: object) -> FloatArray:
    """Round a soft assignment to ``{0, 1}`` at the ``0.5`` threshold."""
    return (np.asarray(x, dtype=float) >= 0.5).astype(float)


def flip_deltas(problem: DiscreteProblem, x: object) -> FloatArray:
    r"""The energy change for flipping each single bit of ``x``.

    Uses ``problem.flip_deltas`` when the problem provides that closed-form fast path,
    otherwise evaluates the ``n`` single-flip neighbours in one batched ``energy`` call
    and subtracts the base energy.
    """
    fast = getattr(problem, "flip_deltas", None)
    if fast is not None:
        return np.asarray(fast(x), dtype=float)
    xv = np.asarray(x, dtype=float)
    n = problem.n
    base = float(problem.energy(xv))
    neighbours = np.tile(xv, (n, 1))
    diag = np.arange(n)
    neighbours[diag, diag] = 1.0 - neighbours[diag, diag]
    neigh_energy = np.asarray(problem.energy(neighbours), dtype=float)
    return neigh_energy - base


def one_flip_descent(problem: DiscreteProblem, x0: object) -> tuple[FloatArray, float]:
    r"""Greedy single-bit-flip descent from ``x0`` to a 1-flip-local minimum.

    ``x0`` is rounded to ``{0, 1}`` first; each step flips the most energy-decreasing
    bit until no single flip improves. Returns ``(x, energy)``.
    """
    x = round_relaxed(x0)
    while True:
        delta = flip_deltas(problem, x)
        i = int(np.argmin(delta))
        if delta[i] >= -1e-12:
            break
        x[i] = 1.0 - x[i]
    return x, float(problem.energy(x))


def decode(
    problem: DiscreteProblem,
    *,
    relaxed: object | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[tuple[int, ...], float]:
    r"""Round + local-search a problem to a valid binary point (an *upper* bound).

    Starts from the rounded ``relaxed`` assignment (when given), the all-zero point, and
    ``n_starts`` deterministic random binary points; each is refined by
    :func:`one_flip_descent` and the best is returned as ``(assignment, energy)``.
    """
    n = problem.n
    rng = np.random.default_rng(seed)
    starts: list[FloatArray] = [np.zeros(n)]
    if relaxed is not None:
        starts.append(round_relaxed(relaxed))
    for _ in range(max(n_starts, 0)):
        starts.append(rng.integers(0, 2, size=n).astype(float))

    best_x: FloatArray | None = None
    best_e = np.inf
    for start in starts:
        x, e = one_flip_descent(problem, start)
        if e < best_e:
            best_x, best_e = x, e
    assert best_x is not None
    return tuple(int(v) for v in best_x), float(best_e)


def brute_force_min(
    problem: DiscreteProblem, *, max_n: int = _MAX_EXACT_N
) -> tuple[tuple[int, ...], float]:
    r"""Exact minimum-energy binary point by enumerating all ``2^n`` assignments.

    Exponential (``O(2^n)`` energy evaluations); intended as the small-``n`` oracle that
    self-checks the certificate sandwich. Raises :class:`ValueError` for ``n > max_n``.
    """
    n = problem.n
    if n > max_n:
        raise ValueError(
            f"brute_force_min is exponential (O(2^n)); n={n} exceeds the {max_n} cap. "
            "Use decode + certify_gap for a certified heuristic instead."
        )
    idx = np.arange(1 << n, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
    x = bits.astype(float)
    e = np.asarray(problem.energy(x), dtype=float)
    best = int(np.argmin(e))
    return tuple(int(v) for v in x[best]), float(e[best])


__all__ = [
    "brute_force_min",
    "decode",
    "energy",
    "flip_deltas",
    "is_binary",
    "one_flip_descent",
    "round_relaxed",
]

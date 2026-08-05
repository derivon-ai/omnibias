# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy) rounding, 1-flip local search, and exact oracle.

The rounding / local-search / oracle machinery now lives in the ``omnibias-discrete``
substrate (:mod:`omnibias.discrete._core.decode`) and is generic over any
``DiscreteProblem``. This module re-exports it and keeps the historical ``decode_qubo``
name; :class:`~omnibias.qubo.problem.QUBOProblem` supplies a closed-form ``flip_deltas``
fast path, so the local search costs one matrix-vector product per sweep exactly as
before.

:func:`brute_force_min` is the exact ``O(2^n)`` optimum -- the ground-truth used to
self-check the certificate sandwich on small ``n`` (keep ``n <= ~20``).
"""

from __future__ import annotations

from omnibias.discrete._core.decode import (
    brute_force_min,
    decode,
    energy,
    is_binary,
    one_flip_descent,
    round_relaxed,
)
from omnibias.qubo.problem import QUBOProblem


def decode_qubo(
    problem: QUBOProblem,
    *,
    relaxed: object | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[tuple[int, ...], float]:
    r"""Round + local-search a QUBO to a valid binary point (an *upper* bound).

    Starts from the rounded ``relaxed`` assignment (when given), the all-zero point, and
    ``n_starts`` deterministic random binary points; each is refined by
    :func:`one_flip_descent` and the best is returned as ``(assignment, energy)``. A thin
    alias of :func:`omnibias.discrete.decode`.
    """
    result: tuple[tuple[int, ...], float] = decode(
        problem, relaxed=relaxed, n_starts=n_starts, seed=seed
    )
    return result


__all__ = [
    "brute_force_min",
    "decode_qubo",
    "energy",
    "is_binary",
    "one_flip_descent",
    "round_relaxed",
]

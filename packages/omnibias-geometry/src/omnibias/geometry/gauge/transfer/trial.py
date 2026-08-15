# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge-invariant holonomy trial spaces for a finite transfer matrix.

Trial vectors are characters of closed holonomies evaluated on the
angle / class-angle grid of a dense transfer matrix.  That puts the
trial space in the physical (gauge-invariant) sector.  Character-basis
heat-kernel matrices are already diagonal in this basis; the lever is
the dense ``angle`` / ``su2_class_angle`` constructions.

A badly conditioned Gram matrix loosens rather than tightens a
variational bound.  Every :class:`TrialSpace` reports its Gram condition
number and is flagged above :data:`GRAM_COND_THRESHOLD`.  A Magnus
remainder, when requested, is carried as an enclosure width and must
enter the certified gap.

Not a continuum gauge claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.geometry.gauge.band._core import (
    abelian_holonomy,
    magnus_truncation_bound,
    su2_transverse_constant,
)
from omnibias.geometry.gauge.transfer.matrices import TransferMatrix

GRAM_COND_THRESHOLD = 1.0e8


@dataclass(frozen=True)
class Loop:
    """A closed holonomy used as one trial direction.

    ``winding`` is the U(1) Fourier mode or the SU(2) Dynkin label
    ``a`` (character ``sin((a+1) θ) / sin θ``).
    """

    winding: int
    regime: str = "abelian"
    components: tuple[float, float, float] = (0.3, -0.2, 0.5)
    length: float = 1.0
    coupling: float = 1.0


@dataclass(frozen=True)
class TrialSpace:
    """Orthonormal-ish trial vectors plus conditioning / remainder metadata."""

    vectors: tuple[tuple[float, ...], ...]
    gram_condition: float
    flagged: bool
    remainder_width: float
    basis: str

    @property
    def dim(self) -> int:
        return len(self.vectors)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _gram_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[_dot(u, v) for v in vectors] for u in vectors]


def _float_inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    n = len(matrix)
    aug = [
        [float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-18:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for row in range(n):
            if row != col and aug[row][col] != 0.0:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


def _frobenius(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(math.fsum(x * x for row in matrix for x in row))


def gram_condition_number(vectors: Sequence[Sequence[float]]) -> float:
    """``||G||_F · ||G^{-1}||_F`` of the trial Gram matrix (``inf`` if singular)."""
    if not vectors:
        return float("inf")
    gram = _gram_matrix(vectors)
    inverse = _float_inverse(gram)
    if inverse is None:
        return float("inf")
    return _frobenius(gram) * _frobenius(inverse)


def su2_holonomy_trace(
    components: tuple[float, float, float],
    *,
    length: float,
    coupling: float,
) -> float:
    """``Re tr U`` of a transverse-constant SU(2) holonomy (closed-form Rodrigues)."""
    u00, _u01, _u10, u11 = su2_transverse_constant(
        components, length=length, coupling=coupling
    )
    return float((u00 + u11).real)


def abelian_holonomy_phase(*, a0: float, lo: float, hi: float, coupling: float) -> complex:
    """Closed-form U(1) holonomy across a tanh slab (spec 02-14)."""
    return abelian_holonomy(a0=a0, lo=lo, hi=hi, coupling=coupling)


def _u1_character_vector(n_points: int, winding: int) -> tuple[float, ...]:
    if winding == 0:
        return tuple(1.0 for _ in range(n_points))
    if winding > 0:
        return tuple(
            math.cos(2.0 * math.pi * winding * index / n_points)
            for index in range(n_points)
        )
    return tuple(
        math.sin(2.0 * math.pi * abs(winding) * index / n_points)
        for index in range(n_points)
    )


def _su2_character_vector(n_points: int, dynkin: int) -> tuple[float, ...]:
    denom = n_points + 1
    mode = abs(int(dynkin)) + 1
    return tuple(
        math.sin(mode * index * math.pi / denom) for index in range(1, n_points + 1)
    )


def default_loops(transfer: TransferMatrix, dim: int) -> tuple[Loop, ...]:
    """Winding / Dynkin family matching the transfer's real eigenbasis.

    U(1) uses the signed Fourier order ``0, +1, -1, +2, -2, ...`` (cosine then
    sine).  A non-negative-only family repeats ``cos(2π n k / N) =
    cos(2π (N-n) k / N)`` and makes the Gram singular.
    """
    if transfer.model.startswith("u1"):
        windings: list[int] = [0]
        mode = 1
        while len(windings) < dim:
            windings.append(mode)
            if len(windings) < dim:
                windings.append(-mode)
            mode += 1
        return tuple(Loop(winding=winding, regime="abelian") for winding in windings)
    return tuple(
        Loop(winding=index, regime="transverse_constant") for index in range(dim)
    )


def holonomy_trial_space(
    transfer: TransferMatrix,
    loops: Sequence[Loop] | None = None,
    *,
    dim: int | None = None,
) -> TrialSpace:
    """Gauge-invariant trial vectors from ``tr W[C]`` on the angle grid.

    ``dim`` caps the number of loops (default: the matrix dimension).
    Reports the Gram condition number; a run above
    :data:`GRAM_COND_THRESHOLD` is flagged rather than silently trusted.
    """
    size = transfer.dimension
    count = size if dim is None else min(int(dim), size)
    if count < 1:
        raise ValueError("trial space needs dim >= 1")
    family = tuple(loops) if loops is not None else default_loops(transfer, count)
    family = family[:count]
    if transfer.model.startswith("u1"):
        vectors = tuple(_u1_character_vector(size, loop.winding) for loop in family)
    else:
        vectors = tuple(_su2_character_vector(size, loop.winding) for loop in family)
    remainder = 0.0
    for loop in family:
        if loop.regime == "magnus":
            bound = magnus_truncation_bound(
                a_norm=math.hypot(*loop.components),
                length=loop.length,
                order=4,
            )
            remainder = max(remainder, bound.hi - bound.lo)
    cond = gram_condition_number(vectors)
    return TrialSpace(
        vectors=vectors,
        gram_condition=cond,
        flagged=cond > GRAM_COND_THRESHOLD or not math.isfinite(cond),
        remainder_width=remainder,
        basis=transfer.basis,
    )


__all__ = [
    "GRAM_COND_THRESHOLD",
    "Loop",
    "TrialSpace",
    "abelian_holonomy_phase",
    "default_loops",
    "gram_condition_number",
    "holonomy_trial_space",
    "su2_holonomy_trace",
]

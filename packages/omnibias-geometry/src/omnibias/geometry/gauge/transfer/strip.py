# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite 2+1-D SU(2) spatial-strip transfer, reflection positivity, cluster tail.

A spatial circle of ``n_sites`` class angles, each discretized into
``n_angles`` bins, gives an ``n_angles^{n_sites}``-dimensional Euclidean
time transfer.  CI uses ``2 × 4 → 16``.  The spectrum is **not** known
in closed form: spatial weights couple the sites, so this is not a
tensor of heat kernels.

Reflection positivity is checked on **this** matrix for a locked angle
inversion and a locked family of test vectors.  It is not
Osterwalder–Seiler reconstruction.  The cluster tail is a geometric
majorant of a locked spatial-bond correlator.  Continuum existence
stays external.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.series import geometric_tail_enclosure
from omnibias.core.verified.transcend import PI_IV, cos_iv, exp_iv
from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
from omnibias.geometry.gauge.transfer.matrices import (
    Scalar,
    TransferMatrix,
    encode_scalar,
)

STRIP_COUPLING_LOCK = Fraction(1, 2)


def _decode(index: int, n_sites: int, n_angles: int) -> tuple[int, ...]:
    sites: list[int] = []
    value = int(index)
    for _ in range(n_sites):
        sites.append(value % n_angles)
        value //= n_angles
    return tuple(sites)


def _encode(sites: Sequence[int], n_angles: int) -> int:
    acc = 0
    power = 1
    for site in sites:
        acc += int(site) * power
        power *= n_angles
    return acc


def _temporal_kernel(beta: Interval, n_angles: int) -> tuple[tuple[Interval, ...], ...]:
    rows: list[tuple[Interval, ...]] = []
    for i in range(n_angles):
        row: list[Interval] = []
        for j in range(n_angles):
            delta = (PI_IV + PI_IV) * Interval.from_value(Fraction(i - j, n_angles))
            row.append(exp_iv(beta * cos_iv(delta)))
        rows.append(tuple(row))
    return tuple(rows)


def _spatial_sqrt_weight(
    sites: Sequence[int], beta: Interval, n_angles: int
) -> Interval:
    action = Interval.point(0.0)
    n_sites = len(sites)
    for slot in range(n_sites):
        left, right = sites[slot], sites[(slot + 1) % n_sites]
        delta = (PI_IV + PI_IV) * Interval.from_value(Fraction(left - right, n_angles))
        action = action + cos_iv(delta)
    return exp_iv(beta * action * Interval.from_value(Fraction(1, 2)))


def su2_spatial_strip_transfer(
    coupling: Scalar,
    *,
    n_sites: int = 2,
    n_angles: int = 4,
    lattice_spacing: float = 1.0,
) -> TransferMatrix:
    """Euclidean-time transfer on a spatial circle of SU(2) class angles.

    Entries are ``√W(α) (⊗_sites K) √W(α')`` with Wilson kernels
    ``K(θ,φ) = exp(β cos(θ-φ))``.  The spectrum is not claimed in closed
    form.  This is one finite matrix, not 4-D Yang-Mills.
    """
    if isinstance(coupling, bool) or not isinstance(coupling, int | float | Fraction):
        raise ValueError(f"coupling must be a positive scalar, got {coupling!r}")
    if float(coupling) <= 0.0:
        raise ValueError(f"coupling must be > 0, got {coupling!r}")
    if n_sites < 2:
        raise ValueError(f"n_sites must be >= 2, got {n_sites}")
    if n_angles < 2:
        raise ValueError(f"n_angles must be >= 2, got {n_angles}")
    beta = Interval.from_value(coupling)
    dim = int(n_angles) ** int(n_sites)
    kernel = _temporal_kernel(beta, n_angles)
    weights = [
        _spatial_sqrt_weight(_decode(index, n_sites, n_angles), beta, n_angles)
        for index in range(dim)
    ]
    raw: list[list[Interval]] = []
    for left in range(dim):
        left_sites = _decode(left, n_sites, n_angles)
        row: list[Interval] = []
        for right in range(dim):
            right_sites = _decode(right, n_sites, n_angles)
            hop = Interval.point(1.0)
            for site_l, site_r in zip(left_sites, right_sites, strict=True):
                hop = hop * kernel[site_l][site_r]
            row.append(weights[left] * hop * weights[right])
        raw.append(row)
    entries = tuple(tuple(cell for cell in row) for row in raw)
    labels = tuple(
        ",".join(str(site) for site in _decode(index, n_sites, n_angles))
        for index in range(dim)
    )
    mid = np.array([[0.5 * (cell.lo + cell.hi) for cell in row] for row in entries])
    _values, vectors = np.linalg.eigh(0.5 * (mid + mid.T))
    order = np.argsort(_values)[::-1]
    perron = tuple(float(x) for x in vectors[:, order[0]])
    subdominant = tuple(
        tuple(float(x) for x in vectors[:, order[k]]) for k in range(1, len(order))
    )
    return TransferMatrix(
        model="su2_spatial_strip",
        basis="angle",
        entries=entries,
        mode_labels=labels,
        exact_eigenvalues=None,
        parameters={
            "builder": "su2_spatial_strip_transfer",
            "coupling": encode_scalar(coupling),
            "n_sites": int(n_sites),
            "n_angles": int(n_angles),
            "lattice_spacing": float(lattice_spacing),
        },
        perron_vector=perron,
        subdominant_vectors=subdominant,
        symmetric=True,
    )


def _reflect_index(index: int, n_sites: int, n_angles: int) -> int:
    sites = _decode(index, n_sites, n_angles)
    reflected = tuple((-site) % n_angles for site in sites)
    return _encode(reflected, n_angles)


def _quadratic_form(
    matrix: Sequence[Sequence[Interval]],
    left: Sequence[float],
    right: Sequence[float],
) -> Interval:
    total = Interval.point(0.0)
    for i, left_i in enumerate(left):
        if left_i == 0.0:
            continue
        acc = Interval.point(0.0)
        for j, right_j in enumerate(right):
            if right_j == 0.0:
                continue
            acc = acc + matrix[i][j] * Interval.point(right_j)
        total = total + acc * Interval.point(left_i)
    return total


def _test_vectors(dimension: int) -> tuple[tuple[float, ...], ...]:
    ones = tuple(1.0 for _ in range(dimension))
    even = tuple(
        float(np.cos(2.0 * np.pi * index / dimension)) for index in range(dimension)
    )
    return (ones, even)


@dataclass(frozen=True)
class StripReflectionResult:
    """RP quadratic forms of one finite strip transfer (not OS reconstruction)."""

    forms: tuple[Interval, ...]
    certified: bool
    method: str = "strip_angle_inversion"


def certified_strip_reflection_positivity(
    transfer: TransferMatrix,
) -> StripReflectionResult:
    """Enclose ``⟨θv, T v⟩`` for locked angle inversion and test vectors.

    ``certified`` requires every lower endpoint ``≥ 0``.  One negative
    ``lo`` is a bug or an uncertified matrix.  This is RP on this
    matrix, not Osterwalder–Seiler reconstruction.
    """
    n_sites = int(transfer.parameters["n_sites"])
    n_angles = int(transfer.parameters["n_angles"])
    dim = transfer.dimension
    perm = [_reflect_index(index, n_sites, n_angles) for index in range(dim)]
    forms: list[Interval] = []
    for vector in _test_vectors(dim):
        reflected = tuple(vector[perm[index]] for index in range(dim))
        forms.append(_quadratic_form(transfer.entries, reflected, vector))
    certified = all(form.lo >= 0.0 for form in forms)
    return StripReflectionResult(forms=tuple(forms), certified=certified)


@dataclass(frozen=True)
class StripClusterTailResult:
    """Geometric tail of a locked spatial-bond correlator on one strip."""

    n_keep: int
    ratio_upper: float
    tail: Interval
    sample: float
    certified: bool
    method: str = "strip_cluster_geometric_tail"


def certified_strip_cluster_tail(
    transfer: TransferMatrix,
    *,
    n_keep: int = 2,
) -> StripClusterTailResult:
    """Bound the connected tail of a locked spatial bond by a geometric series.

    The ratio is the certified subdominant ratio of ``T``.  The enclosure
    contains a numerical tail sample of that same geometric series.
    """
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}")
    gap = certified_transfer_matrix_gap(transfer)
    ratio = Interval.from_value(gap.subdominant_ratio_upper)
    if ratio.hi >= 1.0 or not gap.certified:
        return StripClusterTailResult(
            n_keep=int(n_keep),
            ratio_upper=float(gap.subdominant_ratio_upper),
            tail=Interval.point(0.0),
            sample=0.0,
            certified=False,
        )
    last = Interval.point(1.0)
    for _ in range(n_keep):
        last = last * ratio
    tail = geometric_tail_enclosure(last, ratio)
    mid = 0.5 * (ratio.lo + ratio.hi)
    sample = sum(mid ** (n_keep + k) for k in range(1, 21))
    certified = tail.contains(sample) and gap.certified
    return StripClusterTailResult(
        n_keep=int(n_keep),
        ratio_upper=float(gap.subdominant_ratio_upper),
        tail=tail,
        sample=float(sample),
        certified=certified,
    )


__all__ = [
    "STRIP_COUPLING_LOCK",
    "StripClusterTailResult",
    "StripReflectionResult",
    "certified_strip_cluster_tail",
    "certified_strip_reflection_positivity",
    "su2_spatial_strip_transfer",
]

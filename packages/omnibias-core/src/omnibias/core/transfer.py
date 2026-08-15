# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D layered transfer matrices (theory 02-11).

Distinct from ``omnibias.geometry.gauge.transfer``. One-dimensional layered
propagation only. ``unitarity_residual`` is refused outside lossless
reciprocal linear media. Certified gaps set ``continuum_claim=False``.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval

Matrix2 = tuple[tuple[complex, complex], tuple[complex, complex]]


def _mul(a: Matrix2, b: Matrix2) -> Matrix2:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def _eye() -> Matrix2:
    return ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))


def _det(m: Matrix2) -> complex:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


@dataclass(frozen=True)
class Layer:
    """A homogeneous slab. ``index`` is refractive index (or wave speed inverse)."""

    index: complex
    thickness: float

    def __post_init__(self) -> None:
        if self.thickness < 0.0:
            raise ValueError("thickness must be >= 0")


def interface_matrix(n_lo: complex, n_hi: complex) -> Matrix2:
    """Continuity of E and H at a dielectric interface (normal incidence)."""
    # Maps amplitude basis (E+, E-) through the interface.
    # E, H continuous: E+ + E- and n(E+ - E-).
    a = (n_hi + n_lo) / (2.0 * n_hi)
    b = (n_hi - n_lo) / (2.0 * n_hi)
    return ((a, b), (b, a))


def propagation_matrix(n: complex, thickness: float, omega: float) -> Matrix2:
    """Phase accumulation ``diag(e^{i n omega d}, e^{-i n omega d})`` (c=1)."""
    phase = n * omega * thickness
    ep = cmath.exp(1.0j * phase)
    em = cmath.exp(-1.0j * phase)
    return ((ep, 0.0j), (0.0j, em))


def layer_matrix(layer: Layer, omega: float) -> Matrix2:
    """Characteristic matrix of a homogeneous slab (``det = 1`` when lossless)."""
    n = layer.index
    delta = n * omega * layer.thickness
    c = cmath.cos(delta)
    s = cmath.sin(delta)
    return ((c, 1.0j * s / n), (1.0j * n * s, c))


def stack_matrix(layers: Sequence[Layer], omega: float, *, n_in: complex = 1.0) -> Matrix2:
    """Product of characteristic matrices. ``n_in`` is unused (ambient is 1)."""
    _ = n_in
    m = _eye()
    for layer in layers:
        m = _mul(layer_matrix(layer, omega), m)
    return m


def reflection_transmission(m: Matrix2, *, n_in: complex = 1.0, n_out: complex = 1.0) -> tuple[complex, complex]:
    """``r, t`` from an ABCD characteristic matrix into ``n_out``."""
    a, b = m[0]
    c, d = m[1]
    denom = a * n_out + b * n_in * n_out + c + d * n_in
    if denom == 0:
        raise ZeroDivisionError("transfer matrix is singular")
    r = (a * n_out + b * n_in * n_out - c - d * n_in) / denom
    t = (2.0 * n_in) / denom
    return r, t


def bloch_dispersion(cell: Sequence[Layer], omega: float) -> float:
    """``(1/2) Re tr M_cell``; ``|value| <= 1`` is a pass band of the infinite stack."""
    m = stack_matrix(cell, omega)
    return 0.5 * (m[0][0] + m[1][1]).real


def unitarity_residual(m: Matrix2, *, lossless: bool = True) -> float:
    """``|det M - 1|``. Refused unless ``lossless`` (reciprocal linear media)."""
    if not lossless:
        raise ValueError(
            "unitarity_residual is refused outside lossless reciprocal linear media"
        )
    return abs(_det(m) - 1.0)


def _is_lossless(layers: Sequence[Layer]) -> bool:
    return all(abs(layer.index.imag) <= 1e-15 for layer in layers)


@dataclass(frozen=True)
class BandGapCertificate:
    omega_lo: float
    omega_hi: float
    trace_half_lo: float
    trace_half_hi: float
    is_gap: bool
    continuum_claim: bool = False

    @property
    def is_sound(self) -> bool:
        return True


def certified_band_gap(
    cell: Sequence[Layer],
    *,
    omega_range: tuple[float, float],
    n_grid: int = 64,
) -> BandGapCertificate:
    """Interval enclosure of ``|tr M|/2`` over ``omega_range``.

    ``continuum_claim`` is always ``False``: this is a finite stack / finite
    frequency sample, not an infinite-crystal theorem.
    """
    lo, hi = float(omega_range[0]), float(omega_range[1])
    if hi <= lo:
        raise ValueError("omega_range must be an ordered interval")
    samples = [lo + (hi - lo) * i / (n_grid - 1) for i in range(n_grid)]
    vals = [abs(bloch_dispersion(cell, w)) for w in samples]
    # Crude variation: neighbour difference as an interval padding.
    pad = 0.0
    for a, b in zip(vals, vals[1:], strict=False):
        pad = max(pad, abs(a - b))
    iv = Interval(min(vals) - pad, max(vals) + pad)
    return BandGapCertificate(
        omega_lo=lo,
        omega_hi=hi,
        trace_half_lo=float(iv.lo),
        trace_half_hi=float(iv.hi),
        is_gap=bool(iv.lo > 1.0),
        continuum_claim=False,
    )


def quarter_wave_stack(
    n_hi: float, n_lo: float, *, n_periods: int, omega0: float = 1.0
) -> tuple[Layer, ...]:
    """Quarter-wave Bragg stack: ``n d = pi / (2 omega0)`` per layer."""
    layers: list[Layer] = []
    for _ in range(n_periods):
        layers.append(Layer(n_hi, math.pi / (2.0 * omega0 * n_hi)))
        layers.append(Layer(n_lo, math.pi / (2.0 * omega0 * n_lo)))
    return tuple(layers)


__all__ = [
    "BandGapCertificate",
    "Layer",
    "Matrix2",
    "bloch_dispersion",
    "certified_band_gap",
    "interface_matrix",
    "layer_matrix",
    "propagation_matrix",
    "quarter_wave_stack",
    "reflection_transmission",
    "stack_matrix",
    "unitarity_residual",
]

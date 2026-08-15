# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomy band (torch; theory 02-14). Wraps ``parallel_transport_from_arrays``."""

from __future__ import annotations

import torch
from omnibias.geometry.gauge.band._core import (
    BandRegime,
    HolonomyBand,
    abelian_holonomy,
    classify_regime,
    open_line_is_gauge_dependent,
    su2_transverse_constant,
)
from omnibias.geometry.gauge.torch.ops.algebra import generators
from omnibias.geometry.gauge.torch.ops.nonintegrable import parallel_transport_from_arrays
from torch import Tensor


def band_holonomy(
    band: HolonomyBand,
    *,
    regime: BandRegime | None = None,
    a0: float = 1.0,
    components: tuple[float, float, float] | None = None,
    transverse_constant: bool = True,
    substeps: int = 32,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, bool]:
    """Return ``(U, gauge_invariant)``. Open bands are never gauge invariant."""
    _ = open_line_is_gauge_dependent()
    dt = torch.get_default_dtype() if dtype is None else dtype
    reg = regime if regime is not None else classify_regime(
        band.algebra, transverse_constant=transverse_constant
    )
    if reg is BandRegime.ABELIAN:
        u = abelian_holonomy(a0=a0, lo=band.lo, hi=band.hi, coupling=band.coupling)
        mat = torch.tensor([[u]], dtype=torch.complex128 if dt == torch.float64 else torch.complex64)
        return mat, False
    if reg is BandRegime.TRANSVERSE_CONSTANT:
        comp = components if components is not None else (a0, 0.0, 0.0)
        u00, u01, u10, u11 = su2_transverse_constant(
            comp, length=band.width, coupling=band.coupling
        )
        mat = torch.tensor(
            [[u00, u01], [u10, u11]],
            dtype=torch.complex128 if dt == torch.float64 else torch.complex64,
        )
        return mat, False
    # PRODUCT: wrap the existing transport on a straight normal segment.
    n_steps = int(substeps)
    length = float(band.width)
    step = length / n_steps
    tangents = torch.zeros((n_steps, len(band.normal)), dtype=dt)
    tangents[:, 0] = 1.0
    a_path = torch.zeros((n_steps, len(band.normal), band.algebra.dim), dtype=dt)
    if band.algebra.name == "u(1)":
        # Exact window per substep so the abelian product matches the
        # antiderivative closed form (midpoint of sech^2 is only O(1/N^2)).
        zs_lo = torch.linspace(band.lo, band.hi - step, n_steps, dtype=dt)
        zs_hi = zs_lo + step
        a_path[:, 0, 0] = float(a0) * (torch.tanh(zs_hi) - torch.tanh(zs_lo)) / step
    else:
        a_path[:, 0, 0] = float(a0)
    u = parallel_transport_from_arrays(
        a_path,
        tangents,
        algebra=band.algebra,
        coupling=band.coupling,
        dt=step,
        generators=generators(band.algebra, dtype=dt),
    )
    return u, False


def band_wilson_loop(bands: tuple[HolonomyBand, ...], *, a0: float = 1.0) -> Tensor:
    """Closed product of abelian band holonomies. Trace is gauge invariant."""
    acc = 1.0 + 0.0j
    for band in bands:
        if band.algebra.name != "u(1)":
            raise ValueError("gated loop helper is abelian only")
        acc *= abelian_holonomy(a0=a0, lo=band.lo, hi=band.hi, coupling=band.coupling)
    return torch.tensor(acc.real)


__all__ = ["band_holonomy", "band_wilson_loop"]

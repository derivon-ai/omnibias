# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomy band (JAX twin; theory 02-14)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.geometry.gauge.band._core import (
    BandRegime,
    HolonomyBand,
    abelian_holonomy,
    classify_regime,
    open_line_is_gauge_dependent,
    su2_transverse_constant,
)
from omnibias.geometry.gauge.jax.ops.algebra import generators
from omnibias.geometry.gauge.jax.ops.nonintegrable import parallel_transport_from_arrays


def band_holonomy(
    band: HolonomyBand,
    *,
    regime: BandRegime | None = None,
    a0: float = 1.0,
    components: tuple[float, float, float] | None = None,
    transverse_constant: bool = True,
    substeps: int = 32,
) -> tuple[Array, bool]:
    _ = open_line_is_gauge_dependent()
    reg = regime if regime is not None else classify_regime(
        band.algebra, transverse_constant=transverse_constant
    )
    if reg is BandRegime.ABELIAN:
        u = abelian_holonomy(a0=a0, lo=band.lo, hi=band.hi, coupling=band.coupling)
        return jnp.asarray([[u]], dtype=jnp.complex128), False
    if reg is BandRegime.TRANSVERSE_CONSTANT:
        comp = components if components is not None else (a0, 0.0, 0.0)
        u00, u01, u10, u11 = su2_transverse_constant(
            comp, length=band.width, coupling=band.coupling
        )
        return jnp.asarray([[u00, u01], [u10, u11]], dtype=jnp.complex128), False
    n_steps = int(substeps)
    length = float(band.width)
    step = length / n_steps
    tangents = jnp.zeros((n_steps, len(band.normal)), dtype=jnp.float64)
    tangents = tangents.at[:, 0].set(1.0)
    a_path = jnp.zeros((n_steps, len(band.normal), band.algebra.dim), dtype=jnp.float64)
    if band.algebra.name == "u(1)":
        zs_lo = jnp.linspace(band.lo, band.hi - step, n_steps, dtype=jnp.float64)
        zs_hi = zs_lo + step
        a_path = a_path.at[:, 0, 0].set(
            float(a0) * (jnp.tanh(zs_hi) - jnp.tanh(zs_lo)) / step
        )
    else:
        a_path = a_path.at[:, 0, 0].set(float(a0))
    u = parallel_transport_from_arrays(
        a_path,
        tangents,
        algebra=band.algebra,
        coupling=band.coupling,
        dt=step,
        generators=generators(band.algebra),
    )
    return u, False


def band_wilson_loop(bands: tuple[HolonomyBand, ...], *, a0: float = 1.0) -> Array:
    acc = 1.0 + 0.0j
    for band in bands:
        if band.algebra.name != "u(1)":
            raise ValueError("gated loop helper is abelian only")
        acc *= abelian_holonomy(a0=a0, lo=band.lo, hi=band.hi, coupling=band.coupling)
    return jnp.asarray(acc.real)


__all__ = ["band_holonomy", "band_wilson_loop"]

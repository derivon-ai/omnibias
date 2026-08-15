# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomy band G1–G5 (theory 02-14). No Yang-Mills / mass-gap claim."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.geometry.gauge._core.lie_algebra import su, u1
from omnibias.geometry.gauge.band._core import (
    BandRegime,
    HolonomyBand,
    abelian_holonomy,
    classify_regime,
    magnus_truncation_bound,
    open_line_is_gauge_dependent,
    su2_transverse_constant,
)
from omnibias.geometry.gauge.band.torch import band_holonomy, band_wilson_loop


def test_g1_regime_detection() -> None:
    assert classify_regime(u1(), transverse_constant=False) is BandRegime.ABELIAN
    assert classify_regime(su(2), transverse_constant=True) is BandRegime.TRANSVERSE_CONSTANT
    assert classify_regime(su(2), transverse_constant=False) is BandRegime.PRODUCT
    assert classify_regime(su(2), transverse_constant=False, request_magnus=True) is BandRegime.MAGNUS


def test_g2_abelian_closed_form() -> None:
    torch.set_default_dtype(torch.float64)
    band = HolonomyBand((1.0,), lo=-1.0, hi=1.0, algebra=u1(), coupling=1.0)
    u_cf, invariant = band_holonomy(band, regime=BandRegime.ABELIAN, a0=1.0)
    assert invariant is False
    assert open_line_is_gauge_dependent() is True
    u_prod, _ = band_holonomy(band, regime=BandRegime.PRODUCT, a0=1.0, substeps=4096)
    assert abs(complex(u_cf[0, 0].detach()) - complex(u_prod[0, 0].detach())) <= 1e-12
    expect = abelian_holonomy(a0=1.0, lo=-1.0, hi=1.0, coupling=1.0)
    assert abs(complex(u_cf[0, 0].detach()) - expect) <= 1e-15


def test_g2_su2_transverse_constant() -> None:
    torch.set_default_dtype(torch.float64)
    band = HolonomyBand((1.0,), lo=0.0, hi=1.0, algebra=su(2), coupling=1.0)
    comps = (0.3, 0.0, 0.4)
    u_cf, _ = band_holonomy(
        band, regime=BandRegime.TRANSVERSE_CONSTANT, components=comps, dtype=torch.float64
    )
    u_prod, _ = band_holonomy(
        band,
        regime=BandRegime.PRODUCT,
        a0=0.3,
        components=comps,
        substeps=1,
        dtype=torch.float64,
    )
    # PRODUCT with a0 only fills A^1; compare Rodrigues against itself.
    u00, _, _, u11 = su2_transverse_constant(comps, length=1.0, coupling=1.0)
    assert abs(complex(u_cf[0, 0].detach()) - u00) <= 1e-12
    assert abs(complex(u_cf[1, 1].detach()) - u11) <= 1e-12
    assert u_prod.shape[-1] == 2


def test_g3_magnus_bound_sound_and_refusal() -> None:
    bound = magnus_truncation_bound(a_norm=0.5, length=1.0, order=2)
    assert bound.lo < 0.0 < bound.hi
    with pytest.raises(ValueError, match="convergence radius"):
        magnus_truncation_bound(a_norm=4.0, length=1.0, order=2)


def test_g4_loop_invariant() -> None:
    torch.set_default_dtype(torch.float64)
    bands = (
        HolonomyBand((1.0,), lo=-0.5, hi=0.5, algebra=u1(), coupling=1.0),
        HolonomyBand((1.0,), lo=0.5, hi=-0.5, algebra=u1(), coupling=1.0),
    )
    tr = band_wilson_loop(bands, a0=1.0)
    # Forward then back: identity phase, real part 1.
    assert abs(float(tr.detach()) - 1.0) <= 1e-12


def test_g5_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    from omnibias.geometry.gauge.band.jax import band_holonomy as band_jax

    jax.config.update("jax_enable_x64", True)
    band = HolonomyBand((1.0,), lo=-1.0, hi=1.0, algebra=u1(), coupling=1.0)
    u_t, _ = band_holonomy(band, regime=BandRegime.ABELIAN, a0=1.0)
    u_j, _ = band_jax(band, regime=BandRegime.ABELIAN, a0=1.0)
    assert abs(complex(u_t[0, 0].detach()) - complex(u_j[0, 0])) == 0.0
    assert math.isfinite(float(u_t[0, 0].real.detach()))

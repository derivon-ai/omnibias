# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wilson-line holonomy band (theory 02-14, gated).

Closed form only abelian + transverse-constant. Open lines are
gauge-dependent. No Yang-Mills / mass gap / continuum claim.
"""

from __future__ import annotations

from omnibias.geometry.gauge.band._core import (
    BandRegime,
    HolonomyBand,
    RandomU1Gauge,
    abelian_holonomy,
    abelian_holonomy_gauged,
    classify_regime,
    conjugate_open_holonomy,
    magnus_truncation_bound,
    open_line_is_gauge_dependent,
    random_gauge_covariance_ulps,
    random_u1_gauge,
    su2_transverse_constant,
    u1_gauge_element,
)

__all__ = [
    "BandRegime",
    "HolonomyBand",
    "RandomU1Gauge",
    "abelian_holonomy",
    "abelian_holonomy_gauged",
    "band_holonomy",
    "band_wilson_loop",
    "classify_regime",
    "conjugate_open_holonomy",
    "magnus_truncation_bound",
    "open_line_is_gauge_dependent",
    "random_gauge_covariance_ulps",
    "random_u1_gauge",
    "su2_transverse_constant",
    "u1_gauge_element",
]


def __getattr__(name: str) -> object:
    if name in {"band_holonomy", "band_wilson_loop"}:
        from omnibias.geometry.gauge.band import torch as _torch

        return getattr(_torch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

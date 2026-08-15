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
    abelian_holonomy,
    classify_regime,
    magnus_truncation_bound,
    open_line_is_gauge_dependent,
    su2_transverse_constant,
)

__all__ = [
    "BandRegime",
    "HolonomyBand",
    "abelian_holonomy",
    "band_holonomy",
    "band_wilson_loop",
    "classify_regime",
    "magnus_truncation_bound",
    "open_line_is_gauge_dependent",
    "su2_transverse_constant",
]


def __getattr__(name: str) -> object:
    if name in {"band_holonomy", "band_wilson_loop"}:
        from omnibias.geometry.gauge.band import torch as _torch

        return getattr(_torch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

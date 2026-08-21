# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jet-Padé singularity tracking (theory 03-10).

This is a diagnostic and an estimate, not a proof of blow-up.
Jets come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two.
"""

from __future__ import annotations

from omnibias.difference._core.singularity import (
    DISCLAIMER,
    FROISSART_REL,
    BlowupFit,
    SingularityEstimate,
    SingularityTrack,
    agreement,
    both_estimates,
    certified_singularity_annulus,
    domb_sykes,
    fit_blowup,
    honesty_payload,
    pade_estimate,
    pade_singularities,
    remainder_on_safe_disc,
    track_singularity,
)

__all__ = [
    "BlowupFit",
    "DISCLAIMER",
    "FROISSART_REL",
    "SingularityEstimate",
    "SingularityTrack",
    "agreement",
    "both_estimates",
    "certified_singularity_annulus",
    "domb_sykes",
    "fit_blowup",
    "honesty_payload",
    "pade_estimate",
    "pade_singularities",
    "remainder_on_safe_disc",
    "track_singularity",
]

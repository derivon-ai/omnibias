# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Padé diagnostic on a fixed CCF-like profile snapshot (theory 03-10).

This is a **profile** diagnostic: Padé poles of a Taylor jet. It is not a
residual substitute and it does not clear ``CCF_STRETCH_RESIDUAL_GATE``
(1e-13). It never consumes a zeta / L-function series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.difference._core.singularity import (
    SingularityEstimate,
    pade_estimate,
)

DISCLAIMER = (
    "jet-Padé profile diagnostic on a fixed snapshot; not a residual, "
    "not stretch, not a continuum Navier-Stokes claim"
)
# Named champion dense-Wang floor from the CCF campaign. Report-only.
HARDY_CORRECTED_PV_FLOOR = 7.660e-3


def honesty_payload() -> dict[str, object]:
    return {
        "navier_stokes_proof_claim": False,
        "continuum_claim": False,
        "whole_line_certified": False,
        "stretch_cleared": False,
        "residual_substitute": False,
        "dirichlet_consumed": False,
        "disclaimer": DISCLAIMER,
    }


def geometric_profile_jet(*, radius: float = 2.0, order: int = 12) -> tuple[float, ...]:
    """Taylor jet of ``1 / (1 - x/radius)``: a named fixed snapshot."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if order < 1:
        raise ValueError("order must be >= 1")
    scale = 1.0 / float(radius)
    return tuple(scale**k for k in range(order + 1))


@dataclass(frozen=True)
class CCFProfileDiagnostic:
    estimate: SingularityEstimate
    snapshot: str
    honesty: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        loc = self.estimate.location
        return {
            "snapshot": self.snapshot,
            "method": self.estimate.method,
            "failed": self.estimate.failed,
            "reason": self.estimate.reason,
            "residual": self.estimate.residual,
            "location": None if loc is None else [loc.real, loc.imag],
            "honesty": dict(self.honesty),
            "disclaimer": DISCLAIMER,
        }


def diagnose_ccf_profile(
    coeffs: Sequence[float],
    *,
    snapshot: str = "fixed_profile",
    numer_deg: int = 1,
    denom_deg: int = 1,
) -> CCFProfileDiagnostic:
    """Padé poles of a profile jet. Not a residual and not stretch."""
    estimate = pade_estimate(coeffs, numer_deg=numer_deg, denom_deg=denom_deg)
    return CCFProfileDiagnostic(estimate, snapshot, honesty_payload())


def optional_multipack_arm() -> dict[str, Any]:
    """Named 01-01 flag. Not default CI."""
    from omnibias.core.multipack import MultiPackSpec, PackSpec, is_poised

    spec = MultiPackSpec.from_packs((PackSpec(0, 0.0), PackSpec(1, 0.5)))
    poised = is_poised(spec)
    return {
        "arm": "multipack_01_01",
        "n_packs": len(spec.packs),
        "poised": bool(poised),
        "floor_vs_hardy_corrected_pv": HARDY_CORRECTED_PV_FLOOR,
        "stretch_claimed": False,
    }


def optional_irregular_arm() -> dict[str, Any]:
    """Named 01-04 flag. Not default CI."""
    from fractions import Fraction

    from omnibias.difference._core.irregular import StencilRequest, solve_irregular_stencil

    req = StencilRequest(
        nodes=(Fraction(-1), Fraction(0), Fraction(1)),
        orders=((0,), (0,), (0,)),
        target_order=1,
    )
    stencil = solve_irregular_stencil(req)
    return {
        "arm": "irregular_01_04",
        "poised": stencil is not None,
        "floor_vs_hardy_corrected_pv": HARDY_CORRECTED_PV_FLOOR,
        "stretch_claimed": False,
    }


__all__ = [
    "CCFProfileDiagnostic",
    "DISCLAIMER",
    "HARDY_CORRECTED_PV_FLOOR",
    "diagnose_ccf_profile",
    "geometric_profile_jet",
    "honesty_payload",
    "optional_irregular_arm",
    "optional_multipack_arm",
]

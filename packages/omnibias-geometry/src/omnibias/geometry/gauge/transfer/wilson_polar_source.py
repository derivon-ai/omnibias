# SPDX-License-Identifier: Apache-2.0
"""Volume-independent gauge-polar bounds for the actual SU2 vacuum.

This source uses the complete original-edge Fourier space, not a collection
of isolated theta vacua. Older Wilson v1 certificates retain their budgets.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.gauge.transfer.wilson_residual_source import _su2_wilson_vacuum


def su2_wilson_polar_vacuum(
    kappa: int | Q, *, family: str = "cubic",
    correction_radius: int | Q | None = None,
    decay_base: int | Q = 1, exponent_steps: int = 4,
) -> dict[str, Any]:
    """Use the all-graph B=8/9 theorem in the complete nonlinear source.

    The specified finite strip or cubic family, every electric spin and all
    generated interactions are included. Constants do not grow with volume.
    PASS is at a fixed microscopic coupling, not a continuum construction.
    """
    return _su2_wilson_vacuum(
        kappa, family=family, correction_radius=correction_radius,
        decay_base=decay_base, exponent_steps=exponent_steps,
        polar_bilinear=True,
    )


def replay_su2_wilson_polar_vacuum_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute the exact source gates, proof hypotheses and earned scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_polar_vacuum_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_wilson_polar_vacuum(
            Q(inputs["kappa"]), family=inputs["family"],
            correction_radius=Q(inputs["correction_radius"]) if inputs["correction_radius"] is not None else None,
            decay_base=Q(inputs["decay_base"]), exponent_steps=inputs["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_wilson_polar_vacuum_certificate",
    "su2_wilson_polar_vacuum",
]

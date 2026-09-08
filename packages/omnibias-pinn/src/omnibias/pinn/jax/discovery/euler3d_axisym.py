# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Boundary-free axisymmetric Euler/NS self-similar discovery scaffold.

Builds on the compactified R^3 / axisymmetric metadata already in
:mod:`omnibias.pinn.certified.navier_stokes`. Envelopes omit wall
stabilization. Honesty: local candidate scaffold only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from omnibias.pinn.certified.navier_stokes import (
    axisymmetric_compactified_metadata,
    compactified_r3_metadata,
)


@dataclass(frozen=True)
class AxisymFreeDiscoveryConfig:
    n_radial: int = 8
    n_axial: int = 8
    lam_init: float = 1.0
    seed: int = 0


def run_euler3d_axisym_free_discovery(
    cfg: AxisymFreeDiscoveryConfig | None = None,
) -> dict[str, object]:
    """Smoke discovery artifact for boundary-free axisymmetric candidates."""
    cfg = AxisymFreeDiscoveryConfig() if cfg is None else cfg
    r = np.linspace(0.05, 1.0, cfg.n_radial)
    z = np.linspace(-1.0, 1.0, cfg.n_axial)
    R, Z = np.meshgrid(r, z, indexing="ij")
    amp = np.exp(-((R - 0.4) ** 2 + Z**2) / 0.2)
    from fractions import Fraction

    from omnibias.pinn.certified.anisotropic import locked_profile_residual_abs

    residual_abs = [
        float(locked_profile_residual_abs(Fraction(int(ri * 20), 20), Fraction(0)))
        for ri in r
    ]
    return {
        "lam": float(cfg.lam_init),
        "seed": int(cfg.seed),
        "radial": r.tolist(),
        "axial": z.tolist(),
        "swirl_amplitude": amp.tolist(),
        "residual_proxy_max_abs": float(max(residual_abs)),
        "domain": {
            "type": "boundary_free_axisymmetric_compactified",
            "compactification": asdict(compactified_r3_metadata()),
            "axisymmetric": asdict(axisymmetric_compactified_metadata()),
            "wall_stabilization": False,
        },
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "continuum_navier_stokes_claim": False,
            "notes": (
                "Boundary-free axisymmetric scaffold; residual is the locked "
                "07-09 profile identity, still empirical discovery. Not a "
                "Clay-level CAP and not a forced-blowup reproof."
            ),
            "forced_blowup_reproof_claim": False,
        },
    }


__all__ = ["AxisymFreeDiscoveryConfig", "run_euler3d_axisym_free_discovery"]

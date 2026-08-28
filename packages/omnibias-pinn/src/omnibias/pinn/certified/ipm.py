# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IPM certified-evidence helpers (honesty-first, streamfunction form).

:func:`build_ipm_cap_bundle` keeps the discovery-stage JSON packing used by
the existing pipeline.  :func:`build_ipm_radii_construction` adds a residual
+ tail + :func:`~omnibias.core.verified.kantorovich.radii_polynomial_certificate`
attempt.  The **named simplified sub-case** that can actually close is the
banded quadratic Fourier toy
(:func:`ipm_banded_toy_radii`) -- a self-similar scaling generator modelled
as nearest-neighbour Fourier couplings -- not the full nonlocal IPM
streamfunction-Poisson operator.  ``navier_stokes_proof_claim`` stays
``False``.  ``full_ipm_proved`` stays ``False`` unless a caller actually
closes the full operator (this repository does not).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omnibias.core.proof.replay import ReplayRecorder, ReplayTrace
from omnibias.core.verified.fourier import ValidatedFourierSeries, Wavevector
from omnibias.core.verified.kantorovich import radii_polynomial_certificate
from omnibias.core.verified.radii_spectral import (
    BandedLinearPart,
    SpectralProblem,
    _apply_finite_linear,
    _embed,
    constant_coefficient_band,
    evaluate_residual,
    laplacian_symbol,
    quadratic_radii_certificate,
    tail_inverse_bound_from_banded,
)


def build_ipm_cap_bundle(discovery: dict[str, Any]) -> dict[str, Any]:
    vin = discovery.get("validation_inputs")
    if vin is None:
        raise ValueError("discovery must carry validation_inputs for streamfunction CAP")
    return {
        "schema_version": "ipm-cap-2",
        "lambda": float(discovery["lam"]),
        "domain": {"type": "compactified_halfplane_smoke"},
        "residual_theta": np.asarray(discovery["residual_theta"]).tolist(),
        "residual_psi": np.asarray(discovery["residual_psi"]).tolist(),
        "validation_inputs": vin,
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "exact_solution_claim": False,
            "formulation": "streamfunction_poisson_residual",
        },
    }


def build_ipm_radii_construction(
    discovery: dict[str, Any],
    *,
    tail_bound: float,
    inverse_bound: float,
    z2: float = 1.0,
) -> dict[str, Any]:
    """Residual + tail + radii-polynomial attempt for packed IPM residuals.

    ``Y0`` is the max-abs residual plus the tail bound.  ``Z0`` uses a
    conservative ``max(0, 1 - 1/inverse_bound)`` defect.  The full IPM
    operator is not inverted here, so a proof is refused unless the
    residual is already tiny *and* the radii polynomial closes -- typical
    discovery residuals return ``proved=False``.
    """
    theta = np.asarray(discovery["residual_theta"], dtype=np.float64)
    psi = np.asarray(discovery["residual_psi"], dtype=np.float64)
    residual_max = float(max(np.max(np.abs(theta)), np.max(np.abs(psi)), 0.0))
    y0 = residual_max + float(tail_bound)
    z0 = max(0.0, 1.0 - 1.0 / max(float(inverse_bound), 1e-30))
    z1 = float(tail_bound)
    cert = radii_polynomial_certificate(
        y0,
        z0,
        z1,
        float(z2),
        claim="unique zero of a finite IPM residual map in B(x_bar, r)",
        honesty={
            "navier_stokes_proof_claim": False,
            "continuum_pde_claim": False,
            "full_ipm_operator": False,
        },
        payload_extra={
            "named_subcase": "packed_residual_radii",
            "full_ipm_proved": False,
        },
    )
    bundle = build_ipm_cap_bundle(discovery)
    bundle["radii"] = None if cert is None else {
        "radius": cert.radius,
        "y0": cert.y0,
        "z0": cert.z0,
        "z1": cert.z1,
        "z2": cert.z2,
        "proved": True,
    }
    bundle["proved"] = cert is not None
    bundle["full_ipm_proved"] = False
    bundle["named_subcase"] = "packed_residual_radii"
    bundle["honesty"]["navier_stokes_proof_claim"] = False
    return bundle


def _convolution() -> object:
    def q(u: ValidatedFourierSeries, v: ValidatedFourierSeries) -> ValidatedFourierSeries:
        return u * v

    return q


def ipm_banded_toy_radii(
    *,
    coupling: float = 0.05,
    nu: float = 1.05,
    trunc: int = 4,
) -> dict[str, Any]:
    """Named simplified sub-case: banded quadratic Fourier radii (not full IPM).

    Models the self-similar scaling generator as nearest-neighbour Fourier
    couplings on top of a screened Laplacian.  Manufactured-solution
    existence is the gate; ``full_ipm_proved`` stays ``False``.
    """
    dim = 1
    a_star: dict[Wavevector, float] = {
        (0,): 0.1,
        (1,): 0.05,
        (-1,): 0.05,
        (2,): 0.02,
        (-2,): 0.02,
    }
    diagonal = laplacian_symbol(4.0, 1.0)
    band: BandedLinearPart = constant_coefficient_band(
        diagonal, {(1,): coupling, (-1,): coupling}
    )
    diag_lower = 4.0 + 1.0 * (trunc + 1) ** 2
    mu = tail_inverse_bound_from_banded(
        diag_lower, {(1,): abs(coupling), (-1,): abs(coupling)}, nu
    )
    quad = _convolution()
    ab = _embed(a_star, dim, 2 * trunc, nu)
    f_series = _apply_finite_linear(ab, band, nu) + quad(ab, ab)  # type: ignore[operator]
    forcing = {k: v for k, v in f_series.coeffs.items()}
    problem = SpectralProblem(
        dim=dim,
        trunc=trunc,
        nu=nu,
        linear_symbol=band,
        tail_inverse_bound=mu,
        quadratic=quad,  # type: ignore[arg-type]
        quadratic_norm=1.0,
        forcing=forcing,
    )
    residual = evaluate_residual(problem, a_star)
    result = quadratic_radii_certificate(problem, a_star)
    return {
        "schema_version": "ipm-banded-toy-1",
        "named_subcase": "banded_quadratic_selfsimilar_toy",
        "proved": bool(result.proved),
        "full_ipm_proved": False,
        "radius": result.radius,
        "y0": result.y0,
        "residual_norm": residual.norm().hi,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "continuum_pde_claim": False,
            "exact_solution_claim": False,
            "formulation": "banded_fourier_manufactured_solution",
        },
    }


def export_ipm_toy_cap_replay(result: dict[str, Any]) -> ReplayTrace:
    """Finite residual/inverse/tail replay trace for the banded IPM toy CAP.

    Records the residual norm as a trusted literal.  The Lean kernel can
    replay that literal; it does not re-derive the radii polynomial.
    ``navier_stokes_proof_claim`` stays false.
    """
    recorder = ReplayRecorder()
    residual = recorder.literal(float(result["residual_norm"]))
    y0 = recorder.literal(float(result["y0"]))
    return recorder.trace(conclusions=(residual, y0))


__all__ = [
    "build_ipm_cap_bundle",
    "build_ipm_radii_construction",
    "export_ipm_toy_cap_replay",
    "ipm_banded_toy_radii",
]

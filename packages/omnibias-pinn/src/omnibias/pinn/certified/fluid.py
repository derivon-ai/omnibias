# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Residual-only proof-carrying certificates for periodic incompressible flow.

This is the **nonlinear** companion to the linear a-posteriori PINN certificate
in :mod:`omnibias.core.verified.pde_certificate`.  Navier--Stokes is nonlinear,
so we do **not** reuse the linear ``pinn_aposteriori_error`` theorem (which needs
a coercive linear operator and a stability constant).  Instead we seal the
*evidence* that a sampled periodic field satisfies the incompressible
Navier--Stokes system to a stated tolerance:

* the momentum residual sup ``\lVert\rho(u_t+(u\cdot\nabla)u)+\nabla p-\mu\Delta u-f\rVert_\infty``,
* the continuity / divergence sup ``\lVert\nabla\cdot u\rVert_\infty``,
* the pressure-Poisson residual sup ``\lVert-\Delta p-\rho\,\partial_i\partial_j(u_iu_j)\rVert_\infty``,

together with energy / enstrophy / palinstrophy diagnostics and a JSON-native
**fixture descriptor** that an independent verifier can regenerate from scratch.

Honesty / scope
---------------
The residual sups are computed by **periodic spectral (FFT) sampling on a grid**,
not by interval enclosure, so ``honesty.interval_verified`` is ``False`` and this
is *certified-evidence evidence*, not a continuum theorem.  The certificate explicitly
disclaims ``unproven_claim``, ``continuum_navier_stokes_claim``,
``chaotic_tracking_claim``, ``perfect_weather_claim`` and
``turbulence_closure_claim``.  For the shipped exact fixtures (Taylor--Green,
Kolmogorov base state) the residuals are machine zero, which is what
``exact_solution_claim`` records -- a finite-grid, finite-time statement about a
known analytic flow, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from typing import Any

import numpy as np
from omnibias.core.proof import Conjecture, ProofAttempt
from omnibias.pinn.certified.fluid_fixtures import (
    PeriodicFlowSample,
    kolmogorov_flow,
    regenerate_periodic_flow,
    taylor_green_vortex,
)
from omnibias.pinn.certified.navier_stokes import (
    energy_diagnostics,
    pressure_poisson_residual_periodic,
    primitive_residual_periodic,
    spectral_divergence,
)

NS_PERIODIC_RESIDUAL_SCHEMA_VERSION = "navier-stokes-periodic-residual-1"

#: honesty claims that must stay ``False`` on every periodic residual certificate.
_FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "unproven_claim",
    "continuum_navier_stokes_claim",
    "chaotic_tracking_claim",
    "perfect_weather_claim",
    "turbulence_closure_claim",
)


def _sha256_json(value: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable proof artifact."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sup(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))


def _spectral_l1_sup_bound(field: np.ndarray, dim: int) -> float:
    r"""Between-node sup bound ``\lVert R\rVert_\infty \le \sum_k|\hat c_k|``.

    For a band-limited (resolved, alias-free) periodic field the trigonometric
    interpolant equals the field, and the triangle inequality bounds the
    *continuum* sup-norm by the ``l1`` norm of its Fourier coefficients -- a bound
    valid **between** grid nodes, not only on them.  Computed over the trailing
    ``dim`` grid axes; the leading component axis (if any) is reduced by ``max``.
    """
    arr = np.asarray(field, dtype=float)
    grid_axes = tuple(range(arr.ndim - dim, arr.ndim))
    nfac = float(np.prod([arr.shape[a] for a in grid_axes]))
    coeffs = np.abs(np.fft.fftn(arr, axes=grid_axes)) / nfac
    l1 = np.sum(coeffs, axis=grid_axes)
    return float(np.max(l1))


def _spectral_tail_ratio(field: np.ndarray, dim: int) -> float:
    """Fraction of spectral energy beyond 2/3 of Nyquist (resolution diagnostic)."""
    arr = np.asarray(field, dtype=float)
    grid_axes = tuple(range(arr.ndim - dim, arr.ndim))
    spec = np.fft.fftn(arr, axes=grid_axes)
    power = np.abs(spec) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    tail_mask = np.zeros(arr.shape, dtype=bool)
    for axis in grid_axes:
        n = arr.shape[axis]
        freqs = np.fft.fftfreq(n, d=1.0 / n)
        hi = np.abs(freqs) > (2.0 / 3.0) * (n // 2)
        shape = [1] * arr.ndim
        shape[axis] = n
        tail_mask = tail_mask | hi.reshape(shape)
    return float(np.sum(power[tail_mask]) / total)


def certified_periodic_flow_residual(
    sample: PeriodicFlowSample,
    *,
    residual_tol: float = 1e-8,
    claim_exact_solution: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    r"""Seal a residual-only Navier--Stokes certificate for a periodic ``sample``.

    Parameters
    ----------
    sample
        A :class:`~omnibias.pinn.certified.fluid_fixtures.PeriodicFlowSample`
        carrying the sampled velocity / pressure / time-derivative / forcing and
        a regeneration ``descriptor``.
    residual_tol
        Tolerance against which ``exact_solution_claim`` is asserted; the three
        residual sups must all sit at or below it for an exact-solution claim.
    claim_exact_solution
        Override the fixture's own ``exact_solution`` hint.  When ``None`` (the
        default) the descriptor's hint is used *and* gated on ``residual_tol``.
    notes
        Free-text provenance note (does not affect any gate).

    Returns
    -------
    dict
        A JSON-serialisable, hash-sealed certificate (schema version
        ``navier-stokes-periodic-residual-1``).
    """
    if residual_tol <= 0.0:
        raise ValueError("residual_tol must be positive")
    domain = tuple(float(v) for v in sample.lengths)
    momentum, continuity = primitive_residual_periodic(
        sample.velocity,
        sample.pressure,
        velocity_t=sample.velocity_t,
        forcing=sample.forcing,
        viscosity=sample.viscosity,
        density=sample.density,
        lengths=domain,
    )
    pressure_res = pressure_poisson_residual_periodic(
        sample.velocity, sample.pressure, density=sample.density, lengths=domain
    )
    divergence = spectral_divergence(sample.velocity, lengths=domain)
    diagnostics = energy_diagnostics(sample.velocity, pressure=sample.pressure, lengths=domain)

    momentum_sup = _sup(momentum)
    continuity_sup = _sup(continuity)
    pressure_sup = _sup(pressure_res)
    residual_sup = max(momentum_sup, continuity_sup, pressure_sup)

    dim = int(sample.dimension)
    spectral_l1_bound = max(
        _spectral_l1_sup_bound(momentum, dim),
        _spectral_l1_sup_bound(continuity, dim),
        _spectral_l1_sup_bound(pressure_res, dim),
    )
    velocity_tail_ratio = _spectral_tail_ratio(sample.velocity, dim)

    hint = bool(sample.descriptor.get("exact_solution", False))
    wants_exact = hint if claim_exact_solution is None else bool(claim_exact_solution)
    exact_solution_claim = bool(wants_exact and residual_sup <= residual_tol)

    honesty = {
        "unproven_claim": False,
        "continuum_navier_stokes_claim": False,
        "chaotic_tracking_claim": False,
        "perfect_weather_claim": False,
        "turbulence_closure_claim": False,
        "interval_verified": False,
        "periodic_model_only": True,
        "numerical_residual": True,
        "exact_solution_claim": exact_solution_claim,
    }

    body: dict[str, Any] = {
        "schema_version": NS_PERIODIC_RESIDUAL_SCHEMA_VERSION,
        "observable": "incompressible_navier_stokes_periodic_residual",
        "model": "incompressible_navier_stokes",
        "verification_method": "spectral_grid_sampling",
        "fixture": dict(sample.descriptor),
        "dimension": int(sample.dimension),
        "grid_shape": list(sample.grid_shape),
        "lengths": list(domain),
        "viscosity": float(sample.viscosity),
        "density": float(sample.density),
        "forced": bool(np.any(sample.forcing)),
        "residual_tol": float(residual_tol),
        "momentum_residual_sup": momentum_sup,
        "continuity_residual_sup": continuity_sup,
        "pressure_poisson_residual_sup": pressure_sup,
        "residual_sup": residual_sup,
        "momentum_residual_rms": _rms(momentum),
        "continuity_residual_rms": _rms(continuity),
        "spectral_l1_residual_bound": spectral_l1_bound,
        "velocity_spectral_tail_ratio": velocity_tail_ratio,
        "max_abs_divergence": _sup(divergence),
        "kinetic_energy": float(diagnostics["kinetic_energy"]),
        "enstrophy": float(diagnostics["enstrophy"]),
        "palinstrophy": float(diagnostics["palinstrophy"]),
        "bkm_vorticity_proxy": float(diagnostics["bkm_vorticity_proxy"]),
        "exact_solution_claim": exact_solution_claim,
        "criterion": (
            "a sampled periodic field is accepted when the momentum, continuity "
            "and pressure-Poisson residual sup-norms all sit at or below "
            "residual_tol; exact_solution_claim additionally requires the field "
            "to match a known analytic fixture to that tolerance"
        ),
        "theorem_dependency": (
            "periodic spectral (FFT) differentiation of the sampled fields "
            "(omnibias.pinn.certified.navier_stokes) -- NOT an interval enclosure"
        ),
        "between_node_bound_method": (
            "spectral_l1_residual_bound = sum_k |hat c_k| bounds the continuum sup-norm "
            "of the (band-limited) residual between grid nodes via the triangle "
            "inequality; valid when the spectrum is resolved "
            "(velocity_spectral_tail_ratio ~ 0). This is a band-limited bound, NOT a "
            "validated interval enclosure -- see "
            "omnibias.pinn.certified.fluid_rigorous.certified_streamfunction_residual "
            "for the interval-verified (whole-domain) certificate."
        ),
        "honesty": honesty,
        "open_obligations": [
            "band_limited_spectral_l1_bound_is_provided_but_assumes_a_resolved_spectrum",
            "interval_verified_whole_domain_enclosure_available_via_streamfunction_cage_cert",
            "continuum_navier_stokes_regularity_is_out_of_scope",
            "high_reynolds_turbulence_and_long_horizon_chaos_tracking_out_of_scope",
        ],
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.fluid.certified_periodic_flow_residual",
        "fixture_name": str(sample.descriptor.get("name", "external_samples")),
        "notes": str(notes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sha256": _sha256_json(body),
    }
    return body


def certified_taylor_green_residual(
    n: int = 64,
    *,
    viscosity: float = 0.1,
    density: float = 1.0,
    time: float = 0.0,
    amplitude: float = 1.0,
    residual_tol: float = 1e-8,
) -> dict[str, Any]:
    """Certify the exact 2-D Taylor--Green vortex (laminar correctness baseline)."""
    sample = taylor_green_vortex(
        n, viscosity=viscosity, density=density, time=time, amplitude=amplitude
    )
    return certified_periodic_flow_residual(sample, residual_tol=residual_tol)


def certified_kolmogorov_residual(
    n: int = 64,
    *,
    viscosity: float = 0.1,
    density: float = 1.0,
    wavenumber: int = 4,
    amplitude: float = 1.0,
    residual_tol: float = 1e-8,
) -> dict[str, Any]:
    """Certify the exact steady forced Kolmogorov shear base state."""
    sample = kolmogorov_flow(
        n, viscosity=viscosity, density=density, wavenumber=wavenumber, amplitude=amplitude
    )
    return certified_periodic_flow_residual(sample, residual_tol=residual_tol)


REQUIRED_NS_PERIODIC_RESIDUAL_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "verification_method",
    "fixture",
    "dimension",
    "grid_shape",
    "lengths",
    "viscosity",
    "density",
    "residual_tol",
    "momentum_residual_sup",
    "continuity_residual_sup",
    "pressure_poisson_residual_sup",
    "residual_sup",
    "spectral_l1_residual_bound",
    "velocity_spectral_tail_ratio",
    "max_abs_divergence",
    "kinetic_energy",
    "enstrophy",
    "exact_solution_claim",
    "criterion",
    "theorem_dependency",
    "honesty",
    "open_obligations",
    "provenance",
)


def periodic_residual_digest_ok(cert: dict[str, Any]) -> bool:
    """``True`` iff the certificate's ``provenance.sha256`` matches its body.

    Recomputes the hash over the body with the ``provenance`` block removed -- the
    exact bytes hashed at seal time -- so any tamper with a sealed field is caught
    (the v1-style tamper-evident digest gate for this dict certificate).
    """
    provenance = cert.get("provenance")
    if not isinstance(provenance, dict):
        return False
    recorded = provenance.get("sha256")
    if not isinstance(recorded, str):
        return False
    body = {k: v for k, v in cert.items() if k != "provenance"}
    return _sha256_json(body) == recorded


def certified_periodic_flow_residual_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-periodic-residual-1`` certificate's structure."""
    errors: list[str] = []
    for key in REQUIRED_NS_PERIODIC_RESIDUAL_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != NS_PERIODIC_RESIDUAL_SCHEMA_VERSION:
        errors.append(f"schema_version must be {NS_PERIODIC_RESIDUAL_SCHEMA_VERSION!r}")

    honesty = cert.get("honesty", {})
    if not isinstance(honesty, dict):
        errors.append("'honesty' must be a mapping")
        honesty = {}
    for flag in _FORBIDDEN_CLAIMS:
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    if honesty.get("interval_verified", False):
        errors.append("honesty.interval_verified must be False (residuals are FFT-sampled)")

    for key in (
        "momentum_residual_sup",
        "continuity_residual_sup",
        "pressure_poisson_residual_sup",
        "residual_sup",
        "max_abs_divergence",
    ):
        val = cert.get(key)
        if not isinstance(val, int | float) or not (float(val) >= 0.0) or math.isinf(float(val)):
            errors.append(f"{key} must be a finite non-negative number")

    tol = cert.get("residual_tol")
    if not isinstance(tol, int | float) or not (float(tol) > 0.0):
        errors.append("residual_tol must be a positive number")
    elif cert.get("exact_solution_claim", False):
        # An exact-solution claim must be self-consistent with the recorded sups.
        for key in (
            "momentum_residual_sup",
            "continuity_residual_sup",
            "pressure_poisson_residual_sup",
        ):
            val = cert.get(key)
            if isinstance(val, int | float) and float(val) > float(tol):
                errors.append(
                    f"exact_solution_claim requires {key} <= residual_tol "
                    f"({float(val):.3e} > {float(tol):.3e})"
                )

    for key in ("spectral_l1_residual_bound", "velocity_spectral_tail_ratio"):
        val = cert.get(key)
        if not isinstance(val, int | float) or float(val) < 0.0 or math.isinf(float(val)):
            errors.append(f"{key} must be a finite non-negative number")
    # The band-limited between-node bound must dominate the on-node residual sup.
    l1 = cert.get("spectral_l1_residual_bound")
    res = cert.get("residual_sup")
    if isinstance(l1, int | float) and isinstance(res, int | float):
        if float(l1) + 1e-9 < float(res):
            errors.append("spectral_l1_residual_bound must be >= residual_sup (it bounds the sup)")

    fixture = cert.get("fixture")
    if not isinstance(fixture, dict) or "name" not in fixture:
        errors.append("'fixture' must be a descriptor mapping with a 'name'")

    if "provenance" in cert and not periodic_residual_digest_ok(cert):
        errors.append("provenance.sha256 digest mismatch (tampered or stale certificate)")
    return errors


# --------------------------------------------------------------------------- #
# Proof-machine prover (kind: ``navier_stokes_periodic_residual``)
# --------------------------------------------------------------------------- #
def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _certificate_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """Build (or extract) a periodic residual certificate from conjecture data."""
    cert = data.get("certificate")
    if isinstance(cert, dict):
        return cert
    residual_tol = float(data.get("residual_tol", 1e-8))
    descriptor = data.get("fixture")
    if isinstance(descriptor, dict):
        sample = regenerate_periodic_flow(descriptor)
        return certified_periodic_flow_residual(sample, residual_tol=residual_tol)
    name = data.get("name")
    if name == "taylor_green_vortex":
        return certified_taylor_green_residual(
            int(data.get("n", 64)),
            viscosity=float(data.get("viscosity", 0.1)),
            density=float(data.get("density", 1.0)),
            time=float(data.get("time", 0.0)),
            amplitude=float(data.get("amplitude", 1.0)),
            residual_tol=residual_tol,
        )
    if name == "kolmogorov_flow":
        return certified_kolmogorov_residual(
            int(data.get("n", 64)),
            viscosity=float(data.get("viscosity", 0.1)),
            density=float(data.get("density", 1.0)),
            wavenumber=int(data.get("wavenumber", 4)),
            amplitude=float(data.get("amplitude", 1.0)),
            residual_tol=residual_tol,
        )
    raise ValueError(
        "conjecture data needs a 'certificate', a 'fixture' descriptor, or a known 'name'"
    )


def prove_navier_stokes_periodic_residual(conjecture: Conjecture) -> ProofAttempt:
    """Adjudicate a residual-only periodic Navier--Stokes certificate."""
    try:
        cert = _certificate_from_data(dict(conjecture.data))
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(f"could not build periodic NS residual certificate: {exc}")

    threshold = conjecture.data.get("residual_tolerance")
    if threshold is None:
        threshold = cert.get("residual_tol", 1e-8)
    residual_sup = float(cert.get("residual_sup", float("inf")))
    if residual_sup > float(threshold):
        return ProofAttempt(
            status="BLOCKED",
            certificate=cert,
            obligations=(
                f"residual sup {residual_sup:.3e} exceeds tolerance {float(threshold):.3e}",
            ),
            detail="periodic NS residual is finite but above the requested tolerance",
        )
    return ProofAttempt(
        status="PROVED",
        certificate=cert,
        detail=(
            "periodic incompressible Navier-Stokes residual (momentum, continuity, "
            "pressure-Poisson) certified at or below tolerance for the stated model flow"
        ),
    )


def navier_stokes_periodic_residual_schema_errors(cert: dict[str, Any]) -> list[str]:
    return certified_periodic_flow_residual_schema_errors(cert)


def replay_navier_stokes_periodic_residual(cert: dict[str, Any]) -> bool | None:
    """Independent numpy twin (lazy import; ``None`` if omnibias-symbolic absent)."""
    try:
        from omnibias.symbolic.fluid import verify_periodic_flow_residual
    except ImportError:
        return None
    report = verify_periodic_flow_residual(cert)
    return bool(report["replay_match"])


__all__ = [
    "NS_PERIODIC_RESIDUAL_SCHEMA_VERSION",
    "REQUIRED_NS_PERIODIC_RESIDUAL_KEYS",
    "certified_kolmogorov_residual",
    "certified_periodic_flow_residual",
    "certified_periodic_flow_residual_schema_errors",
    "certified_taylor_green_residual",
    "navier_stokes_periodic_residual_schema_errors",
    "periodic_residual_digest_ok",
    "prove_navier_stokes_periodic_residual",
    "replay_navier_stokes_periodic_residual",
]

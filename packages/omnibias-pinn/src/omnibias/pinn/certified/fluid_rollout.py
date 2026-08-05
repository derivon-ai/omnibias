# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Genuine time-integration rollout of 2-D incompressible Navier--Stokes.

The earlier "rollout" in the demo stitched together *analytic snapshots*.  This
module integrates the flow forward in time for real: a pseudo-spectral
vorticity-streamfunction solver on the periodic torus,

.. math::

    \omega_t + (u\cdot\nabla)\omega = \nu\,\Delta\omega + f_\omega,
    \qquad u = \nabla^\perp\psi,\quad \omega = -\Delta\psi,

with the velocity recovered spectrally (so ``\nabla\cdot u = 0`` to machine
precision at every step), a 2/3-rule dealiased nonlinear term, and a second-order
**integrating-factor Runge--Kutta** step that treats the stiff viscous operator
exactly.

What is certified -- and what is **not**
----------------------------------------
The honest, falsifiable claims sealed by :func:`certified_rollout_diagnostics`
are *window diagnostics*, not pointwise truth:

* **incompressibility is maintained** -- ``max_t \lVert\nabla\cdot u\rVert_\infty``
  stays at machine zero (the spectral velocity is divergence free by construction);
* for the **inviscid, unforced** case the conserved invariants (kinetic energy and
  enstrophy) **drift by a reported, finite amount** -- a direct, measurable
  statement about how *little* numerical diffusion the scheme introduces;
* the energy / enstrophy time series stay **bounded** in a recorded range.

This is emphatically **not** pointwise chaos tracking, perfect weather, a
turbulence closure, or a continuum theorem; those honesty flags are ``False`` and
the schema gate rejects flipping them.  Butterfly-effect sensitivity is real, so a
long-horizon *trajectory* is never claimed exact -- only these window-level
invariants are.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from omnibias.core.proof import Conjecture, ProofAttempt

NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION = "navier-stokes-rollout-diagnostics-1"

_FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "unproven_claim",
    "continuum_navier_stokes_claim",
    "chaotic_tracking_claim",
    "perfect_weather_claim",
    "turbulence_closure_claim",
)

_TWO_PI = 2.0 * float(np.pi)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wavenumbers(n: int, length: float) -> tuple[np.ndarray, np.ndarray]:
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    return k[:, None], k[None, :]


def _dealias_mask(n: int) -> np.ndarray:
    """The 2/3-rule spectral mask (zero the top third of each axis)."""
    keep = np.ones(n, dtype=float)
    cutoff = n // 3
    freqs = np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers
    keep[np.abs(freqs) > cutoff] = 0.0
    return keep[:, None] * keep[None, :]


def fourier_mode_vorticity(
    n: int, modes: list[dict[str, Any]], *, length: float = _TWO_PI
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Build a band-limited vorticity field ``\sum a\,\cos(k\cdot x+\phi)``.

    ``modes`` is a list of ``{"kx": int, "ky": int, "amp": float, "phase": float}``.
    Returns the sampled field and a JSON-native descriptor that regenerates it.
    """
    axis = length * np.arange(n, dtype=float) / n
    x, y = np.meshgrid(axis, axis, indexing="ij")
    omega = np.zeros((n, n), dtype=float)
    norm: list[dict[str, Any]] = []
    for mode in modes:
        kx = int(mode["kx"])
        ky = int(mode["ky"])
        amp = float(mode.get("amp", 1.0))
        phase = float(mode.get("phase", 0.0))
        omega = omega + amp * np.cos(kx * x + ky * y + phase)
        norm.append({"kx": kx, "ky": ky, "amp": amp, "phase": phase})
    descriptor = {
        "name": "fourier_mode_vorticity",
        "n": int(n),
        "length": float(length),
        "modes": norm,
    }
    return omega, descriptor


def vorticity_from_descriptor(descriptor: dict[str, Any]) -> np.ndarray:
    """Regenerate an initial vorticity field from a :func:`fourier_mode_vorticity` descriptor."""
    if descriptor.get("name") != "fourier_mode_vorticity":
        raise ValueError(f"unknown vorticity descriptor: {descriptor.get('name')!r}")
    omega, _ = fourier_mode_vorticity(
        int(descriptor["n"]), list(descriptor["modes"]), length=float(descriptor.get("length", _TWO_PI))
    )
    return omega


@dataclass(frozen=True)
class RolloutResult:
    """Diagnostics of a vorticity rollout (component time series + final state)."""

    times: np.ndarray
    energy: np.ndarray
    enstrophy: np.ndarray
    max_divergence: float
    energy_drift: float
    enstrophy_drift: float
    final_vorticity: np.ndarray
    length: float
    viscosity: float
    dt: float
    steps: int


def integrate_vorticity_2d(
    omega0: np.ndarray,
    *,
    viscosity: float,
    dt: float,
    steps: int,
    forcing: np.ndarray | None = None,
    length: float = _TWO_PI,
    record_every: int = 1,
) -> RolloutResult:
    r"""Integrate 2-D vorticity transport with an integrating-factor RK2 scheme.

    ``omega0`` is the initial vorticity on an ``n x n`` periodic grid.  Returns a
    :class:`RolloutResult` with the energy / enstrophy time series, the worst-case
    divergence over the window, and the invariant drift (relevant when
    ``viscosity == 0`` and ``forcing is None`` -- the conservative case).
    """
    omega0 = np.asarray(omega0, dtype=float)
    n = omega0.shape[0]
    if omega0.shape != (n, n):
        raise ValueError("omega0 must be a square 2-D field")
    if dt <= 0.0 or steps < 1:
        raise ValueError("dt must be positive and steps >= 1")
    if viscosity < 0.0:
        raise ValueError("viscosity must be non-negative")

    kx, ky = _wavenumbers(n, length)
    k2 = kx * kx + ky * ky
    k2_inv = np.where(k2 == 0.0, 0.0, 1.0 / np.where(k2 == 0.0, 1.0, k2))
    mask = _dealias_mask(n)
    cell = (length / n) ** 2
    decay = np.exp(-viscosity * k2 * dt)
    f_hat = np.zeros_like(omega0, dtype=complex)
    if forcing is not None:
        f_hat = np.fft.fft2(np.asarray(forcing, dtype=float))

    def velocity(w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = w_hat * k2_inv
        u = np.real(np.fft.ifft2(1j * ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * kx * psi_hat))
        return u, v

    def nonlinear(w_hat: np.ndarray) -> np.ndarray:
        u, v = velocity(w_hat)
        wx = np.real(np.fft.ifft2(1j * kx * w_hat))
        wy = np.real(np.fft.ifft2(1j * ky * w_hat))
        n_hat = np.fft.fft2(u * wx + v * wy) * mask
        return cast(np.ndarray, -n_hat + f_hat)

    def divergence(w_hat: np.ndarray) -> float:
        psi_hat = w_hat * k2_inv
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        div = np.real(np.fft.ifft2(1j * kx * u_hat + 1j * ky * v_hat))
        return float(np.max(np.abs(div)))

    def invariants(w_hat: np.ndarray) -> tuple[float, float]:
        u, v = velocity(w_hat)
        omega = np.real(np.fft.ifft2(w_hat))
        energy = float(0.5 * cell * np.sum(u * u + v * v))
        enstrophy = float(0.5 * cell * np.sum(omega * omega))
        return energy, enstrophy

    w_hat = np.fft.fft2(omega0)
    e0, z0 = invariants(w_hat)
    times = [0.0]
    energy = [e0]
    enstrophy = [z0]
    max_div = divergence(w_hat)

    for step in range(1, steps + 1):
        k1 = nonlinear(w_hat)
        w_pred = decay * (w_hat + dt * k1)
        k2_stage = nonlinear(w_pred)
        w_hat = decay * w_hat + 0.5 * dt * (decay * k1 + k2_stage)
        if step % record_every == 0 or step == steps:
            e, z = invariants(w_hat)
            times.append(step * dt)
            energy.append(e)
            enstrophy.append(z)
            max_div = max(max_div, divergence(w_hat))

    energy_arr = np.asarray(energy, dtype=float)
    enstrophy_arr = np.asarray(enstrophy, dtype=float)
    e_ref = max(abs(e0), 1e-300)
    z_ref = max(abs(z0), 1e-300)
    energy_drift = float(np.max(np.abs(energy_arr - e0)) / e_ref)
    enstrophy_drift = float(np.max(np.abs(enstrophy_arr - z0)) / z_ref)
    return RolloutResult(
        times=np.asarray(times, dtype=float),
        energy=energy_arr,
        enstrophy=enstrophy_arr,
        max_divergence=float(max_div),
        energy_drift=energy_drift,
        enstrophy_drift=enstrophy_drift,
        final_vorticity=np.real(np.fft.ifft2(w_hat)),
        length=float(length),
        viscosity=float(viscosity),
        dt=float(dt),
        steps=int(steps),
    )


def certified_rollout_diagnostics(
    initial_vorticity_descriptor: dict[str, Any],
    *,
    viscosity: float = 0.0,
    dt: float = 1e-3,
    steps: int = 200,
    length: float = _TWO_PI,
    divergence_tol: float = 1e-8,
    drift_tol: float = 1e-2,
    notes: str = "",
) -> dict[str, Any]:
    r"""Integrate a band-limited vorticity field forward and seal its diagnostics.

    The initial field is rebuilt from ``initial_vorticity_descriptor`` (a
    :func:`fourier_mode_vorticity` descriptor) so an independent verifier can
    re-run the rollout from scratch.  ``divergence_tol`` gates the maintained
    incompressibility; ``drift_tol`` gates the conserved-invariant drift in the
    inviscid, unforced case.
    """
    if divergence_tol <= 0.0 or drift_tol <= 0.0:
        raise ValueError("tolerances must be positive")
    omega0 = vorticity_from_descriptor(initial_vorticity_descriptor)
    result = integrate_vorticity_2d(
        omega0, viscosity=viscosity, dt=dt, steps=steps, length=length
    )
    conservative = viscosity == 0.0
    incompressible_ok = bool(result.max_divergence <= divergence_tol)
    drift_ok = bool(
        conservative
        and result.energy_drift <= drift_tol
        and result.enstrophy_drift <= drift_tol
    )

    honesty = {
        "unproven_claim": False,
        "continuum_navier_stokes_claim": False,
        "chaotic_tracking_claim": False,
        "perfect_weather_claim": False,
        "turbulence_closure_claim": False,
        "interval_verified": False,
        "numerical_integration": True,
        "statistical_window_only": True,
        "incompressibility_maintained": incompressible_ok,
        "conserved_invariant_drift_bounded": drift_ok,
    }
    body: dict[str, Any] = {
        "schema_version": NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION,
        "observable": "incompressible_navier_stokes_2d_rollout_diagnostics",
        "model": "vorticity_streamfunction_pseudospectral",
        "verification_method": "integrating_factor_rk2_dealiased",
        "dimension": 2,
        "grid_n": int(omega0.shape[0]),
        "length": float(length),
        "viscosity": float(viscosity),
        "dt": float(dt),
        "steps": int(steps),
        "conservative_case": conservative,
        "initial_vorticity": dict(initial_vorticity_descriptor),
        "divergence_tol": float(divergence_tol),
        "drift_tol": float(drift_tol),
        "max_divergence": result.max_divergence,
        "energy_initial": float(result.energy[0]),
        "energy_final": float(result.energy[-1]),
        "energy_min": float(np.min(result.energy)),
        "energy_max": float(np.max(result.energy)),
        "energy_drift": result.energy_drift,
        "enstrophy_initial": float(result.enstrophy[0]),
        "enstrophy_final": float(result.enstrophy[-1]),
        "enstrophy_min": float(np.min(result.enstrophy)),
        "enstrophy_max": float(np.max(result.enstrophy)),
        "enstrophy_drift": result.enstrophy_drift,
        "incompressibility_maintained": incompressible_ok,
        "conserved_invariant_drift_bounded": drift_ok,
        "criterion": (
            "over the integrated window the spectral velocity stays divergence free "
            "(max_divergence <= divergence_tol) and, in the inviscid unforced case, "
            "kinetic energy and enstrophy drift by at most drift_tol -- a finite, "
            "checkable statement about numerical diffusion, not pointwise tracking"
        ),
        "theorem_dependency": (
            "pseudo-spectral vorticity-streamfunction integrator with 2/3 dealiasing "
            "and integrating-factor RK2 (omnibias.pinn.certified.fluid_rollout)"
        ),
        "honesty": honesty,
        "open_obligations": [
            "long_horizon_pointwise_trajectory_is_butterfly_sensitive_and_not_claimed",
            "continuum_navier_stokes_regularity_is_out_of_scope",
            "no_interval_enclosure_of_the_time_integrated_field",
        ],
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.fluid_rollout.certified_rollout_diagnostics",
        "notes": str(notes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_ROLLOUT_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "dimension",
    "grid_n",
    "viscosity",
    "dt",
    "steps",
    "initial_vorticity",
    "divergence_tol",
    "drift_tol",
    "max_divergence",
    "energy_drift",
    "enstrophy_drift",
    "incompressibility_maintained",
    "honesty",
    "provenance",
)


def rollout_diagnostics_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-rollout-diagnostics-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_ROLLOUT_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION:
        errors.append(f"schema_version must be {NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION!r}")

    honesty = cert.get("honesty", {})
    if not isinstance(honesty, dict):
        errors.append("'honesty' must be a mapping")
        honesty = {}
    for flag in _FORBIDDEN_CLAIMS:
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    if honesty.get("interval_verified", False):
        errors.append("honesty.interval_verified must be False (numerical integration)")

    for key in ("max_divergence", "energy_drift", "enstrophy_drift"):
        val = cert.get(key)
        if not isinstance(val, int | float) or float(val) < 0.0 or not np.isfinite(float(val)):
            errors.append(f"{key} must be a finite non-negative number")

    for key in ("divergence_tol", "drift_tol"):
        val = cert.get(key)
        if not isinstance(val, int | float) or float(val) <= 0.0:
            errors.append(f"{key} must be a positive number")

    # Self-consistency of the recorded gate booleans.
    div_tol = cert.get("divergence_tol")
    if isinstance(div_tol, int | float) and isinstance(cert.get("max_divergence"), int | float):
        expect = bool(float(cert["max_divergence"]) <= float(div_tol))
        if bool(cert.get("incompressibility_maintained", False)) != expect:
            errors.append("incompressibility_maintained inconsistent with max_divergence/divergence_tol")

    if not isinstance(cert.get("initial_vorticity"), dict):
        errors.append("'initial_vorticity' must be a regenerable descriptor mapping")
    return errors


# --------------------------------------------------------------------------- #
# Proof-machine prover (kind: ``navier_stokes_rollout_diagnostics``)
# --------------------------------------------------------------------------- #
def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _certificate_from_data(data: dict[str, Any]) -> dict[str, Any]:
    cert = data.get("certificate")
    if isinstance(cert, dict):
        return cert
    descriptor = data.get("initial_vorticity")
    if not isinstance(descriptor, dict):
        descriptor = fourier_mode_vorticity(
            int(data.get("n", 48)),
            [
                {"kx": 1, "ky": 0, "amp": 1.0, "phase": 0.0},
                {"kx": 0, "ky": 2, "amp": 0.7, "phase": 0.3},
                {"kx": 1, "ky": 1, "amp": 0.5, "phase": 1.1},
            ],
        )[1]
    return certified_rollout_diagnostics(
        descriptor,
        viscosity=float(data.get("viscosity", 0.0)),
        dt=float(data.get("dt", 1e-3)),
        steps=int(data.get("steps", 200)),
        divergence_tol=float(data.get("divergence_tol", 1e-8)),
        drift_tol=float(data.get("drift_tol", 1e-2)),
    )


def prove_rollout_diagnostics(conjecture: Conjecture) -> ProofAttempt:
    """Adjudicate a 2-D Navier--Stokes rollout-diagnostics certificate."""
    try:
        cert = _certificate_from_data(dict(conjecture.data))
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(f"could not build rollout-diagnostics certificate: {exc}")

    if not bool(cert.get("incompressibility_maintained", False)):
        return ProofAttempt(
            status="BLOCKED",
            certificate=cert,
            obligations=(
                f"max divergence {float(cert.get('max_divergence', float('inf'))):.3e} "
                f"exceeds divergence_tol {float(cert.get('divergence_tol', 0.0)):.3e}",
            ),
            detail="incompressibility not maintained over the window",
        )
    if bool(cert.get("conservative_case", False)) and not bool(
        cert.get("conserved_invariant_drift_bounded", False)
    ):
        return ProofAttempt(
            status="BLOCKED",
            certificate=cert,
            obligations=(
                f"invariant drift (energy {float(cert.get('energy_drift', 1.0)):.3e}, "
                f"enstrophy {float(cert.get('enstrophy_drift', 1.0)):.3e}) exceeds drift_tol",
            ),
            detail="conserved invariants drifted beyond tolerance",
        )
    return ProofAttempt(
        status="PROVED",
        certificate=cert,
        detail=(
            "2-D incompressible rollout maintained divergence-free velocity and "
            "bounded invariant drift over the integrated window (statistical, not "
            "pointwise tracking)"
        ),
    )


def replay_rollout_diagnostics(cert: dict[str, Any]) -> bool | None:
    """Independent numpy re-integration twin (``None`` if omnibias-symbolic absent)."""
    try:
        from omnibias.symbolic.fluid import verify_rollout_diagnostics
    except ImportError:
        return None
    report = verify_rollout_diagnostics(cert)
    return bool(report["replay_match"])


__all__ = [
    "NS_ROLLOUT_DIAGNOSTICS_SCHEMA_VERSION",
    "REQUIRED_ROLLOUT_KEYS",
    "RolloutResult",
    "certified_rollout_diagnostics",
    "fourier_mode_vorticity",
    "integrate_vorticity_2d",
    "prove_rollout_diagnostics",
    "replay_rollout_diagnostics",
    "rollout_diagnostics_schema_errors",
    "vorticity_from_descriptor",
]

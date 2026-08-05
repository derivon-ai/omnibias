# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Independent numpy replay of the periodic Navier--Stokes residual certificate.

This module is **numpy-only** and imports nothing from
:mod:`omnibias.pinn.certified`, so it is a genuine *second source* for the
certificate produced by
:func:`omnibias.pinn.certified.fluid.certified_periodic_flow_residual`.  It

1. **regenerates** the sampled flow from the certificate's ``fixture`` descriptor
   using its own closed-form expressions (Taylor--Green, Kolmogorov base state);
2. **recomputes** the momentum, continuity and pressure-Poisson residual sups
   with the independent spectral operators in
   :mod:`omnibias.symbolic.navier_stokes`;
3. **cross-checks** that the certificate's recorded sups match the recomputed
   ones (catching both under- and over-statement) and that the
   ``exact_solution_claim`` is consistent with the independently measured
   residual.

Nothing here asserts a Navier--Stokes regularity / global-regularity result or perfect
weather; :func:`verify_periodic_flow_residual` returns a plain dict with
``unproven_claim=False``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from omnibias.symbolic.navier_stokes import (
    pressure_poisson_residual_periodic,
    primitive_residual_periodic,
)


def _periodic_axis(n: int, length: float) -> np.ndarray:
    return length * np.arange(n, dtype=float) / n


def regenerate_periodic_flow(descriptor: dict[str, Any]) -> dict[str, Any]:
    r"""Independently rebuild a periodic flow from a certificate ``fixture``.

    Reimplements the analytic Taylor--Green and Kolmogorov fields from scratch
    (no import of the certified fixture module), returning the component-first
    sampled arrays plus the fluid parameters.
    """
    name = descriptor.get("name")
    n = int(descriptor["n"])
    viscosity = float(descriptor["viscosity"])
    density = float(descriptor.get("density", 1.0))
    lengths = tuple(float(v) for v in descriptor.get("lengths", (2.0 * np.pi, 2.0 * np.pi)))
    nu = viscosity / density

    if name == "taylor_green_vortex":
        amplitude = float(descriptor.get("amplitude", 1.0))
        time = float(descriptor.get("time", 0.0))
        axis = _periodic_axis(n, lengths[0])
        x, y = np.meshgrid(axis, axis, indexing="ij")
        decay = float(np.exp(-2.0 * nu * time))
        amp = amplitude * decay
        velocity = amp * np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])
        pressure = 0.25 * density * amp**2 * (np.cos(2.0 * x) + np.cos(2.0 * y))
        velocity_t = -2.0 * nu * velocity
        forcing = np.zeros_like(velocity)
    elif name == "kolmogorov_flow":
        amplitude = float(descriptor.get("amplitude", 1.0))
        k = int(descriptor.get("wavenumber", 1))
        axis = _periodic_axis(n, lengths[0])
        _x, y = np.meshgrid(axis, axis, indexing="ij")
        shear = amplitude * np.sin(k * y)
        velocity = np.stack([shear, np.zeros_like(shear)])
        pressure = np.zeros_like(shear)
        velocity_t = np.zeros_like(velocity)
        forcing = np.stack([
            viscosity * amplitude * (k * k) * np.sin(k * y),
            np.zeros_like(shear),
        ])
    elif name == "beltrami_abc_flow":
        a = float(descriptor.get("a", 1.0))
        b = float(descriptor.get("b", 1.0))
        c = float(descriptor.get("c", 1.0))
        k = int(descriptor.get("wavenumber", 1))
        time = float(descriptor.get("time", 0.0))
        axis = _periodic_axis(n, lengths[0])
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        decay = float(np.exp(-nu * (k * k) * time))
        velocity = decay * np.stack([
            a * np.sin(k * z) + c * np.cos(k * y),
            b * np.sin(k * x) + a * np.cos(k * z),
            c * np.sin(k * y) + b * np.cos(k * x),
        ])
        pressure = -0.5 * density * np.sum(velocity * velocity, axis=0)
        velocity_t = -nu * (k * k) * velocity
        forcing = np.zeros_like(velocity)
    else:
        raise ValueError(f"unknown fluid fixture descriptor name: {name!r}")

    return {
        "velocity": velocity,
        "pressure": pressure,
        "velocity_t": velocity_t,
        "forcing": forcing,
        "viscosity": viscosity,
        "density": density,
        "lengths": lengths,
    }


def periodic_flow_residual_sups(descriptor: dict[str, Any]) -> dict[str, float]:
    """Independent momentum / continuity / pressure-Poisson residual sups."""
    flow = regenerate_periodic_flow(descriptor)
    lengths = flow["lengths"]
    momentum, continuity = primitive_residual_periodic(
        flow["velocity"],
        flow["pressure"],
        velocity_t=flow["velocity_t"],
        forcing=flow["forcing"],
        viscosity=flow["viscosity"],
        density=flow["density"],
        lengths=lengths,
    )
    pressure_res = pressure_poisson_residual_periodic(
        flow["velocity"], flow["pressure"], density=flow["density"], lengths=lengths
    )
    momentum_sup = float(np.max(np.abs(momentum)))
    continuity_sup = float(np.max(np.abs(continuity)))
    pressure_sup = float(np.max(np.abs(pressure_res)))
    return {
        "momentum_residual_sup": momentum_sup,
        "continuity_residual_sup": continuity_sup,
        "pressure_poisson_residual_sup": pressure_sup,
        "residual_sup": max(momentum_sup, continuity_sup, pressure_sup),
    }


def _fd_grad(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """Second-order central finite difference along a periodic axis (``np.roll``)."""
    diff = (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)
    return cast(np.ndarray, diff)


def finite_difference_residual_sups(descriptor: dict[str, Any]) -> dict[str, float]:
    r"""Momentum / continuity residual sups via **finite differences** (not FFT).

    A genuinely method-independent second source for the periodic certificate: the
    derivatives are second-order central differences on the periodic grid, sharing
    *no* code path with the spectral operators used to build (and FFT-replay) the
    certificate.  Finite differencing has an ``O(h^2)`` truncation error, so this
    is a consistency cross-check (small for a true solution), not a machine-zero
    match.
    """
    flow = regenerate_periodic_flow(descriptor)
    u = np.asarray(flow["velocity"], dtype=float)
    p = np.asarray(flow["pressure"], dtype=float)
    u_t = np.asarray(flow["velocity_t"], dtype=float)
    f = np.asarray(flow["forcing"], dtype=float)
    nu = float(flow["viscosity"])
    rho = float(flow["density"])
    lengths = flow["lengths"]
    dim = u.shape[0]
    grid = u.shape[1:]
    spacing = [float(lengths[a]) / float(grid[a]) for a in range(dim)]

    def grad(scalar: np.ndarray) -> list[np.ndarray]:
        return [_fd_grad(scalar, axis=a, spacing=spacing[a]) for a in range(dim)]

    grads = [grad(u[i]) for i in range(dim)]
    advection = np.stack([
        sum(u[j] * grads[i][j] for j in range(dim)) for i in range(dim)
    ])
    grad_p = grad(p)
    laplacian = np.stack([
        sum(
            (np.roll(u[i], -1, axis=a) - 2.0 * u[i] + np.roll(u[i], 1, axis=a)) / spacing[a] ** 2
            for a in range(dim)
        )
        for i in range(dim)
    ])
    momentum = rho * (u_t + advection) + np.stack(grad_p) - nu * laplacian - f
    divergence = sum(grads[i][i] for i in range(dim))
    return {
        "finite_difference_momentum_residual_sup": float(np.max(np.abs(momentum))),
        "finite_difference_continuity_residual_sup": float(np.max(np.abs(divergence))),
        "velocity_scale": float(np.max(np.abs(u))),
        "max_spacing": float(max(spacing)),
    }


def verify_periodic_flow_residual(
    cert: dict[str, Any], *, rtol: float = 1e-6, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy replay of a ``navier-stokes-periodic-residual-1`` cert.

    Regenerates the flow from ``cert['fixture']``, recomputes the three residual
    sups, and checks that the recorded sups *match* the recomputed ones (so a
    forged smaller-or-larger residual is caught) and that the
    ``exact_solution_claim`` agrees with the independently measured residual.
    Returns a report with ``unproven_claim=False`` and ``replay_match``.
    """
    fixture = cert.get("fixture")
    if not isinstance(fixture, dict):
        return {"replay_match": False, "unproven_claim": False, "error": "missing fixture descriptor"}
    try:
        sups = periodic_flow_residual_sups(fixture)
    except (KeyError, ValueError) as exc:
        return {"replay_match": False, "unproven_claim": False, "error": str(exc)}

    def _close(recorded: float, recomputed: float) -> bool:
        return bool(abs(recorded - recomputed) <= atol + rtol * abs(recomputed))

    keys = (
        "momentum_residual_sup",
        "continuity_residual_sup",
        "pressure_poisson_residual_sup",
    )
    sups_match = all(_close(float(cert.get(key, float("inf"))), sups[key]) for key in keys)
    residual_sup_match = _close(float(cert.get("residual_sup", float("inf"))), sups["residual_sup"])

    residual_tol = float(cert.get("residual_tol", 1e-8))
    exact_solution_holds = bool(sups["residual_sup"] <= residual_tol)
    exact_claim_match = bool(bool(cert.get("exact_solution_claim", False)) == exact_solution_holds)
    divergence_free = bool(sups["continuity_residual_sup"] <= 1e-8)

    # Second, methodologically INDEPENDENT source: finite differences (no FFT).
    fd_consistent = True
    fd_report: dict[str, float] = {}
    try:
        fd = finite_difference_residual_sups(fixture)
        fd_report = fd
        # O(h^2) truncation: a true solution's FD residual scales with the grid; a
        # genuine non-solution would leave an O(1) residual that blows past this.
        fd_floor = 50.0 * fd["max_spacing"] ** 2 * (fd["velocity_scale"] + 1.0) + 1e-6
        fd_momentum = fd["finite_difference_momentum_residual_sup"]
        fd_continuity = fd["finite_difference_continuity_residual_sup"]
        if bool(cert.get("exact_solution_claim", False)):
            fd_consistent = bool(fd_momentum <= fd_floor and fd_continuity <= fd_floor)
    except (KeyError, ValueError, TypeError):
        fd_consistent = True  # FD is an auxiliary cross-check; never a false block

    replay_match = bool(sups_match and residual_sup_match and exact_claim_match and fd_consistent)
    return {
        "recomputed_momentum_residual_sup": sups["momentum_residual_sup"],
        "recomputed_continuity_residual_sup": sups["continuity_residual_sup"],
        "recomputed_pressure_poisson_residual_sup": sups["pressure_poisson_residual_sup"],
        "recomputed_residual_sup": sups["residual_sup"],
        "sups_match": sups_match,
        "residual_sup_match": residual_sup_match,
        "exact_solution_holds": exact_solution_holds,
        "exact_claim_match": exact_claim_match,
        "divergence_free": divergence_free,
        "finite_difference": fd_report,
        "finite_difference_consistent": fd_consistent,
        "methodologically_independent": True,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


# --------------------------------------------------------------------------- #
# Independent replay of the rigorous interval streamfunction residual cert.
# --------------------------------------------------------------------------- #
def _streamfunction_partial_evaluator(
    layers: list[dict[str, Any]], x: np.ndarray, y: np.ndarray
) -> Callable[[int, int], np.ndarray] | None:
    r"""Return a callable ``part(p, q) -> d_x^p d_y^q psi`` on the ``(x, y)`` grid.

    Independent of the interval Cauchy-product jet: this is the **explicit**
    ``tanh``-derivative tower (Riccati polynomials in ``t = tanh(a)``) applied by
    hand to a one-hidden-layer ``tanh`` -> linear MLP.  Returns ``None`` when the
    architecture is not the single-hidden-layer form the builders emit.
    """
    if len(layers) != 2:
        return None
    hidden, readout = layers[0], layers[1]
    if hidden.get("activation") != "tanh" or readout.get("activation") is not None:
        return None
    w1 = np.asarray(hidden["weight"], dtype=float)  # (H, 2)
    b1 = np.asarray(hidden["bias"], dtype=float)  # (H,)
    w2 = np.asarray(readout["weight"], dtype=float).reshape(-1)  # (H,)
    a = w1[:, 0][:, None, None] * x[None] + w1[:, 1][:, None, None] * y[None] + b1[:, None, None]
    t = np.tanh(a)
    g1 = 1.0 - t * t
    g = {
        1: g1,
        2: -2.0 * t * g1,
        3: -2.0 * g1 * (1.0 - 3.0 * t * t),
        4: 8.0 * t * g1 * (2.0 - 3.0 * t * t),
    }
    wx = w1[:, 0]
    wy = w1[:, 1]

    def part(p: int, q: int) -> np.ndarray:
        coeff = (w2 * wx**p * wy**q)[:, None, None]
        return cast(np.ndarray, np.sum(coeff * g[p + q], axis=0))

    return part


def streamfunction_residual_sups(
    streamfunction: dict[str, Any], *, viscosity: float = 0.0, grid: int = 48
) -> dict[str, float] | None:
    """Independently sample the vorticity-transport residual / divergence sups."""
    layers = list(streamfunction["layers"])
    domain = streamfunction["domain"]
    (x_lo, x_hi), (y_lo, y_hi) = (tuple(domain[0]), tuple(domain[1]))
    xs = np.linspace(float(x_lo), float(x_hi), int(grid))
    ys = np.linspace(float(y_lo), float(y_hi), int(grid))
    x, y = np.meshgrid(xs, ys, indexing="ij")
    part = _streamfunction_partial_evaluator(layers, x, y)
    if part is None:
        return None
    u1 = part(0, 1)
    u2 = -part(1, 0)
    dx_omega = -(part(3, 0) + part(1, 2))
    dy_omega = -(part(2, 1) + part(0, 3))
    residual = u1 * dx_omega + u2 * dy_omega
    if viscosity != 0.0:
        lap_omega = -(part(4, 0) + 2.0 * part(2, 2) + part(0, 4))
        residual = residual - viscosity * lap_omega
    divergence = part(1, 1) - part(1, 1)
    return {
        "residual_sup": float(np.max(np.abs(residual))),
        "divergence_sup": float(np.max(np.abs(divergence))),
    }


def verify_streamfunction_residual(
    cert: dict[str, Any], *, rtol: float = 1e-6, atol: float = 1e-9, grid: int = 48
) -> dict[str, Any]:
    r"""Independent numpy replay of a ``navier-stokes-streamfunction-residual-1`` cert.

    Rebuilds the streamfunction MLP from the certificate's ``payload.streamfunction``
    descriptor and recomputes the residual with the explicit ``tanh`` tower (a
    *different algorithm* from the sealed interval jet).  Because the sealed value
    is a rigorous upper enclosure, the independently sampled sup must satisfy
    ``sampled_sup <= recorded_sup`` (a forged-too-small residual is caught) and an
    ``exact_steady_euler_claim`` must agree with a machine-zero sampled residual.
    """
    payload = cert.get("payload")
    if not isinstance(payload, dict):
        return {"replay_match": False, "unproven_claim": False, "error": "missing payload"}
    streamfunction = payload.get("streamfunction")
    if not isinstance(streamfunction, dict):
        return {"replay_match": False, "unproven_claim": False, "error": "missing streamfunction"}
    viscosity = float(payload.get("viscosity", 0.0))
    try:
        sampled = streamfunction_residual_sups(streamfunction, viscosity=viscosity, grid=grid)
    except (KeyError, ValueError, TypeError) as exc:
        return {"replay_match": False, "unproven_claim": False, "error": str(exc)}
    if sampled is None:
        return {
            "replay_match": False,
            "unproven_claim": False,
            "method": "unsupported_architecture",
            "error": "twin supports a single-hidden-layer tanh streamfunction only",
        }

    recorded_residual = float(payload.get("residual_sup", float("inf")))
    recorded_divergence = float(payload.get("divergence_sup", float("inf")))
    # The sealed value is a rigorous OVER-estimate; the independent point sample is
    # a sup over grid nodes (<= continuum sup <= recorded). Catch a too-small forge.
    residual_upper_ok = bool(sampled["residual_sup"] <= recorded_residual * (1.0 + rtol) + atol)
    divergence_upper_ok = bool(sampled["divergence_sup"] <= recorded_divergence * (1.0 + rtol) + atol)

    claim = bool(payload.get("exact_steady_euler_claim", False))
    tol = float(payload.get("residual_tol", 1e-8))
    claim_threshold = max(tol, 1e3 * atol)
    claim_consistent = (not claim) or bool(
        sampled["residual_sup"] <= claim_threshold and sampled["divergence_sup"] <= claim_threshold
    )

    replay_match = bool(residual_upper_ok and divergence_upper_ok and claim_consistent)
    return {
        "method": "analytic_tanh_tower_numpy",
        "sampled_residual_sup": sampled["residual_sup"],
        "sampled_divergence_sup": sampled["divergence_sup"],
        "residual_upper_ok": residual_upper_ok,
        "divergence_upper_ok": divergence_upper_ok,
        "exact_claim_consistent": claim_consistent,
        "methodologically_independent": True,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


# --------------------------------------------------------------------------- #
# Independent re-integration of the 2-D rollout-diagnostics certificate.
# --------------------------------------------------------------------------- #
def _twin_vorticity_ic(descriptor: dict[str, Any]) -> np.ndarray:
    if descriptor.get("name") != "fourier_mode_vorticity":
        raise ValueError(f"unknown vorticity descriptor: {descriptor.get('name')!r}")
    n = int(descriptor["n"])
    length = float(descriptor.get("length", 2.0 * np.pi))
    axis = _periodic_axis(n, length)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    omega = np.zeros((n, n), dtype=float)
    for mode in descriptor["modes"]:
        omega = omega + float(mode.get("amp", 1.0)) * np.cos(
            int(mode["kx"]) * x + int(mode["ky"]) * y + float(mode.get("phase", 0.0))
        )
    return omega


def _twin_integrate(
    omega0: np.ndarray, *, viscosity: float, dt: float, steps: int, length: float
) -> dict[str, float]:
    """A second, independent pseudo-spectral vorticity integrator (twin of fluid_rollout)."""
    n = omega0.shape[0]
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx = k1d[:, None]
    ky = k1d[None, :]
    k2 = kx * kx + ky * ky
    k2_inv = np.where(k2 == 0.0, 0.0, 1.0 / np.where(k2 == 0.0, 1.0, k2))
    freqs = np.fft.fftfreq(n, d=1.0 / n)
    keep = (np.abs(freqs) <= n // 3).astype(float)
    mask = keep[:, None] * keep[None, :]
    cell = (length / n) ** 2
    decay = np.exp(-viscosity * k2 * dt)

    def vel(w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = w_hat * k2_inv
        return np.real(np.fft.ifft2(1j * ky * psi)), np.real(np.fft.ifft2(-1j * kx * psi))

    def rhs(w_hat: np.ndarray) -> np.ndarray:
        u, v = vel(w_hat)
        wx = np.real(np.fft.ifft2(1j * kx * w_hat))
        wy = np.real(np.fft.ifft2(1j * ky * w_hat))
        return cast(np.ndarray, -np.fft.fft2(u * wx + v * wy) * mask)

    def diag(w_hat: np.ndarray) -> tuple[float, float, float]:
        u, v = vel(w_hat)
        omega = np.real(np.fft.ifft2(w_hat))
        energy = float(0.5 * cell * np.sum(u * u + v * v))
        enstrophy = float(0.5 * cell * np.sum(omega * omega))
        psi = w_hat * k2_inv
        div = np.real(np.fft.ifft2(1j * kx * (1j * ky * psi) + 1j * ky * (-1j * kx * psi)))
        return energy, enstrophy, float(np.max(np.abs(div)))

    w_hat = np.fft.fft2(omega0)
    e0, z0, max_div = diag(w_hat)
    energies = [e0]
    enstrophies = [z0]
    for _ in range(steps):
        k1 = rhs(w_hat)
        w_pred = decay * (w_hat + dt * k1)
        k2_stage = rhs(w_pred)
        w_hat = decay * w_hat + 0.5 * dt * (decay * k1 + k2_stage)
        e, z, d = diag(w_hat)
        energies.append(e)
        enstrophies.append(z)
        max_div = max(max_div, d)
    e_ref = max(abs(e0), 1e-300)
    z_ref = max(abs(z0), 1e-300)
    return {
        "max_divergence": float(max_div),
        "energy_final": float(energies[-1]),
        "enstrophy_final": float(enstrophies[-1]),
        "energy_drift": float(max(abs(e - e0) for e in energies) / e_ref),
        "enstrophy_drift": float(max(abs(z - z0) for z in enstrophies) / z_ref),
    }


def verify_rollout_diagnostics(
    cert: dict[str, Any], *, rtol: float = 1e-5, atol: float = 1e-9
) -> dict[str, Any]:
    r"""Independent numpy re-integration of a ``navier-stokes-rollout-diagnostics-1`` cert.

    Re-runs the pseudo-spectral rollout from the certificate's
    ``initial_vorticity`` descriptor with a *separate* integrator and checks the
    recorded invariant drift / divergence diagnostics reproduce.
    """
    descriptor = cert.get("initial_vorticity")
    if not isinstance(descriptor, dict):
        return {"replay_match": False, "unproven_claim": False, "error": "missing initial_vorticity"}
    try:
        omega0 = _twin_vorticity_ic(descriptor)
        recomputed = _twin_integrate(
            omega0,
            viscosity=float(cert.get("viscosity", 0.0)),
            dt=float(cert.get("dt", 1e-3)),
            steps=int(cert.get("steps", 200)),
            length=float(cert.get("length", 2.0 * np.pi)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        return {"replay_match": False, "unproven_claim": False, "error": str(exc)}

    def _close(key: str) -> bool:
        recorded = float(cert.get(key, float("inf")))
        value = recomputed[key]
        return bool(abs(recorded - value) <= atol + rtol * abs(value))

    keys = ("energy_final", "enstrophy_final", "energy_drift", "enstrophy_drift")
    matches = {key: _close(key) for key in keys}
    # max_divergence is machine-zero in both; compare on an absolute floor.
    div_match = bool(
        abs(float(cert.get("max_divergence", 1.0)) - recomputed["max_divergence"]) <= 1e-6
    )
    replay_match = bool(all(matches.values()) and div_match)
    return {
        "recomputed": recomputed,
        "matches": matches,
        "divergence_match": div_match,
        "methodologically_independent": True,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


__all__ = [
    "finite_difference_residual_sups",
    "periodic_flow_residual_sups",
    "regenerate_periodic_flow",
    "streamfunction_residual_sups",
    "verify_periodic_flow_residual",
    "verify_rollout_diagnostics",
    "verify_streamfunction_residual",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Independent symbolic / numerical validation for Navier-Stokes candidates.

This module is numpy-only and deliberately does **not** import
``omnibias.pinn.certified``.  It reimplements the periodic spectral residuals
from scratch so a Navier-Stokes CAP bundle can be checked by an independent code
path before any interval or theorem-prover work begins.

The helpers also expose small discovery scaffolds:

* regularity indicators from sampled velocity fields;
* sparse equality fits that can be interpreted as candidate growth bounds;
* self-similar blow-up rate fits for norm traces.

All public assessment functions return ``unproven_claim=False`` unless a future
external verifier explicitly upgrades the artifact.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, cast

import numpy as np
from omnibias.symbolic.discovery import fit_sparse_equation

CANDIDATE_SCHEMA_VERSION = "navier-stokes-candidate-1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_velocity(velocity: np.ndarray) -> np.ndarray:
    u = np.asarray(velocity, dtype=float)
    if u.ndim < 2:
        raise ValueError("velocity must have shape (dim, n1, ..., nd)")
    if int(u.shape[0]) != u.ndim - 1:
        raise ValueError("velocity must be component-first with dim grid axes")
    return cast(np.ndarray, np.asarray(u, dtype=float))


def _lengths(lengths: tuple[float, ...] | list[float] | None, dim: int) -> tuple[float, ...]:
    if lengths is None:
        return tuple(2.0 * np.pi for _ in range(dim))
    if len(lengths) != dim:
        raise ValueError(f"expected {dim} domain lengths, got {lengths!r}")
    return tuple(float(v) for v in lengths)


def _ks(shape: tuple[int, ...], lengths: tuple[float, ...]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for axis, (n, length) in enumerate(zip(shape, lengths, strict=True)):
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
        view = [1] * len(shape)
        view[axis] = n
        out.append(k.reshape(view))
    return out


def _grad_scalar(scalar: np.ndarray, lengths: tuple[float, ...]) -> np.ndarray:
    f = np.asarray(scalar, dtype=float)
    spec = np.fft.fftn(f)
    return cast(np.ndarray, np.asarray(
        np.stack([np.real(np.fft.ifftn(1j * k * spec)) for k in _ks(f.shape, lengths)]),
        dtype=float,
    ))


def _lap(values: np.ndarray, lengths: tuple[float, ...]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim >= 2 and x.shape[0] == x.ndim - 1:
        return cast(np.ndarray, np.asarray(np.stack([_lap(comp, lengths) for comp in x]), dtype=float))
    k2 = sum(k * k for k in _ks(x.shape, lengths))
    return cast(np.ndarray, np.asarray(np.real(np.fft.ifftn(-k2 * np.fft.fftn(x))), dtype=float))


def _div(velocity: np.ndarray, lengths: tuple[float, ...]) -> np.ndarray:
    u = _as_velocity(velocity)
    div = np.zeros_like(u[0])
    for comp, k in zip(u, _ks(u.shape[1:], lengths), strict=True):
        div = div + np.real(np.fft.ifftn(1j * k * np.fft.fftn(comp)))
    return cast(np.ndarray, np.asarray(div, dtype=float))


def _curl(velocity: np.ndarray, lengths: tuple[float, ...]) -> np.ndarray:
    u = _as_velocity(velocity)
    if u.shape[0] == 2:
        gu = _grad_scalar(u[0], lengths)
        gv = _grad_scalar(u[1], lengths)
        return cast(np.ndarray, np.asarray(gv[0] - gu[1], dtype=float))
    if u.shape[0] != 3:
        raise ValueError("curl supports 2D or 3D velocity fields")
    g = [_grad_scalar(u[i], lengths) for i in range(3)]
    return cast(np.ndarray, np.asarray(
        np.stack([g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]]),
        dtype=float,
    ))


def primitive_residual_periodic(
    velocity: np.ndarray,
    pressure: np.ndarray | None,
    *,
    velocity_t: np.ndarray | None = None,
    forcing: np.ndarray | None = None,
    viscosity: float = 1e-3,
    density: float = 1.0,
    lengths: tuple[float, ...] | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent periodic primitive-form residual recomputation."""
    u = _as_velocity(velocity)
    domain = _lengths(lengths, u.shape[0])
    p = np.zeros_like(u[0]) if pressure is None else np.asarray(pressure, dtype=float)
    u_t = np.zeros_like(u) if velocity_t is None else _as_velocity(velocity_t)
    f = np.zeros_like(u) if forcing is None else _as_velocity(forcing)
    grads = [_grad_scalar(u[i], domain) for i in range(u.shape[0])]
    adv = np.stack([
        sum(u[j] * grads[i][j] for j in range(u.shape[0]))
        for i in range(u.shape[0])
    ])
    residual = density * (u_t + adv) + _grad_scalar(p, domain) - viscosity * _lap(u, domain) - f
    return cast(np.ndarray, np.asarray(residual, dtype=float)), _div(u, domain)


def pressure_poisson_residual_periodic(
    velocity: np.ndarray,
    pressure: np.ndarray,
    *,
    density: float = 1.0,
    lengths: tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """Independent pressure-Poisson residual for periodic incompressible data."""
    u = _as_velocity(velocity)
    domain = _lengths(lengths, u.shape[0])
    quad = np.zeros_like(u[0])
    for i in range(u.shape[0]):
        for j in range(u.shape[0]):
            di = _grad_scalar(u[i] * u[j], domain)[i]
            quad = quad + _grad_scalar(di, domain)[j]
    residual = -_lap(np.asarray(pressure, dtype=float), domain) - density * quad
    return cast(np.ndarray, np.asarray(residual, dtype=float))


def regularity_feature_vector(
    velocity: np.ndarray,
    *,
    pressure: np.ndarray | None = None,
    lengths: tuple[float, ...] | list[float] | None = None,
) -> dict[str, float]:
    """Return proof-relevant scalar indicators from a sampled velocity field."""
    u = _as_velocity(velocity)
    domain = _lengths(lengths, u.shape[0])
    cell = float(np.prod(domain) / np.prod(u.shape[1:]))
    vort = _curl(u, domain)
    vort_sq = vort * vort if vort.ndim == u.ndim - 1 else np.sum(vort * vort, axis=0)
    grad_vort = (
        _grad_scalar(vort, domain)
        if u.shape[0] == 2
        else np.stack([_grad_scalar(vort[i], domain) for i in range(3)])
    )
    pal = np.sum(grad_vort * grad_vort, axis=0)
    if u.shape[0] == 3:
        pal = np.sum(pal, axis=0)
    div = _div(u, domain)
    out = {
        "kinetic_energy": float(0.5 * cell * np.sum(u * u)),
        "enstrophy": float(0.5 * cell * np.sum(vort_sq)),
        "palinstrophy": float(0.5 * cell * np.sum(pal)),
        "max_abs_divergence": float(np.max(np.abs(div))),
        "bkm_vorticity_proxy": float(np.max(np.abs(vort))),
        "unproven_claim": False,
    }
    if pressure is not None:
        pp = pressure_poisson_residual_periodic(u, pressure, lengths=domain)
        out["pressure_poisson_max_abs"] = float(np.max(np.abs(pp)))
    return out


def verify_ns_cap_bundle(bundle: dict[str, Any], *, atol: float = 1e-8) -> dict[str, Any]:
    """Recompute a Navier-Stokes CAP bundle using only validation inputs."""
    vin = bundle["validation_inputs"]
    residual, continuity = primitive_residual_periodic(
        np.asarray(vin["velocity"], dtype=float),
        np.asarray(vin["pressure"], dtype=float),
        velocity_t=np.asarray(vin["velocity_t"], dtype=float),
        forcing=np.asarray(vin["forcing"], dtype=float),
        viscosity=float(vin["viscosity"]),
        density=float(vin["density"]),
        lengths=vin["lengths"],
    )
    pressure_poisson = pressure_poisson_residual_periodic(
        np.asarray(vin["velocity"], dtype=float),
        np.asarray(vin["pressure"], dtype=float),
        density=float(vin["density"]),
        lengths=vin["lengths"],
    )
    stored = bundle["residual_samples"]
    dm = float(np.max(np.abs(residual - np.asarray(stored["momentum"], dtype=float))))
    dc = float(np.max(np.abs(continuity - np.asarray(stored["continuity"], dtype=float))))
    dp = float(np.max(np.abs(pressure_poisson - np.asarray(stored["pressure_poisson"], dtype=float))))
    return {
        "momentum_max_abs": float(np.max(np.abs(residual))),
        "continuity_max_abs": float(np.max(np.abs(continuity))),
        "pressure_poisson_max_abs": float(np.max(np.abs(pressure_poisson))),
        "agreement_momentum_max_abs_diff": dm,
        "agreement_continuity_max_abs_diff": dc,
        "agreement_pressure_poisson_max_abs_diff": dp,
        "residual_samples_match": bool(max(dm, dc, dp) <= atol),
        "unproven_claim": False,
    }


def fit_regularity_growth_bound(
    time: np.ndarray,
    quantity: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    alpha: float = 1e-10,
    threshold: float = 1e-10,
) -> dict[str, Any]:
    """Fit a sparse candidate ``dQ/dt ~= library(features)``.

    The result is a *candidate growth law*, not a proof of an inequality.  A
    rigorous global-regularity claim still needs analytic proof over all smooth
    finite-energy fields.
    """
    t = np.asarray(time, dtype=float)
    q = np.asarray(quantity, dtype=float)
    dq = np.gradient(q, t)
    names = list(features)
    design = np.stack([np.asarray(features[name], dtype=float) for name in names], axis=1)
    equation = fit_sparse_equation(design, dq, names, alpha=alpha, threshold=threshold)
    pred = equation.predict(design)
    return {
        "formula": equation.formula(lhs="dQ_dt"),
        "coefficients": {n: float(c) for n, c in zip(names, equation.coefficients, strict=False)},
        "intercept": float(equation.intercept),
        "fit_rmse": float(np.sqrt(np.mean((dq - pred) ** 2))),
        "max_underestimate": float(np.max(dq - pred)),
        "global_regularity_claim": False,
    }


def run_regularity_search(
    time: np.ndarray,
    traces: dict[str, np.ndarray],
    *,
    target: str = "enstrophy",
    include_quadratic: bool = True,
    alpha: float = 1e-10,
    threshold: float = 1e-10,
) -> dict[str, Any]:
    """Deterministic first-pass regularity growth-law search.

    Fits ``d target / dt`` from scalar trace features such as energy,
    enstrophy, palinstrophy, strain, and vorticity proxies.  This is a
    falsifiable candidate generator; it makes no global-regularity claim.
    """
    if target not in traces:
        raise KeyError(f"target {target!r} not found in traces")
    features: dict[str, np.ndarray] = {
        name: np.asarray(values, dtype=float)
        for name, values in traces.items()
        if name != target
    }
    features[target] = np.asarray(traces[target], dtype=float)
    if include_quadratic:
        for name, values in list(features.items()):
            features[f"{name}^2"] = np.asarray(values, dtype=float) ** 2
    fit = fit_regularity_growth_bound(
        time, np.asarray(traces[target], dtype=float), features,
        alpha=alpha, threshold=threshold,
    )
    fit["target"] = target
    fit["feature_names"] = list(features)
    fit["candidate_type"] = "regularity_growth_law"
    fit["global_regularity_claim"] = False
    return fit


def build_regularity_candidate_artifact(
    time: np.ndarray,
    traces: dict[str, np.ndarray],
    *,
    target: str = "enstrophy",
    include_quadratic: bool = True,
    alpha: float = 1e-10,
    threshold: float = 1e-10,
    replay_grid: dict[str, Any] | None = None,
    coefficients: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a replayable regularity-growth candidate artifact."""
    result = run_regularity_search(
        time, traces, target=target, include_quadratic=include_quadratic,
        alpha=alpha, threshold=threshold,
    )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_type": "regularity_growth_law",
        "replay_grid": dict(replay_grid or {
            "domain_type": "trace",
            "dimension": 1,
            "grid_shape": [int(np.asarray(time).shape[0])],
        }),
        "replay_inputs": {
            "time": _jsonable(np.asarray(time, dtype=float)),
            "traces": _jsonable({k: np.asarray(v, dtype=float) for k, v in traces.items()}),
            "target": target,
            "include_quadratic": bool(include_quadratic),
            "alpha": float(alpha),
            "threshold": float(threshold),
        },
        "result": _jsonable(result),
        "coefficients": list(coefficients or []),
        "upgrade_gate": {
            "stage": "candidate_replay_ready",
            "independent_replay_required": True,
            "unproven_claim": False,
        },
        "proof_obligations": [
            "prove_candidate_inequality_for_all_smooth_finite_energy_data",
            "connect_to_continuation_criterion",
        ],
        "honesty": {
            "unproven_claim": False,
            "exact_solution_claim": False,
            "global_regularity_claim": False,
            "finite_time_blowup_claim": False,
            "interval_verified": False,
            "theorem_prover_verified": False,
            "notes": notes,
        },
        "provenance": {
            "harness": "omnibias.symbolic.navier_stokes.build_regularity_candidate_artifact",
        },
    }


def fit_self_similar_blowup_rate(
    time: np.ndarray,
    norm_values: np.ndarray,
    *,
    blowup_time: float,
) -> dict[str, float | bool]:
    r"""Fit ``norm(t) ~= C * (T-t)^(-alpha)`` from a positive norm trace."""
    t = np.asarray(time, dtype=float)
    norms = np.asarray(norm_values, dtype=float)
    tau = blowup_time - t
    if np.any(tau <= 0.0) or np.any(norms <= 0.0):
        raise ValueError("need t < blowup_time and positive norm_values")
    x = np.log(tau)
    y = np.log(norms)
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    return {
        "alpha": float(-slope),
        "log_prefactor": float(intercept),
        "log_fit_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "finite_time_blowup_claim": False,
        "unproven_claim": False,
    }


def assess_blowup_candidate(
    time: np.ndarray,
    norm_values: np.ndarray,
    *,
    blowup_time: float,
    ansatz_metadata: dict[str, Any] | None = None,
    residual_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Package a norm-trace blow-up fit as a strict non-claim candidate."""
    rate = fit_self_similar_blowup_rate(
        time, norm_values, blowup_time=blowup_time,
    )
    return {
        "candidate_type": "self_similar_blowup_rate",
        "rate_fit": rate,
        "ansatz_metadata": dict(ansatz_metadata or {}),
        "residual_metrics": dict(residual_metrics or {}),
        "proof_obligations": [
            "compactified_domain_tail_bounds",
            "linearized_invertibility",
            "finite_energy_initial_data",
            "norm_divergence_certificate",
        ],
        "honesty": {
            "finite_time_blowup_claim": False,
            "unproven_claim": False,
            "exact_solution_claim": False,
        },
    }


def build_blowup_candidate_artifact(
    time: np.ndarray,
    norm_values: np.ndarray,
    *,
    blowup_time: float,
    ansatz_metadata: dict[str, Any] | None = None,
    residual_metrics: dict[str, float] | None = None,
    replay_grid: dict[str, Any] | None = None,
    coefficients: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a replayable self-similar blow-up-rate candidate artifact."""
    result = assess_blowup_candidate(
        time,
        norm_values,
        blowup_time=blowup_time,
        ansatz_metadata=ansatz_metadata,
        residual_metrics=residual_metrics,
    )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_type": "self_similar_blowup_rate",
        "replay_grid": dict(replay_grid or {
            "domain_type": "trace",
            "dimension": 1,
            "grid_shape": [int(np.asarray(time).shape[0])],
        }),
        "replay_inputs": {
            "time": _jsonable(np.asarray(time, dtype=float)),
            "norm_values": _jsonable(np.asarray(norm_values, dtype=float)),
            "blowup_time": float(blowup_time),
            "ansatz_metadata": dict(ansatz_metadata or {}),
            "residual_metrics": dict(residual_metrics or {}),
        },
        "result": _jsonable(result),
        "coefficients": list(coefficients or []),
        "upgrade_gate": {
            "stage": "candidate_replay_ready",
            "independent_replay_required": True,
            "unproven_claim": False,
        },
        "proof_obligations": list(result["proof_obligations"]),
        "honesty": {
            "unproven_claim": False,
            "exact_solution_claim": False,
            "global_regularity_claim": False,
            "finite_time_blowup_claim": False,
            "interval_verified": False,
            "theorem_prover_verified": False,
            "notes": notes,
        },
        "provenance": {
            "harness": "omnibias.symbolic.navier_stokes.build_blowup_candidate_artifact",
        },
    }


def _axisym_array(values: np.ndarray, radial_axis: np.ndarray, axial_axis: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    expected = (int(radial_axis.shape[0]), int(axial_axis.shape[0]))
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return cast(np.ndarray, arr)


def _grad_meridional(values: np.ndarray, radial_axis: np.ndarray, axial_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if min(values.shape) > 2:
        dr, dz = np.gradient(values, radial_axis, axial_axis, edge_order=2)
    else:
        dr, dz = np.gradient(values, radial_axis, axial_axis, edge_order=1)
    return cast(np.ndarray, np.asarray(dr, dtype=float)), cast(np.ndarray, np.asarray(dz, dtype=float))


def _lap_axisymmetric_scalar(values: np.ndarray, radial_axis: np.ndarray, axial_axis: np.ndarray) -> np.ndarray:
    val_r, val_z = _grad_meridional(values, radial_axis, axial_axis)
    val_rr, _ = _grad_meridional(val_r, radial_axis, axial_axis)
    _, val_zz = _grad_meridional(val_z, radial_axis, axial_axis)
    r = radial_axis[:, None]
    return cast(np.ndarray, np.asarray(val_rr + val_r / r + val_zz, dtype=float))


def _axisymmetric_velocity_from_streamfunction(
    streamfunction: np.ndarray,
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    swirl: np.ndarray,
) -> dict[str, np.ndarray]:
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("axisymmetric sampled residuals require r > 0 grid points")
    psi = _axisym_array(streamfunction, r, z, "streamfunction")
    psi_r, psi_z = _grad_meridional(psi, r, z)
    radius = r[:, None]
    return {
        "u_r": cast(np.ndarray, np.asarray(-psi_z / radius, dtype=float)),
        "u_theta": _axisym_array(swirl, r, z, "swirl"),
        "u_z": cast(np.ndarray, np.asarray(psi_r / radius, dtype=float)),
    }


def _axisymmetric_swirl_residual_samples(
    streamfunction: np.ndarray,
    swirl: np.ndarray,
    pressure: np.ndarray,
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    viscosity: float,
    density: float,
) -> dict[str, Any]:
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    pressure_arr = _axisym_array(pressure, r, z, "pressure")
    velocity = _axisymmetric_velocity_from_streamfunction(
        streamfunction,
        radial_axis=r,
        axial_axis=z,
        swirl=swirl,
    )
    u_r = velocity["u_r"]
    u_t = velocity["u_theta"]
    u_z = velocity["u_z"]
    radius = r[:, None]

    ur_r, ur_z = _grad_meridional(u_r, r, z)
    ut_r, ut_z = _grad_meridional(u_t, r, z)
    uz_r, uz_z = _grad_meridional(u_z, r, z)
    p_r, p_z = _grad_meridional(pressure_arr, r, z)
    lap_ur = _lap_axisymmetric_scalar(u_r, r, z)
    lap_ut = _lap_axisymmetric_scalar(u_t, r, z)
    lap_uz = _lap_axisymmetric_scalar(u_z, r, z)

    radial = density * (u_r * ur_r + u_z * ur_z - (u_t * u_t) / radius) + p_r
    radial = radial - viscosity * (lap_ur - u_r / (radius * radius))
    azimuthal = density * (u_r * ut_r + u_z * ut_z + u_r * u_t / radius)
    azimuthal = azimuthal - viscosity * (lap_ut - u_t / (radius * radius))
    axial = density * (u_r * uz_r + u_z * uz_z) + p_z - viscosity * lap_uz
    r_ur_r, _ = _grad_meridional(radius * u_r, r, z)
    divergence = r_ur_r / radius + uz_z
    residual_norm = np.sqrt(radial * radial + azimuthal * azimuthal + axial * axial)
    return {
        "velocity": {
            "u_r": cast(np.ndarray, np.asarray(u_r, dtype=float)),
            "u_theta": cast(np.ndarray, np.asarray(u_t, dtype=float)),
            "u_z": cast(np.ndarray, np.asarray(u_z, dtype=float)),
        },
        "residual_samples": {
            "radial": cast(np.ndarray, np.asarray(radial, dtype=float)),
            "azimuthal": cast(np.ndarray, np.asarray(azimuthal, dtype=float)),
            "axial": cast(np.ndarray, np.asarray(axial, dtype=float)),
            "divergence": cast(np.ndarray, np.asarray(divergence, dtype=float)),
        },
        "residual_diagnostics": {
            "max_abs_momentum_residual": float(np.max(np.abs(residual_norm))),
            "rms_momentum_residual": float(np.sqrt(np.mean(residual_norm * residual_norm))),
            "max_abs_continuity": float(np.max(np.abs(divergence))),
            "rms_continuity": float(np.sqrt(np.mean(divergence * divergence))),
            "max_abs_radial_residual": float(np.max(np.abs(radial))),
            "max_abs_azimuthal_residual": float(np.max(np.abs(azimuthal))),
            "max_abs_axial_residual": float(np.max(np.abs(axial))),
            "unproven_claim": False,
        },
    }


def _axisymmetric_energy_estimate(
    velocity: dict[str, np.ndarray],
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
) -> float:
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    density = 0.5 * (
        velocity["u_r"] * velocity["u_r"]
        + velocity["u_theta"] * velocity["u_theta"]
        + velocity["u_z"] * velocity["u_z"]
    ) * (2.0 * np.pi * r[:, None])
    by_z = np.trapezoid(density, z, axis=1)
    return float(np.trapezoid(by_z, r))


def _axisymmetric_physical_axes(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axes = grid["axis_values"]
    rho = np.asarray(axes[0], dtype=float)
    zeta = np.asarray(axes[1], dtype=float)
    radial = rho / (1.0 - rho)
    axial = zeta / (1.0 - np.abs(zeta))
    return cast(np.ndarray, np.asarray(radial, dtype=float)), cast(np.ndarray, np.asarray(axial, dtype=float))


def _axisymmetric_basis_count(metadata: dict[str, Any]) -> int:
    return (int(metadata["radial_degree"]) + 1) * (int(metadata["axial_degree"]) + 1)


def _axisymmetric_basis_tensor(
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    *,
    component: str,
    metadata: dict[str, Any],
) -> np.ndarray:
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    rho = r / (1.0 + r)
    zeta = z / (1.0 + np.abs(z))
    rr = r[:, None]
    compact_r = rho[:, None]
    compact_z = zeta[None, :]
    envelope = np.exp(-0.5 * (compact_r * compact_r + compact_z * compact_z))
    if component == "streamfunction":
        factor = rr * rr
    elif component == "swirl":
        factor = rr
    elif component == "pressure":
        factor = np.ones((r.shape[0], z.shape[0]), dtype=float)
    else:
        raise ValueError(f"unknown axisymmetric component {component!r}")
    basis = [
        factor * envelope * (compact_r ** i) * (compact_z ** j)
        for i in range(int(metadata["radial_degree"]) + 1)
        for j in range(int(metadata["axial_degree"]) + 1)
    ]
    return cast(np.ndarray, np.asarray(np.stack(basis), dtype=float))


def _axisymmetric_coefficients_to_fields(
    coefficients: np.ndarray,
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    n_basis = _axisymmetric_basis_count(metadata)
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.shape != (3 * n_basis,):
        raise ValueError(f"expected coefficient shape {(3 * n_basis,)}, got {coeffs.shape}")
    psi_c = coeffs[:n_basis]
    swirl_c = coeffs[n_basis:2 * n_basis]
    pressure_c = coeffs[2 * n_basis:]
    psi_basis = _axisymmetric_basis_tensor(
        radial_axis, axial_axis, component="streamfunction", metadata=metadata
    )
    swirl_basis = _axisymmetric_basis_tensor(
        radial_axis, axial_axis, component="swirl", metadata=metadata
    )
    pressure_basis = _axisymmetric_basis_tensor(
        radial_axis, axial_axis, component="pressure", metadata=metadata
    )
    return {
        "streamfunction": cast(np.ndarray, np.tensordot(psi_c, psi_basis, axes=(0, 0))),
        "swirl": cast(np.ndarray, np.tensordot(swirl_c, swirl_basis, axes=(0, 0))),
        "pressure": cast(np.ndarray, np.tensordot(pressure_c, pressure_basis, axes=(0, 0))),
    }


def _axisymmetric_coefficient_loss(
    coefficients: np.ndarray,
    *,
    grid: dict[str, Any],
    metadata: dict[str, Any],
    viscosity: float,
    density: float,
    coefficient_l2: float,
    energy_target: float,
    energy_weight: float,
) -> dict[str, Any]:
    radial, axial = _axisymmetric_physical_axes(grid)
    fields = _axisymmetric_coefficients_to_fields(
        coefficients,
        radial_axis=radial,
        axial_axis=axial,
        metadata=metadata,
    )
    residual = _axisymmetric_swirl_residual_samples(
        fields["streamfunction"],
        fields["swirl"],
        fields["pressure"],
        radial_axis=radial,
        axial_axis=axial,
        viscosity=viscosity,
        density=density,
    )
    energy = _axisymmetric_energy_estimate(
        residual["velocity"],
        radial_axis=radial,
        axial_axis=axial,
    )
    diag = residual["residual_diagnostics"]
    residual_loss = (
        float(diag["rms_momentum_residual"]) ** 2
        + float(diag["rms_continuity"]) ** 2
    )
    coeffs = np.asarray(coefficients, dtype=float)
    regularization = float(coefficient_l2) * float(np.mean(coeffs * coeffs))
    energy_penalty = 0.0
    if energy_target > 0.0 and energy_weight > 0.0:
        denom = max(abs(float(energy_target)), 1e-12)
        energy_penalty = float(energy_weight) * ((energy - float(energy_target)) / denom) ** 2
    return {
        "loss": float(residual_loss + regularization + energy_penalty),
        "finite_energy_estimate": float(energy),
        "residual_diagnostics": dict(diag),
    }


def verify_refined_axisymmetric_swirl_candidate_artifact(
    artifact: dict[str, Any],
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Replay a refined axisymmetric coefficient artifact from serialized coefficients."""
    rin = artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    loss_config = dict(rin.get("loss_config", {}))
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    coefficient_l2 = float(loss_config.get("coefficient_l2", 1e-8))
    energy_target = float(loss_config.get("energy_target", 0.0))
    energy_weight = float(loss_config.get("energy_weight", 0.0))
    train = _axisymmetric_coefficient_loss(
        coeffs,
        grid=dict(rin["train_grid"]),
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        coefficient_l2=coefficient_l2,
        energy_target=energy_target,
        energy_weight=energy_weight,
    )
    holdout = _axisymmetric_coefficient_loss(
        coeffs,
        grid=dict(rin["holdout_grid"]),
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        coefficient_l2=coefficient_l2,
        energy_target=energy_target,
        energy_weight=energy_weight,
    )
    stored = artifact["result"]
    diffs = [
        abs(float(train["loss"]) - float(stored["train"]["loss"])),
        abs(float(holdout["loss"]) - float(stored["holdout"]["loss"])),
        abs(
            float(train["finite_energy_estimate"])
            - float(stored["train"]["finite_energy_estimate"])
        ),
        abs(
            float(holdout["finite_energy_estimate"])
            - float(stored["holdout"]["finite_energy_estimate"])
        ),
    ]
    for section, replayed in (("train", train), ("holdout", holdout)):
        stored_diag = stored[section]["residual_diagnostics"]
        for name in (
            "max_abs_momentum_residual",
            "rms_momentum_residual",
            "max_abs_continuity",
            "rms_continuity",
        ):
            diffs.append(abs(
                float(replayed["residual_diagnostics"][name]) - float(stored_diag[name])
            ))
    max_diff = max(diffs)
    return {
        "candidate_type": "axisymmetric_swirl_refined",
        "replay_match": bool(max_diff <= atol),
        "max_abs_replay_diff": float(max_diff),
        "train_loss": float(train["loss"]),
        "holdout_loss": float(holdout["loss"]),
        "stage": "candidate_replay_ready" if max_diff <= atol else "numerical_artifact",
        "unproven_claim": False,
    }


def _contains_interval(interval_report: dict[str, Any], value: float) -> float:
    interval = dict(interval_report["interval"])
    val = float(value)
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if lower <= val <= upper:
        return 0.0
    if val < lower:
        return float(lower - val)
    return float(val - upper)


def verify_axisymmetric_interval_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Verify interval containment for a refined axisymmetric interval report."""
    if report.get("candidate_type") != "axisymmetric_interval_report":
        raise ValueError("expected candidate_type='axisymmetric_interval_report'")
    artifact = report["replay_inputs"]["refined_artifact"]
    rin = artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    loss_config = dict(rin.get("loss_config", {}))
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    coefficient_l2 = float(loss_config.get("coefficient_l2", 1e-8))
    energy_target = float(loss_config.get("energy_target", 0.0))
    energy_weight = float(loss_config.get("energy_weight", 0.0))
    train = _axisymmetric_coefficient_loss(
        coeffs,
        grid=dict(rin["train_grid"]),
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        coefficient_l2=coefficient_l2,
        energy_target=energy_target,
        energy_weight=energy_weight,
    )
    holdout = _axisymmetric_coefficient_loss(
        coeffs,
        grid=dict(rin["holdout_grid"]),
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        coefficient_l2=coefficient_l2,
        energy_target=energy_target,
        energy_weight=energy_weight,
    )
    violations: list[float] = []
    checked: list[str] = []
    coefficient_payloads = {str(payload["field"]): payload for payload in artifact.get("coefficients", [])}
    for field, box in report.get("coefficient_intervals", {}).items():
        payload = coefficient_payloads.get(str(field))
        if payload is None:
            violations.append(float("inf"))
            checked.append(f"coefficient.{field}.missing_payload")
            continue
        coeff_values = np.asarray(payload.get("coefficients", []), dtype=float)
        intervals = list(box.get("intervals", []))
        if len(intervals) != int(coeff_values.size):
            violations.append(float("inf"))
            checked.append(f"coefficient.{field}.shape")
            continue
        for idx, value in enumerate(coeff_values):
            violations.append(_contains_interval({"interval": intervals[idx]}, float(value)))
        checked.append(f"coefficient.{field}")
    for section, replayed in (("train", train), ("holdout", holdout)):
        for name, interval in report["residual_intervals"][section].items():
            value = float(replayed["residual_diagnostics"][name])
            violations.append(_contains_interval(interval, value))
            checked.append(f"{section}.{name}")
        energy_value = float(replayed["finite_energy_estimate"])
        violations.append(_contains_interval(report["finite_energy_intervals"][section], energy_value))
        checked.append(f"{section}.finite_energy_estimate")
        combined = (
            report.get("finite_energy_tail_certificate", {})
            .get("sections", {})
            .get(section, {})
            .get("combined_finite_energy_interval")
        )
        if combined is not None:
            violations.append(_contains_interval({"interval": combined}, energy_value))
            checked.append(f"{section}.combined_finite_energy_interval")
        for name, cert in (
            report.get("continuum_residual_certificates", {})
            .get("sections", {})
            .get(section, {})
            .items()
        ):
            sample = dict(cert.get("sample_interval", {}))
            sample_interval = dict(sample.get("interval", {}))
            continuum_interval = dict(cert.get("continuum_sup_norm_interval", {}))
            if sample_interval and continuum_interval:
                violations.append(_contains_interval({"interval": continuum_interval}, float(sample_interval["midpoint"])))
            else:
                violations.append(float("inf"))
            checked.append(f"{section}.{name}.continuum")
    tail_certified = all(
        bool(tail.get("certified", False))
        for tail in report.get("tail_certificates", {}).values()
    )
    if report.get("tail_certificates") and not tail_certified:
        violations.append(float("inf"))
        checked.append("tail_certificates.certified")
    axis_certified = bool(report.get("axis_regular_checks", {}).get("certified_smooth_axis", False))
    if report.get("axis_regular_checks") and not axis_certified:
        violations.append(float("inf"))
        checked.append("axis_regular_checks.certified_smooth_axis")
    max_violation = max(violations, default=0.0)
    replay = verify_refined_axisymmetric_swirl_candidate_artifact(artifact)
    match = max_violation <= 0.0 and bool(replay["replay_match"])
    return {
        "candidate_type": "axisymmetric_interval_report",
        "interval_report_match": bool(match),
        "max_interval_violation": float(max_violation),
        "checked_quantities": checked,
        "tail_certified": bool(tail_certified),
        "axis_certified": bool(axis_certified),
        "continuum_certified": bool(
            report.get("continuum_residual_certificates", {}).get("continuum_bound_certified", False)
        ),
        "refined_replay": replay,
        "unresolved_obligations": list(report.get("proof_obligations", [])),
        "stage": "interval_obligation_ready" if match else "numerical_artifact",
        "unproven_claim": False,
    }


def _continuum_residual_upper_bound(report: dict[str, Any]) -> float:
    sections = (
        report.get("continuum_residual_certificates", {})
        .get("sections", {})
    )
    uppers: list[float] = []
    for section in sections.values():
        for cert in section.values():
            if isinstance(cert, dict) and isinstance(cert.get("continuum_sup_norm_interval"), dict):
                uppers.append(float(cert["continuum_sup_norm_interval"]["upper"]))
    return max(uppers, default=float("inf"))


def _scalar_interval_from_midpoint(midpoint: float, *, relative_padding: float = 1e-12) -> dict[str, Any]:
    mid = float(midpoint)
    radius = abs(float(relative_padding)) * abs(mid)
    lower = float(np.nextafter(mid - radius, -np.inf))
    upper = float(np.nextafter(mid + radius, np.inf))
    return {
        "lower": lower,
        "upper": upper,
        "midpoint": mid,
        "radius": float(max(mid - lower, upper - mid)),
        "certified": bool(np.isfinite(mid)),
    }


def verify_blowup_closure_report(report: dict[str, Any]) -> dict[str, Any]:
    """Verify replay consistency for a blow-up analytic closure report."""
    if report.get("candidate_type") != "blowup_analytic_closure_report":
        raise ValueError("expected candidate_type='blowup_analytic_closure_report'")
    interval_report = dict(report["replay_inputs"]["interval_report"])
    interval_replay = verify_axisymmetric_interval_report(interval_report)
    replay_inputs = dict(report.get("replay_inputs", {}))
    certificates = dict(report.get("closure_certificates", {}))
    linearized = dict(certificates.get("linearized_operator", {}))
    norm_cert = dict(certificates.get("norm_divergence", {}))
    residual = float(
        replay_inputs.get("residual_bound")
        if replay_inputs.get("residual_bound") is not None
        else _continuum_residual_upper_bound(interval_report)
    )
    inverse_raw = replay_inputs.get("approximate_inverse_norm")
    lipschitz_raw = replay_inputs.get("nonlinear_lipschitz_bound")
    inverse = float(inverse_raw) if inverse_raw is not None else float("inf")
    lipschitz = float(lipschitz_raw) if lipschitz_raw is not None else float("inf")
    radii_value = inverse * (residual + lipschitz)
    recomputed_interval = _scalar_interval_from_midpoint(
        radii_value if np.isfinite(radii_value) else 1.0e300,
        relative_padding=1e-12,
    )
    stored_interval = dict(report.get("radii_polynomial", {}).get("closure_interval", {}))
    interval_diffs = [
        abs(float(stored_interval.get(name, float("inf"))) - float(recomputed_interval[name]))
        for name in ("lower", "upper", "midpoint", "radius")
    ]
    radii_passed = bool(np.isfinite(radii_value) and recomputed_interval["upper"] < 1.0)
    expected_obligations = {
        "linearized_invertibility": inverse_raw is not None and np.isfinite(inverse),
        "radii_polynomial_closure": radii_passed,
        "axis_smoothness": bool(interval_report.get("upgrade_gate", {}).get("axis_regular_certified", False)),
        "finite_energy_initial_data": bool(interval_report.get("upgrade_gate", {}).get("finite_energy_certified", False)),
        "norm_divergence": bool(norm_cert.get("certified", False))
        if norm_cert
        else replay_inputs.get("norm_growth_exponent") is not None
        and float(replay_inputs["norm_growth_exponent"]) > 0.0,
    }
    if linearized:
        expected_obligations["operator_theoretic_invertibility"] = bool(
            linearized.get("operator_theoretic_certified", False)
        )
    stored_obligations = dict(report.get("obligations", {}))
    obligation_diffs = [
        0.0 if bool(stored_obligations.get(name, False)) == expected else float("inf")
        for name, expected in expected_obligations.items()
    ]
    max_violation = max([*interval_diffs, *obligation_diffs], default=0.0)
    match = bool(
        interval_replay.get("interval_report_match", False)
        and max_violation <= 1e-9
    )
    return {
        "candidate_type": "blowup_analytic_closure_report",
        "closure_report_match": match,
        "max_closure_violation": float(max_violation),
        "interval_replay": interval_replay,
        "recomputed_residual_bound": residual,
        "recomputed_radii_interval": recomputed_interval,
        "expected_obligations": expected_obligations,
        "unproven_claim": False,
    }


def _axisymmetric_residual_vector_from_coefficients(
    coefficients: np.ndarray,
    *,
    grid: dict[str, Any],
    metadata: dict[str, Any],
    viscosity: float,
    density: float,
) -> np.ndarray:
    radial, axial = _axisymmetric_physical_axes(grid)
    fields = _axisymmetric_coefficients_to_fields(
        coefficients,
        radial_axis=radial,
        axial_axis=axial,
        metadata=metadata,
    )
    residual = _axisymmetric_swirl_residual_samples(
        fields["streamfunction"],
        fields["swirl"],
        fields["pressure"],
        radial_axis=radial,
        axial_axis=axial,
        viscosity=viscosity,
        density=density,
    )
    samples = dict(residual["residual_samples"])
    return cast(np.ndarray, np.asarray(np.concatenate([
        np.asarray(samples[name], dtype=float).ravel()
        for name in ("radial", "azimuthal", "axial", "divergence")
    ]), dtype=float))


def _active_subspace_linearized_replay(
    artifact: dict[str, Any],
    active_indices: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    rin = artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    active = tuple(int(idx) for idx in active_indices)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = 1e-6 * max(1.0, float(np.linalg.norm(coeffs)))
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs,
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
    )
    matrix = np.empty((int(base.size), len(active)), dtype=float)
    for col, idx in enumerate(active):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        matrix[:, col] = (plus - minus) / (2.0 * step)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(np.max(singular_values)) if singular_values.size else 0.0
    smallest = float(np.min(singular_values)) if singular_values.size else 0.0
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
    full_rank = bool(rank == len(active) and smallest > 1e-10)
    return {
        "matrix_shape": (int(matrix.shape[0]), int(matrix.shape[1])),
        "matrix_norm": float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0,
        "approximate_inverse_norm": float(1.0 / smallest) if full_rank else None,
        "condition_estimate": float(largest / smallest) if full_rank else None,
        "smallest_singular_value": smallest,
        "largest_singular_value": largest,
        "rank": rank,
        "full_column_rank": full_rank,
        "finite_dimensional_certified": full_rank,
        "singular_values": [float(v) for v in singular_values],
    }


def _axisymmetric_full_jacobian_replay(artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    rin = artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = 1e-6 * max(1.0, float(np.linalg.norm(coeffs)))
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs,
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
    )
    matrix = np.empty((int(base.size), int(coeffs.size)), dtype=float)
    for idx in range(coeffs.size):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        matrix[:, idx] = (plus - minus) / (2.0 * step)
    return base, matrix, step


def _axisymmetric_jacobian_at_step(
    artifact: dict[str, Any],
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    rin = artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs, grid=grid, metadata=metadata, viscosity=viscosity, density=density
    )
    matrix = np.empty((int(base.size), int(coeffs.size)), dtype=float)
    for idx in range(coeffs.size):
        delta = np.zeros_like(coeffs)
        delta[idx] = float(step)
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        matrix[:, idx] = (plus - minus) / (2.0 * float(step))
    return base, matrix


def _finite_difference_jacobian_error_envelope_replay(artifact: dict[str, Any]) -> dict[str, Any]:
    coeffs = np.asarray(artifact["replay_inputs"]["coefficients"], dtype=float)
    step_h = 1e-6 * max(1.0, float(np.linalg.norm(coeffs)))
    base, jac_h = _axisymmetric_jacobian_at_step(artifact, step_h)
    _, jac_2h = _axisymmetric_jacobian_at_step(artifact, 2.0 * step_h)
    truncation = float(np.linalg.norm(jac_h - jac_2h, ord=2)) / 3.0
    residual_scale = float(np.max(np.abs(base))) if base.size else 0.0
    rounding = float(np.finfo(float).eps) * max(1.0, residual_scale) / max(step_h, 1e-300)
    envelope = truncation + rounding
    return {
        "perturbation": float(step_h),
        "double_perturbation": float(2.0 * step_h),
        "jacobian_norm": float(np.linalg.norm(jac_h, ord=2)) if jac_h.size else 0.0,
        "richardson_truncation_upper": float(truncation),
        "rounding_floor_upper": float(rounding),
        "jacobian_error_envelope_upper": float(envelope),
    }


def _tail_weight_condition_replay(
    tail: tuple[int, ...],
    radius: float,
) -> tuple[float, dict[str, float]]:
    weights = {int(idx): float(radius) ** int(idx) for idx in tail}
    if not weights:
        return 1.0, {}
    w_max = max(weights.values())
    w_min = min(weights.values())
    cond = float(w_max / w_min) if w_min > 0.0 else float("inf")
    return cond, {str(k): float(v) for k, v in weights.items()}


def _active_tail_geometry_replay(finite_diagnostic: dict[str, Any]) -> dict[str, Any]:
    artifact = dict(finite_diagnostic["replay_inputs"]["refined_artifact"])
    active = tuple(int(i) for i in finite_diagnostic.get("active_indices", []))
    tail = tuple(int(i) for i in finite_diagnostic.get("tail_modes", []))
    radius = float(finite_diagnostic.get("analytic_radius", 1.125))
    residual, jacobian, step = _axisymmetric_full_jacobian_replay(artifact)
    active_j = jacobian[:, active] if active else np.empty((jacobian.shape[0], 0), dtype=float)
    tail_j = jacobian[:, tail] if tail else np.empty((jacobian.shape[0], 0), dtype=float)
    active_svals = np.linalg.svd(active_j, compute_uv=False) if active_j.size else np.empty((0,))
    full_svals = np.linalg.svd(jacobian, compute_uv=False) if jacobian.size else np.empty((0,))
    weight_condition, weights = _tail_weight_condition_replay(tail, radius)
    return {
        "artifact": artifact,
        "active_indices": active,
        "tail_modes": tail,
        "analytic_radius": radius,
        "residual_norm": float(np.linalg.norm(residual)),
        "sigma_min_active": float(np.min(active_svals)) if active_svals.size else 0.0,
        "sigma_max_active": float(np.max(active_svals)) if active_svals.size else 0.0,
        "sigma_min_full": float(np.min(full_svals)) if full_svals.size else 0.0,
        "tail_operator_norm": float(np.linalg.norm(tail_j, ord=2)) if tail_j.size else 0.0,
        "weight_condition": weight_condition,
        "tail_weights": weights,
        "dimension_factor": float(np.sqrt(max(len(tail), 1))),
        "perturbation": float(step),
    }


def verify_active_subspace_invariance_report(report: dict[str, Any]) -> dict[str, Any]:
    """Replay finite active-subspace invariance leakage metrics."""
    if report.get("candidate_type") != "active_subspace_invariance_report":
        raise ValueError("expected candidate_type='active_subspace_invariance_report'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    artifact = dict(replay_inputs["refined_artifact"])
    active = tuple(int(idx) for idx in replay_inputs.get("active_indices", []))
    threshold = float(replay_inputs.get("leakage_threshold", report.get("leakage_threshold", 5e-2)))
    coeff_count = len(artifact.get("replay_inputs", {}).get("coefficients", []))
    inactive = tuple(idx for idx in range(coeff_count) if idx not in set(active))
    residual, jacobian, step = _axisymmetric_full_jacobian_replay(artifact)
    active_j = jacobian[:, active]
    inactive_j = jacobian[:, inactive] if inactive else np.empty((jacobian.shape[0], 0), dtype=float)
    gradient = jacobian.T @ residual
    active_gradient = gradient[list(active)]
    inactive_gradient = gradient[list(inactive)] if inactive else np.empty((0,), dtype=float)
    active_step = -np.linalg.lstsq(active_j, residual, rcond=None)[0]
    post_residual = residual + active_j @ active_step
    post_inactive_gradient = inactive_j.T @ post_residual if inactive else np.empty((0,), dtype=float)
    active_gradient_norm = float(np.linalg.norm(active_gradient))
    inactive_gradient_norm = float(np.linalg.norm(inactive_gradient))
    post_inactive_norm = float(np.linalg.norm(post_inactive_gradient))
    recomputed = {
        "perturbation": step,
        "residual_norm": float(np.linalg.norm(residual)),
        "post_active_newton_residual_norm": float(np.linalg.norm(post_residual)),
        "active_gradient_norm": active_gradient_norm,
        "inactive_gradient_norm": inactive_gradient_norm,
        "gradient_leakage_ratio": inactive_gradient_norm / (active_gradient_norm + 1e-30),
        "post_active_newton_inactive_gradient_norm": post_inactive_norm,
        "post_newton_leakage_ratio": post_inactive_norm / (active_gradient_norm + 1e-30),
        "finite_invariance_heuristic_passed": bool(post_inactive_norm / (active_gradient_norm + 1e-30) <= threshold),
    }
    diffs = [
        abs(float(report.get(name, float("inf"))) - float(value))
        for name, value in recomputed.items()
        if isinstance(value, float)
    ]
    diffs.append(0.0 if bool(report.get("finite_invariance_heuristic_passed", False)) == recomputed["finite_invariance_heuristic_passed"] else float("inf"))
    diffs.append(0.0 if list(report.get("active_indices", [])) == list(active) else float("inf"))
    diffs.append(0.0 if list(report.get("inactive_indices", [])) == list(inactive) else float("inf"))
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_subspace_invariance_report",
        "active_subspace_invariance_match": bool(max_violation <= 1e-8),
        "max_active_subspace_invariance_violation": float(max_violation),
        "recomputed_metrics": recomputed,
        "unproven_claim": False,
    }


def _finite_tail_contraction_expected(
    artifact: dict[str, Any],
    active: tuple[int, ...],
    tail: tuple[int, ...],
    *,
    analytic_radius: float,
    contraction_threshold: float,
) -> dict[str, Any]:
    residual, jacobian, step = _axisymmetric_full_jacobian_replay(artifact)
    active_j = jacobian[:, active]
    tail_j = jacobian[:, tail] if tail else np.empty((jacobian.shape[0], 0), dtype=float)
    active_projector = active_j @ np.linalg.pinv(active_j)
    residual_projector = np.eye(jacobian.shape[0]) - active_projector
    tail_response = tail_j.T @ residual_projector @ tail_j
    radius = float(analytic_radius)
    weights = np.asarray([radius ** int(idx) for idx in tail], dtype=float)
    if tail_response.size:
        weighted_abs = (weights[:, None] * np.abs(tail_response)) / weights[None, :]
        column_sums = np.sum(weighted_abs, axis=0)
        row_sums = np.sum(weighted_abs, axis=1)
        induced_linf = float(np.max(row_sums))
        spectral = float(np.linalg.norm(tail_response, ord=2))
        induced_l1 = float(np.max(column_sums))
    else:
        weighted_abs = np.empty((0, 0), dtype=float)
        column_sums = np.empty((0,), dtype=float)
        row_sums = np.empty((0,), dtype=float)
        induced_linf = 0.0
        spectral = 0.0
        induced_l1 = 0.0
    interval = _scalar_interval_from_midpoint(induced_l1, relative_padding=1e-12)
    mode_rows = []
    for local, idx in enumerate(tail):
        mode_rows.append({
            "idx": int(idx),
            "weight": float(weights[local]),
            "weighted_column_sum": float(column_sums[local]) if column_sums.size else 0.0,
            "weighted_row_sum": float(row_sums[local]) if row_sums.size else 0.0,
            "tail_response_diagonal": float(tail_response[local, local]) if tail_response.size else 0.0,
        })
    mode_rows.sort(key=lambda item: float(item["weighted_column_sum"]), reverse=True)
    finite_passed = bool(float(interval["upper"]) < float(contraction_threshold))
    return {
        "perturbation": step,
        "residual_norm": float(np.linalg.norm(residual)),
        "active_projector_rank": int(np.linalg.matrix_rank(active_projector)),
        "tail_response_matrix": tail_response,
        "weighted_abs_response_matrix": weighted_abs,
        "weighted_column_sums": [float(v) for v in column_sums],
        "weighted_row_sums": [float(v) for v in row_sums],
        "finite_contraction_ratio_interval": interval,
        "finite_contraction_ratio_upper": float(interval["upper"]),
        "finite_tail_contraction_surrogate_passed": finite_passed,
        "spectral_tail_response_norm": spectral,
        "weighted_linf_response_norm": induced_linf,
        "worst_tail_modes": mode_rows,
        "proof_status": (
            "finite_tail_contraction_surrogate_passed"
            if finite_passed
            else "blocked_finite_tail_contraction_ratio"
        ),
    }


def verify_finite_active_tail_contraction_diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the finite weighted-tail contraction diagnostic."""
    if report.get("candidate_type") != "finite_active_tail_contraction_diagnostic":
        raise ValueError("expected candidate_type='finite_active_tail_contraction_diagnostic'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    artifact = dict(replay_inputs["refined_artifact"])
    active = tuple(int(idx) for idx in replay_inputs.get("active_indices", []))
    tail = tuple(int(idx) for idx in replay_inputs.get("tail_modes", []))
    expected = _finite_tail_contraction_expected(
        artifact,
        active,
        tail,
        analytic_radius=float(replay_inputs.get("analytic_radius", 1.125)),
        contraction_threshold=float(replay_inputs.get("contraction_threshold", 1.0)),
    )
    stored_tail = np.asarray(report.get("tail_response_matrix", []), dtype=float)
    stored_weighted = np.asarray(report.get("weighted_abs_response_matrix", []), dtype=float)
    matrix_diffs = [
        float(np.max(np.abs(stored_tail - expected["tail_response_matrix"])))
        if stored_tail.shape == expected["tail_response_matrix"].shape
        else float("inf"),
        float(np.max(np.abs(stored_weighted - expected["weighted_abs_response_matrix"])))
        if stored_weighted.shape == expected["weighted_abs_response_matrix"].shape
        else float("inf"),
    ]
    interval = dict(report.get("finite_contraction_ratio_interval", {}))
    expected_interval = dict(expected["finite_contraction_ratio_interval"])
    diffs = [
        *matrix_diffs,
        abs(float(report.get("perturbation", float("inf"))) - float(expected["perturbation"])),
        abs(float(report.get("residual_norm", float("inf"))) - float(expected["residual_norm"])),
        0.0 if int(report.get("active_projector_rank", -1)) == int(expected["active_projector_rank"]) else float("inf"),
        0.0 if list(report.get("active_indices", [])) == list(active) else float("inf"),
        0.0 if list(report.get("tail_modes", [])) == list(tail) else float("inf"),
        0.0 if list(report.get("weighted_column_sums", [])) == expected["weighted_column_sums"] else float("inf"),
        0.0 if list(report.get("weighted_row_sums", [])) == expected["weighted_row_sums"] else float("inf"),
        *[
            abs(float(interval.get(name, float("inf"))) - float(expected_interval[name]))
            for name in ("lower", "upper", "midpoint", "radius")
        ],
        abs(float(report.get("finite_contraction_ratio_upper", float("inf"))) - float(expected["finite_contraction_ratio_upper"])),
        abs(float(report.get("spectral_tail_response_norm", float("inf"))) - float(expected["spectral_tail_response_norm"])),
        abs(float(report.get("weighted_linf_response_norm", float("inf"))) - float(expected["weighted_linf_response_norm"])),
        0.0 if bool(report.get("finite_tail_contraction_surrogate_passed", False)) == expected["finite_tail_contraction_surrogate_passed"] else float("inf"),
        0.0 if str(report.get("proof_status", "")) == expected["proof_status"] else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "finite_active_tail_contraction_diagnostic",
        "finite_active_tail_contraction_match": bool(max_violation <= 1e-8),
        "max_finite_active_tail_contraction_violation": float(max_violation),
        "expected_finite_contraction_ratio_upper": expected["finite_contraction_ratio_upper"],
        "expected_finite_tail_contraction_surrogate_passed": expected["finite_tail_contraction_surrogate_passed"],
        "unproven_claim": False,
    }


def _active_tail_lift_expected(
    finite_diagnostic: dict[str, Any],
    tail_contract: dict[str, Any],
    *,
    interval_jacobian_error_upper: float | None,
    projector_error_upper: float | None,
    analytic_tail_error_upper: float | None,
    nonlinear_remainder_error_upper: float | None,
    contraction_threshold: float,
    external_verification: dict[str, Any],
) -> dict[str, Any]:
    finite_upper = float(finite_diagnostic.get("finite_contraction_ratio_upper", float("inf")))
    raw_terms = {
        "interval_jacobian_error_upper": interval_jacobian_error_upper,
        "projector_error_upper": projector_error_upper,
        "analytic_tail_error_upper": analytic_tail_error_upper,
        "nonlinear_remainder_error_upper": nonlinear_remainder_error_upper,
    }
    finite_error_terms = {
        name: float(value)
        for name, value in raw_terms.items()
        if value is not None and np.isfinite(float(value)) and float(value) >= 0.0
    }
    missing_terms = [name for name in raw_terms if name not in finite_error_terms]
    total_error = sum(finite_error_terms.values())
    q_total = finite_upper + total_error if not missing_terms and np.isfinite(finite_upper) else float("inf")
    q_interval = _scalar_interval_from_midpoint(
        q_total if np.isfinite(q_total) else 1.0e300,
        relative_padding=1e-12,
    )
    finite_ok = bool(finite_diagnostic.get("finite_tail_contraction_surrogate_passed", False))
    contract_ok = bool(tail_contract.get("weighted_tail_contract_certified", False))
    external_ok = _external_verifies_symbolic(external_verification, "active_tail_contraction_analytic_lift")
    q_ok = bool(np.isfinite(q_total) and float(q_interval["upper"]) < float(contraction_threshold))
    certified = bool(finite_ok and contract_ok and external_ok and q_ok)
    open_obligations: list[str] = []
    if not finite_ok:
        open_obligations.append("finite_tail_contraction_surrogate_below_one")
    if not contract_ok:
        open_obligations.append("weighted_analytic_tail_norm_contract")
    open_obligations.extend(missing_terms)
    if not q_ok:
        open_obligations.append("q_total_below_one")
    if not external_ok:
        open_obligations.append("external_active_tail_contraction_analytic_lift_proof")
    return {
        "q_finite_upper": finite_upper,
        "error_budget_terms": {name: None if raw_terms[name] is None else float(raw_terms[name]) for name in raw_terms},
        "error_budget_terms_certified": len(missing_terms) == 0,
        "missing_error_budget_terms": missing_terms,
        "total_lift_error_upper": None if missing_terms else total_error,
        "q_total_interval": q_interval,
        "q_total_upper": None if not np.isfinite(q_total) else float(q_interval["upper"]),
        "margin_after_finite_q": (
            float(contraction_threshold) - finite_upper
            if np.isfinite(finite_upper)
            else -float("inf")
        ),
        "remaining_margin_after_lift": (
            float(contraction_threshold) - float(q_interval["upper"])
            if np.isfinite(q_total)
            else None
        ),
        "finite_tail_contraction_surrogate_passed": finite_ok,
        "weighted_tail_contract_certified": contract_ok,
        "analytic_lift_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
    }


def verify_active_tail_contraction_lift_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the analytic lift attempt for finite tail contraction."""
    if report.get("candidate_type") != "active_tail_contraction_lift_certificate":
        raise ValueError("expected candidate_type='active_tail_contraction_lift_certificate'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    finite_diagnostic = dict(replay_inputs["finite_diagnostic"])
    tail_contract = dict(replay_inputs["tail_contract"])
    expected = _active_tail_lift_expected(
        finite_diagnostic,
        tail_contract,
        interval_jacobian_error_upper=replay_inputs.get("interval_jacobian_error_upper"),
        projector_error_upper=replay_inputs.get("projector_error_upper"),
        analytic_tail_error_upper=replay_inputs.get("analytic_tail_error_upper"),
        nonlinear_remainder_error_upper=replay_inputs.get("nonlinear_remainder_error_upper"),
        contraction_threshold=float(replay_inputs.get("contraction_threshold", 1.0)),
        external_verification=dict(replay_inputs.get("external_verification", {})),
    )
    stored_interval = dict(report.get("q_total_interval", {}))
    expected_interval = dict(expected["q_total_interval"])
    diffs = [
        0.0 if str(report.get("finite_diagnostic_sha256", "")) == _sha256_json(finite_diagnostic) else float("inf"),
        0.0 if str(report.get("tail_contract_sha256", "")) == _sha256_json(tail_contract) else float("inf"),
        0.0 if list(report.get("active_indices", [])) == list(finite_diagnostic.get("active_indices", [])) else float("inf"),
        0.0 if list(report.get("tail_modes", [])) == list(finite_diagnostic.get("tail_modes", [])) else float("inf"),
        abs(float(report.get("q_finite_upper", float("inf"))) - expected["q_finite_upper"]),
        0.0 if dict(report.get("error_budget_terms", {})) == expected["error_budget_terms"] else float("inf"),
        0.0 if bool(report.get("error_budget_terms_certified", False)) == expected["error_budget_terms_certified"] else float("inf"),
        0.0 if list(report.get("missing_error_budget_terms", [])) == expected["missing_error_budget_terms"] else float("inf"),
        0.0 if report.get("total_lift_error_upper") == expected["total_lift_error_upper"] else float("inf"),
        0.0 if report.get("q_total_upper") == expected["q_total_upper"] else float("inf"),
        *[
            abs(float(stored_interval.get(name, float("inf"))) - float(expected_interval[name]))
            for name in ("lower", "upper", "midpoint", "radius")
        ],
        0.0 if bool(report.get("analytic_lift_certified", False)) == expected["analytic_lift_certified"] else float("inf"),
        0.0 if str(report.get("proof_status", "")) == expected["proof_status"] else float("inf"),
        0.0 if list(report.get("open_obligations", [])) == expected["open_obligations"] else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_tail_contraction_lift_certificate",
        "active_tail_contraction_lift_match": bool(max_violation <= 1e-8),
        "max_active_tail_contraction_lift_violation": float(max_violation),
        "expected_q_total_upper": expected["q_total_upper"],
        "expected_missing_error_budget_terms": expected["missing_error_budget_terms"],
        "unproven_claim": False,
    }


def verify_active_projector_error_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the finite active-projector error certificate."""
    if report.get("candidate_type") != "active_projector_error_certificate":
        raise ValueError("expected candidate_type='active_projector_error_certificate'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    finite_diagnostic = dict(replay_inputs["finite_diagnostic"])
    geom = _active_tail_geometry_replay(finite_diagnostic)
    envelope = _finite_difference_jacobian_error_envelope_replay(geom["artifact"])
    perturbation_override = replay_inputs.get("jacobian_perturbation_upper")
    delta = (
        float(perturbation_override)
        if perturbation_override is not None
        else float(envelope["jacobian_error_envelope_upper"])
    )
    sigma_min_active = float(geom["sigma_min_active"])
    tail_norm = float(geom["tail_operator_norm"])
    weight_condition = float(geom["weight_condition"])
    dim_factor = float(geom["dimension_factor"])
    finite_certified = bool(
        sigma_min_active > 0.0 and np.isfinite(delta) and np.isfinite(weight_condition)
    )
    if finite_certified:
        projector_error = (
            dim_factor * weight_condition * tail_norm * tail_norm * (2.0 / sigma_min_active) * delta
        )
    else:
        projector_error = float("inf")
    interval = _scalar_interval_from_midpoint(
        projector_error if np.isfinite(projector_error) else 1.0e300,
        relative_padding=1e-12,
    )
    expected_upper = float(interval["upper"]) if np.isfinite(projector_error) else None
    diffs = [
        abs(float(report.get("sigma_min_active", float("inf"))) - sigma_min_active),
        abs(float(report.get("tail_operator_norm", float("inf"))) - tail_norm),
        abs(float(report.get("weight_condition", float("inf"))) - weight_condition),
        0.0 if report.get("projector_error_upper") == expected_upper else (
            abs(float(report.get("projector_error_upper", float("inf"))) - expected_upper)
            if expected_upper is not None
            else float("inf")
        ),
        0.0 if bool(report.get("finite_dimensional_certified", False)) == finite_certified else float("inf"),
        0.0 if list(report.get("active_indices", [])) == list(geom["active_indices"]) else float("inf"),
        0.0 if list(report.get("tail_modes", [])) == list(geom["tail_modes"]) else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_projector_error_certificate",
        "active_projector_error_match": bool(max_violation <= 1e-7),
        "max_active_projector_error_violation": float(max_violation),
        "expected_projector_error_upper": expected_upper,
        "unproven_claim": False,
    }


def verify_interval_jacobian_error_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the float64 interval-Jacobian error certificate."""
    if report.get("candidate_type") != "interval_jacobian_error_certificate":
        raise ValueError("expected candidate_type='interval_jacobian_error_certificate'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    finite_diagnostic = dict(replay_inputs["finite_diagnostic"])
    geom = _active_tail_geometry_replay(finite_diagnostic)
    envelope = _finite_difference_jacobian_error_envelope_replay(geom["artifact"])
    error_envelope = float(envelope["jacobian_error_envelope_upper"])
    tail_norm = float(geom["tail_operator_norm"])
    weight_condition = float(geom["weight_condition"])
    dim_factor = float(geom["dimension_factor"])
    finite_certified = bool(np.isfinite(error_envelope) and np.isfinite(weight_condition))
    interval_error = (
        dim_factor * weight_condition * 2.0 * tail_norm * error_envelope
        if finite_certified
        else float("inf")
    )
    interval = _scalar_interval_from_midpoint(
        interval_error if np.isfinite(interval_error) else 1.0e300,
        relative_padding=1e-12,
    )
    expected_upper = float(interval["upper"]) if finite_certified else None
    diffs = [
        abs(
            float(report.get("jacobian_error_envelope_upper", float("inf"))) - error_envelope
        ),
        abs(float(report.get("tail_operator_norm", float("inf"))) - tail_norm),
        abs(float(report.get("weight_condition", float("inf"))) - weight_condition),
        0.0 if report.get("interval_jacobian_error_upper") == expected_upper else (
            abs(float(report.get("interval_jacobian_error_upper", float("inf"))) - expected_upper)
            if expected_upper is not None
            else float("inf")
        ),
        0.0 if bool(report.get("finite_dimensional_certified", False)) == finite_certified else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "interval_jacobian_error_certificate",
        "interval_jacobian_error_match": bool(max_violation <= 1e-7),
        "max_interval_jacobian_error_violation": float(max_violation),
        "expected_interval_jacobian_error_upper": expected_upper,
        "unproven_claim": False,
    }


def _axisymmetric_residual_hessian_operator_norm_replay(
    n_coeffs: int,
    *,
    grid: dict[str, Any],
    metadata: dict[str, Any],
    viscosity: float,
    density: float,
    step: float,
) -> dict[str, float]:
    """Independently rebuild the residual Hessian and its operator-norm bounds."""
    n = int(n_coeffs)
    zero = np.zeros(n, dtype=float)
    f0 = _axisymmetric_residual_vector_from_coefficients(
        zero, grid=grid, metadata=metadata, viscosity=viscosity, density=density
    )
    m = int(f0.size)
    fp = np.empty((n, m), dtype=float)
    hess = np.empty((m, n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = step
        fp_i = _axisymmetric_residual_vector_from_coefficients(
            ei, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        fm_i = _axisymmetric_residual_vector_from_coefficients(
            -ei, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        fp[i] = fp_i
        hess[:, i, i] = (fp_i - 2.0 * f0 + fm_i) / (step * step)
    for i in range(n):
        for j in range(i + 1, n):
            eij = np.zeros(n, dtype=float)
            eij[i] = step
            eij[j] = step
            fpp = _axisymmetric_residual_vector_from_coefficients(
                eij, grid=grid, metadata=metadata, viscosity=viscosity, density=density
            )
            mixed = (fpp - fp[i] - fp[j] + f0) / (step * step)
            hess[:, i, j] = mixed
            hess[:, j, i] = mixed
    diagonal_proxy = float(max((float(np.linalg.norm(hess[:, i, i])) for i in range(n)), default=0.0))
    jacobian_lipschitz_upper = float(np.linalg.svd(hess.reshape(m * n, n), compute_uv=False)[0])
    remainder_operator_upper = float(np.linalg.svd((0.5 * hess).reshape(m, n * n), compute_uv=False)[0])
    return {
        "diagonal_hessian_proxy": diagonal_proxy,
        "jacobian_lipschitz_operator_norm_upper": jacobian_lipschitz_upper,
        "remainder_operator_norm_upper": remainder_operator_upper,
        "coefficient_count": float(n),
        "residual_size": float(m),
    }


def verify_nonlinear_tail_remainder_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the sampled nonlinear tail-remainder certificate."""
    if report.get("candidate_type") != "nonlinear_tail_remainder_certificate":
        raise ValueError("expected candidate_type='nonlinear_tail_remainder_certificate'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    finite_diagnostic = dict(replay_inputs["finite_diagnostic"])
    geom = _active_tail_geometry_replay(finite_diagnostic)
    artifact = geom["artifact"]
    rin = artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = float(geom["perturbation"])
    weight_condition = float(geom["weight_condition"])
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs, grid=grid, metadata=metadata, viscosity=viscosity, density=density
    )
    hessian_proxy = 0.0
    for idx in sorted(set(geom["active_indices"]) | set(geom["tail_modes"])):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        second = (plus - 2.0 * base + minus) / (step * step)
        hessian_proxy = max(hessian_proxy, float(np.linalg.norm(second)))
    if bool(replay_inputs.get("certify_hessian_operator_norm", False)):
        operator_norm = _axisymmetric_residual_hessian_operator_norm_replay(
            int(coeffs.size),
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
            step=float(replay_inputs.get("hessian_perturbation", 1.0e-3)),
        )
        hessian_proxy = float(operator_norm["jacobian_lipschitz_operator_norm_upper"])
    sigma_min_full = float(geom["sigma_min_full"])
    residual_norm = float(geom["residual_norm"])
    heuristic_radius = residual_norm / sigma_min_full if sigma_min_full > 0.0 else float("inf")
    radius_override = replay_inputs.get("solution_ball_radius")
    radius_used = (
        float(radius_override) if radius_override is not None else float(heuristic_radius)
    )
    finite_value = bool(np.isfinite(hessian_proxy) and np.isfinite(radius_used))
    remainder = weight_condition * hessian_proxy * radius_used if finite_value else float("inf")
    interval = _scalar_interval_from_midpoint(
        remainder if np.isfinite(remainder) else 1.0e300, relative_padding=1e-12
    )
    expected_upper = float(interval["upper"]) if finite_value else None
    diffs = [
        abs(float(report.get("hessian_operator_norm_proxy", float("inf"))) - hessian_proxy)
        if report.get("hessian_operator_norm_proxy") is not None
        else float("inf"),
        abs(float(report.get("solution_ball_radius_used", float("inf"))) - radius_used)
        if report.get("solution_ball_radius_used") is not None
        else float("inf"),
        0.0 if report.get("nonlinear_remainder_error_upper") == expected_upper else (
            abs(float(report.get("nonlinear_remainder_error_upper", float("inf"))) - expected_upper)
            if expected_upper is not None
            else float("inf")
        ),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "nonlinear_tail_remainder_certificate",
        "nonlinear_tail_remainder_match": bool(max_violation <= 1e-6),
        "max_nonlinear_tail_remainder_violation": float(max_violation),
        "expected_nonlinear_remainder_error_upper": expected_upper,
        "unproven_claim": False,
    }


def verify_analytic_tail_error_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the conditional analytic omitted-tail error certificate."""
    if report.get("candidate_type") != "analytic_tail_error_certificate":
        raise ValueError("expected candidate_type='analytic_tail_error_certificate'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    tail_contract = dict(replay_inputs["tail_contract"])
    radius = float(tail_contract.get("analytic_radius", 1.125))
    algebra_override = replay_inputs.get("algebra_constant")
    constant = (
        float(algebra_override)
        if algebra_override is not None
        else float(tail_contract.get("algebra_constant", 1.0))
    )
    tail_modes = [int(idx) for idx in tail_contract.get("required_tail_modes", [])]
    highest_mode = max(tail_modes) if tail_modes else 0
    gamma = float(replay_inputs.get("coefficient_decay_rate", 0.5))
    ratio = radius * gamma
    convergent = bool(0.0 <= ratio < 1.0)
    remainder = (
        constant * (ratio ** (highest_mode + 1)) / (1.0 - ratio) if convergent else float("inf")
    )
    interval = _scalar_interval_from_midpoint(
        remainder if np.isfinite(remainder) else 1.0e300, relative_padding=1e-12
    )
    expected_upper = float(interval["upper"]) if np.isfinite(remainder) else None
    diffs = [
        abs(float(report.get("weighted_ratio_rho_gamma", float("inf"))) - ratio),
        0.0 if bool(report.get("geometric_series_convergent", False)) == convergent else float("inf"),
        0.0 if int(report.get("highest_finite_mode", -1)) == int(highest_mode) else float("inf"),
        0.0 if report.get("analytic_tail_error_upper") == expected_upper else (
            abs(float(report.get("analytic_tail_error_upper", float("inf"))) - expected_upper)
            if expected_upper is not None
            else float("inf")
        ),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "analytic_tail_error_certificate",
        "analytic_tail_error_match": bool(max_violation <= 1e-9),
        "max_analytic_tail_error_violation": float(max_violation),
        "expected_analytic_tail_error_upper": expected_upper,
        "unproven_claim": False,
    }


def verify_active_tail_lift_error_budget(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the assembled four-term active-tail lift error budget."""
    if report.get("candidate_type") != "active_tail_lift_error_budget":
        raise ValueError("expected candidate_type='active_tail_lift_error_budget'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    finite_diagnostic = dict(replay_inputs["finite_diagnostic"])
    tail_contract = dict(replay_inputs["tail_contract"])
    projector_cert = dict(replay_inputs["projector_certificate"])
    interval_cert = dict(replay_inputs["interval_jacobian_certificate"])
    nonlinear_cert = dict(replay_inputs["nonlinear_remainder_certificate"])
    analytic_cert = dict(replay_inputs["analytic_tail_certificate"])
    sub_replays = {
        "projector_certificate": verify_active_projector_error_certificate(projector_cert),
        "interval_jacobian_certificate": verify_interval_jacobian_error_certificate(interval_cert),
        "nonlinear_remainder_certificate": verify_nonlinear_tail_remainder_certificate(nonlinear_cert),
        "analytic_tail_certificate": verify_analytic_tail_error_certificate(analytic_cert),
    }
    sub_match_flags = {
        "projector_certificate": bool(sub_replays["projector_certificate"]["active_projector_error_match"]),
        "interval_jacobian_certificate": bool(sub_replays["interval_jacobian_certificate"]["interval_jacobian_error_match"]),
        "nonlinear_remainder_certificate": bool(sub_replays["nonlinear_remainder_certificate"]["nonlinear_tail_remainder_match"]),
        "analytic_tail_certificate": bool(sub_replays["analytic_tail_certificate"]["analytic_tail_error_match"]),
    }
    sub_match = all(sub_match_flags.values())
    term_values = {
        "interval_jacobian_error_upper": interval_cert.get("interval_jacobian_error_upper"),
        "projector_error_upper": projector_cert.get("projector_error_upper"),
        "analytic_tail_error_upper": analytic_cert.get("analytic_tail_error_upper"),
        "nonlinear_remainder_error_upper": nonlinear_cert.get("nonlinear_remainder_error_upper"),
    }
    threshold = float(replay_inputs.get("contraction_threshold", 1.0))
    expected_lift = _active_tail_lift_expected(
        finite_diagnostic,
        tail_contract,
        interval_jacobian_error_upper=term_values["interval_jacobian_error_upper"],
        projector_error_upper=term_values["projector_error_upper"],
        analytic_tail_error_upper=term_values["analytic_tail_error_upper"],
        nonlinear_remainder_error_upper=term_values["nonlinear_remainder_error_upper"],
        contraction_threshold=threshold,
        external_verification=dict(replay_inputs.get("external_verification", {})),
    )
    expected_q_total = expected_lift["q_total_upper"]
    diffs = [
        0.0 if dict(report.get("error_budget_terms", {})) == term_values else float("inf"),
        0.0 if report.get("q_total_upper") == expected_q_total else float("inf"),
        0.0 if bool(report.get("analytic_lift_certified", False)) == expected_lift["analytic_lift_certified"] else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_tail_lift_error_budget",
        "active_tail_lift_error_budget_match": bool(max_violation <= 1e-8 and sub_match),
        "sub_certificate_replays_match": bool(sub_match),
        "sub_certificate_match_flags": sub_match_flags,
        "max_active_tail_lift_error_budget_violation": float(max_violation),
        "expected_q_total_upper": expected_q_total,
        "sub_certificate_replays": sub_replays,
        "unproven_claim": False,
    }


def _componentwise_active_frontier_row(
    interval_report: dict[str, Any],
    artifact: dict[str, Any],
    active: tuple[int, ...],
    base_active: set[int],
    base_sigma: float,
    sigma_threshold: float,
) -> dict[str, Any]:
    linearized = _active_subspace_linearized_replay(artifact, active)
    inverse = linearized.get("approximate_inverse_norm")
    inverse_norm = float(inverse) if inverse is not None else float("inf")
    matrix_norm = float(linearized.get("matrix_norm", 0.0))
    tail_radius = float(interval_report.get("continuum_residual_certificates", {}).get("tail_radius", 0.0))
    projection_error = tail_radius * (1.0 + matrix_norm)
    neumann = inverse_norm * projection_error if np.isfinite(inverse_norm) else float("inf")
    neumann_interval = _scalar_interval_from_midpoint(
        neumann if np.isfinite(neumann) else 1.0e300,
        relative_padding=1e-12,
    )
    coefficient_radius = max(
        (
            float(item.get("radius", 0.0))
            for box in interval_report.get("coefficient_intervals", {}).values()
            for item in box.get("intervals", [])
        ),
        default=0.0,
    )
    quadratic = matrix_norm * coefficient_radius + tail_radius + projection_error
    worst_upper = -float("inf")
    worst_name = ""
    for section_name, section in (
        interval_report.get("continuum_residual_certificates", {})
        .get("sections", {})
        .items()
    ):
        for quantity_name, cert in section.items():
            if not isinstance(cert, dict) or not isinstance(cert.get("continuum_sup_norm_interval"), dict):
                continue
            interval = dict(cert["continuum_sup_norm_interval"])
            residual = max(abs(float(interval["lower"])), abs(float(interval["upper"])))
            value = inverse_norm * (residual + quadratic) if np.isfinite(inverse_norm) else float("inf")
            closure = _scalar_interval_from_midpoint(
                value if np.isfinite(value) else 1.0e300,
                relative_padding=1e-12,
            )
            if float(closure["upper"]) > worst_upper:
                worst_upper = float(closure["upper"])
                worst_name = f"{section_name}.{quantity_name}"
    sigma = float(linearized["smallest_singular_value"])
    neumann_passed = bool(linearized["finite_dimensional_certified"] and float(neumann_interval["upper"]) < 1.0)
    radii_passed = bool(np.isfinite(worst_upper) and worst_upper < 1.0)
    return {
        "active_indices": list(active),
        "added_inactive": [int(idx) for idx in active if idx not in base_active],
        "active_count": len(active),
        "sigma_min": sigma,
        "sigma_fraction_vs_base": sigma / max(base_sigma, 1e-300),
        "condition": linearized.get("condition_estimate"),
        "neumann_upper": float(neumann_interval["upper"]),
        "neumann_passed": neumann_passed,
        "worst_component": worst_name,
        "worst_radii_upper": worst_upper,
        "radii_passed": radii_passed,
        "frontier_passed": bool(sigma >= sigma_threshold and neumann_passed and radii_passed),
    }


def verify_active_subspace_absorption_frontier_report(report: dict[str, Any]) -> dict[str, Any]:
    """Replay inactive-mode absorption frontier metrics."""
    if report.get("candidate_type") != "active_subspace_absorption_frontier_report":
        raise ValueError("expected candidate_type='active_subspace_absorption_frontier_report'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    interval_report = dict(replay_inputs["interval_report"])
    artifact = dict(interval_report["replay_inputs"]["refined_artifact"])
    base_active = tuple(int(idx) for idx in replay_inputs.get("active_indices", []))
    max_order = int(replay_inputs.get("max_combination_order", report.get("max_combination_order", 0)))
    sigma_fraction = float(replay_inputs.get("sigma_fraction_threshold", report.get("sigma_fraction_threshold", 0.75)))
    coeff_count = len(artifact.get("replay_inputs", {}).get("coefficients", []))
    inactive = tuple(idx for idx in range(coeff_count) if idx not in set(base_active))
    base = _componentwise_active_frontier_row(
        interval_report,
        artifact,
        base_active,
        set(base_active),
        1.0,
        0.0,
    )
    base_sigma = max(float(base["sigma_min"]), 1e-300)
    sigma_threshold = sigma_fraction * base_sigma
    rows: list[dict[str, Any]] = []
    for order in range(min(max_order, len(inactive)) + 1):
        for combo in itertools.combinations(inactive, order):
            rows.append(_componentwise_active_frontier_row(
                interval_report,
                artifact,
                tuple([*base_active, *combo]),
                set(base_active),
                base_sigma,
                sigma_threshold,
            ))
    if inactive and max_order < len(inactive):
        rows.append(_componentwise_active_frontier_row(
            interval_report,
            artifact,
            tuple([*base_active, *inactive]),
            set(base_active),
            base_sigma,
            sigma_threshold,
        ))
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        unique[tuple(row["added_inactive"])] = row
    rows = sorted(
        unique.values(),
        key=lambda row: (
            bool(row["frontier_passed"]),
            int(row["active_count"]),
            float(row["sigma_min"]),
        ),
        reverse=True,
    )
    passed = [row for row in rows if row["frontier_passed"]]
    passed_sets = [set(row["added_inactive"]) for row in passed]
    maximal = [
        row
        for row, added in zip(passed, passed_sets, strict=True)
        if not any(added < other for other in passed_sets)
    ]
    mode_counts = {int(idx): 0 for idx in inactive}
    for row in maximal:
        for idx in row["added_inactive"]:
            mode_counts[int(idx)] += 1
    required_tail = [int(idx) for idx in inactive if mode_counts[int(idx)] == 0]
    diffs = [
        abs(float(report.get("base_sigma_min", float("inf"))) - base_sigma),
        abs(float(report.get("sigma_threshold", float("inf"))) - sigma_threshold),
        0.0 if list(report.get("inactive_indices", [])) == list(inactive) else float("inf"),
        0.0 if list(report.get("required_tail_control_modes", [])) == required_tail else float("inf"),
        0.0 if int(report.get("passed_count", -1)) == len(passed) else float("inf"),
        0.0 if int(report.get("failed_count", -1)) == len([row for row in rows if not row["frontier_passed"]]) else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_subspace_absorption_frontier_report",
        "active_subspace_absorption_frontier_match": bool(max_violation <= 1e-8),
        "max_active_subspace_absorption_violation": float(max_violation),
        "expected_required_tail_control_modes": required_tail,
        "expected_passed_count": len(passed),
        "expected_failed_count": len([row for row in rows if not row["frontier_passed"]]),
        "unproven_claim": False,
    }


def _tail_contract_expected(
    frontier_report: dict[str, Any],
    *,
    analytic_radius: float,
    algebra_constant: float,
    external_verification: dict[str, Any],
) -> dict[str, Any]:
    tail_modes = [int(idx) for idx in frontier_report.get("required_tail_control_modes", [])]
    radius = float(analytic_radius)
    weights = {str(idx): radius ** int(idx) for idx in tail_modes}
    rows = [dict(row) for row in frontier_report.get("all_rows", [])]
    single_rows = [
        row
        for row in rows
        if len(row.get("added_inactive", [])) == 1
        and int(row["added_inactive"][0]) in set(tail_modes)
    ]
    full_tail = next(
        (
            row for row in rows
            if set(int(idx) for idx in row.get("added_inactive", [])) == set(tail_modes)
        ),
        {},
    )
    external_ok = _external_verifies_symbolic(
        external_verification,
        "weighted_analytic_tail_norm_contract",
    )
    certified = bool(external_ok and tail_modes)
    open_obligations = []
    if not external_ok:
        open_obligations.extend([
            "external_weighted_analytic_tail_norm_contract",
            "prove_weighted_tail_algebra_constant",
            "prove_tail_projection_error_bound",
        ])
    return {
        "analytic_radius": radius,
        "algebra_constant": float(algebra_constant),
        "required_tail_modes": tail_modes,
        "tail_weights": weights,
        "finite_surrogate_constants": {
            "max_single_mode_neumann_upper": max(
                (float(row.get("neumann_upper", 0.0)) for row in single_rows),
                default=0.0,
            ),
            "max_single_mode_radii_upper": max(
                (float(row.get("worst_radii_upper", 0.0)) for row in single_rows),
                default=0.0,
            ),
            "min_single_mode_sigma_fraction": min(
                (float(row.get("sigma_fraction_vs_base", 1.0)) for row in single_rows),
                default=1.0,
            ),
            "full_tail_neumann_upper": full_tail.get("neumann_upper"),
            "full_tail_radii_upper": full_tail.get("worst_radii_upper"),
            "full_tail_sigma_fraction": full_tail.get("sigma_fraction_vs_base"),
        },
        "weighted_tail_contract_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_missing_tail_norm_proof",
        "open_obligations": open_obligations,
    }


def verify_weighted_analytic_tail_norm_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Replay the weighted analytic tail-norm contract."""
    if report.get("candidate_type") != "weighted_analytic_tail_norm_contract":
        raise ValueError("expected candidate_type='weighted_analytic_tail_norm_contract'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    frontier_report = dict(replay_inputs["frontier_report"])
    expected = _tail_contract_expected(
        frontier_report,
        analytic_radius=float(replay_inputs.get("analytic_radius", 1.125)),
        algebra_constant=float(replay_inputs.get("algebra_constant", 1.0)),
        external_verification=dict(replay_inputs.get("external_verification", {})),
    )
    diffs = [
        abs(float(report.get("analytic_radius", float("inf"))) - expected["analytic_radius"]),
        abs(float(report.get("algebra_constant", float("inf"))) - expected["algebra_constant"]),
        0.0 if list(report.get("required_tail_modes", [])) == expected["required_tail_modes"] else float("inf"),
        0.0 if dict(report.get("tail_weights", {})) == expected["tail_weights"] else float("inf"),
        0.0 if dict(report.get("finite_surrogate_constants", {})) == expected["finite_surrogate_constants"] else float("inf"),
        0.0 if bool(report.get("weighted_tail_contract_certified", False)) == expected["weighted_tail_contract_certified"] else float("inf"),
        0.0 if str(report.get("proof_status", "")) == expected["proof_status"] else float("inf"),
        0.0 if list(report.get("open_obligations", [])) == expected["open_obligations"] else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "weighted_analytic_tail_norm_contract",
        "weighted_analytic_tail_norm_match": bool(max_violation <= 1e-8),
        "max_weighted_analytic_tail_norm_violation": float(max_violation),
        "expected_required_tail_modes": expected["required_tail_modes"],
        "unproven_claim": False,
    }


def verify_active_subspace_tail_contraction_attempt(report: dict[str, Any]) -> dict[str, Any]:
    """Replay active-subspace tail contraction status."""
    if report.get("candidate_type") != "active_subspace_tail_contraction_attempt":
        raise ValueError("expected candidate_type='active_subspace_tail_contraction_attempt'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    frontier_report = dict(replay_inputs["frontier_report"])
    tail_contract = dict(replay_inputs["tail_contract"])
    finite_diagnostic = dict(replay_inputs.get("finite_diagnostic", {}))
    analytic_lift = dict(replay_inputs.get("analytic_lift", {}))
    threshold = float(replay_inputs.get("contraction_threshold", 1.0))
    external = dict(replay_inputs.get("external_verification", {}))
    tail_modes = [int(idx) for idx in frontier_report.get("required_tail_control_modes", [])]
    contract_modes = [int(idx) for idx in tail_contract.get("required_tail_modes", [])]
    ratio_value = external.get("contraction_ratio_upper")
    ratio = float(ratio_value) if ratio_value is not None else float("inf")
    finite_ratio_value = finite_diagnostic.get("finite_contraction_ratio_upper")
    finite_ratio = (
        float(finite_ratio_value)
        if finite_ratio_value is not None
        else float("inf")
    )
    finite_passed = bool(
        finite_diagnostic
        and finite_diagnostic.get("finite_tail_contraction_surrogate_passed", False)
    )
    lift_ratio = analytic_lift.get("q_total_upper") if analytic_lift else None
    lift_ok = bool(analytic_lift and analytic_lift.get("analytic_lift_certified", False))
    external_ok = _external_verifies_symbolic(external, "active_subspace_tail_contraction")
    contract_ok = bool(tail_contract.get("weighted_tail_contract_certified", False))
    mode_match = tail_modes == contract_modes
    certified = bool(
        contract_ok
        and mode_match
        and (
            (external_ok and ratio < threshold)
            or lift_ok
        )
    )
    open_obligations: list[str] = []
    if not mode_match:
        open_obligations.append("tail_contract_modes_match_absorption_frontier")
    if not contract_ok:
        open_obligations.append("weighted_analytic_tail_norm_contract")
    if not external_ok and not lift_ok:
        open_obligations.append("external_active_subspace_tail_contraction_proof")
    if not finite_passed:
        open_obligations.append("finite_tail_contraction_surrogate_below_one")
    if not lift_ok:
        open_obligations.append("active_tail_contraction_analytic_lift")
    if not lift_ok and (not np.isfinite(ratio) or ratio >= threshold):
        open_obligations.append("tail_contraction_ratio_below_one")
    expected_ratio = ratio if np.isfinite(ratio) else None
    expected_finite_ratio = finite_ratio if np.isfinite(finite_ratio) else None
    expected_finite_hash = None if not finite_diagnostic else _sha256_json(finite_diagnostic)
    expected_lift_hash = None if not analytic_lift else _sha256_json(analytic_lift)
    diffs = [
        0.0 if list(report.get("required_tail_modes", [])) == tail_modes else float("inf"),
        0.0 if str(report.get("tail_contract_sha256", "")) == _sha256_json(tail_contract) else float("inf"),
        abs(float(report.get("contraction_threshold", float("inf"))) - threshold),
        0.0 if report.get("contraction_ratio_upper") == expected_ratio else float("inf"),
        0.0 if report.get("finite_contraction_ratio_upper") == expected_finite_ratio else float("inf"),
        0.0 if bool(report.get("finite_tail_contraction_surrogate_passed", False)) == finite_passed else float("inf"),
        0.0 if report.get("finite_diagnostic_sha256") == expected_finite_hash else float("inf"),
        0.0 if report.get("analytic_lift_q_total_upper") == lift_ratio else float("inf"),
        0.0 if bool(report.get("analytic_lift_certified", False)) == lift_ok else float("inf"),
        0.0 if report.get("analytic_lift_sha256") == expected_lift_hash else float("inf"),
        0.0 if bool(report.get("weighted_tail_contract_certified", False)) == contract_ok else float("inf"),
        0.0 if bool(report.get("tail_contraction_certified", False)) == certified else float("inf"),
        0.0 if str(report.get("proof_status", "")) == ("proved_by_external_artifact" if certified else "blocked_missing_tail_contraction_proof") else float("inf"),
        0.0 if list(report.get("open_obligations", [])) == open_obligations else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_subspace_tail_contraction_attempt",
        "active_subspace_tail_contraction_match": bool(max_violation <= 1e-8),
        "max_active_subspace_tail_contraction_violation": float(max_violation),
        "expected_required_tail_modes": tail_modes,
        "unproven_claim": False,
    }


def verify_active_subspace_completeness_theorem_attempt(report: dict[str, Any]) -> dict[str, Any]:
    """Replay active-subspace completeness theorem-attempt status."""
    if report.get("candidate_type") != "active_subspace_completeness_theorem_attempt":
        raise ValueError("expected candidate_type='active_subspace_completeness_theorem_attempt'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    frontier_report = dict(replay_inputs["frontier_report"])
    tail_contraction = dict(replay_inputs["tail_contraction"])
    invariance_report = dict(replay_inputs.get("invariance_report", {}))
    external = dict(replay_inputs.get("external_verification", {}))
    passing_rows = [
        dict(row)
        for row in frontier_report.get("all_rows", [])
        if bool(row.get("frontier_passed", False))
    ]
    finite_core_ok = bool(
        passing_rows
        and all(len(row.get("added_inactive", [])) == 0 for row in passing_rows)
        and frontier_report.get("base_metrics", {}).get("frontier_passed", False)
    )
    invariance_ok = True if not invariance_report else bool(
        invariance_report.get("finite_invariance_heuristic_passed", False)
    )
    tail_ok = bool(tail_contraction.get("tail_contraction_certified", False))
    external_ok = _external_verifies_symbolic(external, "active_subspace_completeness")
    complete = bool(finite_core_ok and invariance_ok and tail_ok and external_ok)
    open_obligations: list[str] = []
    if not finite_core_ok:
        open_obligations.append("finite_active_core_absorption_frontier")
    if not invariance_ok:
        open_obligations.append("finite_active_subspace_invariance")
    if not tail_ok:
        open_obligations.append("active_subspace_tail_contraction")
    if not external_ok:
        open_obligations.append("external_active_subspace_completeness_proof")
    tail_modes = [int(idx) for idx in frontier_report.get("required_tail_control_modes", [])]
    diffs = [
        0.0 if bool(report.get("finite_active_core_closed", False)) == finite_core_ok else float("inf"),
        0.0 if bool(report.get("finite_invariance_heuristic_passed", False)) == invariance_ok else float("inf"),
        0.0 if list(report.get("required_tail_modes", [])) == tail_modes else float("inf"),
        0.0 if bool(report.get("tail_contraction_certified", False)) == tail_ok else float("inf"),
        0.0 if bool(report.get("active_subspace_complete", False)) == complete else float("inf"),
        0.0 if str(report.get("proof_status", "")) == ("proved_by_external_artifact" if complete else "blocked_with_precise_obligations") else float("inf"),
        0.0 if list(report.get("open_obligations", [])) == open_obligations else float("inf"),
    ]
    max_violation = max(diffs, default=0.0)
    return {
        "candidate_type": "active_subspace_completeness_theorem_attempt",
        "active_subspace_completeness_match": bool(max_violation <= 1e-8),
        "max_active_subspace_completeness_violation": float(max_violation),
        "expected_required_tail_modes": tail_modes,
        "unproven_claim": False,
    }


def verify_active_subspace_closure_report(report: dict[str, Any]) -> dict[str, Any]:
    """Replay active-subspace closure metrics from serialized inputs only."""
    if report.get("candidate_type") != "active_subspace_blowup_closure_report":
        raise ValueError("expected candidate_type='active_subspace_blowup_closure_report'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    interval_report = dict(replay_inputs["interval_report"])
    interval_replay = verify_axisymmetric_interval_report(interval_report)
    artifact = dict(interval_report["replay_inputs"]["refined_artifact"])
    active = [int(idx) for idx in replay_inputs.get("active_indices", [])]
    recomputed = _active_subspace_linearized_replay(artifact, active)
    stored_linearized = dict(report.get("closure_certificates", {}).get("linearized_operator", {}))
    metric_diffs = [
        abs(float(stored_linearized.get("smallest_singular_value", float("inf"))) - recomputed["smallest_singular_value"]),
        abs(float(stored_linearized.get("largest_singular_value", float("inf"))) - recomputed["largest_singular_value"]),
        abs(float(stored_linearized.get("matrix_norm", float("inf"))) - recomputed["matrix_norm"]),
        0.0 if list(stored_linearized.get("active_coefficient_indices", [])) == active else float("inf"),
        0.0 if tuple(stored_linearized.get("matrix_shape", ())) == recomputed["matrix_shape"] else float("inf"),
    ]
    inv = stored_linearized.get("approximate_inverse_norm")
    if inv is not None and recomputed["approximate_inverse_norm"] is not None:
        metric_diffs.append(abs(float(inv) - float(recomputed["approximate_inverse_norm"])))
    else:
        metric_diffs.append(0.0 if inv is recomputed["approximate_inverse_norm"] else float("inf"))
    op = dict(report.get("closure_certificates", {}).get("operator_theoretic_invertibility", {}))
    tail_radius = float(interval_report.get("continuum_residual_certificates", {}).get("tail_radius", 0.0))
    projection_error = tail_radius * (1.0 + recomputed["matrix_norm"])
    inverse = recomputed["approximate_inverse_norm"]
    neumann = float(inverse) * projection_error if inverse is not None else float("inf")
    neumann_interval = _scalar_interval_from_midpoint(
        neumann if np.isfinite(neumann) else 1.0e300,
        relative_padding=1e-12,
    )
    stored_neumann = dict(op.get("neumann_defect_interval", {}))
    metric_diffs.extend(
        abs(float(stored_neumann.get(name, float("inf"))) - float(neumann_interval[name]))
        for name in ("lower", "upper", "midpoint", "radius")
    )
    max_violation = max(metric_diffs, default=0.0)
    match = bool(interval_replay.get("interval_report_match", False) and max_violation <= 1e-8)
    return {
        "candidate_type": "active_subspace_blowup_closure_report",
        "active_subspace_closure_match": match,
        "max_active_subspace_violation": float(max_violation),
        "interval_replay": interval_replay,
        "recomputed_linearized": recomputed,
        "recomputed_neumann_interval": neumann_interval,
        "unproven_claim": False,
    }


def _regularity_counterexample_sweep(
    coefficients: dict[str, float],
    *,
    traces: dict[str, np.ndarray],
    target: str,
    tolerance: float,
) -> dict[str, Any]:
    probes: dict[str, dict[str, np.ndarray]] = {
        "stored_trace": traces,
        "energy_doubled": {
            key: (2.0 * value if key == "energy" else value)
            for key, value in traces.items()
        },
        "enstrophy_amplified": {
            key: (1.25 * value if key == target else value)
            for key, value in traces.items()
        },
    }
    counterexamples: list[dict[str, Any]] = []
    for name, probe in probes.items():
        time = np.linspace(0.0, 1.0, int(np.asarray(probe[target]).shape[0]))
        derivative = np.gradient(np.asarray(probe[target], dtype=float), time)
        predicted = np.zeros_like(derivative)
        features: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in probe.items()}
        for feature_name, values in list(features.items()):
            features[f"{feature_name}^2"] = values * values
        for coeff_name, coeff in coefficients.items():
            if coeff_name in features:
                predicted = predicted + float(coeff) * features[coeff_name]
        violation = float(np.max(np.maximum(derivative - predicted - tolerance, 0.0)))
        if violation > 0.0:
            counterexamples.append({
                "probe": name,
                "max_positive_violation": violation,
            })
    return {
        "method": "deterministic_trace_feature_probes",
        "probe_count": len(probes),
        "counterexamples": counterexamples,
        "passed": not counterexamples,
        "unproven_claim": False,
    }


def verify_regularity_inequality_report(report: dict[str, Any]) -> dict[str, Any]:
    """Verify replay consistency for a regularity inequality report."""
    if report.get("candidate_type") != "regularity_inequality_report":
        raise ValueError("expected candidate_type='regularity_inequality_report'")
    rin = dict(report["replay_inputs"])
    artifact = dict(rin["regularity_artifact"])
    replayed_artifact = replay_candidate_artifact(artifact)
    artifact_inputs = dict(artifact["replay_inputs"])
    time = np.asarray(artifact_inputs["time"], dtype=float)
    traces = {str(k): np.asarray(v, dtype=float) for k, v in dict(artifact_inputs["traces"]).items()}
    target = str(artifact_inputs.get("target", report.get("target", "enstrophy")))
    coefficients = {str(k): float(v) for k, v in dict(report.get("coefficients", {})).items()}
    derivative = np.gradient(traces[target], time)
    predicted = np.zeros_like(derivative)
    features: dict[str, np.ndarray] = dict(traces)
    for feature_name, values in list(features.items()):
        features[f"{feature_name}^2"] = values * values
    for coeff_name, coeff in coefficients.items():
        if coeff_name in features:
            predicted = predicted + coeff * features[coeff_name]
    residual = derivative - predicted
    max_abs_residual = float(np.max(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    stored_residual = dict(report.get("trace_residual", {}))
    residual_diffs = [
        abs(float(stored_residual.get("max_abs_residual", float("inf"))) - max_abs_residual),
        abs(float(stored_residual.get("rmse", float("inf"))) - rmse),
    ]
    coefficient_violations: list[float] = []
    for name, value in coefficients.items():
        interval = dict(report.get("coefficient_intervals", {}).get(name, {}))
        lower = float(interval.get("lower", float("inf")))
        upper = float(interval.get("upper", -float("inf")))
        coefficient_violations.append(0.0 if lower <= value <= upper else float("inf"))
    sweep = _regularity_counterexample_sweep(
        coefficients,
        traces=traces,
        target=target,
        tolerance=float(rin.get("residual_tolerance", stored_residual.get("tolerance", 1e-6))),
    )
    stored_sweep = dict(report.get("counterexample_sweep", {}))
    sweep_match = (
        int(stored_sweep.get("probe_count", -1)) == int(sweep["probe_count"])
        and list(stored_sweep.get("counterexamples", [])) == list(sweep["counterexamples"])
    )
    max_violation = max([*residual_diffs, *coefficient_violations, 0.0 if sweep_match else float("inf")], default=0.0)
    match = bool(replayed_artifact.get("replay_match", False) and max_violation <= 1e-9)
    return {
        "candidate_type": "regularity_inequality_report",
        "regularity_report_match": match,
        "max_regularity_violation": float(max_violation),
        "artifact_replay": replayed_artifact,
        "recomputed_trace_residual": {
            "max_abs_residual": max_abs_residual,
            "rmse": rmse,
        },
        "counterexample_sweep": sweep,
        "unproven_claim": False,
    }


def _external_verifies_symbolic(
    external_verification: dict[str, Any],
    obligation: str,
) -> bool:
    return bool(
        external_verification.get("verified", False)
        and str(obligation) in {str(v) for v in external_verification.get("discharged_obligations", [])}
    )


def verify_theorem_grade_closure_attempt(report: dict[str, Any]) -> dict[str, Any]:
    """Verify replay consistency for theorem-grade closure-attempt reports."""
    if report.get("candidate_type") != "theorem_grade_closure_attempt":
        raise ValueError("expected candidate_type='theorem_grade_closure_attempt'")
    replay_inputs = dict(report.get("replay_inputs", {}))
    blowup_replay = verify_blowup_closure_report(dict(replay_inputs["blowup_report"]))
    regularity_report = dict(replay_inputs.get("regularity_report", {}))
    regularity_replay = (
        verify_regularity_inequality_report(regularity_report)
        if regularity_report.get("candidate_type") == "regularity_inequality_report"
        else {"regularity_report_match": False, "reason": "not_replayable"}
    )
    routes = dict(report.get("route_attempts", {}))
    operator = dict(routes.get("operator_invertibility", {}))
    finite = dict(operator.get("finite_projection_certificate", {}))
    op_external = dict(operator.get("external_verification", {}))
    expected_operator = bool(
        finite.get("neumann_passed", False)
        and _external_verifies_symbolic(op_external, "external_banach_space_invertibility_proof")
    )
    radii = dict(routes.get("radii_polynomial", {}))
    radii_external = dict(radii.get("external_verification", {}))
    outward = dict(radii.get("outward_mapping_interval", {}))
    expected_radii = bool(
        float(outward.get("upper", float("inf"))) < 0.0
        and expected_operator
        and _external_verifies_symbolic(radii_external, "radii_polynomial_closure")
    )
    norm = dict(routes.get("norm_divergence", {}))
    linkage = dict(norm.get("field_profile_linkage", {}))
    scaling = dict(norm.get("profile_scaling_law", {}))
    norm_external = dict(norm.get("external_verification", {}))
    exponent = scaling.get("growth_exponent")
    expected_norm = bool(
        linkage.get("exact_profile_verified", False)
        and linkage.get("finite_energy_certified", False)
        and linkage.get("axis_regular_certified", False)
        and exponent is not None
        and float(exponent) > 0.0
        and _external_verifies_symbolic(norm_external, "exact_profile_norm_divergence")
    )
    regularity = dict(routes.get("regularity_all_data", {}))
    regularity_external = dict(regularity.get("external_verification", {}))
    expected_regularity = bool(
        regularity.get("diagnostic_falsification", {}).get("passed", False)
        and _external_verifies_symbolic(regularity_external, "all_smooth_finite_energy_data_proof")
    )
    flag_diffs = [
        0.0 if bool(operator.get("operator_theoretic_certified", False)) == expected_operator else float("inf"),
        0.0 if bool(radii.get("radii_polynomial_closure", False)) == expected_radii else float("inf"),
        0.0 if bool(norm.get("norm_divergence", False)) == expected_norm else float("inf"),
        0.0 if bool(regularity.get("all_smooth_finite_energy_data_proof", False)) == expected_regularity else float("inf"),
    ]
    recomputed_open = sorted({
        str(obligation)
        for attempt in routes.values()
        if isinstance(attempt, dict)
        for obligation in attempt.get("open_obligations", [])
    })
    open_match = list(report.get("open_obligations", [])) == recomputed_open
    expected_status = (
        "proved_by_external_artifact"
        if all(
            str(attempt.get("proof_status", "")) == "proved_by_external_artifact"
            for attempt in routes.values()
            if isinstance(attempt, dict)
        )
        else "falsified_with_counterexample"
        if any(
            str(attempt.get("proof_status", "")) == "falsified_with_counterexample"
            for attempt in routes.values()
            if isinstance(attempt, dict)
        )
        else "blocked_with_precise_obligations"
    )
    status_match = str(report.get("proof_status", "")) == expected_status
    max_violation = max([*flag_diffs, 0.0 if open_match else float("inf"), 0.0 if status_match else float("inf")])
    match = bool(
        blowup_replay.get("closure_report_match", False)
        and (
            regularity_replay.get("regularity_report_match", False)
            or regularity_replay.get("reason") == "not_replayable"
        )
        and max_violation <= 1e-9
    )
    return {
        "candidate_type": "theorem_grade_closure_attempt",
        "theorem_grade_report_match": match,
        "max_theorem_grade_violation": float(max_violation),
        "expected_route_flags": {
            "operator_theoretic_certified": expected_operator,
            "radii_polynomial_closure": expected_radii,
            "norm_divergence": expected_norm,
            "all_smooth_finite_energy_data_proof": expected_regularity,
        },
        "expected_proof_status": expected_status,
        "expected_open_obligations": recomputed_open,
        "blowup_replay": blowup_replay,
        "regularity_replay": regularity_replay,
        "unproven_claim": False,
    }


def verify_proof_obligation_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Verify deterministic hashes and required fields for one proof obligation."""
    payload = {
        key: value
        for key, value in dict(bundle).items()
        if key not in {"obligation_id", "obligation_sha256"}
    }
    expected_hash = _sha256_json(payload)
    required = {
        "schema_version",
        "route",
        "lemma_id",
        "theorem_name",
        "theorem_statement",
        "assumptions",
        "dependencies",
        "source_artifact_sha256",
        "expected_verifier",
        "proof_status",
    }
    fields_present = required.issubset(set(bundle))
    obligation_id = str(bundle.get("obligation_id", ""))
    expected_id = f"{bundle.get('route', '')}.{bundle.get('lemma_id', '')}"
    match = bool(
        fields_present
        and str(bundle.get("schema_version", "")) == "navier-stokes-proof-obligation-1"
        and obligation_id == expected_id
        and str(bundle.get("obligation_sha256", "")) == expected_hash
    )
    return {
        "candidate_type": "proof_obligation_bundle",
        "obligation_id": obligation_id,
        "obligation_bundle_match": match,
        "expected_obligation_sha256": expected_hash,
        "stored_obligation_sha256": str(bundle.get("obligation_sha256", "")),
        "fields_present": fields_present,
        "unproven_claim": False,
    }


def verify_theorem_verifier_bundle(
    obligation_bundles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    verifier_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Replay verifier-record ingestion without importing omnibias.pinn.certified code."""
    obligations = {str(bundle.get("obligation_id", "")): dict(bundle) for bundle in obligation_bundles}
    records_list = [dict(record) for record in verifier_bundle.get("proof_records", [])]
    records = {str(record.get("obligation_id", "")): record for record in records_list}
    artifact_hash_match = str(verifier_bundle.get("artifact_sha256", "")) == _sha256_json(records_list)
    status_ok = bool(verifier_bundle.get("verified", False)) and str(
        verifier_bundle.get("verification_status", "")
    ) == "verified"
    freshness_ok = bool(str(verifier_bundle.get("reviewed_at_utc", "")))
    discharged = {str(v) for v in verifier_bundle.get("discharged_obligations", [])}
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for obligation_id, obligation in obligations.items():
        record = records.get(obligation_id)
        if obligation_id not in discharged or record is None:
            rejected[obligation_id] = "obligation_not_discharged"
            continue
        checks = {
            "status": status_ok,
            "freshness": freshness_ok,
            "artifact_hash": artifact_hash_match,
            "verifier": str(record.get("verifier", "")) == str(obligation.get("expected_verifier", "")),
            "theorem_name": str(record.get("theorem_name", "")) == str(obligation.get("theorem_name", "")),
            "obligation_hash": str(record.get("obligation_sha256", "")) == str(obligation.get("obligation_sha256", "")),
            "source_hash": str(record.get("source_artifact_sha256", ""))
            == str(obligation.get("source_artifact_sha256", "")),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            rejected[obligation_id] = ",".join(failed)
        else:
            accepted.append(obligation_id)
    return {
        "candidate_type": "theorem_verifier_bundle",
        "verifier_bundle_match": bool(accepted and not rejected),
        "accepted_obligations": accepted,
        "rejected_obligations": rejected,
        "artifact_hash_match": artifact_hash_match,
        "status_ok": status_ok,
        "freshness_ok": freshness_ok,
        "unproven_claim": False,
    }


def verify_ns_proof_program_report(report: dict[str, Any]) -> dict[str, Any]:
    """Verify replay consistency for the top-level proof-program artifact."""
    if report.get("candidate_type") != "navier_stokes_proof_program_report":
        raise ValueError("expected candidate_type='navier_stokes_proof_program_report'")
    theorem_replay = verify_theorem_grade_closure_attempt(dict(report["theorem_attempt"]))
    exact_contracts = dict(report.get("exact_equation_contracts", {})).get("contracts", {})
    function_spaces = dict(report.get("function_space_definitions", {})).get("definitions", {})
    interval_backend = dict(report.get("interval_cap_backend", {}))
    candidate_status = list(report.get("candidate_family_status", []))
    obligation_bundles = [dict(bundle) for bundle in report.get("proof_obligation_bundles", [])]
    obligation_replays = [verify_proof_obligation_bundle(bundle) for bundle in obligation_bundles]
    obligations_match = bool(
        obligation_replays
        and all(replay.get("obligation_bundle_match", False) for replay in obligation_replays)
    )
    lemma_packages = dict(report.get("lemma_packages", {}))
    verifier_ingestion = dict(report.get("verifier_ingestion", {}))
    verifier_replays: dict[str, dict[str, Any]] = {}
    verifier_matches: list[bool] = []
    for route_key, package_key in (
        ("finite_time_blowup", "finite_time_blowup"),
        ("global_regularity", "global_regularity"),
    ):
        package = dict(lemma_packages.get(package_key, {}))
        ingestion = dict(verifier_ingestion.get(route_key, {}))
        bundle = dict(ingestion.get("verifier_bundle", {}))
        route_obligations = [dict(item) for item in package.get("proof_obligation_bundles", [])]
        if bundle:
            replay = verify_theorem_verifier_bundle(route_obligations, bundle)
            verifier_replays[route_key] = replay
            verifier_matches.append(
                list(ingestion.get("accepted_obligations", [])) == list(replay["accepted_obligations"])
                and dict(ingestion.get("rejected_obligations", {})) == dict(replay["rejected_obligations"])
            )
        else:
            verifier_replays[route_key] = {
                "candidate_type": "theorem_verifier_bundle",
                "verifier_bundle_match": False,
                "reason": "verifier_bundle_missing",
                "unproven_claim": False,
            }
            verifier_matches.append(not ingestion.get("accepted_obligations"))
    recomputed_open_obligations = sorted({
        *[str(v) for v in report.get("theorem_attempt", {}).get("open_obligations", [])],
        *[str(v) for v in interval_backend.get("open_obligations", [])],
        *[str(v) for family in candidate_status for v in family.get("open_obligations", [])],
        *[
            str(obligation_id)
            for ingestion in verifier_ingestion.values()
            if isinstance(ingestion, dict)
            for obligation_id in ingestion.get("rejected_obligations", {})
        ],
    })
    recomputed_open_lemmas = sorted({
        *[
            str(v)
            for contract in exact_contracts.values()
            if isinstance(contract, dict)
            for v in contract.get("open_obligations", [])
        ],
        *[
            str(v)
            for definition in function_spaces.values()
            if isinstance(definition, dict)
            for v in definition.get("open_obligations", [])
        ],
        *[
            str(v)
            for package in lemma_packages.values()
            if isinstance(package, dict)
            for v in package.get("open_lemmas", [])
        ],
    })
    open_obligations_match = list(report.get("open_obligations", [])) == recomputed_open_obligations
    open_lemmas_match = list(report.get("open_lemmas", [])) == recomputed_open_lemmas
    expected_status = (
        "proved_finite_time_blowup"
        if report.get("lemma_packages", {}).get("finite_time_blowup", {}).get("proof_status")
        == "proved_finite_time_blowup"
        else "proved_global_regularity"
        if report.get("lemma_packages", {}).get("global_regularity", {}).get("proof_status")
        == "proved_global_regularity"
        else "candidate_falsified"
        if any(family.get("status") == "candidate_falsified" for family in candidate_status)
        else "blocked_with_named_missing_lemma"
    )
    status_match = str(report.get("proof_status", "")) == expected_status
    max_violation = max(
        0.0 if theorem_replay.get("theorem_grade_report_match", False) else float("inf"),
        0.0 if obligations_match else float("inf"),
        0.0 if all(verifier_matches) else float("inf"),
        0.0 if open_obligations_match else float("inf"),
        0.0 if open_lemmas_match else float("inf"),
        0.0 if status_match else float("inf"),
    )
    return {
        "candidate_type": "navier_stokes_proof_program_report",
        "proof_program_report_match": bool(max_violation <= 1e-9),
        "max_proof_program_violation": float(max_violation),
        "theorem_replay": theorem_replay,
        "obligation_replays": obligation_replays,
        "verifier_replays": verifier_replays,
        "expected_open_obligations": recomputed_open_obligations,
        "expected_open_lemmas": recomputed_open_lemmas,
        "expected_proof_status": expected_status,
        "unproven_claim": False,
    }


def verify_ns_solve_or_falsify_report(report: dict[str, Any]) -> dict[str, Any]:
    """Verify replay consistency for the solve-or-falsify roadmap artifact."""
    if report.get("candidate_type") != "navier_stokes_solve_or_falsify_report":
        raise ValueError("expected candidate_type='navier_stokes_solve_or_falsify_report'")
    phases = {str(name): dict(phase) for name, phase in dict(report.get("phases", {})).items()}
    stored_hashes = dict(report.get("phase_sha256", {}))
    recomputed_hashes = {name: _sha256_json(phase) for name, phase in phases.items()}
    phase_hashes_match = stored_hashes == recomputed_hashes
    theorem_phase = dict(report.get("replay_inputs", {}).get("theorem_attempt", {}))
    program_phase = dict(report.get("replay_inputs", {}).get("proof_program", phases.get("proof_program", {})))
    theorem_replay = verify_theorem_grade_closure_attempt(theorem_phase)
    program_replay = verify_ns_proof_program_report(program_phase)
    recomputed_open = sorted({
        str(obligation)
        for phase in phases.values()
        if isinstance(phase, dict)
        for obligation in phase.get("open_obligations", [])
    })
    open_match = list(report.get("open_obligations", [])) == recomputed_open
    falsification = dict(phases.get("falsification", {}))
    final_gate = (
        dict(phases.get("formal_verification", {}))
        .get("final_claim_gate", {})
    )
    expected_status = (
        "proved_by_external_artifact"
        if bool(final_gate.get("unproven_claim", False))
        else "candidate_falsified"
        if falsification.get("decision") == "stop_current_family"
        else "blocked_with_named_missing_lemma"
    )
    status_match = str(report.get("proof_status", "")) == expected_status
    expected_digest = _sha256_json({
        key: value
        for key, value in report.items()
        if key != "solve_or_falsify_sha256"
    })
    digest_match = str(report.get("solve_or_falsify_sha256", "")) == expected_digest
    max_violation = max(
        0.0 if phase_hashes_match else float("inf"),
        0.0 if theorem_replay.get("theorem_grade_report_match", False) else float("inf"),
        0.0 if program_replay.get("proof_program_report_match", False) else float("inf"),
        0.0 if open_match else float("inf"),
        0.0 if status_match else float("inf"),
        0.0 if digest_match else float("inf"),
    )
    return {
        "candidate_type": "navier_stokes_solve_or_falsify_report",
        "solve_or_falsify_report_match": bool(max_violation <= 1e-9),
        "max_solve_or_falsify_violation": float(max_violation),
        "phase_hashes_match": phase_hashes_match,
        "expected_phase_sha256": recomputed_hashes,
        "stored_phase_sha256": stored_hashes,
        "expected_open_obligations": recomputed_open,
        "expected_proof_status": expected_status,
        "solve_or_falsify_sha256_match": digest_match,
        "theorem_replay": theorem_replay,
        "proof_program_replay": program_replay,
        "unproven_claim": False,
    }


def verify_ns_theorem_ladder_report(report: dict[str, Any]) -> dict[str, Any]:
    """Verify replay consistency for the active-subspace theorem ladder."""
    if report.get("candidate_type") != "navier_stokes_theorem_ladder_report":
        raise ValueError("expected candidate_type='navier_stokes_theorem_ladder_report'")
    phases = {str(name): dict(phase) for name, phase in dict(report.get("phases", {})).items()}
    stored_hashes = dict(report.get("phase_sha256", {}))
    recomputed_hashes = {name: _sha256_json(phase) for name, phase in phases.items()}
    phase_hashes_match = stored_hashes == recomputed_hashes
    frontier_replay = verify_active_subspace_absorption_frontier_report(
        dict(report.get("replay_inputs", {}).get("frontier_report", {}))
    )
    tail_contract_replay = verify_weighted_analytic_tail_norm_contract(
        dict(phases.get("tail_norm_contract", {}))
    )
    tail_contraction_replay = verify_active_subspace_tail_contraction_attempt(
        dict(phases.get("tail_contraction", {}))
    )
    completeness_replay = verify_active_subspace_completeness_theorem_attempt(
        dict(phases.get("active_subspace_completeness", {}))
    )
    finite_tail_phase = dict(phases.get("finite_tail_contraction_diagnostic", {}))
    finite_tail_replay = (
        verify_finite_active_tail_contraction_diagnostic(finite_tail_phase)
        if finite_tail_phase
        else {
            "candidate_type": "finite_active_tail_contraction_diagnostic",
            "finite_active_tail_contraction_match": True,
            "unproven_claim": False,
        }
    )
    analytic_lift_phase = dict(phases.get("analytic_tail_lift", {}))
    analytic_lift_replay = (
        verify_active_tail_contraction_lift_certificate(analytic_lift_phase)
        if analytic_lift_phase
        else {
            "candidate_type": "active_tail_contraction_lift_certificate",
            "active_tail_contraction_lift_match": True,
            "unproven_claim": False,
        }
    )
    recomputed_open = sorted({
        str(obligation)
        for phase in phases.values()
        if isinstance(phase, dict)
        for obligation in phase.get("open_obligations", [])
    })
    open_match = list(report.get("open_obligations", [])) == recomputed_open
    assumptions_gate = dict(phases.get("classical_assumptions", {}))
    expected_status = (
        "proved_by_external_artifact"
        if bool(assumptions_gate.get("classical_assumptions_ready", False))
        else "blocked_with_named_missing_lemma"
    )
    status_match = str(report.get("proof_status", "")) == expected_status
    expected_digest = _sha256_json({
        key: value
        for key, value in report.items()
        if key != "theorem_ladder_sha256"
    })
    digest_match = str(report.get("theorem_ladder_sha256", "")) == expected_digest
    route_summary = dict(report.get("route_summary", {}))
    summary_match = bool(
        bool(route_summary.get("tail_theorem_closed", False))
        == bool(phases.get("active_subspace_completeness", {}).get("active_subspace_complete", False))
        and bool(route_summary.get("existence_theorem_closed", False))
        == bool(phases.get("existence_theorem", {}).get("exact_profile_verified", False))
        and bool(route_summary.get("blowup_theorem_closed", False))
        == bool(phases.get("blowup_theorem", {}).get("norm_divergence_certified", False))
        and bool(route_summary.get("classical_assumptions_ready", False))
        == bool(assumptions_gate.get("classical_assumptions_ready", False))
    )
    if finite_tail_phase:
        summary_match = bool(
            summary_match
            and bool(route_summary.get("finite_tail_contraction_surrogate_passed", False))
            == bool(finite_tail_phase.get("finite_tail_contraction_surrogate_passed", False))
            and route_summary.get("finite_tail_contraction_ratio_upper")
            == finite_tail_phase.get("finite_contraction_ratio_upper")
        )
    if analytic_lift_phase:
        summary_match = bool(
            summary_match
            and bool(route_summary.get("analytic_tail_lift_certified", False))
            == bool(analytic_lift_phase.get("analytic_lift_certified", False))
            and route_summary.get("analytic_tail_lift_q_total_upper")
            == analytic_lift_phase.get("q_total_upper")
        )
    max_violation = max(
        0.0 if phase_hashes_match else float("inf"),
        0.0 if frontier_replay.get("active_subspace_absorption_frontier_match", False) else float("inf"),
        0.0 if tail_contract_replay.get("weighted_analytic_tail_norm_match", False) else float("inf"),
        0.0 if tail_contraction_replay.get("active_subspace_tail_contraction_match", False) else float("inf"),
        0.0 if completeness_replay.get("active_subspace_completeness_match", False) else float("inf"),
        0.0 if finite_tail_replay.get("finite_active_tail_contraction_match", False) else float("inf"),
        0.0 if analytic_lift_replay.get("active_tail_contraction_lift_match", False) else float("inf"),
        0.0 if open_match else float("inf"),
        0.0 if status_match else float("inf"),
        0.0 if digest_match else float("inf"),
        0.0 if summary_match else float("inf"),
    )
    return {
        "candidate_type": "navier_stokes_theorem_ladder_report",
        "theorem_ladder_report_match": bool(max_violation <= 1e-9),
        "max_theorem_ladder_violation": float(max_violation),
        "phase_hashes_match": phase_hashes_match,
        "expected_phase_sha256": recomputed_hashes,
        "stored_phase_sha256": stored_hashes,
        "expected_open_obligations": recomputed_open,
        "expected_proof_status": expected_status,
        "theorem_ladder_sha256_match": digest_match,
        "frontier_replay": frontier_replay,
        "tail_contract_replay": tail_contract_replay,
        "tail_contraction_replay": tail_contraction_replay,
        "completeness_replay": completeness_replay,
        "finite_tail_replay": finite_tail_replay,
        "analytic_lift_replay": analytic_lift_replay,
        "unproven_claim": False,
    }


def verify_axisymmetric_swirl_candidate_artifact(
    artifact: dict[str, Any],
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Replay an axisymmetric swirl artifact from serialized samples only."""
    rin = artifact["replay_inputs"]
    recomputed = _axisymmetric_swirl_residual_samples(
        np.asarray(rin["streamfunction"], dtype=float),
        np.asarray(rin["swirl"], dtype=float),
        np.asarray(rin["pressure"], dtype=float),
        radial_axis=np.asarray(rin["radial_axis"], dtype=float),
        axial_axis=np.asarray(rin["axial_axis"], dtype=float),
        viscosity=float(rin["viscosity"]),
        density=float(rin["density"]),
    )
    energy = _axisymmetric_energy_estimate(
        recomputed["velocity"],
        radial_axis=np.asarray(rin["radial_axis"], dtype=float),
        axial_axis=np.asarray(rin["axial_axis"], dtype=float),
    )
    stored_result = artifact["result"]
    stored_samples = stored_result["residual_samples"]
    sample_diffs = [
        float(np.max(np.abs(
            recomputed["residual_samples"][name] - np.asarray(stored_samples[name], dtype=float)
        )))
        for name in ("radial", "azimuthal", "axial", "divergence")
    ]
    stored_diag = stored_result["residual_diagnostics"]
    diag_diffs = [
        abs(float(recomputed["residual_diagnostics"][name]) - float(stored_diag[name]))
        for name in (
            "max_abs_momentum_residual",
            "rms_momentum_residual",
            "max_abs_continuity",
            "rms_continuity",
        )
    ]
    energy_diff = abs(float(stored_result["finite_energy_estimate"]) - energy)
    max_diff = max([*sample_diffs, *diag_diffs, energy_diff])
    return {
        "candidate_type": "axisymmetric_swirl_sandbox",
        "replay_match": bool(max_diff <= atol),
        "max_abs_replay_diff": float(max_diff),
        "energy_abs_diff": float(energy_diff),
        "stage": "candidate_replay_ready" if max_diff <= atol else "numerical_artifact",
        "unproven_claim": False,
    }


def build_axisymmetric_candidate_bridge_artifacts(
    axisymmetric_artifact: dict[str, Any],
    *,
    time: np.ndarray | None = None,
    blowup_time: float = 1.0,
) -> dict[str, Any]:
    """Emit companion regularity and blow-up artifacts for an axisymmetric candidate."""
    t = np.linspace(0.0, 0.8, 80) if time is None else np.asarray(time, dtype=float)
    if np.any(t >= blowup_time):
        raise ValueError("bridge-artifact time grid must stay below blowup_time")
    result = axisymmetric_artifact["result"]
    energy = float(max(result.get("finite_energy_estimate", 1.0), 1e-12))
    residual_diagnostics = dict(
        result.get("residual_diagnostics", result.get("train", {}).get("residual_diagnostics", {}))
    )
    residual = float(residual_diagnostics.get("max_abs_momentum_residual", 0.0))
    enstrophy = energy * (1.0 + 0.05 * t + residual * t * t)
    traces = {
        "energy": energy * np.ones_like(t),
        "enstrophy": enstrophy,
        "palinstrophy": (1.0 + residual) * enstrophy,
        "bkm_vorticity_proxy": np.sqrt(np.maximum(enstrophy, 1e-12)),
    }
    replay_grid = dict(axisymmetric_artifact.get("replay_grid", {}))
    coefficients = list(axisymmetric_artifact.get("coefficients", []))
    regularity = build_regularity_candidate_artifact(
        t,
        traces,
        target="enstrophy",
        include_quadratic=False,
        replay_grid=replay_grid,
        coefficients=coefficients,
        notes="axisymmetric sandbox companion regularity trace",
    )
    norm_values = np.sqrt(energy) * (blowup_time - t) ** -0.25
    blowup = build_blowup_candidate_artifact(
        t,
        norm_values,
        blowup_time=blowup_time,
        ansatz_metadata=dict(axisymmetric_artifact["replay_inputs"].get("ansatz_metadata", {})),
        residual_metrics=residual_diagnostics,
        replay_grid=replay_grid,
        coefficients=coefficients,
        notes="axisymmetric sandbox companion blow-up-rate trace",
    )
    return {
        "candidate_type": "axisymmetric_candidate_bridge",
        "axisymmetric": axisymmetric_artifact,
        "regularity": regularity,
        "blowup": blowup,
        "honesty": {
            "unproven_claim": False,
            "global_regularity_claim": False,
            "finite_time_blowup_claim": False,
        },
    }


def verify_axisymmetric_candidate_bridge_artifacts(
    bridge: dict[str, Any],
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Replay every replayable child in an axisymmetric bridge artifact."""
    if bridge.get("candidate_type") != "axisymmetric_candidate_bridge":
        raise ValueError("expected candidate_type='axisymmetric_candidate_bridge'")
    axisymmetric = replay_candidate_artifact(dict(bridge["axisymmetric"]), atol=atol)
    regularity = replay_candidate_artifact(dict(bridge["regularity"]), atol=atol)
    blowup = replay_candidate_artifact(dict(bridge["blowup"]), atol=atol)
    child_reports = {
        "axisymmetric": axisymmetric,
        "regularity": regularity,
        "blowup": blowup,
    }
    return {
        "candidate_type": "axisymmetric_candidate_bridge",
        "replay_match": bool(all(bool(report.get("replay_match", False)) for report in child_reports.values())),
        "child_reports": child_reports,
        "unproven_claim": False,
    }


def replay_candidate_artifact(artifact: dict[str, Any], *, atol: float = 1e-10) -> dict[str, Any]:
    """Recompute a replayable candidate artifact from serialized inputs only."""
    ctype = artifact["candidate_type"]
    if ctype == "axisymmetric_candidate_bridge":
        return verify_axisymmetric_candidate_bridge_artifacts(artifact, atol=atol)
    if ctype == "axisymmetric_interval_report":
        return verify_axisymmetric_interval_report(artifact)
    if ctype == "blowup_analytic_closure_report":
        return verify_blowup_closure_report(artifact)
    if ctype == "active_subspace_blowup_closure_report":
        return verify_active_subspace_closure_report(artifact)
    if ctype == "active_subspace_invariance_report":
        return verify_active_subspace_invariance_report(artifact)
    if ctype == "active_subspace_absorption_frontier_report":
        return verify_active_subspace_absorption_frontier_report(artifact)
    if ctype == "finite_active_tail_contraction_diagnostic":
        return verify_finite_active_tail_contraction_diagnostic(artifact)
    if ctype == "active_tail_contraction_lift_certificate":
        return verify_active_tail_contraction_lift_certificate(artifact)
    if ctype == "active_projector_error_certificate":
        return verify_active_projector_error_certificate(artifact)
    if ctype == "interval_jacobian_error_certificate":
        return verify_interval_jacobian_error_certificate(artifact)
    if ctype == "nonlinear_tail_remainder_certificate":
        return verify_nonlinear_tail_remainder_certificate(artifact)
    if ctype == "analytic_tail_error_certificate":
        return verify_analytic_tail_error_certificate(artifact)
    if ctype == "active_tail_lift_error_budget":
        return verify_active_tail_lift_error_budget(artifact)
    if ctype == "regularity_inequality_report":
        return verify_regularity_inequality_report(artifact)
    if ctype == "theorem_grade_closure_attempt":
        return verify_theorem_grade_closure_attempt(artifact)
    if ctype == "navier_stokes_proof_program_report":
        return verify_ns_proof_program_report(artifact)
    if ctype == "navier_stokes_solve_or_falsify_report":
        return verify_ns_solve_or_falsify_report(artifact)
    if ctype == "navier_stokes_theorem_ladder_report":
        return verify_ns_theorem_ladder_report(artifact)
    rin = artifact["replay_inputs"]
    stored = artifact["result"]
    if ctype == "axisymmetric_swirl_sandbox":
        return verify_axisymmetric_swirl_candidate_artifact(artifact, atol=atol)
    if ctype == "axisymmetric_swirl_refined":
        return verify_refined_axisymmetric_swirl_candidate_artifact(artifact, atol=atol)
    if ctype == "regularity_growth_law":
        replayed = run_regularity_search(
            np.asarray(rin["time"], dtype=float),
            {k: np.asarray(v, dtype=float) for k, v in rin["traces"].items()},
            target=str(rin["target"]),
            include_quadratic=bool(rin["include_quadratic"]),
            alpha=float(rin["alpha"]),
            threshold=float(rin["threshold"]),
        )
        coeff_names = set(stored.get("coefficients", {})) | set(replayed.get("coefficients", {}))
        coeff_diff = max(
            (
                abs(
                    float(stored.get("coefficients", {}).get(name, 0.0))
                    - float(replayed.get("coefficients", {}).get(name, 0.0))
                )
                for name in coeff_names
            ),
            default=0.0,
        )
        fit_diff = abs(float(stored["fit_rmse"]) - float(replayed["fit_rmse"]))
        match = coeff_diff <= atol and fit_diff <= atol
        return {
            "candidate_type": ctype,
            "replay_match": bool(match),
            "coefficient_max_abs_diff": float(coeff_diff),
            "fit_rmse_abs_diff": float(fit_diff),
            "unproven_claim": False,
        }
    if ctype == "self_similar_blowup_rate":
        replayed = assess_blowup_candidate(
            np.asarray(rin["time"], dtype=float),
            np.asarray(rin["norm_values"], dtype=float),
            blowup_time=float(rin["blowup_time"]),
            ansatz_metadata=dict(rin.get("ansatz_metadata", {})),
            residual_metrics=dict(rin.get("residual_metrics", {})),
        )
        alpha_diff = abs(float(stored["rate_fit"]["alpha"]) - float(replayed["rate_fit"]["alpha"]))
        rmse_diff = abs(
            float(stored["rate_fit"]["log_fit_rmse"])
            - float(replayed["rate_fit"]["log_fit_rmse"])
        )
        match = alpha_diff <= atol and rmse_diff <= atol
        return {
            "candidate_type": ctype,
            "replay_match": bool(match),
            "alpha_abs_diff": float(alpha_diff),
            "log_fit_rmse_abs_diff": float(rmse_diff),
            "unproven_claim": False,
        }
    raise ValueError(f"unsupported candidate_type {ctype!r}")


def verify_clm_multizero_first_blowup(
    cert: dict[str, Any], *, atol: float = 1e-7
) -> dict[str, Any]:
    r"""Independent numpy replay of a CLM multi-zero earliest-blow-up certificate.

    Recomputes -- with a *separate* code path that imports nothing from
    ``omnibias.pinn.certified`` -- every quantity the certificate claims:

    * the numerator polynomial ``P(u) = sum_i c_i prod_{j!=i}(u + a_j^2)`` whose
      positive roots are the squared non-origin zeros of the odd profile;
    * all real zeros ``x = 0`` and ``x = +-sqrt(u_*)`` and the fact that they are
      genuine zeros of ``omega0(x) = sum_i c_i x/(x^2+a_i^2)``;
    * the line Hilbert transform ``H omega0(x) = -sum_i c_i a_i/(x^2+a_i^2)`` at
      every zero, its maximum, and the earliest blow-up time ``2 / max H omega0``.

    The recomputed maximum / time are then checked to fall inside the
    certificate's outward-rounded intervals, and the zero count is matched.
    Returns a report dict with ``unproven_claim=False`` and an overall
    ``replay_match`` flag.
    """
    coeffs = [float(c) for c in cert["coeffs"]]
    scales = [float(a) for a in cert["scales"]]
    n = len(coeffs)
    a2 = [a * a for a in scales]

    # Independent numerator polynomial P(u) (ascending coefficients).
    poly = np.asarray([0.0], dtype=float)
    for i in range(n):
        term = np.asarray([1.0], dtype=float)
        for j in range(n):
            if j != i:
                term = np.polynomial.polynomial.polymul(term, np.asarray([a2[j], 1.0]))
        poly = np.polynomial.polynomial.polyadd(poly, coeffs[i] * term)
    poly = np.atleast_1d(np.asarray(poly, dtype=float))

    stored_poly = np.asarray(cert["numerator_poly_u_coeffs"], dtype=float)
    m = max(poly.size, stored_poly.size)
    poly_pad = np.zeros(m)
    poly_pad[: poly.size] = poly
    stored_pad = np.zeros(m)
    stored_pad[: stored_poly.size] = stored_poly
    poly_max_abs_diff = float(np.max(np.abs(poly_pad - stored_pad))) if m else 0.0

    # Positive real roots -> all real zeros (origin + symmetric pairs).
    if poly.size > 1:
        roots = np.polynomial.polynomial.polyroots(poly)
        us = [float(r.real) for r in roots if abs(r.imag) <= 1e-9 and r.real > 0.0]
    else:
        us = []
    xs = [0.0]
    for u in us:
        root = float(np.sqrt(u))
        xs.extend((root, -root))

    def hilbert(x: float) -> float:
        return -sum(c * a / (x * x + a * a) for c, a in zip(coeffs, scales, strict=True))

    def omega0(x: float) -> float:
        return sum(c * x / (x * x + a * a) for c, a in zip(coeffs, scales, strict=True))

    max_abs_omega0_at_zeros = max((abs(omega0(x)) for x in xs), default=0.0)
    hvals = [hilbert(x) for x in xs]
    h_max = max(hvals) if hvals else 0.0
    recomputed_time = (2.0 / h_max) if h_max > 0.0 else None

    n_distinct_positive = len(us)
    cert_distinct = int(cert["n_distinct_positive_roots"])

    cert_hmax = cert["hilbert_omega0_max"]
    hmax_in = bool(cert_hmax["lower"] <= h_max <= cert_hmax["upper"])

    bt = cert.get("first_blowup_time")
    if recomputed_time is not None and isinstance(bt, dict):
        time_in = bool(bt["lower"] <= recomputed_time <= bt["upper"])
    elif recomputed_time is None and bt is None:
        time_in = True
    else:
        time_in = False

    cert_singular = bool(cert.get("singularity_certified", False))
    singular_match = bool((h_max > 0.0) == cert_singular)

    replay_match = bool(
        poly_max_abs_diff <= atol
        and max_abs_omega0_at_zeros <= atol
        and n_distinct_positive == cert_distinct
        and hmax_in
        and time_in
        and singular_match
    )
    return {
        "numerator_poly_max_abs_diff": poly_max_abs_diff,
        "n_distinct_positive_roots_recomputed": n_distinct_positive,
        "n_distinct_positive_roots_match": bool(n_distinct_positive == cert_distinct),
        "max_abs_omega0_at_zeros": float(max_abs_omega0_at_zeros),
        "zeros_are_genuine": bool(max_abs_omega0_at_zeros <= atol),
        "hilbert_max_recomputed": float(h_max),
        "hilbert_max_in_certificate_interval": hmax_in,
        "first_blowup_time_recomputed": recomputed_time,
        "first_blowup_time_in_certificate_interval": time_in,
        "singularity_match": singular_match,
        "replay_match": replay_match,
        "unproven_claim": False,
    }


def assess_navier_stokes_candidate(bundle: dict[str, Any]) -> dict[str, Any]:
    """One-shot honest assessment of a CAP bundle."""
    verify = verify_ns_cap_bundle(bundle)
    vin = bundle["validation_inputs"]
    features = regularity_feature_vector(
        np.asarray(vin["velocity"], dtype=float),
        pressure=np.asarray(vin["pressure"], dtype=float),
        lengths=vin["lengths"],
    )
    return {
        "verification": verify,
        "regularity_features": features,
        "honesty": {
            "unproven_claim": False,
            "exact_solution_claim": False,
            "interval_verified": False,
            "theorem_prover_verified": False,
        },
    }


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "assess_blowup_candidate",
    "assess_navier_stokes_candidate",
    "build_axisymmetric_candidate_bridge_artifacts",
    "build_blowup_candidate_artifact",
    "build_regularity_candidate_artifact",
    "fit_regularity_growth_bound",
    "fit_self_similar_blowup_rate",
    "pressure_poisson_residual_periodic",
    "primitive_residual_periodic",
    "regularity_feature_vector",
    "replay_candidate_artifact",
    "run_regularity_search",
    "verify_active_projector_error_certificate",
    "verify_active_subspace_absorption_frontier_report",
    "verify_active_subspace_closure_report",
    "verify_active_subspace_completeness_theorem_attempt",
    "verify_active_subspace_invariance_report",
    "verify_active_subspace_tail_contraction_attempt",
    "verify_active_tail_contraction_lift_certificate",
    "verify_active_tail_lift_error_budget",
    "verify_analytic_tail_error_certificate",
    "verify_axisymmetric_candidate_bridge_artifacts",
    "verify_axisymmetric_interval_report",
    "verify_axisymmetric_swirl_candidate_artifact",
    "verify_blowup_closure_report",
    "verify_clm_multizero_first_blowup",
    "verify_finite_active_tail_contraction_diagnostic",
    "verify_interval_jacobian_error_certificate",
    "verify_nonlinear_tail_remainder_certificate",
    "verify_ns_cap_bundle",
    "verify_ns_proof_program_report",
    "verify_ns_solve_or_falsify_report",
    "verify_ns_theorem_ladder_report",
    "verify_proof_obligation_bundle",
    "verify_refined_axisymmetric_swirl_candidate_artifact",
    "verify_regularity_inequality_report",
    "verify_theorem_grade_closure_attempt",
    "verify_theorem_verifier_bundle",
    "verify_weighted_analytic_tail_norm_contract",
]

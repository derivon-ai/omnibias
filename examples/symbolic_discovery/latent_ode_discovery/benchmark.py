# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Latent-ODE discovery benchmark: recover a hidden oscillator from one channel.

We observe a **single coordinate** of a 2-D linear oscillator -- undamped
(``z1' = z2, z2' = -omega^2 z1``) and damped
(``z2' = -omega^2 z1 - 2 gamma z2``) -- and reconstruct the governing law with the
Takens delay-embedding + autoencoder + ``FieldLawDiscoverer`` pipeline.

The latent coordinates are only defined up to a diffeomorphism, so the honest,
checkable target is the **spectrum**: the recovered latent system matrix is
similar to the truth, hence its eigenvalues (the oscillation frequency and the
decay rate) are coordinate-invariant and match the ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from omnibias.symbolic.latent import discover_latent_ode


def _harmonic_series(*, omega: float, dt: float, t_max: float) -> np.ndarray:
    t = np.arange(0.0, t_max, dt)
    return np.cos(omega * t)


def _damped_series(*, omega: float, gamma: float, dt: float, t_max: float) -> np.ndarray:
    t = np.arange(0.0, t_max, dt)
    omega_d = np.sqrt(omega * omega - gamma * gamma)
    return np.exp(-gamma * t) * np.cos(omega_d * t)


def _spectrum(eigenvalues: np.ndarray) -> dict[str, float]:
    return {
        "max_abs_imag": float(np.max(np.abs(eigenvalues.imag))),
        "mean_real": float(np.mean(eigenvalues.real)),
    }


def evaluate_benchmark(*, dt: float = 0.01) -> dict[str, Any]:
    """Recover the undamped and damped oscillator spectra from one coordinate."""
    omega = 1.7
    undamped = discover_latent_ode(
        _harmonic_series(omega=omega, dt=dt, t_max=40.0),
        dt=dt,
        latent_dim=2,
        embedding_dim=4,
        delay=5,
        max_degree=2,
    )
    und_spec = _spectrum(undamped.eigenvalues)

    omega2, gamma = 2.0, 0.15
    omega_d = float(np.sqrt(omega2 * omega2 - gamma * gamma))
    damped = discover_latent_ode(
        _damped_series(omega=omega2, gamma=gamma, dt=0.005, t_max=30.0),
        dt=0.005,
        latent_dim=2,
        embedding_dim=5,
        delay=8,
        max_degree=2,
    )
    dmp_spec = _spectrum(damped.eigenvalues)

    return {
        "undamped": {
            "true_omega": omega,
            "recovered_frequency": und_spec["max_abs_imag"],
            "frequency_abs_error": abs(und_spec["max_abs_imag"] - omega),
            "recovered_growth_rate": und_spec["mean_real"],
            "reconstruction_rmse": undamped.reconstruction_rmse,
            "component_formulas": list(undamped.component_formulas),
        },
        "damped": {
            "true_omega_d": omega_d,
            "true_gamma": gamma,
            "recovered_frequency": dmp_spec["max_abs_imag"],
            "recovered_growth_rate": dmp_spec["mean_real"],
            "frequency_abs_error": abs(dmp_spec["max_abs_imag"] - omega_d),
            "growth_rate_abs_error": abs(dmp_spec["mean_real"] + gamma),
            "reconstruction_rmse": damped.reconstruction_rmse,
            "component_formulas": list(damped.component_formulas),
        },
        "honesty_note": undamped.note,
    }


def write_artifacts(results: dict[str, Any], out_dir: Path) -> None:
    """Write the benchmark report to ``out_dir/report.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2, sort_keys=True))

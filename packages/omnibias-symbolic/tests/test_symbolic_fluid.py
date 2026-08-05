# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Second-source (numpy-only) replay of the proof-carrying fluid certificates.

These tests import **only** ``omnibias.symbolic`` (never ``omnibias.pinn``), so
they exercise the independent verifier in isolation: 3-D fixture regeneration, the
finite-difference residual cross-check, the analytic ``tanh``-tower streamfunction
replay, and the independent rollout re-integration.
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.symbolic.fluid import (
    finite_difference_residual_sups,
    periodic_flow_residual_sups,
    regenerate_periodic_flow,
    streamfunction_residual_sups,
    verify_periodic_flow_residual,
    verify_rollout_diagnostics,
    verify_streamfunction_residual,
)

_TWO_PI = 2.0 * math.pi


def _beltrami_descriptor(n: int = 16) -> dict:
    return {
        "name": "beltrami_abc_flow",
        "dimension": 3,
        "n": n,
        "lengths": [_TWO_PI, _TWO_PI, _TWO_PI],
        "viscosity": 0.05,
        "density": 1.0,
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "wavenumber": 1,
        "time": 0.2,
    }


def test_regenerate_3d_beltrami_is_divergence_free() -> None:
    flow = regenerate_periodic_flow(_beltrami_descriptor())
    assert flow["velocity"].shape == (3, 16, 16, 16)
    sups = periodic_flow_residual_sups(_beltrami_descriptor())
    assert sups["residual_sup"] < 1e-9  # exact 3-D solution


def test_finite_difference_residual_is_small_for_exact_fixture() -> None:
    desc = _beltrami_descriptor()
    fd = finite_difference_residual_sups(desc)
    floor = 50.0 * fd["max_spacing"] ** 2 * (fd["velocity_scale"] + 1.0) + 1e-6
    assert fd["finite_difference_momentum_residual_sup"] <= floor
    assert fd["finite_difference_continuity_residual_sup"] <= floor


def test_periodic_flow_twin_round_trip() -> None:
    desc = _beltrami_descriptor()
    sups = periodic_flow_residual_sups(desc)
    cert = {
        "fixture": desc,
        "momentum_residual_sup": sups["momentum_residual_sup"],
        "continuity_residual_sup": sups["continuity_residual_sup"],
        "pressure_poisson_residual_sup": sups["pressure_poisson_residual_sup"],
        "residual_sup": sups["residual_sup"],
        "residual_tol": 1e-8,
        "exact_solution_claim": True,
    }
    rep = verify_periodic_flow_residual(cert)
    assert rep["replay_match"] is True
    assert rep["finite_difference_consistent"] is True
    cert["momentum_residual_sup"] = 0.5  # forge an impossibly large residual
    assert verify_periodic_flow_residual(cert)["replay_match"] is False


def _shear_streamfunction_descriptor() -> dict:
    # y-only tanh streamfunction -> exact steady-Euler shear
    return {
        "name": "shear_streamfunction",
        "domain": [[0.0, _TWO_PI], [0.0, _TWO_PI]],
        "layers": [
            {"weight": [[0.0, 0.7], [0.0, -1.1], [0.0, 0.5]],
             "bias": [0.2, -0.3, 0.6], "activation": "tanh"},
            {"weight": [[0.9, -0.4, 1.2]], "bias": [0.0], "activation": None},
        ],
    }


def test_streamfunction_twin_replays_exact_shear() -> None:
    sf = _shear_streamfunction_descriptor()
    sups = streamfunction_residual_sups(sf, viscosity=0.0)
    assert sups["residual_sup"] < 1e-9  # advection vanishes for y-only psi
    cert = {
        "payload": {
            "streamfunction": sf,
            "viscosity": 0.0,
            "residual_sup": max(sups["residual_sup"], 1e-12),
            "divergence_sup": max(sups["divergence_sup"], 1e-12),
            "exact_steady_euler_claim": True,
            "residual_tol": 1e-8,
        }
    }
    rep = verify_streamfunction_residual(cert)
    assert rep["replay_match"] is True
    assert rep["methodologically_independent"] is True
    # a forged exact claim with a non-shear-sized residual must be rejected
    cert["payload"]["residual_sup"] = 1e-14
    cert["payload"]["streamfunction"] = {
        "domain": [[0.0, _TWO_PI], [0.0, _TWO_PI]],
        "layers": [
            {"weight": [[0.6, 0.7], [-0.8, -1.1]], "bias": [0.2, -0.3], "activation": "tanh"},
            {"weight": [[0.9, -0.4]], "bias": [0.0], "activation": None},
        ],
    }
    assert verify_streamfunction_residual(cert)["replay_match"] is False


def test_rollout_twin_reintegrates_consistently() -> None:
    descriptor = {
        "name": "fourier_mode_vorticity",
        "n": 32,
        "length": _TWO_PI,
        "modes": [
            {"kx": 1, "ky": 0, "amp": 1.0, "phase": 0.0},
            {"kx": 0, "ky": 2, "amp": 0.7, "phase": 0.3},
        ],
    }
    base = {"initial_vorticity": descriptor, "viscosity": 0.0, "dt": 2e-3, "steps": 120, "length": _TWO_PI}
    probe = verify_rollout_diagnostics({**base, "max_divergence": 0.0})
    recomputed = probe["recomputed"]
    cert = {**base, **recomputed}
    rep = verify_rollout_diagnostics(cert)
    assert rep["replay_match"] is True
    assert recomputed["max_divergence"] < 1e-10
    assert recomputed["energy_drift"] < 1e-3

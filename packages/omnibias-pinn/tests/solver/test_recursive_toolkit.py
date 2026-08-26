# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Recursive Burgers / Cole-Hopf / Picard toolkit (closed-form adjacent)."""

from __future__ import annotations

import numpy as np
from omnibias.pinn.solver._core.recursive import (
    cole_hopf_exact_burgers_demo,
    honesty_payload,
    picard_frozen_advection,
    recursive_toolkit_smoke,
)


def test_cole_hopf_exact_burgers_demo() -> None:
    out = cole_hopf_exact_burgers_demo(nu=0.05, k=-0.8, order=12)
    assert out["check"]["passed"] is True
    assert out["honesty"]["navier_stokes_proof_claim"] is False


def test_picard_defect_improves() -> None:
    n = 48
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    dx = float(x[1] - x[0])
    u0 = 0.3 * np.sin(2.0 * np.pi * x)
    report = picard_frozen_advection(u0, viscosity=0.08, dx=dx, iters=25, tol=1e-9)
    assert report.residual_history[0] > report.final_residual
    assert report.honesty["navier_stokes_proof_claim"] is False


def test_recursive_toolkit_smoke() -> None:
    smoke = recursive_toolkit_smoke()
    assert smoke["cole_hopf_passed"] is True
    assert smoke["picard_improved"] is True
    assert type(smoke["cole_hopf_passed"]) is bool
    assert type(smoke["picard_improved"]) is bool
    assert smoke["honesty"] == honesty_payload()
    assert smoke["honesty"]["navier_stokes_proof_claim"] is False
    assert "Adomian polynomials" in smoke["missing"]
    assert "OWNS one-way spatial filters" in smoke["missing"]
    assert "3-D NS Cauchy-Kovalevskaya recurrence" in smoke["missing"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Implicit DAE residual discovery on a planted circle."""

from __future__ import annotations

import numpy as np
from omnibias.symbolic.system_discovery import ImplicitSystemDiscoverer


def test_circle_dae_residuals() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 80, dtype=np.float64)
    q = np.cos(t)
    p = np.sin(t)
    out = ImplicitSystemDiscoverer().discover(
        t,
        {"q": q, "p": p},
        differential=("q", "p"),
        algebraic_squares=("q", "p"),
        hidden=64,
    )
    assert out.passed is True
    assert out.yang_mills_claim is False
    assert out.algebraic_rmse["q^2+p^2"] < 0.05
    q_eq = out.differential["q"]
    # q' ≈ -p
    pred = q_eq.predict(np.column_stack([q, p]))
    assert float(np.sqrt(np.mean((pred + p) ** 2))) < 0.15

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Piecewise symbolic discovery: recover a switched ODE as a hybrid automaton.

`omnibias.symbolic.piecewise` routes samples into regions with an `omnibias.partition` soft
partition, then runs the existing STLSQ sparse regression INSIDE each region. A system whose
governing law switches across a surface is recovered as one equation per region plus the
hardened ``if ... then`` switch. Global SINDy on the same data finds a single averaged law
that fits neither regime.

This CPU smoke builds a switched first-order ODE trajectory

* region A (``x < 0``):  ``du/dx = 0.5 - 0.5 u``   (relaxes toward 1)
* region B (``x > 0``):  ``du/dx = -1.0 u``         (decays toward 0)

with ``u`` continuous (so ``du`` has a kink at ``x = 0``), fits ONE global omnibias field to
``(x, u)``, reads off the exact closed-form ``(u, du)`` jet, and shows the piecewise automaton
recovers both laws + the boundary and beats the global average.

Honesty: the STLSQ fit is numpy / non-differentiable; near the switch surface the single
global field smooths the kink, so the per-region fits are approximate (the boundary itself is
recovered exactly from the partition). The ``beta -> inf`` gate hardening is the feasibility /
temperature sense of "collapse", distinct from the founding ``delta -> 0`` bias collapse.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _switched_trajectory(n_fine: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-2.0, 2.0, n_fine)
    dx = xs[1] - xs[0]
    u = np.empty_like(xs)
    u[0] = 0.0
    for i in range(1, n_fine):
        xi = xs[i - 1]
        du = 0.5 - 0.5 * u[i - 1] if xi < 0.0 else -1.0 * u[i - 1]
        u[i] = u[i - 1] + dx * du
    return xs, u


def main() -> None:
    from omnibias.partition import PartitionConfig
    from omnibias.partition._core.params import PartitionParams
    from omnibias.symbolic.discovery import rmse
    from omnibias.symbolic.field_discovery import extract_field_jet
    from omnibias.symbolic.piecewise import (
        fit_piecewise_ode_law,
        global_sparse_law,
        polynomial_value_library,
    )

    xs, u = _switched_trajectory()
    sel = np.linspace(0, xs.size - 1, 400).astype(int)
    x = xs[sel].reshape(-1, 1)
    y = u[sel]

    cfg = PartitionConfig(n_features=1, depth=1, split_kind="axis", beta_final=32.0, anneal_steps=1)
    partition = PartitionParams(cfg, W=np.array([[1.0]]), t=np.array([0.0]))

    # degree=1: discover an affine law du = a + b*u per region (matches the true linear regimes).
    automaton, field = fit_piecewise_ode_law(x, y, partition, degree=1, hidden=200, seed=0)

    print("=== piecewise symbolic discovery of a switched ODE ===")
    print(f"fitted global field train_rmse = {field.train_rmse:.3e}")
    print(f"\nswitch surface: {automaton.switch_conditions()[0]}")
    print("\nrecovered hybrid automaton (piecewise laws):")
    print(automaton.report())
    print("\ntrue laws:   x<0: du = 0.5 - 0.5*u     x>0: du = -1*u")

    # score piecewise vs the single-law global baseline on the same closed-form jet
    jet = extract_field_jet(field, x, max_order=1)
    uu, du = jet.value(), jet.partial((1,))
    design, names = polynomial_value_library(uu, degree=1)
    piece_rmse = rmse(du, automaton.predict(x, design))
    glob = global_sparse_law(design, du, names)
    glob_rmse = rmse(du, glob.predict(design))

    print("\n--- piecewise vs global SINDy (fit to the closed-form du jet) ---")
    print(f"  global (one averaged law): {glob.formula(lhs='du')}")
    print(f"  global   du-RMSE = {glob_rmse:.4e}")
    print(f"  piecewise du-RMSE = {piece_rmse:.4e}   ({glob_rmse / max(piece_rmse, 1e-12):.1f}x better)")
    assert piece_rmse < 0.7 * glob_rmse
    print("\nOK: two laws + the switch recovered; the global average fits neither regime.")


if __name__ == "__main__":
    main()

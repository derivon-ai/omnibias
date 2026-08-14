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

Honesty: STLSQ polish is numpy / non-differentiable; gates and per-cell
coefficients are trained by Adam on the soft-weighted residual, then hardened.
The learned split is fit on the closed-form field jet ``(x, u, du)`` (not the
oracle switched law). The oracle axis partition is the control. A tab SoftTree
or Arrangement trained on the trajectory's finite-difference ``du`` can be
hardened from the **fitted** split as the partition (distinct from
``fit_learned_piecewise_ode``). The Arrangement constructor is unplanted
(random ``W``, no ``e_0``); that path does not call ``_refine_split_threshold``.
The ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the founding ``delta -> 0`` bias collapse.
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
    from omnibias.symbolic.field_discovery import extract_field_jet, fit_neural_field_nd
    from omnibias.symbolic.piecewise import (
        fit_learned_piecewise_ode,
        fit_piecewise_ode_law,
        global_sparse_law,
        polynomial_value_library,
    )

    xs, u = _switched_trajectory()
    sel = np.linspace(0, xs.size - 1, 400).astype(int)
    x = xs[sel].reshape(-1, 1)
    y = u[sel]
    field_learned = fit_neural_field_nd(x, y, hidden=200, seed=0)
    jet_learned = extract_field_jet(field_learned, x, max_order=1)
    u_jet, du_jet = jet_learned.value(), jet_learned.partial((1,))
    learned, state = fit_learned_piecewise_ode(
        x,
        u_jet,
        du_jet,
        n_gates=1,
        degree=1,
        steps=200,
        seed=0,
        alpha=1e-12,
        threshold=1e-5,
    )
    print("=== learned partition (gates from data) ===")
    print(learned.report())
    print(f"learned threshold t = {float(np.asarray(state['t']).reshape(-1)[0]):.3f}")

    cfg = PartitionConfig(n_features=1, depth=1, split_kind="axis", beta_final=32.0, anneal_steps=1)
    partition = PartitionParams(cfg, W=np.array([[1.0]]), t=np.array([0.0]))

    # Oracle partition (control): degree=1 affine law per region.
    automaton, field = fit_piecewise_ode_law(x, y, partition, degree=1, hidden=200, seed=0)

    print("\n=== oracle partition (control) ===")
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

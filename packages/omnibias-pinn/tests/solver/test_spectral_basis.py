# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stage 3: readout seam + basis='spectral' for the frozen-feature solver."""

from __future__ import annotations

import math

import numpy as np
import omnibias.pinn.solver as pde
import omnibias.pinn.solver.jax as pj
import omnibias.pinn.solver.torch as pt
import pytest
import torch
from omnibias.pinn._core.constrained import HardCondition, dirichlet
from omnibias.pinn.solver.torch.assemble import build_plan, eval_plan_rows
from omnibias.pinn.solver.torch.readout import (
    freeze_features,
    readout_size,
    set_readout,
)
from omnibias.pinn.torch.cage.constrained import ConstrainedExpressionField
from omnibias.pinn.torch.fields.jet_mlp import JetMLPVectorField
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

DTYPE = torch.float64
SMALL = pde.CollocationSpec(n_interior=8, n_boundary=6)


def _periodic_heat():
    dom = pde.Domain(
        ("t", "x"),
        ((0.0, 0.15), (0.0, 1.0)),
        periodic=(False, True),
        time_axis="t",
    )

    def initial(c):
        xp = pde.array_namespace(c)
        return xp.sin(2.0 * math.pi * c[:, 1])

    base = pde.heat(dom, diffusivity=0.1, initial=initial)
    from dataclasses import replace

    return replace(
        base,
        boundary=(pde.BoundaryCondition(component="u", kind="periodic", axis="x"),),
    )


def _poisson():
    return pde.poisson(pde.Domain(("x",), ((0.0, 1.0),)), source=-1.0, boundary=0.0)


def test_spectral_column_sweep_matches_fresh_field_per_column() -> None:
    """Falsifier for the old readout-dependent spectral cache bug.

    Reusing one CollocationPlan while sweeping the readout must match a
    freshly-built field at each column (the contract Stage 0 restored).
    """
    system = _periodic_heat()
    torch.manual_seed(0)
    field = SpectralVectorField(
        coordinate_spec=system.domain.coordinate_spec,
        components=system.component_spec(),
        K=2,
        L=1.0,
        time_hidden=4,
        time_depth=1,
        activation="tanh",
        weight_init_scale=0.5,
        dtype=DTYPE,
    )
    with torch.no_grad():
        field.W_t.normal_(0.0, 0.4)
        field.beta_t.normal_(0.0, 0.05)
        field.V.normal_(0.0, 0.05)
        field.b_t.zero_()
    freeze_features(field)
    _, _, n_unknowns = readout_size(field)
    plan = build_plan(field, system, SMALL)
    e_k = torch.zeros(n_unknowns, dtype=DTYPE)
    set_readout(field, e_k)
    r0 = eval_plan_rows(plan)

    for k in range(n_unknowns):
        e_k.zero_()
        e_k[k] = 1.0
        set_readout(field, e_k)
        reused = eval_plan_rows(plan) - r0

        torch.manual_seed(0)
        fresh = SpectralVectorField(
            coordinate_spec=system.domain.coordinate_spec,
            components=system.component_spec(),
            K=2,
            L=1.0,
            time_hidden=4,
            time_depth=1,
            activation="tanh",
            weight_init_scale=0.5,
            dtype=DTYPE,
        )
        with torch.no_grad():
            fresh.W_t.copy_(field.W_t)
            fresh.beta_t.copy_(field.beta_t)
            set_readout(fresh, torch.zeros(n_unknowns, dtype=DTYPE))
        plan0 = build_plan(fresh, system, SMALL)
        r0_fresh = eval_plan_rows(plan0)
        set_readout(fresh, e_k)
        plan_k = build_plan(fresh, system, SMALL)
        expected = eval_plan_rows(plan_k) - r0_fresh
        assert torch.allclose(reused, expected, atol=1e-12, rtol=0.0), k


def test_basis_spectral_end_to_end_torch() -> None:
    sol = pt.solve_least_squares(
        _periodic_heat(),
        basis="spectral",
        K=3,
        L=1.0,
        hidden=8,
        time_depth=1,
        seed=0,
        collocation=SMALL,
        ridge=1e-6,
    )
    assert sol.method == "least_squares"
    assert isinstance(sol.field, SpectralVectorField)
    assert math.isfinite(sol.residual_norm)
    # Exact mode-1 heat solution should be approached even with a tiny budget.
    grid = np.linspace(0.0, 1.0, 16, endpoint=False)
    ts = np.linspace(0.0, 0.15, 5)
    xx, tt = np.meshgrid(grid, ts, indexing="ij")
    pts = np.stack([tt.ravel(), xx.ravel()], axis=-1)
    u = sol.evaluate(pts, "u").detach().numpy()
    exact = np.exp(-0.1 * (2.0 * math.pi) ** 2 * pts[:, 0]) * np.sin(
        2.0 * math.pi * pts[:, 1]
    )
    rel = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    assert rel < 0.35, f"spectral heat relL2 too large: {rel}"


def test_basis_spectral_end_to_end_jax() -> None:
    sol = pj.solve_least_squares(
        _periodic_heat(),
        basis="spectral",
        K=3,
        L=1.0,
        hidden=8,
        time_depth=1,
        seed=0,
        collocation=SMALL,
        ridge=1e-6,
    )
    assert sol.method == "least_squares"
    assert type(sol.field).__name__ == "SpectralVectorField"
    assert math.isfinite(sol.residual_norm)


def test_cage_over_spectral_works_with_seam() -> None:
    system = _periodic_heat()
    field = pt.build_field(
        system,
        basis="spectral",
        K=2,
        L=1.0,
        hidden=6,
        seed=1,
        hard_conditions=pde.plan_hard_conditions(system, mode="auto"),
    )
    assert isinstance(field, ConstrainedExpressionField)
    assert isinstance(field.base, SpectralVectorField)
    freeze_features(field)
    n_out, n_feat, n = readout_size(field)
    assert n == n_out * n_feat + n_out
    theta = torch.zeros(n, dtype=DTYPE)
    theta[0] = 0.42
    set_readout(field, theta)
    assert torch.isclose(field.V.reshape(-1)[0], torch.tensor(0.42, dtype=DTYPE))
    sol = pt.solve_least_squares(
        system,
        basis="spectral",
        K=2,
        L=1.0,
        hidden=6,
        seed=1,
        collocation=SMALL,
        hard_conditions="auto",
    )
    assert isinstance(sol.field, ConstrainedExpressionField)
    assert math.isfinite(sol.residual_norm)


def test_spectral_refuses_steady_poisson() -> None:
    with pytest.raises(ValueError, match="time axis"):
        pt.build_field(_poisson(), basis="spectral", K=2, hidden=4)
    with pytest.raises(ValueError, match="time axis"):
        pj.build_field(_poisson(), basis="spectral", K=2, hidden=4)
    with pytest.raises(ValueError, match="time axis"):
        pt.solve_least_squares(_poisson(), basis="spectral", K=2, hidden=4)


def test_jet_field_is_refused_by_seam() -> None:
    torch.manual_seed(0)
    field = JetMLPVectorField(
        coordinate_spec=_poisson().domain.coordinate_spec,
        components=_poisson().component_spec(),
        hidden=4,
        depth=2,
        base="tanh",
        jet_order=1,
        dtype=DTYPE,
    )
    with pytest.raises(TypeError, match="no supported linear readout"):
        readout_size(field)


def test_dirichlet_conditions_on_spectral() -> None:
    """Dirichlet on a non-periodic heat system with a spectral basis."""
    conditions = (
        HardCondition("u", 1, dirichlet(0.0), 0.0),
        HardCondition("u", 1, dirichlet(1.0), 0.0),
    )
    system = _periodic_heat()
    # Use the spectral base without requiring the periodic plan.
    base = SpectralVectorField(
        coordinate_spec=system.domain.coordinate_spec,
        components=system.component_spec(),
        K=2,
        L=1.0,
        time_hidden=4,
        time_depth=1,
        activation="tanh",
        dtype=DTYPE,
    )
    cage = ConstrainedExpressionField(
        base=base,
        conditions=conditions,
        bounds=system.domain.bounds,
        certify=False,
    )
    freeze_features(cage)
    _, _, n = readout_size(cage)
    set_readout(cage, torch.randn(n, dtype=DTYPE) * 0.01)
    coords = torch.rand(4, 2, dtype=DTYPE)
    state = cage(coords)
    assert state.ops.value(state, "u").shape == (4,)

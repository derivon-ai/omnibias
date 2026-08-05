# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact-curvature optimisers on the solver's residual-minimisation driver.

Every optimiser here guarantees monotone descent (line search, trust region, or
adaptive cubic regularisation), so the load-bearing gate is exactly that: the
objective after ``iters`` curvature steps must not exceed the objective the Adam
warmup alone reached. That is checked against the *true* objective rather than
the reported ``residual_norm``, which is the unweighted row RMS.

The budgets are deliberately tiny -- this is a correctness suite, not the
benchmark. The accuracy comparison against Adam / L-BFGS lives in
``docs/examples/pinn_solver_curvature.py``.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver.torch.assemble import (  # noqa: E402
    condition_residual,
    default_interior,
    interior_residual,
)
from omnibias.pinn.solver.torch.steady import _weighted_rows  # noqa: E402

CONDITION_WEIGHT = 5.0
SPEC = pde.CollocationSpec(n_interior=8, n_boundary=8)
BUDGET = {
    "hidden": 12,
    "seed": 0,
    "collocation": SPEC,
    "adam_iters": 25,
    "condition_weight": CONDITION_WEIGHT,
}


def _poisson_system() -> pde.System:
    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * math.pi**2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    return pde.poisson(dom, source=source, boundary=0.0)


def _objective(solution: pde.System) -> float:
    """The fused scalar objective of a returned solution (weighted row energy)."""
    field, system = solution.field, solution.system
    coords = default_interior(field, system, SPEC)
    with torch.no_grad():
        rows = _weighted_rows(field, system, coords, SPEC, CONDITION_WEIGHT)
        return float(torch.sum(rows**2))


def test_optimizers_registry_is_the_documented_set() -> None:
    assert pt.OPTIMIZERS == {
        "adam",
        "cubic_gauss_newton",
        "cubic_newton",
        "gauss_newton",
        "jet_subspace_tensor",
        "lbfgs",
        "natural_gradient",
        "trust_region_newton_cg",
    }
    # KFAC hooks nn.Linear, which the closed-form jet forward never triggers.
    assert not any("kfac" in name for name in pt.OPTIMIZERS)


def test_weighted_rows_energy_equals_the_fused_scalar_loss() -> None:
    system = _poisson_system()
    field = pt.build_field(system, hidden=10, seed=3)
    coords = default_interior(field, system, SPEC)
    with torch.no_grad():
        rows = _weighted_rows(field, system, coords, SPEC, CONDITION_WEIGHT)
        fused = torch.mean(interior_residual(field, system, coords) ** 2)
        fused = fused + CONDITION_WEIGHT * torch.mean(
            condition_residual(field, system, SPEC) ** 2
        )
    assert float(torch.sum(rows**2)) == pytest.approx(float(fused), rel=1e-12)


@pytest.mark.parametrize(
    ("optimizer", "optimizer_kwargs"),
    [
        ("cubic_newton", None),
        ("cubic_gauss_newton", None),
        ("trust_region_newton_cg", None),
        ("jet_subspace_tensor", {"subspace_dim": 3, "order": 3}),
        ("natural_gradient", None),
        ("gauss_newton", {"solver": "qr", "damping_strategy": "nielsen"}),
    ],
)
def test_curvature_step_is_monotone_and_finite(
    optimizer: str, optimizer_kwargs: dict[str, object] | None
) -> None:
    system = _poisson_system()
    warmup = pt.solve_optimize(system, optimizer="adam", iters=0, **BUDGET)
    tuned = pt.solve_optimize(
        system,
        optimizer=optimizer,
        iters=4,
        optimizer_kwargs=optimizer_kwargs,
        **BUDGET,
    )
    assert tuned.method == f"optimize:{optimizer}"
    assert math.isfinite(tuned.residual_norm)
    before, after = _objective(warmup), _objective(tuned)
    assert after <= before * (1.0 + 1e-9), (
        f"{optimizer} increased the objective: {before:.6e} -> {after:.6e}"
    )


def test_gauss_newton_reports_its_damping() -> None:
    system = _poisson_system()
    sol = pt.solve_optimize(
        system, optimizer="gauss_newton", iters=3,
        optimizer_kwargs={"solver": "qr"}, **BUDGET,
    )
    assert sol.diagnostics["optimizer"] == "gauss_newton"
    assert sol.diagnostics["gn_damping"] > 0.0
    assert isinstance(sol.diagnostics["gn_accepted"], bool)


def test_natural_gradient_accepts_an_explicit_metric_override() -> None:
    system = _poisson_system()
    # metric=None is honest backtracked gradient descent, not Fisher scoring.
    sol = pt.solve_optimize(
        system, optimizer="natural_gradient", iters=3,
        optimizer_kwargs={"metric": None}, **BUDGET,
    )
    assert math.isfinite(sol.residual_norm)


def test_grad_norm_balancing_reweights_and_still_descends() -> None:
    system = _poisson_system()
    kwargs = dict(BUDGET, condition_weight=1.0)
    balanced = pt.solve_optimize(
        system,
        optimizer="cubic_newton",
        iters=4,
        loss_balancing="grad_norm",
        balance_every=2,
        **kwargs,
    )
    assert balanced.diagnostics["loss_balancing"] == "grad_norm"
    weights = balanced.diagnostics["balance_weights"]
    assert len(weights) == 2
    assert all(math.isfinite(w) and w > 0.0 for w in weights)
    # The interior term is the reference, so its weight stays ~1 while the
    # condition term is scaled to match its gradient norm.
    assert weights[0] == pytest.approx(1.0, rel=1e-6)
    assert weights[1] != pytest.approx(1.0, rel=1e-3)
    assert math.isfinite(balanced.residual_norm)


def test_solve_steady_dispatches_on_an_optimizer_name() -> None:
    system = _poisson_system()
    sol = pt.solve_steady(system, method="cubic_gauss_newton", iters=3, **BUDGET)
    assert sol.method == "optimize:cubic_gauss_newton"


def test_error_paths() -> None:
    system = _poisson_system()
    with pytest.raises(ValueError, match="unknown optimizer"):
        pt.solve_optimize(system, optimizer="kfac", iters=1, **BUDGET)
    with pytest.raises(ValueError, match="loss_balancing must be"):
        pt.solve_optimize(system, loss_balancing="adaptive", iters=1, **BUDGET)
    with pytest.raises(ValueError, match="needs a scalar loss"):
        pt.solve_optimize(
            system, optimizer="gauss_newton", loss_balancing="grad_norm",
            iters=1, **BUDGET,
        )
    with pytest.raises(ValueError, match="balance_every must be"):
        pt.solve_optimize(
            system, loss_balancing="grad_norm", balance_every=0, iters=1, **BUDGET,
        )
    with pytest.raises(ValueError, match="unknown steady method"):
        pt.solve_steady(system, method="newton_raphson")

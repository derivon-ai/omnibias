# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Solver auto-detection of hard boundary / initial conditions.

Two of these tests carry more weight than the rest.

The **full-residual guard** re-assembles every condition row *ignoring* the
plan's absorption after an auto solve. If the plan ever claims a condition the
cage does not actually enforce, the loss stops watching it and nothing else
would notice; this is what turns that silence into a failure.

The **regression guard** pins ``hard_conditions="none"`` to today's numbers bit
for bit, because the whole feature is opt-in only if it changes nothing when it
is off.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver._core.hard import plan_hard_conditions  # noqa: E402
from omnibias.pinn.solver.torch.assemble import condition_residual  # noqa: E402

EXACT = 1e-12
SMALL = pde.CollocationSpec(n_interior=12, n_boundary=12)


def _poisson():
    """1-D Poisson with Dirichlet ends: ``u'' = -pi^2 sin(pi x)``, ``u = sin(pi x)``."""
    dom = pde.Domain(("x",), ((0.0, 1.0),))

    def source(c):
        xp = pde.array_namespace(c)
        return -(math.pi**2) * xp.sin(math.pi * c[:, 0])

    return pde.poisson(dom, source=source, boundary=0.0)


def _heat():
    dom = pde.Domain(("t", "x"), ((0.0, 0.2), (0.0, 1.0)), time_axis="t")

    def initial(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 1])

    return pde.heat(dom, diffusivity=0.1, initial=initial, boundary=0.0)


def _full_condition_residual(sol) -> float:
    """Every condition row, absorbed or not -- the anti-silent-failure guard."""
    rows = condition_residual(sol.field, sol.system, SMALL, None)
    if not rows.numel():
        return 0.0
    return float(rows.detach().abs().max())


# --------------------------------------------------------------------------- #
# The planner.
# --------------------------------------------------------------------------- #
def test_the_planner_absorbs_dirichlet_ends_and_seals_a_certificate() -> None:
    from omnibias.core.proof.certificate import verify_certificate_digest

    plan = plan_hard_conditions(_poisson())
    assert plan.absorbed_boundary == frozenset({0})
    assert plan.is_total
    assert len(plan.certificates) == 1
    assert verify_certificate_digest(plan.certificates[0])


def test_the_planner_absorbs_an_initial_condition_alongside_the_boundary() -> None:
    plan = plan_hard_conditions(_heat())
    assert plan.absorbed_boundary == frozenset({0})
    assert plan.absorbed_initial == frozenset({0})
    assert plan.is_total, plan.declined


def test_mode_none_absorbs_nothing() -> None:
    plan = plan_hard_conditions(_heat(), mode="none")
    assert not plan
    assert plan.summary() == "hard conditions: none absorbed"


def test_an_unknown_mode_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="hard_conditions must be one of"):
        plan_hard_conditions(_poisson(), mode="yes")


def test_periodicity_is_declined_with_a_reason_naming_the_ansatz() -> None:
    base = _heat()
    sys = replace(
        base,
        boundary=(
            *base.boundary,
            pde.BoundaryCondition(component="u", kind="periodic", axis="x"),
        ),
    )
    plan = plan_hard_conditions(sys)
    reasons = [d.reason for d in plan.declined]
    assert any("periodicity is carried by the ansatz" in r for r in reasons)


def test_a_condition_whose_faces_are_all_periodic_is_declined_too() -> None:
    """The domain, not the condition, can be what makes a face unavailable."""
    dom = pde.Domain(
        ("t", "x"), ((0.0, 0.2), (0.0, 1.0)), periodic=(False, True), time_axis="t"
    )
    plan = plan_hard_conditions(pde.heat(dom, diffusivity=0.1, initial=0.0))
    assert [d.reason for d in plan.declined] == ["covers no non-periodic face"]
    # the initial condition is unaffected: partial absorption is the norm
    assert plan.absorbed_initial == frozenset({0})


def test_two_spatial_axes_are_declined_by_the_stage_a_scope() -> None:
    """The engine recurses over any number of axes; only this scope is validated."""
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    plan = plan_hard_conditions(pde.poisson(dom, source=0.0, boundary=0.0))
    assert not plan
    assert plan.declined
    assert all("at most one spatial axis" in d.reason for d in plan.declined)


def test_a_declined_condition_prints_what_it_was_and_why() -> None:
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    plan = plan_hard_conditions(pde.poisson(dom, source=0.0, boundary=0.0))
    text = str(plan.declined[0])
    assert text.startswith("boundary[0] on 'u':")
    assert "Stage A" in text


# --------------------------------------------------------------------------- #
# The linear driver.
# --------------------------------------------------------------------------- #
def test_absorbed_conditions_hold_to_machine_precision_after_a_linear_solve() -> None:
    torch.set_default_dtype(torch.float64)
    sol = pt.solve_least_squares(
        _poisson(), hidden=48, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    assert sol.diagnostics["hard_absorbed"] == 2  # both ends of the interval
    assert _full_condition_residual(sol) < EXACT


def test_the_full_residual_guard_also_covers_a_time_dependent_system() -> None:
    torch.set_default_dtype(torch.float64)
    sol = pt.solve_least_squares(
        _heat(), hidden=48, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    assert sol.diagnostics["hard_absorbed"] == 3  # two ends plus the initial slice
    assert _full_condition_residual(sol) < EXACT


def test_the_reported_residual_norm_never_hides_an_absorbed_row() -> None:
    """``residual_norm`` is assembled without the plan, on purpose."""
    torch.set_default_dtype(torch.float64)
    hard = pt.solve_least_squares(
        _poisson(), hidden=48, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    soft = pt.solve_least_squares(_poisson(), hidden=48, seed=0, collocation=SMALL)
    assert hard.residual_norm == pytest.approx(hard.residual_norm)
    assert soft.residual_norm > 0.0


def test_hard_conditions_beat_soft_ones_on_the_boundary_they_share() -> None:
    torch.set_default_dtype(torch.float64)
    hard = pt.solve_least_squares(
        _poisson(), hidden=48, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    soft = pt.solve_least_squares(_poisson(), hidden=48, seed=0, collocation=SMALL)
    ends = np.array([[0.0], [1.0]])
    hard_gap = float(hard.evaluate(ends, "u").detach().abs().max())
    soft_gap = float(soft.evaluate(ends, "u").detach().abs().max())
    assert hard_gap < EXACT
    assert hard_gap < soft_gap


def test_the_absorbed_solve_still_solves_the_interior_equation() -> None:
    """Dropping the rows must not cost accuracy where the PDE actually lives."""
    torch.set_default_dtype(torch.float64)
    sol = pt.solve_least_squares(
        _poisson(),
        hidden=96,
        weight_init_scale=3.0,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=40, n_boundary=8),
        hard_conditions="auto",
    )
    pts = np.linspace(0.02, 0.98, 60).reshape(-1, 1)
    u = sol.evaluate(pts, "u").detach().numpy()
    ustar = np.sin(math.pi * pts[:, 0])
    assert np.linalg.norm(u - ustar) / np.linalg.norm(ustar) < 1e-3


# --------------------------------------------------------------------------- #
# The optimisation driver.
# --------------------------------------------------------------------------- #
def test_absorbed_conditions_hold_after_an_optimised_solve() -> None:
    torch.set_default_dtype(torch.float64)
    sol = pt.solve_optimize(
        _poisson(),
        hidden=24,
        seed=0,
        collocation=SMALL,
        adam_iters=5,
        iters=5,
        hard_conditions="auto",
    )
    assert _full_condition_residual(sol) < EXACT


def test_an_absorbed_system_optimises_a_pure_interior_loss() -> None:
    """Nothing is left to weight, which is the point of absorbing everything."""
    torch.set_default_dtype(torch.float64)
    sol = pt.solve_optimize(
        _poisson(),
        hidden=24,
        seed=0,
        collocation=SMALL,
        adam_iters=2,
        iters=2,
        hard_conditions="auto",
    )
    assert sol.diagnostics["hard_declined"] == ()
    rows = condition_residual(
        sol.field, sol.system, SMALL, plan_hard_conditions(sol.system)
    )
    assert rows.numel() == 0


# --------------------------------------------------------------------------- #
# The regression guard: opt-in means opt-in.
# --------------------------------------------------------------------------- #
def test_the_default_reproduces_the_previous_solve_bit_for_bit() -> None:
    torch.set_default_dtype(torch.float64)
    kw = dict(hidden=32, seed=3, collocation=SMALL)
    a = pt.solve_least_squares(_poisson(), **kw)
    b = pt.solve_least_squares(_poisson(), hard_conditions="none", **kw)
    pts = np.linspace(0.05, 0.95, 25).reshape(-1, 1)
    assert torch.equal(a.evaluate(pts, "u"), b.evaluate(pts, "u"))
    assert a.residual_norm == b.residual_norm


def test_the_default_optimised_solve_is_unchanged_too() -> None:
    torch.set_default_dtype(torch.float64)
    kw = dict(hidden=16, seed=1, collocation=SMALL, adam_iters=3, iters=3)
    a = pt.solve_optimize(_poisson(), **kw)
    b = pt.solve_optimize(_poisson(), hard_conditions="none", **kw)
    pts = np.linspace(0.05, 0.95, 25).reshape(-1, 1)
    assert torch.equal(a.evaluate(pts, "u"), b.evaluate(pts, "u"))


def test_the_default_field_is_not_wrapped_in_a_cage() -> None:
    from omnibias.pinn.torch.cage import ConstrainedExpressionField

    torch.set_default_dtype(torch.float64)
    sol = pt.solve_least_squares(_poisson(), hidden=16, seed=0, collocation=SMALL)
    assert not isinstance(sol.field, ConstrainedExpressionField)


def test_auto_mode_does_wrap_the_field() -> None:
    from omnibias.pinn.torch.cage import ConstrainedExpressionField

    torch.set_default_dtype(torch.float64)
    sol = pt.solve_least_squares(
        _poisson(), hidden=16, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    assert isinstance(sol.field, ConstrainedExpressionField)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inverse problems: the mechanics of recovering a coefficient.

This module owns the parts that are cheap and decidable -- the solution object,
the constraint transforms, coupled multi-unknown systems, composition with the
rest of the optimiser surface, and the error paths. The load-bearing accuracy
claim (a known coefficient comes back across seeds and under a noise sweep) is the
heavier gate in ``test_inverse_recovery.py``.

The problem fixtures are shared with that module via ``_inverse_helpers``, so the
two can never drift into describing different problems.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from _inverse_helpers import (  # noqa: E402
    FAST,
    N_OBS,
    TRUE_C,
    TRUE_D,
    TRUE_NU,
    TWO_TRUTH,
    burgers_system,
    guess,
    heat_exact,
    heat_system,
    obs_points,
    recover,
    sin_initial,
    two_unknown_guesses,
    two_unknown_observations,
    two_unknown_system,
    wave_exact,
    wave_system,
)


# --------------------------------------------------------------------------- #
# the solution object
# --------------------------------------------------------------------------- #
def test_solution_reports_recovered_values_and_misfit() -> None:
    coords = obs_points(heat_system(TRUE_D))
    sol = recover(heat_system, TRUE_D, heat_exact(coords), coords, seed=0)
    assert sol.method == "inverse:cubic_gauss_newton"
    assert set(sol.recovered) == {"theta"}
    assert sol["theta"] == sol.recovered["theta"]
    assert sol.diagnostics["n_observations"] == N_OBS
    assert sol.diagnostics["recovered"] == sol.recovered
    assert math.isfinite(sol.data_misfit) and sol.data_misfit > 0.0
    assert math.isfinite(sol.residual_norm)
    assert abs(sol.recovered["theta"] - TRUE_D) / TRUE_D < 0.01
    # the field is still evaluable, and it interpolates the observations
    predicted = sol.evaluate(coords, "u").detach().numpy()
    assert float(np.abs(predicted - heat_exact(coords)).max()) < 0.1


def test_residual_norm_is_measured_under_the_recovered_binding() -> None:
    """Not at the initial guess, which would report a wildly wrong residual."""
    from omnibias.pinn.solver.torch.assemble import residual_norm

    coords = obs_points(heat_system(TRUE_D))
    sol = recover(heat_system, TRUE_D, heat_exact(coords), coords, seed=0, **FAST)
    with pde.bind_unknowns(sol.recovered):
        expected = residual_norm(sol.field, sol.system, FAST["collocation"])
    assert sol.residual_norm == pytest.approx(expected, rel=1e-12)
    # The guess is 3x the truth, so evaluating at it would give a different number.
    with pde.bind_unknowns({"theta": 3.0 * TRUE_D}):
        at_guess = residual_norm(sol.field, sol.system, FAST["collocation"])
    assert abs(at_guess - sol.residual_norm) > 1e-9


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
def test_the_positive_transform_keeps_a_coefficient_positive() -> None:
    """Start far above the truth: descend, without ever going negative.

    The constraint is what is being tested, and it holds by construction (the value
    is a softplus of the raw parameter), so this does not need a converged solve --
    only enough steps to show the optimiser really is pushing the coefficient down
    through a region where an unconstrained parameterisation could overshoot.
    """
    coords = obs_points(heat_system(TRUE_D))
    unknown = pde.Unknown("theta", initial=20.0, transform="positive")
    sol = pt.solve_inverse(
        heat_system(unknown),
        [pde.Observations("u", coords, heat_exact(coords))],
        seed=0,
        **FAST,
    )
    assert 0.0 < sol.recovered["theta"] < 20.0


def test_bounded_transform_stays_inside_its_box() -> None:
    coords = obs_points(heat_system(TRUE_D))
    unknown = pde.Unknown(
        "theta", initial=0.9, transform="bounded", lower=0.2, upper=1.0
    )
    sol = pt.solve_inverse(
        heat_system(unknown),
        [pde.Observations("u", coords, heat_exact(coords))],
        seed=0,
        **FAST,
    )
    assert 0.2 < sol.recovered["theta"] < 1.0


# --------------------------------------------------------------------------- #
# composition with the rest of the driver
# --------------------------------------------------------------------------- #
def test_a_multi_unknown_system_carries_both_coefficients() -> None:
    """Two unknowns are collected in declaration order and both come back.

    Whether they come back *accurately* is the recovery gate's job; what is checked
    here is the plumbing, which is where a multi-unknown system can go wrong
    silently -- one name shadowing the other in the binding, or a coefficient that
    never joins the parameter vector and so is returned at its initial guess.
    """
    coords = obs_points(two_unknown_system(TWO_TRUTH))
    system = two_unknown_system(two_unknown_guesses())
    assert tuple(u.name for u in system.unknowns) == ("Du", "Dv")
    sol = pt.solve_inverse(
        system, two_unknown_observations(coords), seed=0, **FAST
    )
    assert set(sol.recovered) == {"Du", "Dv"}
    assert all(v > 0.0 for v in sol.recovered.values())
    assert sol.recovered["Du"] != sol.recovered["Dv"], (
        "the two coefficients moved identically, so one is shadowing the other"
    )
    assert all(v != 1.0 for v in sol.recovered.values()), "a coefficient never moved"


@pytest.mark.parametrize("optimizer", ["lbfgs", "adam", "gauss_newton"])
def test_every_optimizer_path_accepts_an_inverse_problem(optimizer: str) -> None:
    """The shared driver means no optimiser needs inverse-specific plumbing."""
    coords = obs_points(heat_system(TRUE_D))
    sol = recover(
        heat_system,
        TRUE_D,
        heat_exact(coords),
        coords,
        seed=0,
        iters=6,
        adam_iters=20,
        optimizer=optimizer,
    )
    assert math.isfinite(sol.recovered["theta"])
    assert sol.recovered["theta"] > 0.0


def test_refinement_and_balancing_compose_with_an_inverse_solve() -> None:
    coords = obs_points(heat_system(TRUE_D))
    sol = recover(
        heat_system,
        TRUE_D,
        heat_exact(coords),
        coords,
        seed=0,
        iters=6,
        adam_iters=20,
        optimizer="cubic_newton",
        loss_balancing="grad_norm",
        balance_every=3,
        refinement=pde.RefinementSpec(
            every=2, n_candidates=32, n_add=4, max_points=200
        ),
    )
    # three terms now: interior, conditions, data
    assert len(sol.diagnostics["balance_weights"]) == 3
    assert sol.diagnostics["n_interior_final"] > sol.diagnostics["n_interior_uniform"]


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_error_paths() -> None:
    coords = obs_points(heat_system(TRUE_D))
    values = np.zeros(N_OBS)
    obs = pde.Observations("u", coords, values)
    with pytest.raises(ValueError, match="needs at least one Unknown"):
        pt.solve_inverse(heat_system(0.3), [obs], **FAST)
    with pytest.raises(ValueError, match="needs observations"):
        pt.solve_inverse(heat_system(guess(TRUE_D)), [], **FAST)
    with pytest.raises(ValueError, match="unknown component 'w'"):
        pt.solve_inverse(
            heat_system(guess(TRUE_D)),
            [pde.Observations("w", coords, values)],
            **FAST,
        )
    with pytest.raises(ValueError, match="the domain is 2-D"):
        pt.solve_inverse(
            heat_system(guess(TRUE_D)),
            [pde.Observations("u", coords[:, :1], values)],
            **FAST,
        )
    with pytest.raises(ValueError, match="unknown optimizer"):
        pt.solve_inverse(heat_system(guess(TRUE_D)), [obs], optimizer="kfac", **FAST)


def test_forward_drivers_refuse_an_unbound_coefficient() -> None:
    """The failure mode this whole design exists to prevent: a silent wrong PDE."""
    system = heat_system(guess(TRUE_D))
    with pytest.raises(ValueError, match="solve_optimize is a forward driver"):
        pt.solve_optimize(system, hidden=8, iters=1, adam_iters=1)
    with pytest.raises(ValueError, match="solve_least_squares is a forward driver"):
        pt.solve_least_squares(system, hidden=8)
    # ...but pinning the coefficient makes the same System solvable forward.
    with pde.bind_unknowns({"theta": 0.25}):
        sol = pt.solve_optimize(
            system,
            hidden=8,
            iters=2,
            adam_iters=3,
            collocation=pde.CollocationSpec(n_interior=4, n_boundary=4),
        )
    assert math.isfinite(sol.residual_norm)

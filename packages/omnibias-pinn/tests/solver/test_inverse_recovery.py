# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Does a known coefficient actually come back? Multi-seed recovery gate.

Three recoveries covering the structural cases rather than the flattering ones:

* ``heat`` / ``diffusivity`` -- a linear coefficient, checked against the analytic
  solution ``exp(-D pi^2 t) sin(pi x)``.
* ``wave`` / ``speed`` -- enters as ``speed ** 2``, so only the magnitude is
  identifiable; checked against ``sin(pi x) cos(pi c t)``.
* ``burgers`` / ``viscosity`` -- a nonlinear problem with no closed form, so the
  observations come from a forward solve pinned at the true value with
  ``bind_unknowns`` (the standard synthetic-inverse protocol).

Every recovery starts from a deliberately wrong initial guess -- 3x the truth --
so passing cannot be an artefact of initialising near the answer. Every tolerance
is read off a measured distribution (6 seeds x a four-level noise sweep on a
GPU-cluster node, tabulated in ``docs/benchmarks.md``) with roughly a 2x safety
margin, not guessed.

Marked ``slow``: this is dozens of solves. The cheap mechanical guarantees are the
fast suite in ``test_inverse.py``, whose problem fixtures this module reuses so the
two can never drift into describing different problems.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from _inverse_helpers import (  # noqa: E402
    BUDGET,
    TRUE_C,
    TRUE_D,
    TRUE_NU,
    TWO_TRUTH,
    burgers_system,
    heat_exact,
    heat_system,
    obs_points,
    recover,
    two_unknown_guesses,
    two_unknown_observations,
    two_unknown_system,
    wave_exact,
    wave_system,
)

pytestmark = pytest.mark.slow

SEEDS = range(5)


def _noisy(values: np.ndarray, noise: float, seed: int) -> np.ndarray:
    """Gaussian noise at ``noise`` times the solution amplitude."""
    if noise <= 0.0:
        return values
    rng = np.random.default_rng(9000 + seed)
    return values + rng.normal(0.0, noise * float(np.abs(values).max()), values.shape)


def _rel_errors(build, truth, values, coords, **kw) -> list[float]:
    return [
        abs(recover(build, truth, values, coords, seed=seed, **kw).recovered["theta"]
            - truth)
        / truth
        for seed in SEEDS
    ]


def test_recovers_heat_diffusivity() -> None:
    coords = obs_points(heat_system(TRUE_D))
    errors = _rel_errors(heat_system, TRUE_D, heat_exact(coords), coords)
    # measured max over 6 seeds: 0.03%
    assert max(errors) < 0.01, f"diffusivity not recovered: {errors}"


def test_recovers_wave_speed_magnitude() -> None:
    coords = obs_points(wave_system(TRUE_C))
    errors = _rel_errors(wave_system, TRUE_C, wave_exact(coords), coords)
    # measured max over 6 seeds: 0.09%
    assert max(errors) < 0.01, f"wave speed not recovered: {errors}"


def test_recovers_burgers_viscosity_from_a_pinned_forward_solve() -> None:
    system = burgers_system(
        pde.Unknown("theta", initial=TRUE_NU, transform="positive")
    )
    coords = obs_points(system)
    # Pin the unknown at the truth to generate the data: this is the *same* System
    # object the inverse solve will use, which is the point of the binding.
    with pde.bind_unknowns({"theta": TRUE_NU}):
        forward = pt.solve_optimize(
            system,
            hidden=48,
            seed=12345,
            collocation=pde.CollocationSpec(n_interior=24, n_boundary=16),
            iters=300,
            adam_iters=2000,
            condition_weight=50.0,
        )
        values = pde.sample_observations(forward, "u", coords).values
    errors = _rel_errors(burgers_system, TRUE_NU, values, coords)
    # measured max over 6 seeds: 5.5%; the nonlinear problem is the hard case.
    assert max(errors) < 0.12, f"viscosity not recovered: {errors}"


def test_recovers_two_coefficients_at_once() -> None:
    """Two coefficients, each visible in its own observed component.

    Harder than it looks: the optimiser must separate curvature across two scalars
    that differ by ~3x, on top of the network weights. Measured worst-of-the-two
    error over 6 seeds: median 0.16%, max 0.28% (L-BFGS at the same budget: median
    19%, max 70%).
    """
    coords = obs_points(two_unknown_system(TWO_TRUTH))
    sol = pt.solve_inverse(
        two_unknown_system(two_unknown_guesses()),
        two_unknown_observations(coords),
        seed=0,
        **{**BUDGET, "iters": 60},
    )
    for name, want in zip(("Du", "Dv"), TWO_TRUTH, strict=True):
        assert abs(sol.recovered[name] - want) / want < 0.02, sol.recovered


def test_noise_degrades_recovery_gracefully() -> None:
    """A noise sweep, and the honest shape of the degradation.

    Measured medians on ``heat``: 0.02% / 0.75% / 3.8% / 11.5% at noise levels
    0 / 1% / 5% / 15% of the solution amplitude. The gate is that the recovery
    stays usable well past the noise level *and* that more noise really does cost
    accuracy -- a recovery that ignored the data would be flat instead.
    """
    coords = obs_points(heat_system(TRUE_D))
    exact = heat_exact(coords)
    medians: dict[float, float] = {}
    for noise, ceiling in ((0.0, 0.01), (0.01, 0.03), (0.05, 0.10), (0.15, 0.25)):
        errors = [
            abs(
                recover(
                    heat_system, TRUE_D, _noisy(exact, noise, seed), coords, seed=seed
                ).recovered["theta"]
                - TRUE_D
            )
            / TRUE_D
            for seed in SEEDS
        ]
        medians[noise] = float(np.median(errors))
        assert medians[noise] < ceiling, (
            f"noise={noise}: median error {medians[noise]:.3%} exceeds {ceiling:.0%}"
        )
    assert medians[0.15] > medians[0.0], (
        f"error did not grow with noise, so the fit ignores the data: {medians}"
    )


def test_gauss_newton_beats_lbfgs_at_recovering_the_coefficient() -> None:
    """Why ``cubic_gauss_newton`` is the default here but not in solve_optimize.

    A lone scalar coefficient and a few hundred network weights have curvature on
    completely different scales, so one shared step size cannot serve both. At this
    budget the measured medians are 0.02% (Gauss-Newton) against 17% (L-BFGS).
    """
    coords = obs_points(heat_system(TRUE_D))
    exact = heat_exact(coords)
    curvature = _rel_errors(heat_system, TRUE_D, exact, coords)
    first_order = _rel_errors(heat_system, TRUE_D, exact, coords, optimizer="lbfgs")
    assert float(np.median(curvature)) < 0.1 * float(np.median(first_order))

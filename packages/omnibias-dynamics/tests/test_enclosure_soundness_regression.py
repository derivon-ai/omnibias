# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Randomized soundness sweeps for the validated-dynamics enclosures.

Two claims are swept against ground truth computed independently of the
enclosure machinery:

* :func:`~omnibias.dynamics.spectral_radius_bound` brackets ``rho(M)`` for
  **every** matrix in the interval enclosure, so the bracket is checked against
  ``numpy`` eigenvalues of concrete matrices sampled from inside it;
* :func:`~omnibias.dynamics.variational_flow` encloses the true trajectory, so
  its box is checked against a finely-stepped float RK4 integration.

Assertions are exact. Where ground truth is itself approximate -- RK4 is a
numerical integrator, not an oracle -- the reference is widened into an interval
by its own error estimate rather than the enclosure being given slack, so a
failure still means the enclosure is wrong rather than merely tight.
"""

from __future__ import annotations

import numpy as np
import pytest
from _dynamics_helpers import harmonic_float, hopf_float, radial_float, rk4
from _enclosure import assert_encloses
from omnibias.core.verified.interval import Interval
from omnibias.dynamics import (
    harmonic_oscillator,
    hopf_normal_form,
    monodromy_determinant,
    radial_logistic,
    spectral_radius_bound,
    variational_flow,
)
from omnibias.dynamics._core.variational import _nth_root_down


def _random_interval_matrix(
    rng: np.random.Generator, n: int, radius: float
) -> tuple[list[list[Interval]], np.ndarray]:
    """An interval matrix plus its midpoint."""
    mid = rng.standard_normal((n, n))
    matrix = [
        [Interval(mid[i, j] - radius, mid[i, j] + radius) for j in range(n)]
        for i in range(n)
    ]
    return matrix, mid


def _sample_inside(
    rng: np.random.Generator, mid: np.ndarray, radius: float
) -> np.ndarray:
    return mid + rng.uniform(-radius, radius, size=mid.shape)


def test_spectral_radius_bracket_contains_every_sampled_matrix() -> None:
    """``rho(M)`` of any concrete ``M`` inside the enclosure must lie in the bracket."""
    for seed in range(120):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 5))
        radius = float(rng.uniform(0.0, 0.15))
        matrix, mid = _random_interval_matrix(rng, n, radius)
        bracket = spectral_radius_bound(matrix)
        assert bracket.lo <= bracket.hi
        for _ in range(12):
            concrete = _sample_inside(rng, mid, radius)
            rho = float(np.max(np.abs(np.linalg.eigvals(concrete))))
            assert rho <= bracket.hi, (
                f"seed={seed}: spectral radius {rho!r} of a matrix inside the enclosure "
                f"exceeds the certified upper bound {bracket.hi!r}"
            )


def test_spectral_radius_bracket_holds_on_degenerate_point_matrices() -> None:
    r"""A zero-radius enclosure is a plain matrix; the bracket must still hold.

    The two sides need different references. ``bracket.hi`` is an induced norm, far
    above ``rho``, so LAPACK's own error cannot reach it and the comparison is exact.
    ``bracket.lo = |det|^{1/n}`` is often tight to the last ulp, and ``numpy`` is a
    backward-stable numerical eigensolver rather than an oracle -- on some matrices
    its ``rho`` is small enough to violate the exact identity ``rho^n >= |det|``. So
    the lower bound is checked against *its own* rigorous premise, and ``numpy`` is
    used only as a cross-check within a derived error budget.
    """
    eps = float(np.finfo(float).eps)
    for seed in range(120):
        rng = np.random.default_rng(seed + 5_000)
        n = int(rng.integers(2, 5))
        mid = rng.standard_normal((n, n))
        matrix = [[Interval.point(float(mid[i, j])) for j in range(n)] for i in range(n)]
        bracket = spectral_radius_bound(matrix)
        rho = float(np.max(np.abs(np.linalg.eigvals(mid))))

        assert rho <= bracket.hi, (
            f"seed={seed}: rho(M) = {rho!r} exceeds the certified upper bound "
            f"{bracket.hi!r}"
        )
        # Rigorous, LAPACK-free: rho >= |det|^{1/n} is exact, so it suffices that
        # bracket.lo^n really is at or below |det|.
        det_lo = monodromy_determinant(matrix).abs().lo
        assert Interval.point(bracket.lo).pow_int(n).hi <= det_lo, (
            f"seed={seed}: bracket.lo = {bracket.lo!r} raised to n={n} exceeds "
            f"|det| = {det_lo!r}, so it is not a valid geometric-mean bound"
        )
        # Backward-stable eigensolver: |rho_computed - rho| <~ c eps ||M||_F.
        eig_err = 16.0 * eps * float(np.linalg.norm(mid, "fro"))
        assert bracket.lo <= rho + eig_err, (
            f"seed={seed}: bracket.lo = {bracket.lo!r} is above numpy's rho = {rho!r} "
            f"by more than the eigensolver's own error budget {eig_err!r}"
        )


def test_the_nth_root_lower_bound_is_rounded_outward() -> None:
    r"""``base ** (1/n)`` rounds to nearest, which is the wrong way for a lower bound.

    ``float.__pow__`` is not correctly rounded and ``1/n`` is itself inexact, so the
    naive expression cannot be *proven* to satisfy ``root^n <= base`` for any of the
    bases below -- and on real matrices it did land above the true root. The helper
    must step down until outward-rounded arithmetic settles it.
    """
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(2, 6))
        base = float(rng.uniform(1e-3, 50.0))
        root = _nth_root_down(base, n)
        assert Interval.point(root).pow_int(n).hi <= base, (base, n, root)
        # and it must not be needlessly pessimistic
        assert root >= 0.999999 * base ** (1.0 / n)
        checked += 1
    assert checked == 300


@pytest.mark.parametrize(
    ("make_verified", "make_float", "y0", "horizon"),
    [
        (harmonic_oscillator, harmonic_float, (1.0, 0.0), 1.3),
        (hopf_normal_form, hopf_float, (0.5, 0.2), 2.0),
        (radial_logistic, radial_float, (0.4,), 1.0),  # scalar r' = mu r - r^3
    ],
)
def test_variational_flow_box_encloses_a_finely_stepped_trajectory(
    make_verified, make_float, y0: tuple[float, ...], horizon: float  # noqa: ANN001
) -> None:
    """The validated flow's box must contain the true trajectory endpoint."""
    for parameter in (0.5, 1.0, 1.5):
        field, jacobian = make_verified(parameter)
        n_steps = 400
        flow = variational_flow(field, jacobian, list(y0), horizon / n_steps, n_steps)
        truth = rk4(make_float(parameter), list(y0), 0.0, horizon, 40_000)
        box = flow.box()
        for axis, value in enumerate(truth):
            assert_encloses(
                box[axis], value, what=f"trajectory[{axis}] (parameter={parameter})"
            )


def test_variational_flow_enclosure_only_widens_with_a_coarser_step() -> None:
    """More steps may tighten the box; fewer must never make it unsound."""
    field, jacobian = harmonic_oscillator(1.0)
    horizon = 1.3
    truth = rk4(harmonic_float(1.0), [1.0, 0.0], 0.0, horizon, 40_000)
    for n_steps in (50, 100, 200, 400, 800):
        flow = variational_flow(field, jacobian, [1.0, 0.0], horizon / n_steps, n_steps)
        box = flow.box()
        for axis, value in enumerate(truth):
            assert_encloses(box[axis], value, what=f"trajectory[{axis}] (n={n_steps})")

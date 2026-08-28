# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the one-sided radii-polynomial existence closure.

Mirrors ``tests/test_verified_radii_spectral.py``'s manufactured-solution
strategy (pick a finite ``a*``, set the forcing ``f = ell a* + a* * a*`` so that
``F(a*) = 0`` exactly, then check :func:`series_radii_certificate` both proves
existence and certifies a ball that actually contains the known true zero) for
the one-sided :mod:`omnibias.core.verified.radii_series` pipeline.
"""

from __future__ import annotations

import random

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.banded import banded_tail_inverse_bound
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.radii_series import (
    SeriesProblem,
    SeriesRadiiResult,
    Symbol,
    _apply_diagonal_symbol,
    _assemble,
    _embed,
    _invert_real,
    constant_symbol,
    constant_tail_inverse_bound,
    evaluate_residual,
    series_radii_certificate,
)


# --------------------------------------------------------------------------- #
# Helpers: build a manufactured-solution problem.
# --------------------------------------------------------------------------- #
def _forcing_for(a_star: list[float], trunc: int, nu: float, symbol: Symbol) -> list[Interval]:
    """``f = ell*a* + a* * a*`` so that ``F(a*) = 0`` exactly."""
    ab = _embed(a_star, 2 * trunc, nu)
    f_series = _apply_diagonal_symbol(ab, symbol) + (ab * ab)
    return list(f_series.coeffs)


def _scalar_problem(
    lam: float = 4.0,
    nu: float = 1.05,
    trunc: int = 4,
    a_star: list[float] | None = None,
) -> tuple[SeriesProblem, list[float]]:
    if a_star is None:
        a_star = [0.1, 0.05, 0.02, 0.01, 0.005]
    symbol = constant_symbol(lam)
    mu = constant_tail_inverse_bound(lam)
    forcing = _forcing_for(a_star, trunc, nu, symbol)
    problem = SeriesProblem(
        trunc=trunc,
        nu=nu,
        linear_symbol=symbol,
        tail_inverse_bound=mu,
        forcing=forcing,
    )
    return problem, a_star


def _distance(a: list[float], b: list[float], trunc: int, nu: float) -> float:
    n = max(len(a), len(b))
    diff = [(a[k] if k < len(a) else 0.0) - (b[k] if k < len(b) else 0.0) for k in range(n)]
    return _embed(diff, 2 * trunc, nu).norm().hi


# --------------------------------------------------------------------------- #
# Core: manufactured-solution existence proof.
# --------------------------------------------------------------------------- #
def test_manufactured_solution_is_an_exact_zero() -> None:
    problem, a_star = _scalar_problem()
    residual = evaluate_residual(problem, a_star)
    assert residual.norm().hi < 1e-12


def test_exact_solution_proved() -> None:
    problem, a_star = _scalar_problem()
    result = series_radii_certificate(problem, a_star)
    assert isinstance(result, SeriesRadiiResult)
    assert result.proved
    assert result.certificate is not None
    assert result.y0 < 1e-12  # exact defect
    assert result.z0 < 1.0  # finite-block inverse is good
    assert result.radius is not None and result.radius > 0.0


def test_all_bounds_nonnegative() -> None:
    problem, a_star = _scalar_problem()
    r = series_radii_certificate(problem, a_star)
    assert min(r.y0, r.z0, r.z1, r.z2, r.a_op_norm, r.residual_norm) >= 0.0


def test_true_solution_lies_in_certified_ball() -> None:
    # Certify at a *perturbed* approximation; the genuine zero a* must be captured.
    problem, a_star = _scalar_problem()
    a_bar = list(a_star)
    a_bar[-1] += 1e-3

    result = series_radii_certificate(problem, a_bar)
    assert result.proved
    assert result.radius is not None
    dist = _distance(a_star, a_bar, problem.trunc, problem.nu)
    assert dist > 0.0  # genuinely perturbed
    assert dist <= result.radius  # the true zero is inside the unique-solution ball


def test_certificate_is_sealed_and_verifiable() -> None:
    problem, a_star = _scalar_problem()
    result = series_radii_certificate(problem, a_star)
    assert result.certificate is not None
    cert = result.certificate.certificate
    assert cert["payload"]["type"] == "radii_polynomial"
    assert verify_certificate_digest(cert)
    assert result.certificate.kappa < 1.0


def test_non_constant_diagonal_symbol_manufactured_proof() -> None:
    """A genuinely non-constant diagonal symbol ell(n) = 4 + n (Euler-operator style)."""
    trunc, nu = 3, 1.05

    def symbol(n: int) -> float:
        return 4.0 + n

    # sup_{n>N} 1/(4+n) is attained at n=N+1 (ell increasing); the same banded
    # bound (s=0) used by constant_tail_inverse_bound applies here too.
    mu = banded_tail_inverse_bound(4.0 + trunc + 1, 0.0).hi
    a_star = [0.08, 0.03, 0.02, 0.01]
    forcing = _forcing_for(a_star, trunc, nu, symbol)
    problem = SeriesProblem(
        trunc=trunc, nu=nu, linear_symbol=symbol, tail_inverse_bound=mu, forcing=forcing
    )
    assert evaluate_residual(problem, a_star).norm().hi < 1e-12
    result = series_radii_certificate(problem, a_star)
    assert result.proved
    assert result.radius is not None and result.radius > 0.0


# --------------------------------------------------------------------------- #
# Soundness cross-checks of the operator-norm bounds.
# --------------------------------------------------------------------------- #
def test_z0_upper_bounds_finite_direction_defect() -> None:
    # For finite h, ||(I - A DF(a))h|| / ||h|| <= Z0 (the finite-input op norm).
    problem, a_star = _scalar_problem()
    result = series_radii_certificate(problem, a_star)
    system = _assemble(problem, a_star)
    rng = random.Random(20260828)
    n = problem.trunc + 1
    worst = 0.0
    for _ in range(300):
        coeffs = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        h = _embed(coeffs, problem.work_trunc, problem.nu)
        ratio = (system.defect_apply(h).norm() / h.norm()).hi
        worst = max(worst, ratio)
    assert worst <= result.z0 * (1.0 + 1e-9)


def test_z1_z2_match_their_closed_form() -> None:
    problem, a_star = _scalar_problem()
    result = series_radii_certificate(problem, a_star)
    ab_norm = _embed(a_star, problem.work_trunc, problem.nu).norm().hi
    # Z1 = 2 ||A|| C_Q ||a||,  Z2 = ||A|| C_Q   (C_Q = 1 here).
    assert result.z1 == pytest.approx(2.0 * result.a_op_norm * ab_norm, rel=1e-9)
    assert result.z2 == pytest.approx(result.a_op_norm, rel=1e-9)


def test_y0_bounded_by_operator_norm_times_residual() -> None:
    # ||A F(a)|| <= ||A|| ||F(a)||.
    problem, a_star = _scalar_problem()
    a_bar = list(a_star)
    a_bar[-1] += 2e-3
    result = series_radii_certificate(problem, a_bar)
    assert result.y0 <= result.a_op_norm * result.residual_norm * (1.0 + 1e-9) + 1e-300


# --------------------------------------------------------------------------- #
# Negative space: when the closure must NOT prove.
# --------------------------------------------------------------------------- #
def test_non_coercive_symbol_is_not_proved() -> None:
    # Weak linear part -> large mu / Z1 > 1 -> no contraction.
    problem, a_star = _scalar_problem(lam=0.05)
    result = series_radii_certificate(problem, a_star)
    assert not result.proved
    assert result.certificate is None


def test_large_amplitude_breaks_contraction() -> None:
    big = [5.0, 3.0, 3.0, 0.0, 0.0]
    problem, a_star = _scalar_problem(a_star=big)
    result = series_radii_certificate(problem, a_star)
    # Defect is still ~0 (manufactured), but Z1 = 2||A|| ||a|| is far above 1.
    assert result.z1 > 1.0
    assert not result.proved


def test_unproved_result_still_reports_bounds() -> None:
    problem, a_star = _scalar_problem(lam=0.05)
    result = series_radii_certificate(problem, a_star)
    assert result.certificate is None
    assert result.radius is None
    assert result.z1 > 0.0
    assert result.a_op_norm > 0.0


# --------------------------------------------------------------------------- #
# Validation / error paths.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"trunc": 0}, "trunc"),
        ({"nu": 0.0}, "nu"),
        ({"nu": -1.0}, "nu"),
        ({"tail_inverse_bound": -1.0}, "tail_inverse_bound"),
        ({"forcing": [0.0] * 100}, "forcing"),
    ],
)
def test_seriesproblem_validation(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = dict(
        trunc=2,
        nu=1.05,
        linear_symbol=constant_symbol(4.0),
        tail_inverse_bound=0.25,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SeriesProblem(**base)  # type: ignore[arg-type]


def test_a_bar_longer_than_trunc_plus_one_raises() -> None:
    problem, _ = _scalar_problem(trunc=2, a_star=[0.1, 0.05, 0.02])
    too_long = [0.1, 0.05, 0.02, 0.01]  # length 4 > trunc+1 = 3
    with pytest.raises(ValueError, match="trunc"):
        evaluate_residual(problem, too_long)
    with pytest.raises(ValueError, match="trunc"):
        series_radii_certificate(problem, too_long)


def test_constant_tail_inverse_bound_value() -> None:
    # mu = 1/|value|, value=4 -> 0.25, matching banded_tail_inverse_bound(4, 0).
    mu = constant_tail_inverse_bound(4.0)
    assert mu == pytest.approx(0.25, rel=1e-12)
    assert mu == banded_tail_inverse_bound(4.0, 0.0).hi


def test_constant_tail_inverse_bound_requires_nonzero() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        constant_tail_inverse_bound(0.0)


def test_singular_finite_block_raises() -> None:
    # Zero symbol and zero a_bar -> B_N = 0 -> non-invertible.
    trunc, nu = 2, 1.05
    problem = SeriesProblem(
        trunc=trunc,
        nu=nu,
        linear_symbol=constant_symbol(0.0),
        tail_inverse_bound=1.0,
        forcing=None,
    )
    with pytest.raises(ValueError, match="singular"):
        series_radii_certificate(problem, [0.0] * (trunc + 1))


def test_invert_real_round_trip() -> None:
    mat = [[2.0, 1.0], [0.0, 3.0]]
    inv = _invert_real(mat)
    prod = [[sum(mat[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert abs(prod[0][0] - 1.0) < 1e-12
    assert abs(prod[1][1] - 1.0) < 1e-12
    assert abs(prod[0][1]) < 1e-12
    assert abs(prod[1][0]) < 1e-12


def test_invert_real_singular_raises() -> None:
    with pytest.raises(ValueError, match="singular"):
        _invert_real([[1.0, 2.0], [2.0, 4.0]])

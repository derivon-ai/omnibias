# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the spectral radii-polynomial existence closure.

The workhorse strategy is the **manufactured solution**: pick a finite
trigonometric polynomial ``a*``, set the forcing ``f = ell a* + Q(a*, a*)`` so that
``F(a*) = 0`` *exactly*, and then check that
:func:`quadratic_radii_certificate` (a) proves existence and (b) certifies a ball
that actually contains the known true zero ``a*``.  This validates the full
``Y0/Z0/Z1/Z2`` pipeline end to end.
"""

from __future__ import annotations

import random

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.fourier import ValidatedFourierSeries, Wavevector
from omnibias.core.verified.radii_spectral import (
    SpectralProblem,
    SpectralRadiiResult,
    _apply_finite_symbol,
    _assemble,
    _embed,
    _invert_complex,
    evaluate_residual,
    laplacian_symbol,
    laplacian_tail_inverse_bound,
    quadratic_radii_certificate,
)


# --------------------------------------------------------------------------- #
# Helpers: build a manufactured-solution problem with a convolution nonlinearity.
# --------------------------------------------------------------------------- #
def _convolution() -> object:
    def q(u: ValidatedFourierSeries, v: ValidatedFourierSeries) -> ValidatedFourierSeries:
        return u * v

    return q


def _forcing_for(
    a_star: dict[Wavevector, float],
    dim: int,
    trunc: int,
    nu: float,
    symbol: object,
    quad: object,
) -> dict[Wavevector, ComplexInterval]:
    """``f = ell a* + Q(a*, a*)`` so that ``F(a*) = 0`` exactly."""
    ab = _embed(a_star, dim, 2 * trunc, nu)
    f_series = _apply_finite_symbol(ab, symbol) + quad(ab, ab)  # type: ignore[operator]
    return {k: v for k, v in f_series.coeffs.items()}


def _scalar_problem(
    c0: float = 4.0,
    c2: float = 1.0,
    nu: float = 1.05,
    trunc: int = 4,
    a_star: dict[Wavevector, float] | None = None,
) -> tuple[SpectralProblem, dict[Wavevector, float]]:
    dim = 1
    if a_star is None:
        a_star = {(0,): 0.1, (1,): 0.05, (-1,): 0.05, (2,): 0.02, (-2,): 0.02}
    symbol = laplacian_symbol(c0, c2)
    mu = laplacian_tail_inverse_bound(dim, trunc, c0, c2)
    quad = _convolution()
    forcing = _forcing_for(a_star, dim, trunc, nu, symbol, quad)
    problem = SpectralProblem(
        dim=dim,
        trunc=trunc,
        nu=nu,
        linear_symbol=symbol,
        tail_inverse_bound=mu,
        quadratic=quad,  # type: ignore[arg-type]
        quadratic_norm=1.0,
        forcing=forcing,
    )
    return problem, a_star


def _distance(
    a: dict[Wavevector, float], b: dict[Wavevector, float], dim: int, trunc: int, nu: float
) -> float:
    diff = {k: a.get(k, 0.0) - b.get(k, 0.0) for k in set(a) | set(b)}
    return _embed(diff, dim, 2 * trunc, nu).norm().hi


# --------------------------------------------------------------------------- #
# Core: manufactured-solution existence proof.
# --------------------------------------------------------------------------- #
def test_manufactured_solution_is_an_exact_zero() -> None:
    problem, a_star = _scalar_problem()
    residual = evaluate_residual(problem, a_star)
    assert residual.norm().hi < 1e-12


def test_exact_solution_proved() -> None:
    problem, a_star = _scalar_problem()
    result = quadratic_radii_certificate(problem, a_star)
    assert isinstance(result, SpectralRadiiResult)
    assert result.proved
    assert result.certificate is not None
    assert result.y0 < 1e-12  # exact defect
    assert result.z0 < 1.0  # finite-block inverse is good
    assert result.radius is not None and result.radius > 0.0


def test_all_bounds_nonnegative() -> None:
    problem, a_star = _scalar_problem()
    r = quadratic_radii_certificate(problem, a_star)
    assert min(r.y0, r.z0, r.z1, r.z2, r.a_op_norm, r.residual_norm) >= 0.0


def test_true_solution_lies_in_certified_ball() -> None:
    # Certify at a *perturbed* approximation; the genuine zero a* must be captured.
    problem, a_star = _scalar_problem()
    a_bar = dict(a_star)
    a_bar[(3,)] = a_bar.get((3,), 0.0) + 1e-3
    a_bar[(-3,)] = a_bar.get((-3,), 0.0) + 1e-3

    result = quadratic_radii_certificate(problem, a_bar)
    assert result.proved
    assert result.radius is not None
    dist = _distance(a_star, a_bar, problem.dim, problem.trunc, problem.nu)
    assert dist > 0.0  # genuinely perturbed
    assert dist <= result.radius  # the true zero is inside the unique-solution ball


def test_certificate_is_sealed_and_verifiable() -> None:
    problem, a_star = _scalar_problem()
    result = quadratic_radii_certificate(problem, a_star)
    assert result.certificate is not None
    cert = result.certificate.certificate
    assert cert["payload"]["type"] == "radii_polynomial"
    assert verify_certificate_digest(cert)
    assert result.certificate.kappa < 1.0


# --------------------------------------------------------------------------- #
# Soundness cross-checks of the operator-norm bounds.
# --------------------------------------------------------------------------- #
def test_z0_upper_bounds_finite_direction_defect() -> None:
    # For finite h, ||(I - A DF(a))h|| / ||h|| <= Z0 (the finite-input op norm).
    problem, a_star = _scalar_problem()
    result = quadratic_radii_certificate(problem, a_star)
    system = _assemble(problem, a_star)
    rng = random.Random(20260630)
    finite_modes = [(j,) for j in range(-problem.trunc, problem.trunc + 1)]
    worst = 0.0
    for _ in range(300):
        coeffs = {m: rng.uniform(-1.0, 1.0) for m in finite_modes}
        h = _embed(coeffs, problem.dim, problem.work_trunc, problem.nu)
        ratio = (system.defect_apply(h).norm() / h.norm()).hi
        worst = max(worst, ratio)
    assert worst <= result.z0 * (1.0 + 1e-9)


def test_z1_z2_match_their_closed_form() -> None:
    problem, a_star = _scalar_problem()
    result = quadratic_radii_certificate(problem, a_star)
    ab_norm = _embed(a_star, problem.dim, problem.work_trunc, problem.nu).norm().hi
    # Z1 = 2 ||A|| C_Q ||a||,  Z2 = ||A|| C_Q   (C_Q = 1 here).
    assert result.z1 == pytest.approx(2.0 * result.a_op_norm * ab_norm, rel=1e-9)
    assert result.z2 == pytest.approx(result.a_op_norm, rel=1e-9)


def test_y0_bounded_by_operator_norm_times_residual() -> None:
    # ||A F(a)|| <= ||A|| ||F(a)||.
    problem, a_star = _scalar_problem()
    a_bar = dict(a_star)
    a_bar[(3,)] = 2e-3
    a_bar[(-3,)] = 2e-3
    result = quadratic_radii_certificate(problem, a_bar)
    assert result.y0 <= result.a_op_norm * result.residual_norm * (1.0 + 1e-9) + 1e-300


# --------------------------------------------------------------------------- #
# Negative space: when the closure must NOT prove.
# --------------------------------------------------------------------------- #
def test_non_coercive_symbol_is_not_proved() -> None:
    # Weak linear part -> large mu / Z1 > 1 -> no contraction.
    problem, a_star = _scalar_problem(c0=0.05, c2=0.01)
    result = quadratic_radii_certificate(problem, a_star)
    assert not result.proved
    assert result.certificate is None


def test_large_amplitude_breaks_contraction() -> None:
    big = {(0,): 5.0, (1,): 3.0, (-1,): 3.0}
    problem, a_star = _scalar_problem(a_star=big)
    result = quadratic_radii_certificate(problem, a_star)
    # Defect is still ~0 (manufactured), but Z1 = 2||A|| ||a|| is far above 1.
    assert result.z1 > 1.0
    assert not result.proved


# --------------------------------------------------------------------------- #
# Dimension-generality and dressed (Riesz) nonlinearities.
# --------------------------------------------------------------------------- #
def test_two_dimensional_manufactured_proof() -> None:
    dim, trunc, nu = 2, 2, 1.05
    c0, c2 = 5.0, 1.0
    symbol = laplacian_symbol(c0, c2)
    mu = laplacian_tail_inverse_bound(dim, trunc, c0, c2)
    quad = _convolution()
    a_star = {
        (0, 0): 0.08,
        (1, 0): 0.03,
        (-1, 0): 0.03,
        (0, 1): 0.03,
        (0, -1): 0.03,
    }
    forcing = _forcing_for(a_star, dim, trunc, nu, symbol, quad)
    problem = SpectralProblem(
        dim, trunc, nu, symbol, mu, quad, 1.0, forcing  # type: ignore[arg-type]
    )
    result = quadratic_radii_certificate(problem, a_star)
    assert result.proved
    assert result.radius is not None and result.radius > 0.0


def test_riesz_dressed_symmetric_quadratic_proof() -> None:
    # Symmetric, bounded dressed nonlinearity Q(u,v) = 1/2 (R0 u * v + R0 v * u).
    dim, trunc, nu = 1, 4, 1.05
    c0, c2 = 4.0, 1.0
    symbol = laplacian_symbol(c0, c2)
    mu = laplacian_tail_inverse_bound(dim, trunc, c0, c2)

    def quad(u: ValidatedFourierSeries, v: ValidatedFourierSeries) -> ValidatedFourierSeries:
        return (u.riesz(0) * v + v.riesz(0) * u).scale(0.5)

    a_star = {(0,): 0.0, (1,): 0.05, (-1,): 0.05, (2,): 0.02, (-2,): 0.02}
    forcing = _forcing_for(a_star, dim, trunc, nu, symbol, quad)
    problem = SpectralProblem(
        dim, trunc, nu, symbol, mu, quad, 1.0, forcing  # type: ignore[arg-type]
    )
    assert evaluate_residual(problem, a_star).norm().hi < 1e-12
    result = quadratic_radii_certificate(problem, a_star)
    assert result.proved


# --------------------------------------------------------------------------- #
# Validation / error paths.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dim": 0}, "dim"),
        ({"trunc": 0}, "trunc"),
        ({"nu": 0.5}, "nu"),
        ({"tail_inverse_bound": -1.0}, "tail_inverse_bound"),
        ({"quadratic_norm": -1.0}, "quadratic_norm"),
    ],
)
def test_spectralproblem_validation(kwargs: dict[str, object], match: str) -> None:
    base = dict(
        dim=1,
        trunc=2,
        nu=1.05,
        linear_symbol=laplacian_symbol(1.0, 1.0),
        tail_inverse_bound=0.1,
        quadratic=_convolution(),
        quadratic_norm=1.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SpectralProblem(**base)  # type: ignore[arg-type]


def test_laplacian_tail_inverse_bound_value() -> None:
    # mu = 1/(c0 + c2 (N+1)^2), N=4 -> 1/(4 + 25) = 1/29.
    mu = laplacian_tail_inverse_bound(1, 4, 4.0, 1.0)
    assert mu == pytest.approx(1.0 / 29.0, rel=1e-12)
    assert mu >= 1.0 / 29.0  # outward rounded upper bound


def test_laplacian_tail_inverse_bound_requires_coercivity() -> None:
    with pytest.raises(ValueError, match="coercive"):
        laplacian_tail_inverse_bound(1, 4, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        laplacian_tail_inverse_bound(1, 4, -1.0, 1.0)


def test_singular_finite_block_raises() -> None:
    # Zero symbol and zero nonlinearity -> B_N = 0 -> non-invertible.
    dim, trunc, nu = 1, 2, 1.05

    def zero_symbol(k: Wavevector) -> ComplexInterval:
        return ComplexInterval.zero()

    def zero_quad(
        u: ValidatedFourierSeries, v: ValidatedFourierSeries
    ) -> ValidatedFourierSeries:
        return ValidatedFourierSeries.zero(u.dim, u.trunc, u.nu)

    problem = SpectralProblem(
        dim, trunc, nu, zero_symbol, 1.0, zero_quad, 1.0, None  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="singular"):
        quadratic_radii_certificate(problem, {(0,): 0.0})


def test_invert_complex_round_trip() -> None:
    mat = [[2.0 + 0j, 1.0 + 0j], [0.0 + 0j, 3.0 - 1j]]
    inv = _invert_complex(mat)
    prod = [
        [sum(mat[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)
    ]
    assert abs(prod[0][0] - 1.0) < 1e-12
    assert abs(prod[1][1] - 1.0) < 1e-12
    assert abs(prod[0][1]) < 1e-12
    assert abs(prod[1][0]) < 1e-12


def test_invert_complex_singular_raises() -> None:
    with pytest.raises(ValueError, match="singular"):
        _invert_complex([[1.0 + 0j, 2.0 + 0j], [2.0 + 0j, 4.0 + 0j]])


# --------------------------------------------------------------------------- #
# Tighter truncation refines the radius bracket.
# --------------------------------------------------------------------------- #
def test_unproved_result_still_reports_bounds() -> None:
    problem, a_star = _scalar_problem(c0=0.05, c2=0.01)
    result = quadratic_radii_certificate(problem, a_star)
    assert result.certificate is None
    assert result.radius is None
    # bounds are still computed and inspectable
    assert result.z1 > 0.0
    assert result.a_op_norm > 0.0

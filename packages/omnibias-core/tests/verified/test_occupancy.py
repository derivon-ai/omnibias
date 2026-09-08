# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 04-03: certified Fermi occupancy, gates G2/G3/G4 (certified half).

G1 (float exactness) and the float half of G4/G5 live in
``packages/omnibias-core/tests/test_occupancy.py``.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.occupancy import (
    FermiModel,
    entropy_derivatives,
    entropy_per_state,
    grand_potential_density,
    occupancy,
    occupancy_derivatives,
    occupancy_mu_derivatives,
    occupancy_window,
    sommerfeld_coefficient,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.occupancy import (
    PolynomialDensityOfStates,
    certified_chemical_potential,
    constant_density_of_states,
    electron_count_enclosure,
    entropy_enclosure,
    grand_potential_enclosure,
    occupancy_enclosure,
    occupancy_mu_enclosure,
    occupancy_window_enclosure,
    sommerfeld_coefficient_enclosure,
    sommerfeld_moment_enclosure,
)

_RNG = random.Random(20260908)

_CASES = (
    (2.0, 0.5, 0.3),
    (0.7, -1.0, 2.5),
    (5.0, 0.0, 0.0),
    (10.0, 2.0, 1.9),
    (0.3, -3.0, -2.5),
)


# ----- G2: enclosure soundness (dense grid + random sample) -----------------


@pytest.mark.parametrize("beta,mu,_", _CASES)
def test_g2_occupancy_enclosure_contains_grid_and_random_float(
    beta: float, mu: float, _: float
) -> None:
    model = FermiModel(beta=beta, mu=mu)
    grid = [-6.0 + 0.25 * k for k in range(49)]
    sample = grid + [_RNG.uniform(-6.0, 6.0) for _ in range(30)]
    for e in sample:
        enc_tower = occupancy_enclosure(beta, mu, e, order=3)
        float_tower = occupancy_derivatives(model, e, order=3)
        for enc, val in zip(enc_tower, float_tower, strict=True):
            assert enc.lo <= val <= enc.hi, (beta, mu, e, enc, val)


@pytest.mark.parametrize("beta,mu,_", _CASES)
def test_g2_occupancy_mu_enclosure_contains_grid_and_random_float(
    beta: float, mu: float, _: float
) -> None:
    model = FermiModel(beta=beta, mu=mu)
    grid = [-4.0 + 0.2 * k for k in range(41)]
    sample = grid + [_RNG.uniform(-4.0, 4.0) for _ in range(30)]
    for e in sample:
        enc_tower = occupancy_mu_enclosure(beta, mu, e, order=2)
        float_tower = occupancy_mu_derivatives(model, e, order=2)
        for enc, val in zip(enc_tower, float_tower, strict=True):
            assert enc.lo <= val <= enc.hi, (beta, mu, e, enc, val)


@pytest.mark.parametrize("beta,mu,_", _CASES)
def test_g2_entropy_enclosure_contains_grid_and_random_float(
    beta: float, mu: float, _: float
) -> None:
    model = FermiModel(beta=beta, mu=mu)
    grid = [-6.0 + 0.25 * k for k in range(49)]
    sample = grid + [_RNG.uniform(-6.0, 6.0) for _ in range(30)]
    for e in sample:
        enc_tower = entropy_enclosure(beta, mu, e, order=2)
        float_tower = entropy_derivatives(model, e, order=2)
        for enc, val in zip(enc_tower, float_tower, strict=True):
            assert enc.lo <= val <= enc.hi, (beta, mu, e, enc, val)


@pytest.mark.parametrize("beta,mu,_", _CASES)
def test_g2_grand_potential_enclosure_contains_grid_and_random_float(
    beta: float, mu: float, _: float
) -> None:
    model = FermiModel(beta=beta, mu=mu)
    grid = [-6.0 + 0.25 * k for k in range(49)]
    sample = grid + [_RNG.uniform(-6.0, 6.0) for _ in range(30)]
    for e in sample:
        enc = grand_potential_enclosure(beta, mu, e)
        val = grand_potential_density(model, e)
        assert enc.lo <= val <= enc.hi, (beta, mu, e, enc, val)


@pytest.mark.parametrize("beta,mu", [(2.0, 0.5), (0.6, -1.5), (4.0, 0.2)])
def test_g2_occupancy_window_enclosure_contains_grid_and_random_float(
    beta: float, mu: float
) -> None:
    model = FermiModel(beta=beta, mu=mu)
    lo_grid = [-6.0 + 0.5 * k for k in range(13)]
    sample = lo_grid + [_RNG.uniform(-6.0, -0.5) for _ in range(15)]
    for e_lo in sample:
        e_hi = -e_lo  # symmetric window, always e_lo <= e_hi
        enc = occupancy_window_enclosure(beta, mu, e_lo, e_hi)
        val = occupancy_window(model, e_lo, e_hi)
        assert enc.lo <= val <= enc.hi, (beta, mu, e_lo, e_hi, enc, val)


def test_g2_electron_count_enclosure_contains_scipy_quadrature_and_is_sound() -> None:
    scipy_integrate = pytest.importorskip("scipy.integrate")
    beta, mu, g0 = 2.0, 0.3, 2.0
    dos = constant_density_of_states(g0)
    model = FermiModel(beta=beta, mu=mu)
    e_lo, e_hi = -5.0, 5.0
    enc = electron_count_enclosure(beta, mu, dos, e_lo, e_hi)
    ref, _err = scipy_integrate.quad(
        lambda e: g0 * occupancy(model, e), e_lo, e_hi
    )
    assert enc.lo <= ref <= enc.hi
    assert enc.width < 0.2  # tight, not merely sound


def test_g2_electron_count_enclosure_contains_random_seeds() -> None:
    scipy_integrate = pytest.importorskip("scipy.integrate")
    dos = constant_density_of_states(1.5)
    for _ in range(10):
        beta = _RNG.uniform(0.3, 8.0)
        mu = _RNG.uniform(-3.0, 3.0)
        model = FermiModel(beta=beta, mu=mu)
        e_lo = _RNG.uniform(-8.0, -1.0)
        e_hi = _RNG.uniform(1.0, 8.0)
        enc = electron_count_enclosure(beta, mu, dos, e_lo, e_hi)
        ref, _err = scipy_integrate.quad(
            lambda e, model=model: 1.5 * occupancy(model, e), e_lo, e_hi
        )
        assert enc.lo <= ref <= enc.hi, (beta, mu, e_lo, e_hi, enc, ref)


# ----- G3: certified chemical potential -------------------------------------


def _bisection_root(
    beta: float, dos: PolynomialDensityOfStates, e_lo: float, e_hi: float, n_target: float
) -> float:
    """Fine-bisection oracle for ``N(mu) = n_target`` (float reference)."""
    g0 = dos.coefficients[0]

    def n_of_mu(mu: float) -> float:
        model = FermiModel(beta=beta, mu=mu)
        return g0 * occupancy_window(model, e_lo, e_hi)

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n_of_mu(mid) < n_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_g3_certified_chemical_potential_ball_contains_bisection_root() -> None:
    beta, g0 = 2.0, 2.0
    dos = constant_density_of_states(g0)
    e_lo, e_hi = -3.0, 3.0
    mu_true = 0.5
    model = FermiModel(beta=beta, mu=mu_true)
    n_target = g0 * occupancy_window(model, e_lo, e_hi)

    decision = certified_chemical_potential(
        beta, dos, e_lo, e_hi, n_target, mu_bar=0.51, r_max=0.1
    )
    assert decision.accepted is True
    assert decision.reason == "ball"
    assert decision.certificate is not None

    root = _bisection_root(beta, dos, e_lo, e_hi, n_target)
    assert root == pytest.approx(mu_true, abs=1e-6)
    radius = decision.certificate.radius
    lo, hi = 0.51 - radius, 0.51 + radius
    assert lo <= root <= hi
    assert lo <= mu_true <= hi


@pytest.mark.parametrize(
    "beta,mu_true,e_lo,e_hi,r_max",
    [
        (0.8, -1.2, -6.0, 6.0, 0.2),
        (5.0, 0.7, -1.0, 1.0, 0.1),
        (1.5, 0.0, -4.0, 4.0, 0.15),
        (3.0, 1.3, -2.5, 2.5, 0.1),
    ],
)
def test_g3_certified_chemical_potential_matches_bisection_across_regimes(
    beta: float, mu_true: float, e_lo: float, e_hi: float, r_max: float
) -> None:
    g0 = 1.0
    dos = constant_density_of_states(g0)
    model = FermiModel(beta=beta, mu=mu_true)
    n_target = g0 * occupancy_window(model, e_lo, e_hi)

    # A trial displaced by a small, fixed absolute amount from the truth;
    # each (beta, window, r_max) here is tuned to a nonempty ball -- the
    # empty-ball halt itself is exercised separately below.
    mu_bar = mu_true + 0.02
    decision = certified_chemical_potential(
        beta, dos, e_lo, e_hi, n_target, mu_bar=mu_bar, r_max=r_max
    )
    assert decision.accepted is True
    assert decision.certificate is not None
    root = _bisection_root(beta, dos, e_lo, e_hi, n_target)
    radius = decision.certificate.radius
    assert mu_bar - radius <= root <= mu_bar + radius
    assert mu_bar - radius <= mu_true <= mu_bar + radius


def test_g3_certified_chemical_potential_reports_empty_ball_as_a_halt() -> None:
    """A too-tight r_max for the tower-derived Lipschitz bound is an honest
    reported halt (`reason="empty"`), never an exception or silent widening."""
    beta, g0 = 2.0, 2.0
    dos = constant_density_of_states(g0)
    e_lo, e_hi = -3.0, 3.0
    model = FermiModel(beta=beta, mu=0.5)
    n_target = g0 * occupancy_window(model, e_lo, e_hi)

    decision = certified_chemical_potential(
        beta, dos, e_lo, e_hi, n_target, mu_bar=0.51, r_max=0.01
    )
    assert decision.accepted is False
    assert decision.reason == "empty"
    assert decision.certificate is None


def test_g3_certified_chemical_potential_requires_positive_r_max() -> None:
    dos = constant_density_of_states(1.0)
    with pytest.raises(ValueError, match="r_max"):
        certified_chemical_potential(1.0, dos, -1.0, 1.0, 0.5, mu_bar=0.0, r_max=0.0)


# ----- G4: certified Sommerfeld coefficients --------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_g4_sommerfeld_coefficient_enclosure_contains_float(n: int) -> None:
    enc = sommerfeld_coefficient_enclosure(n)
    val = sommerfeld_coefficient(n)
    assert enc.lo <= val <= enc.hi
    assert enc.width < 1e-9


def test_g4_sommerfeld_enclosures_match_ashcroft_mermin_closed_form() -> None:
    a1 = sommerfeld_coefficient_enclosure(1)
    a2 = sommerfeld_coefficient_enclosure(2)
    assert a1.lo <= math.pi**2 / 6.0 <= a1.hi
    assert a2.lo <= 7.0 * math.pi**4 / 360.0 <= a2.hi


def test_g4_sommerfeld_moment_enclosure_zero_and_one() -> None:
    assert sommerfeld_moment_enclosure(0) == Interval.point(1.0)
    assert sommerfeld_moment_enclosure(1) == Interval.point(0.0)
    assert sommerfeld_moment_enclosure(3) == Interval.point(0.0)


def test_g4_sommerfeld_beats_finite_difference_on_tightness() -> None:
    """The closed-form certified enclosure is far tighter than a naive
    finite-difference numerical-integration arm of the same moment."""
    enc = sommerfeld_coefficient_enclosure(2)
    # A crude finite-difference/quadrature estimate over a truncated
    # integration domain, mimicking a non-certified numerical arm.
    scipy_integrate = pytest.importorskip("scipy.integrate")

    def sigma_prime(z: float) -> float:
        s = 1.0 / (1.0 + math.exp(-z))
        return s * (1.0 - s)

    fd_estimate, _err = scipy_integrate.quad(
        lambda z: sigma_prime(z) * z**4, -40.0, 40.0
    )
    fd_a2 = 2.0 * fd_estimate / math.factorial(4)
    true_a2 = 7.0 * math.pi**4 / 360.0
    assert abs(fd_a2 - true_a2) > enc.width  # certified enclosure is tighter


def test_g4_sommerfeld_moment_enclosure_rejects_negative_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 0"):
        sommerfeld_moment_enclosure(-1)


def test_g4_sommerfeld_coefficient_enclosure_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        sommerfeld_coefficient_enclosure(0)


# ----- PolynomialDensityOfStates --------------------------------------------


def test_polynomial_density_of_states_derivative_enclosure_matches_analytic() -> None:
    dos = PolynomialDensityOfStates((1.0, 2.0, 3.0))  # 1 + 2e + 3e^2
    assert dos.degree == 2
    for e in (-2.0, 0.0, 1.5):
        val = dos.value_enclosure(e)
        assert val.lo <= (1.0 + 2.0 * e + 3.0 * e * e) <= val.hi
        d1 = dos.derivative_enclosure(1, e)
        assert d1.lo <= (2.0 + 6.0 * e) <= d1.hi
        d2 = dos.derivative_enclosure(2, e)
        assert d2.lo <= 6.0 <= d2.hi
        d3 = dos.derivative_enclosure(3, e)
        assert d3 == Interval.point(0.0)


def test_polynomial_density_of_states_rejects_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PolynomialDensityOfStates(())


def test_constant_density_of_states_rejects_non_positive_g0() -> None:
    with pytest.raises(ValueError, match="g0 must be > 0"):
        constant_density_of_states(0.0)
    with pytest.raises(ValueError, match="g0 must be > 0"):
        constant_density_of_states(-1.0)


def test_electron_count_enclosure_rejects_reversed_bounds() -> None:
    dos = constant_density_of_states(1.0)
    with pytest.raises(ValueError, match="e_lo <= e_hi"):
        electron_count_enclosure(1.0, 0.0, dos, 1.0, -1.0)

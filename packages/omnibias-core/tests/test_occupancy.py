# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 04-03: Fermi occupancy and thermodynamic potentials, gates G1/G4/G5.

G2 (enclosure soundness) and G3 (certified chemical potential) live in
``packages/omnibias-core/tests/verified/test_occupancy.py`` against
:mod:`omnibias.core.verified.occupancy`.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.collapse import (
    CollapseSpec,
    register_collapse,
    reset_collapse_registry,
)
from omnibias.core.occupancy import (
    FermiModel,
    entropy_derivatives,
    entropy_per_state,
    grand_potential_density,
    honesty_payload,
    occupancy,
    occupancy_derivatives,
    occupancy_mu_derivatives,
    occupancy_window,
    reduced_argument,
    sommerfeld_coefficient,
    sommerfeld_moment,
    thermal_broadening,
    zero_temperature_occupancy,
)

mpmath = pytest.importorskip("mpmath")

_CASES = (
    (2.0, 0.5, 0.3),
    (0.7, -1.0, 2.5),
    (5.0, 0.0, 0.0),
    (10.0, 2.0, 1.9),
    (0.3, -3.0, -2.5),
)


def _mp_sigmoid(z: object) -> object:
    return 1 / (1 + mpmath.exp(-z))


def _mp_softplus(z: object) -> object:
    return mpmath.log1p(mpmath.exp(z))


def _mp_occupancy_of_energy(beta: float, mu: float, energy: object) -> object:
    z = -beta * (energy - mu)
    return _mp_sigmoid(z)


def _mp_entropy_of_energy(beta: float, mu: float, energy: object) -> object:
    z = -beta * (energy - mu)
    return _mp_softplus(z) - z * _mp_sigmoid(z)


def _mp_omega_of_mu(beta: float, mu: object, energy: float) -> object:
    z = -beta * (energy - mu)
    return -_mp_softplus(z) / beta


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


# ----- FermiModel validation -------------------------------------------------


def test_fermi_model_rejects_non_positive_or_non_finite_beta() -> None:
    with pytest.raises(ValueError, match="beta must be finite and > 0"):
        FermiModel(beta=0.0)
    with pytest.raises(ValueError, match="beta must be finite and > 0"):
        FermiModel(beta=-1.0)
    with pytest.raises(ValueError, match="beta must be finite and > 0"):
        FermiModel(beta=math.inf)


def test_fermi_model_rejects_non_finite_mu() -> None:
    with pytest.raises(ValueError, match="mu must be finite"):
        FermiModel(beta=1.0, mu=math.nan)


def test_reduced_argument_matches_definition() -> None:
    model = FermiModel(beta=2.0, mu=0.5)
    assert reduced_argument(model, 0.3) == pytest.approx(-2.0 * (0.3 - 0.5))


# ----- G1: exactness ----------------------------------------------------------


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_g1_entropy_identity_matches_binary_entropy(
    beta: float, mu: float, energy: float
) -> None:
    model = FermiModel(beta=beta, mu=mu)
    f = occupancy(model, energy)
    s = entropy_per_state(model, energy)
    if 0.0 < f < 1.0:
        ref = -f * math.log(f) - (1.0 - f) * math.log(1.0 - f)
        assert s == pytest.approx(ref, abs=1e-12)


def test_g1_entropy_special_values() -> None:
    model = FermiModel(beta=1.0, mu=0.0)
    assert entropy_per_state(model, 0.0) == pytest.approx(math.log(2.0), abs=1e-14)
    # s'(0) = 0 (maximum entropy point).
    assert entropy_derivatives(model, 0.0, order=1)[1] == pytest.approx(0.0, abs=1e-12)
    # Both tails vanish.
    assert entropy_per_state(model, 200.0) == pytest.approx(0.0, abs=1e-12)
    assert entropy_per_state(model, -200.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_g1_occupancy_derivatives_match_high_precision_reference(
    beta: float, mu: float, energy: float
) -> None:
    mpmath.mp.dps = 50
    model = FermiModel(beta=beta, mu=mu)
    tower = occupancy_derivatives(model, energy, order=8)

    def f_of_e(e: object) -> object:
        return _mp_occupancy_of_energy(beta, mu, e)

    for n in range(9):
        ref = float(mpmath.diff(f_of_e, mpmath.mpf(str(energy)), n))
        assert tower[n] == pytest.approx(ref, rel=1e-8, abs=1e-8), (beta, mu, energy, n)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_g1_entropy_derivatives_match_high_precision_reference(
    beta: float, mu: float, energy: float
) -> None:
    mpmath.mp.dps = 50
    model = FermiModel(beta=beta, mu=mu)
    tower = entropy_derivatives(model, energy, order=6)

    def s_of_e(e: object) -> object:
        return _mp_entropy_of_energy(beta, mu, e)

    for n in range(7):
        ref = float(mpmath.diff(s_of_e, mpmath.mpf(str(energy)), n))
        assert tower[n] == pytest.approx(ref, rel=1e-6, abs=1e-6), (beta, mu, energy, n)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_g1_occupancy_mu_derivatives_match_high_precision_reference(
    beta: float, mu: float, energy: float
) -> None:
    mpmath.mp.dps = 50
    model = FermiModel(beta=beta, mu=mu)
    tower = occupancy_mu_derivatives(model, energy, order=4)

    def f_of_mu(m: object) -> object:
        return _mp_occupancy_of_energy(beta, m, energy)

    for n in range(5):
        ref = float(mpmath.diff(f_of_mu, mpmath.mpf(str(mu)), n))
        assert tower[n] == pytest.approx(ref, rel=1e-8, abs=1e-8), (beta, mu, energy, n)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_g1_domega_dmu_equals_negative_occupancy(
    beta: float, mu: float, energy: float
) -> None:
    mpmath.mp.dps = 50
    model = FermiModel(beta=beta, mu=mu)

    def omega_of_mu(m: object) -> object:
        return _mp_omega_of_mu(beta, m, energy)

    d_omega_d_mu = float(mpmath.diff(omega_of_mu, mpmath.mpf(str(mu)), 1))
    assert d_omega_d_mu == pytest.approx(-occupancy(model, energy), abs=1e-10)


def test_g1_thermal_broadening_is_beta_times_sigma_prime() -> None:
    model = FermiModel(beta=3.0, mu=0.2)
    got = thermal_broadening(model, 0.1)
    _, d1 = occupancy_derivatives(model, 0.1, order=1)
    assert got == pytest.approx(-d1)
    assert got > 0.0  # thermal broadening is a positive smearing width


@pytest.mark.parametrize("beta,mu", [(2.0, 0.5), (0.5, -1.0), (4.0, 0.0)])
def test_g1_occupancy_window_matches_quadrature(beta: float, mu: float) -> None:
    model = FermiModel(beta=beta, mu=mu)
    e_lo, e_hi = -8.0, 8.0
    got = occupancy_window(model, e_lo, e_hi)

    mpmath.mp.dps = 30
    ref = float(
        mpmath.quad(lambda e: _mp_occupancy_of_energy(beta, mu, e), [e_lo, mu, e_hi])
    )
    assert got == pytest.approx(ref, rel=1e-8)
    assert got >= 0.0


def test_g1_occupancy_window_rejects_reversed_bounds() -> None:
    model = FermiModel(beta=1.0, mu=0.0)
    with pytest.raises(ValueError, match="e_lo <= e_hi"):
        occupancy_window(model, 1.0, -1.0)


# ----- G4: certified Sommerfeld coefficients (float half) -------------------


def test_g4_sommerfeld_moment_zero_and_odd_orders() -> None:
    assert sommerfeld_moment(0) == pytest.approx(1.0)
    for order in (1, 3, 5, 7):
        assert sommerfeld_moment(order) == 0.0


def test_g4_sommerfeld_moments_match_numeric_quadrature() -> None:
    mpmath.mp.dps = 40
    for order in (2, 4, 6):
        ref = float(
            mpmath.quad(
                lambda z, order=order: mpmath.diff(_mp_sigmoid, z) * z**order,
                [-mpmath.inf, 0, mpmath.inf],
            )
        )
        assert sommerfeld_moment(order) == pytest.approx(ref, rel=1e-9)


def test_g4_sommerfeld_coefficients_match_ashcroft_mermin() -> None:
    a1 = sommerfeld_coefficient(1)
    a2 = sommerfeld_coefficient(2)
    assert a1 == pytest.approx(math.pi**2 / 6.0, rel=1e-12)
    assert a2 == pytest.approx(7.0 * math.pi**4 / 360.0, rel=1e-12)


def test_g4_sommerfeld_coefficient_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        sommerfeld_coefficient(0)


# ----- zero-temperature limit (named, not requested as a new collapse) ------


def test_zero_temperature_occupancy_is_the_step_function() -> None:
    model = FermiModel(beta=50.0, mu=1.0)
    assert zero_temperature_occupancy(model, 0.5) == 1.0
    assert zero_temperature_occupancy(model, 1.5) == 0.0
    assert zero_temperature_occupancy(model, 1.0) == 0.5


# ----- G5: honesty non-vacuity ------------------------------------------------


def test_g5_honesty_keys_are_all_false() -> None:
    payload = honesty_payload()
    assert payload == {
        "founding_bias_collapse": False,
        "temperature_collapse": False,
        "requests_new_collapse_registry_slot": False,
        "dft_solved_claim": False,
        "many_body_solved_claim": False,
        "interacting_system_claim": False,
        "thermodynamic_limit_taken": False,
        "phase_transition_proved": False,
        "theorem_prover_verified": False,
    }


def test_g5_honesty_keys_never_set_true_in_source() -> None:
    """Static scan: no permanently-false honesty key is ever assigned True
    anywhere in the occupancy modules (core or verified)."""
    import re
    from pathlib import Path

    forbidden_keys = (
        "dft_solved_claim",
        "many_body_solved_claim",
        "interacting_system_claim",
        "thermodynamic_limit_taken",
        "phase_transition_proved",
        "founding_bias_collapse",
        "temperature_collapse",
        "requests_new_collapse_registry_slot",
        "theorem_prover_verified",
    )
    repo_root = Path(__file__).resolve().parents[3]
    sources = [
        repo_root / "packages/omnibias-core/src/omnibias/core/occupancy.py",
        repo_root / "packages/omnibias-core/src/omnibias/core/verified/occupancy.py",
    ]
    pattern = re.compile(
        r'"(' + "|".join(forbidden_keys) + r')"\s*:\s*True'
    )
    for path in sources:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        assert matches == [], f"{path} sets a forbidden honesty key True: {matches}"


def test_g5_collapse_registry_refuses_a_beta_indicator_spec() -> None:
    """Non-vacuity: this module does not smuggle in a tenth named collapse.

    ``beta -> inf`` with a 0/1 ``indicator`` surviving object is *exactly*
    the founding temperature collapse; the registry must refuse it as a
    rebrand rather than silently accepting a new name for the same limit.
    """
    spec = CollapseSpec(
        name="occupancy_zero_temperature",
        parameter="beta",
        limit="inf",
        surviving_object="indicator",
        failure="not a certificate",
        home="omnibias.core.occupancy",
        register="differentiable",
    )
    with pytest.raises(ValueError, match="rebrand of 'temperature'"):
        register_collapse(spec)

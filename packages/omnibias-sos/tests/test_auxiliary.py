# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Auxiliary-functional (background) method: sound for-all-data time-average bounds.

Every "proved" bound is independently re-checked two ways: (1) the certified SOS
Gram matrix must expand *exactly* to the residual ``S = C - Phi - grad(V).f`` (so the
proof is about the right polynomial), and (2) ``S`` must be nonnegative on a dense
grid plus random samples.  The flagship is the energy-conserving Galerkin triad,
whose certified bound matches the analytic background-method optimum ``C = 1/(2 nu^2)``.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import (
    canonical_json,
    verify_certificate_digest,
)
from omnibias.sos.auxiliary import (
    AuxiliaryBoundCertificate,
    PolynomialSystem,
    certify_time_average_bound,
    energy_conserving_triad_system,
    energy_observable,
    seal_auxiliary_bound,
)
from omnibias.sos.certify import rational_gram
from omnibias.sos.honesty import GALERKIN_TRUNCATION, SOSScope
from omnibias.sos.monomials import gram_to_poly
from omnibias.sos.problem import Polynomial, RationalPolynomial


def _residual(cert: AuxiliaryBoundCertificate, system: PolynomialSystem, phi: Polynomial):
    """The exact residual polynomial ``S = C - Phi - grad(V).f`` for a proved bound."""
    v = cert.auxiliary_polynomial()
    return (
        RationalPolynomial.constant(Fraction(cert.bound), system.n_vars)
        - RationalPolynomial.from_polynomial(phi)
        - system.lie_derivative_rational(v)
    )


def _assert_sound(cert: AuxiliaryBoundCertificate, system: PolynomialSystem, phi: Polynomial) -> None:
    """The certified SOS Gram must expand to exactly S, and S must sample nonnegative."""
    assert cert.certified
    assert cert.sos_certificate is not None
    residual = _residual(cert, system, phi)

    # (1) the certified Gram is a proof about *this* residual (exact coefficient match).
    gram = rational_gram(cert.sos_certificate)
    assert gram is not None
    expanded = gram_to_poly(gram, list(cert.sos_certificate.basis), system.n_vars)
    keys = set(residual.support) | set(expanded.support)
    assert max((abs(float(residual.coefficient(k)) - expanded.coefficient(k)) for k in keys), default=0.0) < 1e-9

    # (2) S >= 0 on a dense grid + random samples (empirical cross-check of the proof).
    s_float = residual.to_float()
    axis = [i / 3.0 for i in range(-9, 10)]

    def grid(dim: int, prefix: list[float]) -> float:
        if dim == 0:
            return s_float.evaluate(prefix)
        return min(grid(dim - 1, [*prefix, a]) for a in axis)

    assert grid(system.n_vars, []) >= -1e-9
    rng = random.Random(0)
    for _ in range(2000):
        point = [rng.uniform(-3.0, 3.0) for _ in range(system.n_vars)]
        assert s_float.evaluate(point) >= -1e-9


def test_linear_decay_average_is_certified_near_zero() -> None:
    # dx = -x, Phi = x^2 : x(t) -> 0, so the infinite-time average of x^2 is 0.
    system = PolynomialSystem(1, (Polynomial(1, {(1,): -1.0}),))
    phi = Polynomial(1, {(2,): 1.0})
    cert = certify_time_average_bound(system, phi)
    assert cert.certified
    assert 0.0 <= cert.bound_value < 1e-3
    assert cert.applies_to_all_initial_data  # coercive V => bounded below
    _assert_sound(cert, system, phi)


@pytest.mark.parametrize("forcing", [1.0, 2.0, 3.0])
def test_forced_scalar_recovers_stationary_second_moment(forcing: float) -> None:
    # dx = -x + F : the stationary state is x = F, so the mean of x^2 is exactly F^2.
    system = PolynomialSystem(1, (Polynomial(1, {(1,): -1.0, (0,): forcing}),))
    phi = Polynomial(1, {(2,): 1.0})
    cert = certify_time_average_bound(system, phi)
    assert cert.certified
    # A sound upper bound that is tight to the exact stationary value F^2.
    assert cert.bound_value >= forcing * forcing - 1e-6
    assert cert.bound_value == pytest.approx(forcing * forcing, abs=5e-3)
    assert cert.applies_to_all_initial_data
    _assert_sound(cert, system, phi)


@pytest.mark.parametrize("viscosity", [0.5, 1.0, 2.0, 4.0])
def test_triad_energy_bound_matches_background_optimum(viscosity: float) -> None:
    # Energy-conserving Galerkin triad with forcing on mode 1: the background-method
    # optimum for the mean energy is exactly C = f_1^2 / (2 nu^2)  (f_1 = 1).
    system = energy_conserving_triad_system(viscosities=[viscosity] * 3, forcing=[1.0, 0.0, 0.0])
    phi = energy_observable(3)
    cert = certify_time_average_bound(system, phi)
    assert cert.certified
    optimum = 1.0 / (2.0 * viscosity**2)
    assert cert.bound_value >= optimum - 1e-9  # sound upper bound
    assert cert.bound_value == pytest.approx(optimum, rel=2e-3)
    assert cert.applies_to_all_initial_data
    _assert_sound(cert, system, phi)


def test_triad_bound_is_monotone_in_viscosity() -> None:
    phi = energy_observable(3)
    bounds = []
    for nu in (0.5, 1.0, 2.0, 4.0):
        system = energy_conserving_triad_system(viscosities=[nu] * 3, forcing=[1.0, 0.0, 0.0])
        bounds.append(certify_time_average_bound(system, phi).bound_value)
    assert all(a > b for a, b in zip(bounds, bounds[1:], strict=False))  # more damping -> tighter


def test_growth_system_is_bounded_trajectory_only() -> None:
    # dx = +x : the SOS fact S >= 0 is provable, but the optimal V is not bounded
    # below, so the bound is honestly scoped to compact/bounded trajectories, NOT all data.
    system = PolynomialSystem(1, (Polynomial(1, {(1,): 1.0}),))
    phi = Polynomial(1, {(2,): 1.0})
    cert = certify_time_average_bound(system, phi)
    assert cert.certified  # the global SOS fact holds
    assert not cert.applies_to_all_initial_data
    assert cert.v_lower_bound == ""
    assert "compact set" in cert.detail


def test_odd_observable_is_inconclusive_not_false() -> None:
    # Phi = x^3 (odd) cannot be made SOS by a quadratic auxiliary functional; the method
    # must return inconclusive (soundness over completeness), never a false bound.
    system = PolynomialSystem(1, (Polynomial(1, {(1,): -1.0}),))
    phi = Polynomial(1, {(3,): 1.0})
    cert = certify_time_average_bound(system, phi, auxiliary_degree=2)
    assert not cert.certified
    assert cert.status == "inconclusive"
    assert cert.sos_certificate is None


def test_seal_auxiliary_bound_is_finite_dimensional_and_makes_no_open_claim() -> None:
    system = energy_conserving_triad_system(viscosities=[1.0] * 3, forcing=[1.0, 0.0, 0.0])
    phi = energy_observable(3)
    cert = certify_time_average_bound(system, phi)
    assert cert.certified

    scope = SOSScope(GALERKIN_TRUNCATION, truncation_order=3, system="energy-conserving triad")
    sealed = seal_auxiliary_bound(
        cert, claim="mean energy of the 3-mode Galerkin triad is bounded", scope=scope
    )

    honesty = sealed["honesty"]
    assert honesty["unproven_claim"] is False
    assert honesty["continuum_pde_claim"] is False
    assert honesty["finite_dimensional"] is True

    meta = sealed["meta"]
    assert meta["sos"]["scope"] == GALERKIN_TRUNCATION
    assert meta["sos"]["truncation_order"] == 3
    assert meta["auxiliary_bound"]["applies_to_all_initial_data"] is True

    # tamper-evident + canonical
    assert verify_certificate_digest(sealed)
    assert canonical_json(sealed)


def test_cannot_seal_inconclusive_auxiliary_bound() -> None:
    system = PolynomialSystem(1, (Polynomial(1, {(1,): -1.0}),))
    phi = Polynomial(1, {(3,): 1.0})
    cert = certify_time_average_bound(system, phi, auxiliary_degree=2)
    assert not cert.certified
    with pytest.raises(ValueError, match="inconclusive"):
        seal_auxiliary_bound(cert, claim="unreachable")

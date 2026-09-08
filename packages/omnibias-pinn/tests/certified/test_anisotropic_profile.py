# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-09: anisotropic similarity profile, axis germ, source jets."""

from __future__ import annotations

from fractions import Fraction

from omnibias.pinn.certified.anisotropic import (
    LOCKED_H,
    AxisRegularProfile,
    SimilarityScales,
    apply_T_b,
    apply_Z_b,
    assert_honesty,
    axis_germ,
    coefficient_source_jet,
    honesty_payload,
    leading_tangential_residual,
    lemma_41_values,
    locked_axis_regular_profile,
    locked_profile_residual_abs,
    locked_residual_poly,
    profile_jet_coeffs,
    profile_operators,
    radial_div,
    residual_plus_div,
    stress_from_residual_poly,
)


def test_g1_operators_match_lemma_41() -> None:
    profile = locked_axis_regular_profile()
    assert profile.h == LOCKED_H
    scales = SimilarityScales(LOCKED_H)
    assert scales.A == Fraction(1, 2) + LOCKED_H
    assert scales.D == Fraction(1, 2) - LOCKED_H
    X, eta, b = Fraction(1), Fraction(0), Fraction(0)
    jet = profile.jet_at(X, eta)
    ops = profile_operators(LOCKED_H)
    T = ops["T_b"](b, X, eta, jet)
    Z = ops["Z_b"](b, X, eta, jet)
    # F = 1 + X so F_X = 1, F_eta = 0, L = 1, D_X F = 1.
    assert T == Fraction(1)
    assert Z == Fraction(0)
    values = lemma_41_values(profile, X=X, eta=eta, b=b)
    assert values["T_b"] == T
    assert values["Z_b"] == Z
    eta2 = Fraction(1, 2)
    jet2 = profile.jet_at(X, eta2)
    L = 1 - 2 * LOCKED_H * eta2 * eta2
    assert apply_T_b(LOCKED_H, b, X, eta2, jet2) == 1 / L
    assert apply_Z_b(LOCKED_H, b, X, eta2, jet2) == -1 / L


def test_g2_residual_equals_minus_div_T() -> None:
    r = Fraction(1)
    assert residual_plus_div(r) == 0
    T = stress_from_residual_poly(locked_residual_poly())
    assert T[-1] != 0
    R = leading_tangential_residual(locked_axis_regular_profile(), Fraction(1), Fraction(0), r=r)
    assert R == Fraction(1)
    assert R + radial_div(T, r) == 0


def test_g3_axis_germ_remainder_is_zero() -> None:
    germ = axis_germ(locked_axis_regular_profile(), order=2)
    assert germ.remainder.lo == 0.0
    assert germ.remainder.hi == 0.0
    assert germ.coeffs[0].contains(1.0)
    assert germ.coeffs[1].contains(1.0)
    assert germ.coeffs[2].contains(0.0)


def test_g4_jet_multiply_matches_cauchy_product() -> None:
    report = coefficient_source_jet(profile_jet_coeffs(order=1), order=1)
    assert report["cauchy"] == (Fraction(1), Fraction(2))
    assert report["matches"] is True


def test_g5_honesty_no_parent_flag() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
    assert flags["continuum_navier_stokes_claim"] is False
    sealed = assert_honesty(flags)
    assert sealed["navier_stokes_proof_claim"] is False


def test_honesty_refuses_parent_true() -> None:
    import pytest

    with pytest.raises(ValueError, match="navier_stokes_proof_claim"):
        assert_honesty({"navier_stokes_proof_claim": True})


def test_pi_is_exact_integral_of_F_squared() -> None:
    profile = locked_axis_regular_profile()
    X = Fraction(1)
    # F = 1 + X, F^2 = 1 + 2X + X^2, int_0^X = X + X^2 + X^3/3.
    assert profile.Pi(X) == 1 + 1 + Fraction(1, 3)
    assert profile.V0(X, Fraction(1, 2)) == -X * Fraction(1, 2)


def test_locked_residual_abs_is_zero() -> None:
    assert locked_profile_residual_abs() == 0


def test_custom_profile_is_axis_regular() -> None:
    profile = AxisRegularProfile(h=LOCKED_H, c=Fraction(2), a=Fraction(3))
    germ = axis_germ(profile, order=1)
    assert germ.remainder.lo == germ.remainder.hi == 0.0
    assert germ.coeffs[0].contains(2.0)
    assert germ.coeffs[1].contains(6.0)

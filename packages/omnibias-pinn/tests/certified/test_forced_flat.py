# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-13: jet-flat forced concentrating field."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.pulse_envelope import mollifier_tail_contains_truth
from omnibias.pinn.certified.forced_flat import (
    LOCKED_H,
    LOCKED_TAU_HI,
    LOCKED_TAU_LO,
    JetFlatProfile,
    assert_honesty,
    axis_sources,
    axis_T0,
    core_energy_scale,
    core_linfty_scale,
    correct_axis_stress,
    energy_box_prefactor,
    forced_field,
    from_rest_ramp,
    honesty_payload,
    leading_stress_T0,
    linear_T_b_at,
    locked_jet_flat_profile,
    logarithmic_slope,
    on_window_ramp,
    uncorrected_jet_flat_profile,
)


def test_g1_linfty_ratio_monomial() -> None:
    profile = uncorrected_jet_flat_profile()
    s1 = core_linfty_scale(LOCKED_TAU_HI, profile)
    s2 = core_linfty_scale(LOCKED_TAU_LO, profile)
    assert s1.prefactor == s2.prefactor == profile.F(Fraction(1))
    assert s1.prefactor == 2
    assert s1.exponent == s2.exponent == -profile.scales.A
    assert profile.scales.A == Fraction(1, 2) + LOCKED_H


def test_g2_energy_prefactor_and_exponent() -> None:
    profile = uncorrected_jet_flat_profile()
    pref = energy_box_prefactor(profile)
    assert pref == Fraction(17, 3)
    e1 = core_energy_scale(LOCKED_TAU_HI, profile)
    e2 = core_energy_scale(LOCKED_TAU_LO, profile)
    assert e1.prefactor == e2.prefactor == pref
    assert e1.exponent == Fraction(1, 2) - 3 * LOCKED_H
    assert e1.exponent == Fraction(97, 200)
    assert e1.exponent > 0
    assert LOCKED_TAU_LO < LOCKED_TAU_HI


def test_g3_uncorrected_axis_jet_nonzero() -> None:
    jet = axis_T0(uncorrected_jet_flat_profile())
    assert jet == (Fraction(599, 400), Fraction(0))
    assert jet != (0, 0)
    src = axis_sources(uncorrected_jet_flat_profile())
    assert src["S_q"] == -(1 + LOCKED_H)
    assert src["Q_s"] == src["S_q"] / 2
    assert src["N_s"] == 0


def test_g4_corrected_axis_jet_is_zero() -> None:
    corrected, a_star = correct_axis_stress(uncorrected_jet_flat_profile())
    assert a_star == (1 + LOCKED_H) / 4
    assert a_star == Fraction(201, 800)
    assert axis_T0(corrected) == (0, 0)
    assert axis_T0(locked_jet_flat_profile()) == (0, 0)
    assert leading_stress_T0(corrected, 0, 0) == (0, 0)


def test_g5_mollifier_and_from_rest_ramp() -> None:
    assert mollifier_tail_contains_truth(half_width=3.0)
    assert from_rest_ramp().value() == 0
    assert on_window_ramp().value() == 1


def test_g6_honesty_and_named_leftover() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
    assert flags["continuum_navier_stokes_claim"] is False
    assert flags["c_infinity_through_t1_leftover"] is True
    assert flags["pulses_leftover"] is True
    assert flags["joining_leftover"] is True
    assert flags["uniqueness_leftover"] is True
    assert flags["remaining_q_powers_leftover"] is True
    sealed = assert_honesty(flags)
    assert sealed["forced_blowup_reproof_claim"] is False


def test_honesty_refuses_parent_true() -> None:
    with pytest.raises(ValueError, match="forced_blowup_reproof_claim"):
        assert_honesty({"forced_blowup_reproof_claim": True})


def test_b_does_not_enter_axis_jet() -> None:
    base = JetFlatProfile(h=LOCKED_H, c=1, a=1, b=0)
    bumped = JetFlatProfile(h=LOCKED_H, c=1, a=1, b=5)
    assert axis_T0(base) == axis_T0(bumped)


def test_off_axis_t0_is_leftover() -> None:
    with pytest.raises(ValueError, match="axis jet"):
        leading_stress_T0(uncorrected_jet_flat_profile(), 1, 0)


def test_logarithmic_slope_and_linear_T_b() -> None:
    profile = uncorrected_jet_flat_profile()
    assert logarithmic_slope(profile, 0) == 1
    # F = 1+X so l = 1 + X/(1+X); at X=1, l = 3/2.
    assert logarithmic_slope(profile, 1) == Fraction(3, 2)
    assert linear_T_b_at(profile) == 1


def test_forced_field_payload() -> None:
    payload = forced_field(LOCKED_TAU_HI, cutoff=True, ramp=True, corrected=True)
    assert payload["kind"] == "jet_flat_forced_blowup"
    assert payload["axis_T0"] == (0, 0)
    assert payload["ramp_at_rest"] == 0
    assert payload["ramp_on_window"] == 1
    assert payload["cutoff_ok"] is True
    assert payload["honesty"]["forced_blowup_reproof_claim"] is False
    assert payload["leftover"]["c_infinity_through_t1"] is True
    control = forced_field(LOCKED_TAU_HI, corrected=False)
    assert control["axis_T0"][0] != 0


def test_catalog_kind_is_registered() -> None:
    import omnibias.pinn.certified.machine  # noqa: F401
    from omnibias.core.proof.catalog import catalog_entry

    entry = catalog_entry("jet_flat_forced_blowup")
    assert entry is not None
    assert entry.parent == "Navier-Stokes forced blowup (Clay C/D)"
    assert entry.parent_status == "already_true"
    assert entry.kind != "stress_cone"
    assert entry.mode == "exact_replay"

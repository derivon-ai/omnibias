# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact scale-dichotomy identities; not a C2 or G1 certificate."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.chart_cells import g1_from_cells, ledger_payload
from omnibias.dynamics.hilbert16_identities import (
    identity_verdict,
    replay_hilbert16_identities,
)
from omnibias.dynamics.scale_dichotomy import (
    EXISTING_SECTION_POWERS,
    chi_on_kill,
    existing_section_explosion,
    first_root,
    hk_leading_jet,
    identity_verdicts,
    kill_L,
    kill_sep,
    kill_sequence_admission,
    log_inner_coordinate,
    residual_blowup_height,
    residual_event_leading,
    residual_joint_sep_r1_sum,
    rstar,
    scale_dichotomy_witness,
)


def test_blowup_and_joint_identities_are_exact() -> None:
    eps, sigma, eta = Fraction(1, 2), Fraction(1, 3), Fraction(2)
    assert residual_blowup_height(eps, sigma, eta, eps**3 * sigma**2 * eta) == 0
    assert residual_joint_sep_r1_sum(Fraction(-4), Fraction(2), Fraction(1)) == 0
    assert residual_event_leading(
        Fraction(13, 6), Fraction(1, 2), Fraction(3), Fraction(1), Fraction(1, 3), Fraction(2)
    ) == 0


def test_identity_verdicts_and_discovery_keep_parent_open() -> None:
    verdicts = identity_verdicts()
    assert set(verdicts) >= {
        "blowup_height_scale",
        "blowup_height_ratio",
        "event_leading_kappa",
        "joint_sep_r1_sum",
        "hk_second_kappa_vanishes",
    }
    assert all(status == "PROVED" for status in verdicts.values())
    assert identity_verdict("joint_sep_r1_sum").status == "PROVED"
    result = replay_hilbert16_identities()
    assert result.status == "PROVED"
    assert result.statement.parent_status == "open"
    assert result.check is not None
    assert result.check.payload["honesty"]["g1_passed"] is False
    assert result.check.payload["honesty"]["scale_dichotomy_c2_remainder"] is False


def test_log_inner_and_dichotomy_on_kill_sequence() -> None:
    eps = 0.05
    sep = kill_sep(eps)
    L = kill_L(eps)
    assert sep**2 - ((-3.0) ** 2 - 4.0 * L) == 0.0
    assert L > 2.0
    assert abs(log_inner_coordinate(eps, sep) - 1.0 / eps) < 1e-12
    chi = chi_on_kill(-3.0, sep, 1.0 / sep)
    assert abs(chi - 1.0 / rstar(-3.0)) < 1e-9
    witness = scale_dichotomy_witness(eps=eps)
    assert witness["dichotomy_on_tested_scales"] is True
    assert witness["some_scale_bounds_both"] is False
    assert witness["scales"]["separation"]["event_bounded"] is True
    assert witness["scales"]["fold_sqrt_eps"]["event_bounded"] is False


def test_admission_and_existing_sections_do_not_pass_g1() -> None:
    admission = kill_sequence_admission()
    assert admission["incoming_first_hit_retained"] is True
    assert admission["quadratic_relation_residual"] == 0.0
    assert admission["fixed_L_1_tuple_is_invalid"] is True
    assert admission["inadmissible"] is False
    assert admission["still_g1_falsifier"] is True
    assert admission["hk_theorem_24_hypotheses"] is False
    sections = existing_section_explosion()
    assert sections["all_existing_sections_explode"] is True
    assert set(EXISTING_SECTION_POWERS) == {0, 3, 4}
    jet = hk_leading_jet()
    assert jet["second_vanishes"] is True
    assert jet["physical_c2_remainder"] is False
    names = {cell["name"] for cell in ledger_payload()["cells"]}
    assert {
        "LI",
        "WL",
        "kill_super_small_sep",
        "kill_shrinking_root",
        "product_bound",
        "fold_imap",
        "canonical_zeta",
        "fold_zeta",
        "physical_c2",
        "z_x_gap",
        "z_v_bound",
        "z_slow_v",
        "fold_z_x",
        "stage_b",
        "stage_a",
        "chi_b",
        "dx_e_leading",
        "dx_e_unif",
        "stage_c",
        "stage_c_exit",
        "stage_c_th",
        "stage_c_gap",
        "stage_c_env",
        "stage_c_if",
        "stage_c_int",
        "stage_c_lo",
        "stage_c_k",
        "stage_c_boot",
        "stage_c_rect",
        "stage_c_hit",
        "stage_c_sec",
        "stage_c_oneshot",
        "stage_c_oneshot_eps",
        "stage_c_eps_span",
        "stage_c_origin",
        "stage_c_origin_span",
        "stage_c_origin_iface",
        "stage_c_origin_near",
        "stage_c_origin_x32",
        "stage_c_compare",
        "stage_c_uniform",
        "stage_c_interface",
        "sep_spre",
        "dx_e_off",
        "dx_e_ray",
        "dx_e_near",
        "dx_e_open",
        "weighted_section",
        "quasihomogeneous_dichotomy",
        "ln_format_barrier",
        "outgoing_corridor",
        "post_corridor",
        "height_envelope",
        "q_ratio_c2",
        "k_zeta_remainder",
        "kill_zeta",
        "cancelled_n",
        "height_mix",
        "orbit_th",
        "th_integral",
        "vh_orbit",
        "e_out_section",
        "e_out_eps",
        "e_out_speed",
        "e_sigma_speed",
        "e_sigma_in",
        "e_sigma_hit",
        "e_sigma_from0",
        "e_sigma_unif",
        "e_sigma_wall",
        "e_sigma_box",
        "e_sigma_span",
        "e_sigma_pack",
        "e_sigma_eps",
        "e_sigma_oneshot",
        "e_sigma_oneshot_eps",
        "e_sigma_eps_span",
        "e_sigma_eps_lo",
        "O",
    } <= names
    assert g1_from_cells() is False


def test_joint_axis_forbids_simultaneous_smallness() -> None:
    lam1 = Fraction(-2)
    # sep + 2 r1 = 2. If sep < 1 and r1 < 1/2 the sum is < 2.
    sep, r1 = Fraction(1, 3), Fraction(1, 5)
    assert 2 * r1 + sep < -lam1
    r1_from_sep = first_root(float(lam1), float(Fraction(1, 10)))
    assert r1_from_sep > 0.5

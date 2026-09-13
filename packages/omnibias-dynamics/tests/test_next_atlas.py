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
    assert {"LI", "WL", "kill_super_small_sep", "kill_shrinking_root"} <= names
    assert g1_from_cells() is False


def test_joint_axis_forbids_simultaneous_smallness() -> None:
    lam1 = Fraction(-2)
    # sep + 2 r1 = 2. If sep < 1 and r1 < 1/2 the sum is < 2.
    sep, r1 = Fraction(1, 3), Fraction(1, 5)
    assert 2 * r1 + sep < -lam1
    r1_from_sep = first_root(float(lam1), float(Fraction(1, 10)))
    assert r1_from_sep > 0.5

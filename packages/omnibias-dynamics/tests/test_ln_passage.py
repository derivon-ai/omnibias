# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Regression tests for the honest LN/exp coalescing-passage assessment."""

from __future__ import annotations

import pytest
from omnibias.dynamics.ln_passage import (
    admission_compatibility,
    assert_g1_passed,
    certify_regular_arc_model,
    coordinate_ln_evidence,
    coordinate_monomial_replay,
    coordinate_sup_enclosures,
    corrected_kill_a,
    invalid_plan_kill_a,
    kill_b,
    report,
    shrinking_root_radius,
)


def test_printed_kill_a_tuple_is_rejected_and_corrected_path_is_canonical() -> None:
    invalid = invalid_plan_kill_a(0.1)
    assert invalid["is_quadratic_family_point"] is False
    assert abs(float(invalid["relation_residual"])) > 4.9

    corrected = corrected_kill_a(0.1)
    assert corrected.pointwise_in_cell is True
    assert corrected.values["quadratic_relation_holds"] is True
    assert corrected.uniform_format_bounded is False
    assert corrected.failing_chain_function == "physical_outgoing_matching_W_ratio"


def test_corrected_kill_a_log_coordinate_and_w_ratio_diverge() -> None:
    coarse = corrected_kill_a(0.1)
    fine = corrected_kill_a(0.05)
    assert coarse.values["tau"] == pytest.approx(10.0)
    assert fine.values["tau"] == pytest.approx(20.0)
    for power in (0, 3, 4):
        key = f"log_W_ratio_eps_{power}"
        assert float(fine.values[key]) > float(coarse.values[key])


def test_proposed_shrinking_root_radius_is_eventually_negative() -> None:
    placement = kill_b(32, theta=0.125)
    assert placement.pointwise_in_cell is False
    assert placement.positive_radius_margin is False
    assert float(placement.values["proposed_radius"]) < 0
    assert float(placement.values["relative_radius"]) > 0
    assert placement.values["quadratic_relation_holds"] is True
    assert placement.failing_chain_function == "a(L,lambda1)=r1-theta*sep"


def test_shrinking_root_radius_refuses_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="theta"):
        shrinking_root_radius(0.1, theta=0)
    with pytest.raises(ValueError, match="first-root"):
        shrinking_root_radius(1.0, theta=0.1)
    with pytest.raises(ValueError, match="integer"):
        kill_b(1)


def test_finite_chain_guards_and_regular_arc_replay_without_overclaim() -> None:
    assert coordinate_monomial_replay() is True
    evidence = coordinate_ln_evidence()
    assert evidence["chain_valid"] is True
    assert evidence["certificate_valid"] is True
    assert evidence["physical_phi_member"] is False
    assert evidence["physical_uniform_c2"] is False
    assert set(evidence["sup_enclosures"]) == {"epsilon", "sep", "W", "radius"}
    with pytest.raises(ValueError, match="0 < delta < 1"):
        coordinate_sup_enclosures(delta=0)  # type: ignore[arg-type]
    admission = admission_compatibility()
    assert admission["finite_compatible"] is True
    assert admission["ambiguous_control"].status == "unresolved"
    assert admission["small_label_tube"]["strict"] is True
    assert admission["physical_first_hit_complete"] is False

    regular = certify_regular_arc_model()
    assert regular["certified"] is True
    assert regular["replayed"] is True
    assert regular["physical_singular_tail"] is False


def test_g1_assertion_refuses_partial_items_and_report_stays_blocked() -> None:
    with pytest.raises(ValueError, match="all four"):
        assert_g1_passed((True, True, False, True))
    assert_g1_passed((True, True, True, True))
    payload = report()
    assert payload["coordinate_chain_closure"] is True
    assert payload["g1_items_closed"] == (False, False, False, False)
    assert payload["g1_passed"] is False
    assert payload["full_hilbert16_solved"] is False

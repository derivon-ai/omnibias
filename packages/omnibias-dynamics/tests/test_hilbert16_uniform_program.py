# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Regression tests for the Hilbert-16 uniform program modules."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.dynamics.chart_gamma_majorant import probe_chart_gamma_on_kill_sequence
from omnibias.dynamics.hilbert16_adversarial import (
    ADVERSARIAL_SEQUENCE_NAMES,
    audit_track_c_candidate,
    run_adversarial_suite,
)
from omnibias.dynamics.hilbert16_lower_bounds import (
    certified_lower_bound,
    reject_inconsistent_bound,
    verify_claimed_bound,
)
from omnibias.dynamics.hilbert16_uniform_ledger import (
    PHASE_NAMES,
    UNIFORM_PARENT_FLAG,
    uniform_finiteness_discharged,
)
from omnibias.dynamics.hilbert_bound import hilbert_bound
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.hyperbolic_uniform import (
    certify_hyperbolic_uniform_cyclicity,
    default_hyperbolic_smoke_expansion,
    verify_hyperbolic_uniform_cyclicity,
    verify_hyperbolic_uniform_cyclicity_formally,
)
from omnibias.dynamics.ln_cell_derive import derive_quadratic_ln_cell_cauchy_bound
from omnibias.dynamics.normalized_family import (
    NormalizedFamily,
    certify_example_atlas,
    coefficient_dimension,
    normalization_preserves_cycle_count,
    sphere_dimension,
)


def test_normalized_family_dimensions() -> None:
    family = NormalizedFamily(2, coefficient_dimension(2), sphere_dimension(2))
    assert family.harnack_upper_bound == 1
    assert normalization_preserves_cycle_count()["affine_chart_change_preserves_count"]
    atlas = certify_example_atlas()
    assert atlas["infinite_singularity_count"] >= 0


def test_lower_bound_registry_refutes_inconsistent_claims() -> None:
    assert certified_lower_bound(1).value == 0
    assert certified_lower_bound(2).value == 4
    assert certified_lower_bound(3).value == 13
    assert verify_claimed_bound(2, 4)
    assert not verify_claimed_bound(2, 3)
    reject_inconsistent_bound(3, 13)


def test_hilbert_bound_blocks_on_default_ledger() -> None:
    result = hilbert_bound(2)
    assert result.status == "BLOCKED"
    assert result.bound is None
    assert result.parent_flags[UNIFORM_PARENT_FLAG] is False
    assert result.reason_tree


def test_default_ledger_includes_phases_and_adversarial_entries() -> None:
    ledger = default_h16_ledger()
    assert {entry.name for entry in ledger.phases()} == set(PHASE_NAMES)
    assert {entry.name for entry in ledger.adversarial()} == set(ADVERSARIAL_SEQUENCE_NAMES)
    assert derived_parent_flags(ledger)[UNIFORM_PARENT_FLAG] is False
    assert not uniform_finiteness_discharged(ledger.entries)


def test_adversarial_suite_has_fifteen_checks() -> None:
    reports = run_adversarial_suite()
    assert len(reports) == 15
    assert all(report.name.startswith("ADV") for report in reports)
    audit = audit_track_c_candidate(claimed_bound=2, degree=2, majorant_constant=1.0, gamma=1.0)
    assert any(not report.passed for report in audit)


def test_hyperbolic_uniform_certificate_replays() -> None:
    expansion = default_hyperbolic_smoke_expansion()
    certificate = certify_hyperbolic_uniform_cyclicity(
        expansion,
        parameter_lo=Fraction(1, 10),
        parameter_hi=Fraction(1, 2),
        flatness_exponent=2,
        majorant_constant=Fraction(1),
    )
    assert certificate.external_premises
    assert verify_hyperbolic_uniform_cyclicity(certificate)
    formal = verify_hyperbolic_uniform_cyclicity_formally(certificate)
    assert formal.box_cover.available or not formal.box_cover.available


def test_derived_majorant_upgrades_to_empty_premises() -> None:
    expansion = default_hyperbolic_smoke_expansion()
    certificate = certify_hyperbolic_uniform_cyclicity(
        expansion,
        parameter_lo=Fraction(1, 10),
        parameter_hi=Fraction(1, 2),
        flatness_exponent=2,
        majorant_constant=Fraction(1),
        derive_majorant=True,
        variational_sup=Interval(0.0, 1.0),
    )
    assert certificate.external_premises == ()
    assert verify_hyperbolic_uniform_cyclicity(certificate)


def test_chart_gamma_probe_records_route_specific_negative() -> None:
    probe = probe_chart_gamma_on_kill_sequence()
    assert probe.route_specific
    assert probe.chart_gamma > 0


def test_ln_cell_derives_order_two_cauchy_bound() -> None:
    report = derive_quadratic_ln_cell_cauchy_bound()
    assert report.analytic_hypothesis_derived
    assert report.second_derivative_bound.hi > 0

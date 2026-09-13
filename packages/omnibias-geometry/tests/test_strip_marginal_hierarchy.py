# SPDX-License-Identifier: Apache-2.0
"""Exact source, refusal and scope regressions for strip marginal closure."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import strip_marginal_hierarchy as module
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.strip_marginal_hierarchy import (
    replay_su2_strip_marginal_hierarchy_certificate as replay,
)
from omnibias.geometry.gauge.transfer.strip_marginal_hierarchy import (
    su2_strip_compression_geometry as geometry,
)
from omnibias.geometry.gauge.transfer.strip_marginal_hierarchy import (
    su2_strip_marginal_hierarchy as certify,
)


def _good() -> dict[str, Any]:
    return certify(24, correction_radius=Q(1, 8), decay_base=Q(5, 4))


def test_actual_weighted_source_and_uniform_margin_are_independently_exact() -> None:
    result = _good()
    w, a = result["witness"], result["witness"]["arithmetic"]
    source = w["source_certificate"]["payload"]["witness"]
    sa = source["arithmetic"]
    g, b, r, rho = Q(1, 144), Q(5, 4), Q(1, 8), Q(1, 2)
    forcing = 16 * g * b**2
    assert Q(sa["seed_norm_upper"]) == forcing == Q(25, 144)
    assert r - Q(4, 3) * (forcing + r)**2 == Q(sa["self_map_slack"]) == Q(95, 15552)
    assert Q(8, 3) * (forcing + r) == Q(sa["contraction_upper"]) == Q(43, 54)
    m = g * (1 + b) / 3 + 2 * r / 3
    assert Q(a["actual_weighted_hessian_row_upper"]) == m == Q(17, 192)
    assert Q(sa["actual_log_vacuum_hessian_row_upper"]) != m
    assert Q(a["uniform_marginal_curvature_lower"]) == rho - 2 * m == Q(31, 96)
    assert Q(a["weighted_covariance_kernel_row_upper"]) == 1 / (rho - 2 * m) == Q(96, 31)
    assert Q(result["all_compressed_physical_gap_lower"]) == 12 * (rho - 2 * m) == Q(31, 8)
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


@pytest.mark.parametrize("kappa, radius, base", [(20, Q(1, 8), Q(5, 4)), (24, Q(1, 8), Q(2)),
                                               (100, Q(1), Q(1))])
def test_source_failure_never_becomes_an_actual_marginal_bound(kappa: int, radius: Q, base: Q) -> None:
    result = certify(kappa, correction_radius=radius, decay_base=base)
    assert result["status"] == "INCONCLUSIVE"
    assert "actual_vacuum_fixed_point" in result["witness"]["failed_constraints"]
    assert result["all_compressed_physical_gap_lower"] is None
    assert result["actual_weighted_hessian_bound_verified"] is False
    assert result["arbitrary_depth_strip_marginal_closure_verified"] is False
    assert result["witness"]["arithmetic"]["actual_weighted_hessian_row_upper"] is None
    assert replay(result["certificate"])


def test_unweighted_ball_does_not_earn_spatial_exponential_decay() -> None:
    result = certify(24, correction_radius=Q(1, 8))
    assert result["status"] == "PASS"
    assert result["spatial_exponential_covariance_bound_verified"] is False
    assert result["witness"]["arithmetic"]["tail_rate"] is None


@pytest.mark.parametrize("kappa, base", [(19, Q(1)), (24, Q(5, 4)), (40, Q(2)), (100, Q(5))])
def test_automatic_radius_is_a_constructive_witness_for_the_exact_criterion(kappa: int, base: Q) -> None:
    result = certify(kappa, decay_base=base)
    a = result["witness"]["arithmetic"]
    forcing = 64 * base**2 / kappa**2
    assert Q(result["witness"]["inputs"]["correction_radius"]) == forcing
    assert Q(a["source_radius_feasibility_slack"]) == 3 * kappa**2 - 1024 * base**2 > 0
    assert a["source_radius_exists_for_criterion"] is True
    assert result["status"] == "PASS"
    assert Q(a["actual_weighted_hessian_row_upper"]) < Q(17, 128)
    assert replay(result["certificate"])


@pytest.mark.parametrize("kappa, base", [(18, Q(1)), (23, Q(5, 4)), (30, Q(2))])
def test_automatic_refusal_excludes_only_the_named_source_criterion(kappa: int, base: Q) -> None:
    result = certify(kappa, decay_base=base)
    a = result["witness"]["arithmetic"]
    assert a["source_radius_exists_for_criterion"] is False
    assert Q(a["source_radius_feasibility_slack"]) < 0
    assert result["status"] == "INCONCLUSIVE"
    assert result["all_compressed_physical_gap_lower"] is None
    assert result["yang_mills_mass_gap_claim"] is False
    assert replay(result["certificate"])


def test_source_gap_fields_are_not_a_premise(monkeypatch: pytest.MonkeyPatch) -> None:
    original = invariant_vacuum_fourier_family

    def erase_gaps(*args: Any, **kwargs: Any) -> dict[str, Any]:
        source = original(*args, **kwargs)
        for key in list(source["witness"]["arithmetic"]):
            if "gap" in key or "factorization" in key:
                source["witness"]["arithmetic"].pop(key)
        source["status"] = "INCONCLUSIVE"
        return source

    monkeypatch.setattr(module, "invariant_vacuum_fourier_family", erase_gaps)
    result = _good()
    assert result["status"] == "PASS"
    assert result["all_compressed_physical_gap_lower"] == "31/8"


def test_strict_row_gate_remains_separate_from_source_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    original = invariant_vacuum_fourier_family

    def isolate_row_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        source = original(*args, **kwargs)
        source["witness"]["arithmetic"]["fixed_point_verified"] = True
        return source

    monkeypatch.setattr(module, "invariant_vacuum_fourier_family", isolate_row_gate)
    result = certify(100, correction_radius=Q(1))
    assert result["status"] == "INCONCLUSIVE"
    assert result["witness"]["failed_constraints"] == ["strict_weighted_hessian_row_ball"]
    assert result["all_compressed_physical_gap_lower"] is None


@pytest.mark.parametrize("key", ["continuum_claim", "uniform_in_a_claim", "infinite_volume_claim",
                               "yang_mills_claim", "yang_mills_mass_gap_claim", "all_scale_refinement_claim"])
def test_resealed_parent_flag_promotion_is_rejected(key: str) -> None:
    result = _good()
    assert result[key] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][key] = True
    assert not replay(seal_certificate(certificate))


def test_resealed_derivative_or_source_changes_are_rejected() -> None:
    for key in ("actual_weighted_hessian_row_upper", "all_compressed_physical_gap_lower"):
        certificate = deepcopy(_good()["certificate"])
        certificate["payload"]["witness"]["arithmetic"][key] = "0"
        assert not replay(seal_certificate(certificate))
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["source_certificate"]["payload"]["witness"]["decay_base"] = "1"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("value", [True, 24.0, "24", 0, -1])
def test_kappa_input_guards(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(value, correction_radius=Q(1, 8))


@pytest.mark.parametrize("field, value", [("correction_radius", True), ("correction_radius", 0.125),
                                         ("correction_radius", 0), ("decay_base", True),
                                         ("decay_base", 1.25), ("decay_base", Q(3, 4))])
def test_exact_positive_parameter_guards(field: str, value: Any) -> None:
    args: dict[str, Any] = {"correction_radius": Q(1, 8), "decay_base": Q(5, 4)}
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        certify(24, **args)


def test_compressed_geometry_uses_original_lengths_not_unit_coarse_edges() -> None:
    n = 10**6
    result = geometry(n, [0, 3, 11, n])
    assert result["cut_weights"] == [3, 8, n - 11]
    assert sum(result["cut_weights"]) == n
    assert result["vertical_weights"] == ["1"] * 4
    assert result["distances"][0][-1] == n
    assert result["n_original_edges"] == 3 * n + 1
    assert result["physical_kinetic_floor"] == "1"


@pytest.mark.parametrize("n, retained", [(0, [0]), (True, [0, 1]), (4.0, [0, 4]),
                                        (4, [1, 4]), (4, [0, 3]), (4, [0, 2, 2, 4]),
                                        (4, [0, 3, 2, 4]), (4, [0, True, 4]),
                                        (4, [0, Q(2), 4]), (4, [0, 6, 4])])
def test_geometry_refuses_invalid_or_implicit_coarse_charts(n: Any, retained: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        geometry(n, retained)


@pytest.mark.parametrize("certificate", [None, [], {}, {"payload": None}])
def test_replay_malformed_input_is_false(certificate: Any) -> None:
    assert replay(certificate) is False

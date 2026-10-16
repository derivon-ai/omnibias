# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Projected group evidence, covariance corrections and query reuse contracts."""

from copy import deepcopy
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import certificate_is_unconditional, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_grouped_directional import (
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_projected_equilibrium import (
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.interval import Interval


def _equilibrium(budget=None):
    source = GaussianHeatMap(((-1, 0, 0), (1, 0, 0), (0, -1, 4), (0, 1, 4)),
                             (F(1, 4), F(1, 4), F(21, 4), F(21, 4)), budget=budget)
    base = restrict_gaussian_heat(source, ((0, 0, 1),), (0,))
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1, 2, 0), (0, 1, 0)), (0, 0), (0,))
    assert face is not None
    result = certify_gaussian_projected_equilibrium(face, (1, 2, 0), (0, 0, 0))
    assert result is not None
    return result


def test_compile_reuses_sealed_weights_without_transcendentals_and_query_never_solves(monkeypatch):
    equilibrium = _equilibrium()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")

    def fail(*args, **kwargs):
        pytest.fail("compiled projected evidence was not reused")

    monkeypatch.setattr(module, "_probabilities", fail)
    monkeypatch.setattr(module, "_strict", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    face = compile_gaussian_grouped_critical_face(equilibrium)
    assert face is not None
    assert face.posterior is equilibrium.posterior
    assert face.group_probabilities is equilibrium.group_probabilities
    assert face.certificate["meta"]["transcend_backend"] == "mpmath"
    for name in ("solve_affine", "factor_positive_definite", "_moment", "_between_group_covariance"):
        monkeypatch.setattr(module, name, fail)
    result = certify_gaussian_grouped_directional(face, (1, 0, 0), (1, -0.5, 0))
    assert result is not None and result.error_bound > 0
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert certificate_is_unconditional(result.certificate)
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


def test_strict_sealing_policy_still_requires_an_available_backend(monkeypatch):
    equilibrium = _equilibrium()
    face = compile_gaussian_grouped_critical_face(equilibrium)
    assert face is not None
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_grouped_critical_face(equilibrium)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_grouped_directional(face, (1, 0, 0), (1, -0.5, 0))
    assert transcend.strict_backend()


def test_projected_centroid_identity_is_replayed_before_curvature(monkeypatch):
    equilibrium = _equilibrium()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "_linear", lambda *_: (F(1), F(0), F(0)))
    monkeypatch.setattr(module, "_moment", lambda *_: pytest.fail("failed projected identity reached curvature"))
    assert compile_gaussian_grouped_critical_face(equilibrium) is None


@pytest.mark.parametrize("dependency", ["equilibrium", "face", "base"])
def test_live_projected_dependencies_are_checked_before_geometry_or_queries(dependency, monkeypatch):
    equilibrium = _equilibrium()
    critical = compile_gaussian_grouped_critical_face(equilibrium)
    assert critical is not None
    target = {"equilibrium": equilibrium, "face": equilibrium.face,
              "base": equilibrium.face.base_restriction}[dependency]
    target.certificate["payload"]["damage"] = object()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "solve_affine", lambda *_: pytest.fail("damaged evidence reached geometry"))
    monkeypatch.setattr(module, "_linear", lambda *_: pytest.fail("damaged evidence reached query actions"))
    assert compile_gaussian_grouped_critical_face(equilibrium) is None
    assert certify_gaussian_grouped_directional(critical, (1, 0, 0), (1, -0.5, 0)) is None


def test_coefficient_proven_strong_row_stays_strong_when_probability_underflows():
    source = GaussianHeatMap(((0,), (1,)), (0, F(1, 4)-1000))
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    primal_face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), (0,))
    assert primal_face is not None
    equilibrium = certify_gaussian_projected_equilibrium(primal_face, (0,), (0,))
    assert equilibrium is not None
    assert equilibrium.group_multipliers == ((0,), (1,))
    assert equilibrium.inequality_multipliers[0].lo == 0
    assert equilibrium.strong_indices == (0,) and equilibrium.weak_indices == ()
    critical = compile_gaussian_grouped_critical_face(equilibrium)
    assert critical is not None and critical.projector == ((0,),)
    for direction in ((1,), (-1,)):
        result = certify_gaussian_grouped_directional(critical, direction, (99,))
        assert result is not None and result.value == (0,)
        assert result.enclosure == (Interval.point(0),) and result.error_bound == 0
        assert result.equality_multipliers == (Interval.point(direction[0]),)


def test_full_hessian_retains_between_group_covariance_while_projected_blocks_do_not():
    equilibrium = _equilibrium()
    face = compile_gaussian_grouped_critical_face(equilibrium)
    assert face is not None
    # Every within-group z variance is zero. The true negative full-space
    # normal curvature therefore depends entirely on the between-group term.
    assert all(row[2] == 0 for row in equilibrium.centered_centers)
    assert face.hessian[2][2].hi < 0
    assert face.projected_hessian[2] == (Interval.point(0),)*3
    assert face.multiplier_hessian[0] == (Interval.point(0),)*3
    assert face.hessian[0][2] == face.hessian[2][0] == Interval.point(0)


def test_projected_compile_extra_storage_preflight_precedes_geometry(monkeypatch):
    equilibrium = _equilibrium(RidgeBudget(max_matrix_entries=40000))
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "solve_affine", lambda *_: pytest.fail("projected storage preflight missed geometry"))
    assert compile_gaussian_grouped_critical_face(equilibrium) is None


@pytest.mark.parametrize("budget", [RidgeBudget(max_matrix_entries=70000), RidgeBudget(max_elimination_work=650000)])
def test_projected_query_extra_evidence_and_work_are_preflighted(budget, monkeypatch):
    equilibrium = _equilibrium(budget)
    critical = compile_gaussian_grouped_critical_face(equilibrium)
    assert critical is not None
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "_linear", lambda *_: pytest.fail("projected query preflight missed matrix action"))
    assert certify_gaussian_grouped_directional(critical, (1, 0, 0), (1, -0.5, 0)) is None


def test_projected_certificate_dependencies_are_snapshots():
    equilibrium = _equilibrium()
    critical = compile_gaussian_grouped_critical_face(equilibrium)
    assert critical is not None
    result = certify_gaussian_grouped_directional(critical, (1, 0, 0), (1, -0.5, 0))
    assert result is not None
    saved = deepcopy(result.certificate)
    equilibrium.certificate["payload"]["damage"] = object()
    critical.certificate["payload"]["damage"] = object()
    assert saved == result.certificate and verify_certificate_digest(result.certificate)

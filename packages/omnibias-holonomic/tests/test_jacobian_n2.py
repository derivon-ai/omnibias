# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Step A: finite lie for Jacobian n=2 — polarity, honesty, box scope."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
from omnibias.core.proof import Conjecture, ExactCheck, catalog_entry, discover, run_discovery
from omnibias.core.proof.discovery import DiscoveryResult
from omnibias.holonomic._core.poly_n import PolyN, identical_jacobian_constant
from omnibias.holonomic.jacobian_n2 import (
    CI_COEFF_HEIGHT,
    CI_HOMOG_DEGREE,
    CI_HOMOG_HEIGHT,
    CI_MAX_DEGREE,
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    JACOBIAN_N2_HOMOG_KIND,
    JACOBIAN_N2_KIND,
    JACOBIAN_N2_PARENT,
    MOH_DEGREE_BOUND,
    JacobianN2DegreeFamily,
    JacobianN2HomogeneousFamily,
    default_collision_grid,
    escalate_n2_result,
    identity_plus_homogeneous,
    jacobian_n2_box_statement,
    jacobian_n2_homog_statement,
    jacobian_n2_honesty,
    n2_counterexample_earned,
    n2_violation_payload,
    plane_monomials,
    rational_grid_collision,
    reject_jacobian_proof_claim,
    seal_jacobian_honesty,
)


def test_statement_is_universal_and_parent_stays_open() -> None:
    statement = jacobian_n2_box_statement(max_degree=1, coeff_height=1)
    assert statement.existential is False
    assert statement.parent == JACOBIAN_N2_PARENT
    assert statement.parent_status == "open"
    assert statement.name == f"{JACOBIAN_N2_KIND}_d1_h1"
    assert "identically a nonzero constant" in statement.obligation
    assert "not injectivity on Q^2" in statement.obligation
    assert "is not the parent" in statement.obligation
    assert str(MOH_DEGREE_BOUND) in statement.obligation
    assert "height 1" in statement.obligation
    assert "degree 1" in statement.obligation


def test_statement_embeds_the_collision_grid() -> None:
    grid = (Fraction(-1), Fraction(0), Fraction(1, 2))
    statement = jacobian_n2_box_statement(max_degree=2, coeff_height=3, grid=grid)
    assert statement.name == f"{JACOBIAN_N2_KIND}_d2_h3"
    assert "1/2" in statement.obligation
    assert "height 3" in statement.obligation


def test_ci_box_is_below_moh_bound() -> None:
    assert CI_MAX_DEGREE < MOH_DEGREE_BOUND
    assert CI_COEFF_HEIGHT >= 1
    assert MOH_DEGREE_BOUND == 100


def test_default_grid_is_symmetric_integers() -> None:
    axis = default_collision_grid()
    assert axis[0] == Fraction(-2)
    assert axis[-1] == Fraction(2)
    assert len(axis) == 5


def test_honesty_miss_never_earns_parent_or_n2() -> None:
    honesty = jacobian_n2_honesty(discovered=False)
    assert honesty["discovered_by_omnibias"] is False
    assert honesty["jacobian_n2_claim"] is False
    assert honesty["jacobian_conjecture_proof_claim"] is False
    assert honesty["moh_degree_100_settled_claim"] is False
    assert honesty["no_condition_exists_claim"] is False


def test_honesty_counterexample_disproves_and_does_not_prove() -> None:
    honesty = jacobian_n2_honesty(discovered=True, n2_counterexample=True)
    assert honesty["discovered_by_omnibias"] is True
    assert honesty["jacobian_n2_claim"] is True
    assert honesty["jacobian_conjecture_proof_claim"] is False


def test_honesty_rejects_counterexample_without_discovery() -> None:
    with pytest.raises(ValueError, match="n2_counterexample"):
        jacobian_n2_honesty(discovered=False, n2_counterexample=True)


def test_proof_claim_unearned_until_gate() -> None:
    assert JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED is False
    with pytest.raises(ValueError, match="unearned"):
        reject_jacobian_proof_claim({"jacobian_conjecture_proof_claim": True})
    sealed = seal_jacobian_honesty({"jacobian_conjecture_proof_claim": False, "jacobian_n2_claim": True})
    assert sealed["jacobian_conjecture_proof_claim"] is False
    with pytest.raises(TypeError):
        sealed["jacobian_conjecture_proof_claim"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unearned"):
        seal_jacobian_honesty({"jacobian_conjecture_proof_claim": True})


def test_n2_counterexample_requires_identical_jac_and_collision() -> None:
    assert n2_counterexample_earned({}) is False
    assert (
        n2_counterexample_earned(
            {
                "jacobian_identity": "probe",
                "jacobian_nonzero_constant": True,
                "rational_preimages": [[0, 0], [1, 0]],
                "rational_image": [0, 0],
            }
        )
        is False
    )
    assert (
        n2_counterexample_earned(
            {
                "jacobian_identity": "identical",
                "jacobian_nonzero_constant": False,
                "rational_preimages": [[0, 0], [1, 0]],
                "rational_image": [0, 0],
            }
        )
        is False
    )
    assert (
        n2_counterexample_earned(
            {
                "jacobian_identity": "identical",
                "jacobian_nonzero_constant": True,
                "rational_preimages": [[0, 0]],
                "rational_image": [0, 0],
            }
        )
        is False
    )
    assert (
        n2_counterexample_earned(
            {
                "jacobian_identity": "identical",
                "jacobian_nonzero_constant": True,
                "rational_preimages": [[0, 0], [0, 0]],
                "rational_image": [0, 0],
            }
        )
        is False
    )
    assert (
        n2_counterexample_earned(
            {
                "jacobian_identity": "identical",
                "jacobian_nonzero_constant": True,
                "rational_preimages": [[0, 1], ["1/2", -1]],
                "rational_image": [3, 4],
            }
        )
        is True
    )


def _xy() -> tuple[PolyN, PolyN]:
    return PolyN.var(2, 0), PolyN.var(2, 1)


def test_shear_has_identical_unit_jacobian_and_no_grid_collision() -> None:
    x, y = _xy()
    shear = (x, y + x**2)
    assert identical_jacobian_constant(shear) == 1
    assert rational_grid_collision(shear, default_collision_grid()) is None
    payload = n2_violation_payload(shear)
    assert payload["jacobian_identity"] == "identical"
    assert payload["jacobian_nonzero_constant"] is True
    assert n2_counterexample_earned(payload) is False
    assert payload["honesty"]["jacobian_n2_claim"] is False


def test_folding_map_collides_but_jacobian_is_not_constant() -> None:
    x, y = _xy()
    folding = (x**2, y)
    assert identical_jacobian_constant(folding) is None
    collision = rational_grid_collision(folding, default_collision_grid())
    assert collision is not None
    image, preimages = collision
    assert len(set(preimages)) == 2
    assert image[1] == preimages[0][1]
    payload = n2_violation_payload(folding)
    assert payload["jacobian_nonzero_constant"] is False
    assert n2_counterexample_earned(payload) is False
    assert payload["honesty"]["jacobian_n2_claim"] is False


def test_singular_projection_is_not_a_counterexample() -> None:
    x, y = _xy()
    projection = (x, PolyN.zero(2))
    assert identical_jacobian_constant(projection) == 0
    payload = n2_violation_payload(projection)
    assert payload["jacobian_nonzero_constant"] is False
    assert n2_counterexample_earned(payload) is False


def test_axis_probes_can_lie_when_det_is_1_plus_y() -> None:
    x, y = _xy()
    # det JF = 1+y. Every probe with y=0 returns 1; the polynomial is not constant.
    skewed = (x + x * y, y)
    from omnibias.holonomic._core.poly_n import jacobian_det

    det = jacobian_det(skewed)
    assert det.constant_value() is None
    assert identical_jacobian_constant(skewed) is None
    axis_probes = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(2), Fraction(0)),
        (Fraction(-1), Fraction(0)),
    )
    samples = {det.eval(pt) for pt in axis_probes}
    assert samples == {1}


def test_linear_automorphism_is_not_a_counterexample() -> None:
    x, y = _xy()
    linear = (x + y, y)
    payload = n2_violation_payload(linear)
    assert payload["jacobian_constant"] == "1"
    assert rational_grid_collision(linear, default_collision_grid()) is None
    assert n2_counterexample_earned(payload) is False


def test_plane_monomials_are_total_degree() -> None:
    assert plane_monomials(0) == ((0, 0),)
    assert set(plane_monomials(1)) == {(0, 0), (1, 0), (0, 1)}
    assert plane_monomials(1)[0] == (0, 0)
    assert (2, 0) in plane_monomials(2)
    assert (1, 1) in plane_monomials(2)


def test_degree_zero_box_proves_finite_universal() -> None:
    family = JacobianN2DegreeFamily(max_degree=0, coeff_height=1)
    assert family.complete is True
    assert family.cardinality() == 9
    assert family.statement.existential is False
    identity_check = family.check((0, 1, 0, 0, 0, 1))
    assert identity_check is None
    result = run_discovery(family.statement, family, "score_guided", budget=16)
    assert result.status == "PROVED"
    assert result.detail == "no counterexample in complete family"
    assert result.search_incomplete is False
    assert result.characterization is not None
    assert result.characterization.exhausted is True
    assert result.characterization.family_complete is True


def test_degree_one_identity_is_not_a_counterexample() -> None:
    family = JacobianN2DegreeFamily(max_degree=1, coeff_height=1)
    assert family.complete is True
    assert family.cardinality() == 729
    checked = family.check((0, 1, 0, 0, 0, 1))
    assert checked is not None
    assert checked.ok is False
    assert checked.payload["jacobian_constant"] == "1"
    assert checked.payload["honesty"]["jacobian_n2_claim"] is False
    assert family.score((0, 1, 0, 0, 0, 1)) == 2


def test_degree_one_box_proves_finite_universal() -> None:
    family = JacobianN2DegreeFamily(max_degree=1, coeff_height=1)
    result = run_discovery(family.statement, family, "score_guided", budget=800)
    assert result.status == "PROVED"
    assert result.detail == "no counterexample in complete family"
    assert result.evaluated == family.cardinality()
    assert result.search_incomplete is False
    assert result.statement.parent_status == "open"
    assert (result.as_dict().get("honesty") or {}).get("jacobian_n2_claim", False) is False


def test_zero_budget_is_incomplete() -> None:
    family = JacobianN2DegreeFamily(max_degree=0, coeff_height=1)
    result = run_discovery(family.statement, family, "score_guided", budget=0)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True
    assert result.evaluated == 0


def test_degree_two_slice_is_incomplete_and_has_no_hit() -> None:
    family = JacobianN2DegreeFamily(max_degree=2, coeff_height=1)
    assert family.complete is False
    assert family.statement.name.endswith("_slice")
    fold = family.check(("fold", 0))
    assert fold is not None and fold.ok is False
    shear = family.check(("shear", 0, 0, 0, 1))
    assert shear is not None and shear.ok is False
    assert shear.payload["jacobian_constant"] == "1"
    result = run_discovery(family.statement, family, "score_guided", budget=32)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True
    assert result.detail == "search_incomplete"


def test_catalog_and_machine_discharge_degree_zero_box() -> None:
    import omnibias.holonomic.proofmachine as machine_mod
    from omnibias.holonomic.proofmachine import (
        JACOBIAN_N2_DEGREE_BOX,
        _schema_errors,
        build_holonomic_machine,
    )

    assert JACOBIAN_N2_DEGREE_BOX in machine_mod.FAMILY_CATALOG
    entry = catalog_entry(JACOBIAN_N2_DEGREE_BOX)
    assert entry is not None
    assert entry.mode == "exact_search"
    assert entry.parent_status == "open"
    assert entry.complete is True
    found = discover(JACOBIAN_N2_DEGREE_BOX, max_degree=0, coeff_height=1, budget=16)
    assert found.status == "PROVED"
    assert found.detail == "no counterexample in complete family"
    machine = build_holonomic_machine()
    verdict = machine.evaluate(
        Conjecture(
            name="n2-box",
            kind=JACOBIAN_N2_DEGREE_BOX,
            data={"max_degree": 0, "coeff_height": 1, "budget": 16},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.certificate is not None
    assert verdict.certificate.get("honesty", {}).get("jacobian_n2_claim", False) is False
    assert _schema_errors({"honesty": {"jacobian_n2_claim": True}})
    assert _schema_errors({"honesty": {"jacobian_conjecture_proof_claim": True}})
    assert _schema_errors({"honesty": {"jacobian_n2_claim": False}}) == []


def test_escalate_only_on_earned_disproof() -> None:
    family = JacobianN2DegreeFamily(max_degree=0, coeff_height=1)
    proved = run_discovery(family.statement, family, "score_guided", budget=16)
    sealed = escalate_n2_result(proved)
    assert sealed["escalate_parent"] is False
    assert sealed["n2_counterexample"] is False
    assert sealed["honesty"]["jacobian_n2_claim"] is False
    assert sealed["honesty"]["jacobian_conjecture_proof_claim"] is False
    assert "stays open" in sealed["note"]
    fake = DiscoveryResult(
        status="DISPROVED",
        statement=family.statement,
        family=family.name,
        proposer="score_guided",
        budget=1,
        evaluated=1,
        candidate=(0, 0),
        check=ExactCheck(
            ok=True,
            payload={
                "jacobian_identity": "identical",
                "jacobian_nonzero_constant": True,
                "rational_preimages": [[0, 0], [1, 0]],
                "rational_image": [0, 1],
            },
        ),
        detail="counterexample to universal obligation",
    )
    hit = escalate_n2_result(fake)
    assert hit["escalate_parent"] is True
    assert hit["honesty"]["jacobian_n2_claim"] is True
    assert hit["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_cubic_homogeneous_shear_is_not_a_counterexample() -> None:
    family = JacobianN2HomogeneousFamily(degree=3, coeff_height=1)
    assert family.complete is True
    assert family.cardinality() == 6561
    assert family.statement.parent_status == "open"
    assert family.statement.existential is False
    shear = (0, 0, 0, 0, 1, 0, 0, 0)
    decoded = family.decode(shear)
    assert decoded is not None
    x, y = _xy()
    assert decoded == (x, y + x**3)
    checked = family.check(shear)
    assert checked is not None
    assert checked.ok is False
    assert checked.payload["gabber_inverse_ok"] is True
    assert checked.payload["honesty"]["jacobian_n2_claim"] is False


def test_homog_statement_does_not_claim_the_parent() -> None:
    statement = jacobian_n2_homog_statement(degree=3, coeff_height=1)
    assert statement.name == f"{JACOBIAN_N2_HOMOG_KIND}_d3_h1"
    assert "not jacobian_conjecture_n2" in statement.obligation
    assert CI_HOMOG_DEGREE == 3
    assert CI_HOMOG_HEIGHT == 1


def test_identity_plus_homogeneous_roundtrip() -> None:
    x, y = _xy()
    mapped = identity_plus_homogeneous((0, 0, 0, 1, 0, 0, 0, 0), degree=3)
    assert mapped == (x + y**3, y)


def test_cubic_homogeneous_height_one_box_has_no_violator() -> None:
    family = JacobianN2HomogeneousFamily(degree=3, coeff_height=1)
    keller = 0
    for coeffs in product((-1, 0, 1), repeat=8):
        checked = family.check(coeffs)
        assert checked is not None
        assert checked.ok is False
        if checked.payload["jacobian_nonzero_constant"]:
            keller += 1
            assert checked.payload["gabber_inverse_ok"] is True
    assert keller == 5


def test_homog_zero_budget_is_incomplete() -> None:
    family = JacobianN2HomogeneousFamily(degree=3, coeff_height=1)
    result = run_discovery(family.statement, family, "score_guided", budget=0)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True


def test_rejects_invalid_box_parameters() -> None:
    with pytest.raises(ValueError, match="max_degree"):
        jacobian_n2_box_statement(max_degree=-1, coeff_height=1)
    with pytest.raises(ValueError, match="coeff_height"):
        jacobian_n2_box_statement(max_degree=1, coeff_height=-1)
    with pytest.raises(ValueError, match="halfwidth"):
        default_collision_grid(halfwidth=-1)
    with pytest.raises(ValueError, match="non-empty"):
        jacobian_n2_box_statement(max_degree=1, coeff_height=1, grid=())

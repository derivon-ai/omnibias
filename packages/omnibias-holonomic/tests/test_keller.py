# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Alpöge / Gallagher replay, blind tangent sweep, and fiber degree."""

from __future__ import annotations

import ast
from pathlib import Path

from omnibias.core.proof import Conjecture
from omnibias.holonomic._core.poly_n import PolyN, jacobian_det
from omnibias.holonomic._core.rational_poly import to_poly
from omnibias.holonomic.keller import (
    alpoge_map,
    fiber_report,
    matches_known_replay,
    search_hit_certificate,
    verify_alpoge_map,
    verify_gallagher_map,
)
from omnibias.holonomic.keller_search import (
    build_sweep_map,
    generic_fiber_degree,
    search_tangent_sweep,
    solve_side_conditions,
    tangency_polynomial,
)
from omnibias.holonomic.proofmachine import (
    FAMILY_CATALOG,
    KELLER_ALPOGE_REPLAY,
    KELLER_TANGENT_SWEEP,
    KELLER_TANGENT_SWEEP_DEG3,
    build_holonomic_machine,
)


def test_alpoge_replay_identities() -> None:
    cert = verify_alpoge_map()
    assert cert.replay_ok
    assert cert.jacobian_constant == -2
    assert cert.honesty["keller_n_ge_3_replay"] is True
    assert cert.honesty["discovered_by_omnibias"] is False
    assert cert.honesty["jacobian_conjecture_proof_claim"] is False
    assert cert.honesty["jacobian_n2_claim"] is False


def test_alpoge_sweep_constructor_matches_replay() -> None:
    replay = alpoge_map()
    built = build_sweep_map(to_poly([0, 4, -3]), 2, -3, -1, component_order="alpoge")
    assert built is not None
    assert built == replay
    assert jacobian_det(built).constant_value() == -2


def test_gallagher_replay() -> None:
    cert = verify_gallagher_map()
    assert cert.replay_ok
    assert cert.jacobian_constant == 2
    assert cert.generic_fiber == 4
    assert cert.honesty["jacobian_conjecture_proof_claim"] is False


def test_side_conditions_recover_gamma0() -> None:
    assert solve_side_conditions(to_poly([0, 4, -3])) == 2


def test_fiber_degree_alpoge() -> None:
    assert generic_fiber_degree(to_poly([0, 4, -3])) == 3
    report = fiber_report()
    assert report["degree"] == 3
    assert report["leading_coeff_constant"] is True
    w = tangency_polynomial(to_poly([0, 4, -3]), 0, 0)
    assert len(w) - 1 == 3


def test_sweep_jacobian_gate_is_identical_not_probes() -> None:
    from omnibias.holonomic.keller_search import _eval_jacobian_constant

    assert _eval_jacobian_constant(alpoge_map()) == -2
    x, y, z = PolyN.var(3, 0), PolyN.var(3, 1), PolyN.var(3, 2)
    # det = 1+z is 1 on the plane z=0; probes there would lie.
    assert _eval_jacobian_constant((x + x * z, y, z)) is None
    text = Path(__file__).resolve().parents[1] / "src/omnibias/holonomic/keller_search.py"
    source = text.read_text(encoding="utf-8")
    assert "identical_jacobian_constant" in source
    assert "def det_at" not in source


def test_blind_sweep_recovers_a_map() -> None:
    hits = search_tangent_sweep(deg_p=2, height=6, max_hits=1)
    assert hits
    hit = hits[0]
    payload = search_hit_certificate(hit)
    assert payload["replay_ok"] is True
    assert payload["honesty"]["jacobian_conjecture_proof_claim"] is False
    if matches_known_replay(hit):
        assert payload["rediscovered_known_witness"] is True
        assert payload["discovered_by_omnibias"] is False
    else:
        assert payload["discovered_by_omnibias"] is True


def test_search_helpers_are_blind() -> None:
    src = Path(__file__).resolve().parents[1] / "src/omnibias/holonomic/keller_search.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    banned = {
        "(1+xy)^3 z + y^2(1+xy)(4+3xy)",
        "2x-3x^2 y-x^3 z",
        "-1/4",
        "13/2",
    }
    literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.add(node.value)
    assert banned.isdisjoint(literals)
    text = src.read_text(encoding="utf-8")
    assert "Fraction(-1, 4)" not in text
    assert "Fraction(13, 2)" not in text
    assert "[0, 4, -3]" not in text.replace(" ", "")


def test_holonomic_machine_alpoge() -> None:
    machine = build_holonomic_machine()
    assert set(machine.kinds()) >= {
        KELLER_ALPOGE_REPLAY,
        KELLER_TANGENT_SWEEP,
        KELLER_TANGENT_SWEEP_DEG3,
    }
    assert FAMILY_CATALOG[KELLER_ALPOGE_REPLAY]["parent_status"] == "already_false"
    assert FAMILY_CATALOG[KELLER_TANGENT_SWEEP_DEG3]["complete"] == "False"
    verdict = machine.evaluate(Conjecture(name="alpoge", kind=KELLER_ALPOGE_REPLAY))
    assert verdict.status == "PROVED"
    assert verdict.replay_ok is True


def test_holonomic_machine_sweep() -> None:
    machine = build_holonomic_machine()
    verdict = machine.evaluate(
        Conjecture(name="sweep", kind=KELLER_TANGENT_SWEEP, data={"height": 6, "max_hits": 1})
    )
    assert verdict.status == "PROVED"
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["jacobian_n2_claim"] is False


def test_deg3_sweep_recovers_constant_jac_map() -> None:
    hits = search_tangent_sweep(deg_p=3, height=4, max_hits=1)
    assert hits
    hit = hits[0]
    assert hit.jacobian_constant != 0
    assert hit.generic_fiber >= 3
    machine = build_holonomic_machine()
    verdict = machine.evaluate(
        Conjecture(name="deg3", kind=KELLER_TANGENT_SWEEP_DEG3, data={"height": 4})
    )
    assert verdict.status == "PROVED"
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_optional_proposers_on_deg2() -> None:
    hits = search_tangent_sweep(
        deg_p=2, height=6, max_hits=1, proposer="coordinate_newton", budget=32
    )
    if not hits:
        hits = search_tangent_sweep(deg_p=2, height=6, max_hits=1, proposer="score_guided")
    assert hits
    assert hits[0].jacobian_constant != 0

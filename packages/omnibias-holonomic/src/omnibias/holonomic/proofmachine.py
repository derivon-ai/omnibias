# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Register Keller replay / tangent-sweep provers on :class:`ProofMachine`.

===============================  ==========================================
kind                             prover
===============================  ==========================================
``keller_alpoge_replay``         :func:`omnibias.holonomic.keller.verify_alpoge_map`
``keller_tangent_sweep``         :func:`omnibias.holonomic.keller_search.search_tangent_sweep` (deg 2)
``keller_tangent_sweep_deg3``    deg-3 sweep via :func:`omnibias.core.proof.run_discovery`
``jacobian_n2_degree_box``       finite universal ``C_box`` via :class:`JacobianN2DegreeFamily`
``jacobian_n2_homogeneous``      ``I+`` homogeneous box + Gabber inverse test
===============================  ==========================================

``PROVED`` certifies the **finite obligation** (Jacobian identity, or a
constant-Jac 3-to-1 map in the sweep family). It does not mean the Jacobian
conjecture is solved. ``parent_status`` is ``already_false`` for ``n>=3``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omnibias.core.proof import (
    CatalogEntry,
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    register_catalog,
    run_discovery,
)
from omnibias.holonomic.conditions import DfiniteConditionFamily, OreConditionFamily
from omnibias.holonomic.families import (
    HolonomicAlgebraicGuessFamily,
    HolonomicDFiniteGuessFamily,
    HolonomicRecurrenceGuessFamily,
    exp_series,
    fibonacci_samples,
    linear_series,
)
from omnibias.holonomic.jacobian_n2 import (
    CI_COEFF_HEIGHT,
    CI_HOMOG_DEGREE,
    CI_HOMOG_HEIGHT,
    CI_MAX_DEGREE,
    JACOBIAN_N2_HOMOG_KIND,
    JACOBIAN_N2_KIND,
    JacobianN2DegreeFamily,
    JacobianN2HomogeneousFamily,
    n2_counterexample_earned,
)
from omnibias.holonomic.keller import (
    search_hit_certificate,
    verify_alpoge_map,
    verify_gallagher_map,
)
from omnibias.holonomic.keller_search import search_tangent_sweep

KELLER_ALPOGE_REPLAY = "keller_alpoge_replay"
KELLER_TANGENT_SWEEP = "keller_tangent_sweep"
KELLER_TANGENT_SWEEP_DEG3 = "keller_tangent_sweep_deg3"
HOLONOMIC_RECURRENCE_GUESS = "holonomic_recurrence_guess"
HOLONOMIC_DFINITE_GUESS = "holonomic_dfinite_guess"
HOLONOMIC_ALGEBRAIC_GUESS = "holonomic_algebraic_guess"
CONDITION_ORE = "condition_ore"
CONDITION_DFINITE = "condition_dfinite"
JACOBIAN_N2_DEGREE_BOX = JACOBIAN_N2_KIND
JACOBIAN_N2_HOMOGENEOUS = JACOBIAN_N2_HOMOG_KIND

FAMILY_CATALOG: dict[str, dict[str, str]] = {
    KELLER_ALPOGE_REPLAY: {
        "parent_status": "already_false",
        "obligation": "det J(F)+2 ≡ 0 and an explicit 3-to-1 rational witness",
    },
    KELLER_TANGENT_SWEEP: {
        "parent_status": "already_false",
        "obligation": "a deg-2 sweep map with constant nonzero Jacobian and a 3-to-1 witness",
        "complete": "False",
    },
    KELLER_TANGENT_SWEEP_DEG3: {
        "parent_status": "already_false",
        "obligation": "a deg-3 sweep map with constant nonzero Jacobian and a multi-to-one fiber",
        "complete": "False",
    },
    HOLONOMIC_RECURRENCE_GUESS: {
        "parent_status": "already_true",
        "obligation": "a prefix-verified P-recurrence in the (order, degree) box",
        "complete": "True",
    },
    HOLONOMIC_DFINITE_GUESS: {
        "parent_status": "already_true",
        "obligation": "a prefix-verified differential annihilator in the degree box",
        "complete": "True",
    },
    HOLONOMIC_ALGEBRAIC_GUESS: {
        "parent_status": "already_true",
        "obligation": "a prefix-verified algebraic equation P(x, y)=0 in the degree box",
        "complete": "True",
    },
    JACOBIAN_N2_DEGREE_BOX: {
        "parent_status": "open",
        "obligation": (
            "every integer-coefficient map in the CI n=2 degree/height box "
            "either has det JF not identically a nonzero constant or has no "
            "G-collision and passes Gabber; a miss is not the Jacobian conjecture"
        ),
        "complete": "True",
    },
    JACOBIAN_N2_HOMOGENEOUS: {
        "parent_status": "open",
        "obligation": (
            "every I+homogeneous integer map in the CI degree/height box "
            "either is not Keller or has a Gabber-degree polynomial inverse; "
            "a miss is not the Jacobian conjecture"
        ),
        "complete": "True",
    },
}


def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _prove_alpoge(conjecture: Conjecture) -> ProofAttempt:
    name = str(conjecture.data.get("name", "alpoge"))
    cert = verify_gallagher_map() if name == "gallagher" else verify_alpoge_map()
    payload = cert.as_dict()
    if not cert.replay_ok:
        return _blocked(f"{cert.name} identities failed")
    return ProofAttempt(status="PROVED", certificate=payload, detail=cert.name)


def _prove_sweep(conjecture: Conjecture) -> ProofAttempt:
    height = int(conjecture.data.get("height", 6))
    max_hits = int(conjecture.data.get("max_hits", 1))
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    hits = search_tangent_sweep(
        deg_p=2, height=height, max_hits=max_hits, proposer=proposer
    )
    if not hits:
        return _blocked("no constant-Jacobian 3-to-1 sweep map in the coefficient box")
    payload = search_hit_certificate(hits[0])
    return ProofAttempt(status="PROVED", certificate=payload, detail=payload["search"])


def _prove_sweep_deg3(conjecture: Conjecture) -> ProofAttempt:
    height = int(conjecture.data.get("height", 4))
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    hits = search_tangent_sweep(deg_p=3, height=height, max_hits=1, proposer=proposer)
    if not hits:
        return _blocked("search_incomplete")
    payload = search_hit_certificate(hits[0])
    return ProofAttempt(status="PROVED", certificate=payload, detail=payload["search"])


def _schema_errors(certificate: Certificate) -> list[str]:
    errors: list[str] = []
    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping):
        return ["honesty must be a mapping"]
    for key in (
        "jacobian_conjecture_proof_claim",
        "navier_stokes_proof_claim",
        "ten_proofs_formalization_claim",
        "no_condition_exists_claim",
        "unnamed_condition_complete_claim",
    ):
        if honesty.get(key):
            errors.append(f"{key} must be False")
    try:
        from omnibias.holonomic.jacobian_n2 import reject_jacobian_proof_claim

        reject_jacobian_proof_claim(honesty)
    except ValueError as exc:
        errors.append(str(exc))
    if honesty.get("jacobian_n2_claim"):
        inner = certificate.get("payload")
        payload = inner if isinstance(inner, Mapping) else certificate
        if not n2_counterexample_earned(payload):
            errors.append("jacobian_n2_claim requires an exact n=2 violator payload")
    return errors


def _replay_alpoge(certificate: Certificate) -> bool | None:
    name = certificate.get("name", "alpoge")
    fresh = verify_gallagher_map() if name == "gallagher" else verify_alpoge_map()
    return fresh.replay_ok and str(fresh.jacobian_constant) == str(
        certificate.get("jacobian_constant")
    )


def _replay_sweep(certificate: Certificate) -> bool | None:
    from fractions import Fraction

    from omnibias.holonomic._core.poly_n import jacobian_det
    from omnibias.holonomic._core.rational_poly import to_poly
    from omnibias.holonomic.keller_search import build_sweep_map

    try:
        p = to_poly([Fraction(c) for c in certificate["p"]])
        built = build_sweep_map(
            p,
            Fraction(certificate["gamma0"]),
            Fraction(certificate["a"]),
            Fraction(certificate["b"]),
            component_order="alpoge",
        )
    except (KeyError, ValueError, TypeError):
        return False
    if built is None:
        return False
    constant = jacobian_det(built).constant_value()
    return constant is not None and str(constant) == str(certificate.get("jacobian_constant"))


def _from_discovery(result: object) -> ProofAttempt:
    payload = result.as_dict()  # type: ignore[attr-defined]
    status = result.status  # type: ignore[attr-defined]
    detail = result.detail  # type: ignore[attr-defined]
    if status == "PROVED":
        return ProofAttempt(status="PROVED", certificate=payload, detail=detail)
    if status == "DISPROVED":
        return ProofAttempt(status="DISPROVED", certificate=payload, detail=detail)
    return _blocked(detail)


def _prove_recurrence_guess(conjecture: Conjecture) -> ProofAttempt:
    family = HolonomicRecurrenceGuessFamily(
        samples=fibonacci_samples(),
        max_order=int(conjecture.data.get("max_order", 2)),
        max_index_degree=int(conjecture.data.get("max_index_degree", 1)),
    )
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    budget = int(conjecture.data.get("budget", 8))
    return _from_discovery(run_discovery(family.statement, family, proposer, budget=budget))


def _prove_dfinite_guess(conjecture: Conjecture) -> ProofAttempt:
    family = HolonomicDFiniteGuessFamily(
        series=exp_series(),
        max_order=int(conjecture.data.get("max_order", 2)),
        max_degree=int(conjecture.data.get("max_degree", 1)),
    )
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    budget = int(conjecture.data.get("budget", 8))
    return _from_discovery(run_discovery(family.statement, family, proposer, budget=budget))


def _jacobian_n2_family(**kwargs: Any) -> JacobianN2DegreeFamily:
    return JacobianN2DegreeFamily(
        max_degree=int(kwargs.get("max_degree", CI_MAX_DEGREE)),
        coeff_height=int(kwargs.get("coeff_height", CI_COEFF_HEIGHT)),
    )


def _discover_jacobian_n2(**kwargs: Any) -> object:
    family = _jacobian_n2_family(**kwargs)
    budget = int(kwargs.get("budget", family.cardinality() + 8))
    return run_discovery(
        family.statement,
        family,
        str(kwargs.get("proposer", "score_guided")),
        budget=budget,
    )


def _prove_jacobian_n2(conjecture: Conjecture) -> ProofAttempt:
    return _from_discovery(_discover_jacobian_n2(**dict(conjecture.data)))


def _jacobian_n2_homog_family(**kwargs: Any) -> JacobianN2HomogeneousFamily:
    return JacobianN2HomogeneousFamily(
        degree=int(kwargs.get("degree", CI_HOMOG_DEGREE)),
        coeff_height=int(kwargs.get("coeff_height", CI_HOMOG_HEIGHT)),
    )


def _discover_jacobian_n2_homog(**kwargs: Any) -> object:
    family = _jacobian_n2_homog_family(**kwargs)
    budget = int(kwargs.get("budget", family.cardinality() + 8))
    return run_discovery(
        family.statement,
        family,
        str(kwargs.get("proposer", "score_guided")),
        budget=budget,
    )


def _prove_jacobian_n2_homog(conjecture: Conjecture) -> ProofAttempt:
    return _from_discovery(_discover_jacobian_n2_homog(**dict(conjecture.data)))


def _prove_algebraic_guess(conjecture: Conjecture) -> ProofAttempt:
    family = HolonomicAlgebraicGuessFamily(
        series=linear_series(),
        max_x_degree=int(conjecture.data.get("max_x_degree", 1)),
        max_y_degree=int(conjecture.data.get("max_y_degree", 1)),
    )
    proposer = str(conjecture.data.get("proposer", "score_guided"))
    budget = int(conjecture.data.get("budget", 6))
    return _from_discovery(run_discovery(family.statement, family, proposer, budget=budget))


def holonomic_provers() -> list[FunctionProver]:
    return [
        FunctionProver(
            name="keller_alpoge_replay",
            kinds=frozenset({KELLER_ALPOGE_REPLAY}),
            prove_fn=_prove_alpoge,
            schema_fn=_schema_errors,
            replay_fn=_replay_alpoge,
        ),
        FunctionProver(
            name="keller_tangent_sweep",
            kinds=frozenset({KELLER_TANGENT_SWEEP}),
            prove_fn=_prove_sweep,
            schema_fn=_schema_errors,
            replay_fn=_replay_sweep,
        ),
        FunctionProver(
            name="keller_tangent_sweep_deg3",
            kinds=frozenset({KELLER_TANGENT_SWEEP_DEG3}),
            prove_fn=_prove_sweep_deg3,
            schema_fn=_schema_errors,
            replay_fn=_replay_sweep,
        ),
        FunctionProver(
            name=HOLONOMIC_RECURRENCE_GUESS,
            kinds=frozenset({HOLONOMIC_RECURRENCE_GUESS}),
            prove_fn=_prove_recurrence_guess,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name=HOLONOMIC_DFINITE_GUESS,
            kinds=frozenset({HOLONOMIC_DFINITE_GUESS}),
            prove_fn=_prove_dfinite_guess,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
        FunctionProver(
            name=JACOBIAN_N2_DEGREE_BOX,
            kinds=frozenset({JACOBIAN_N2_DEGREE_BOX}),
            prove_fn=_prove_jacobian_n2,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")) and not c.get("honesty", {}).get(
                "jacobian_conjecture_proof_claim"
            ),
        ),
        FunctionProver(
            name=JACOBIAN_N2_HOMOGENEOUS,
            kinds=frozenset({JACOBIAN_N2_HOMOGENEOUS}),
            prove_fn=_prove_jacobian_n2_homog,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")) and not c.get("honesty", {}).get(
                "jacobian_conjecture_proof_claim"
            ),
        ),
        FunctionProver(
            name=HOLONOMIC_ALGEBRAIC_GUESS,
            kinds=frozenset({HOLONOMIC_ALGEBRAIC_GUESS}),
            prove_fn=_prove_algebraic_guess,
            schema_fn=_schema_errors,
            replay_fn=lambda c: bool(c.get("replay_ok")),
        ),
    ]


def build_holonomic_machine() -> ProofMachine:
    machine = ProofMachine()
    for prover in holonomic_provers():
        machine.register(prover)
    return machine


def _register() -> None:
    modes = {
        KELLER_ALPOGE_REPLAY: "exact_replay",
        KELLER_TANGENT_SWEEP: "exact_search",
        KELLER_TANGENT_SWEEP_DEG3: "exact_search",
        HOLONOMIC_RECURRENCE_GUESS: "exact_search",
        HOLONOMIC_DFINITE_GUESS: "exact_search",
        HOLONOMIC_ALGEBRAIC_GUESS: "exact_search",
        JACOBIAN_N2_DEGREE_BOX: "exact_search",
        JACOBIAN_N2_HOMOGENEOUS: "exact_search",
    }
    parents = {
        KELLER_ALPOGE_REPLAY: "Jacobian conjecture n>=3",
        KELLER_TANGENT_SWEEP: "Jacobian conjecture n>=3",
        KELLER_TANGENT_SWEEP_DEG3: "Jacobian conjecture n>=3",
        HOLONOMIC_RECURRENCE_GUESS: "P-recursive sequences",
        HOLONOMIC_DFINITE_GUESS: "D-finite functions",
        HOLONOMIC_ALGEBRAIC_GUESS: "algebraic functions",
        JACOBIAN_N2_DEGREE_BOX: "jacobian_conjecture_n2",
        JACOBIAN_N2_HOMOGENEOUS: "jacobian_conjecture_n2",
    }
    factories = {
        KELLER_ALPOGE_REPLAY: lambda **_k: verify_alpoge_map().as_dict(),
        KELLER_TANGENT_SWEEP: lambda **kwargs: search_tangent_sweep(
            deg_p=2,
            height=int(kwargs.get("height", 6)),
            max_hits=int(kwargs.get("max_hits", 1)),
        ),
        HOLONOMIC_RECURRENCE_GUESS: lambda **kwargs: run_discovery(
            HolonomicRecurrenceGuessFamily(samples=fibonacci_samples()).statement,
            HolonomicRecurrenceGuessFamily(samples=fibonacci_samples()),
            "score_guided",
            budget=int(kwargs.get("budget", 8)),
        ),
        HOLONOMIC_DFINITE_GUESS: lambda **kwargs: run_discovery(
            HolonomicDFiniteGuessFamily(series=exp_series()).statement,
            HolonomicDFiniteGuessFamily(series=exp_series()),
            "score_guided",
            budget=int(kwargs.get("budget", 8)),
        ),
        HOLONOMIC_ALGEBRAIC_GUESS: lambda **kwargs: run_discovery(
            HolonomicAlgebraicGuessFamily(series=linear_series()).statement,
            HolonomicAlgebraicGuessFamily(series=linear_series()),
            "score_guided",
            budget=int(kwargs.get("budget", 6)),
        ),
        JACOBIAN_N2_DEGREE_BOX: _discover_jacobian_n2,
        JACOBIAN_N2_HOMOGENEOUS: _discover_jacobian_n2_homog,
    }
    for kind, meta in FAMILY_CATALOG.items():
        register_catalog(
            CatalogEntry(
                kind=kind,
                obligation=meta["obligation"],
                parent=parents[kind],
                parent_status=meta["parent_status"],  # type: ignore[arg-type]
                package="omnibias.holonomic",
                mode=modes[kind],  # type: ignore[arg-type]
                complete=meta.get("complete", "True") != "False",
            ),
            factories.get(kind),
        )


_register()
register_catalog(
    CatalogEntry(
        kind=CONDITION_ORE,
        obligation="a prefix-verified P-recurrence in the Ore-degree box",
        parent="P-recursive sequences",
        parent_status="already_true",
        package="omnibias.holonomic",
        mode="exact_search",
        complete=False,
    ),
    lambda **kwargs: run_discovery(
        OreConditionFamily().statement,
        OreConditionFamily(),
        str(kwargs.get("proposer", "score_guided")),
        budget=int(kwargs.get("budget", 8)),
    ),
)
register_catalog(
    CatalogEntry(
        kind=CONDITION_DFINITE,
        obligation="a prefix-verified differential annihilator in the degree box",
        parent="D-finite functions",
        parent_status="already_true",
        package="omnibias.holonomic",
        mode="exact_search",
        complete=False,
    ),
    lambda **kwargs: run_discovery(
        DfiniteConditionFamily().statement,
        DfiniteConditionFamily(),
        str(kwargs.get("proposer", "score_guided")),
        budget=int(kwargs.get("budget", 8)),
    ),
)


__all__ = [
    "CONDITION_DFINITE",
    "CONDITION_ORE",
    "FAMILY_CATALOG",
    "HOLONOMIC_ALGEBRAIC_GUESS",
    "HOLONOMIC_DFINITE_GUESS",
    "HOLONOMIC_RECURRENCE_GUESS",
    "JACOBIAN_N2_DEGREE_BOX",
    "JACOBIAN_N2_HOMOGENEOUS",
    "KELLER_ALPOGE_REPLAY",
    "KELLER_TANGENT_SWEEP",
    "KELLER_TANGENT_SWEEP_DEG3",
    "build_holonomic_machine",
    "holonomic_provers",
]

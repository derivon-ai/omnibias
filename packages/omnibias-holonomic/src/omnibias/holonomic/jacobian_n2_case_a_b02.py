# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Case A leftover: the ``(b02, b03, b04)`` axis subsystem over ``Q``.

After the Case A fixes ``b20=b30=b40=0``, the ``(b11,b21,b31)`` origin,
and forcing ``b22=b12=b13=0``, the leftover lives in ``Q[b02,b03,b04]``.
Those three generators vanish over ``Q`` only at the origin. Substituting
that origin empties the Case A chart. That is a local seal, not the
Jacobian conjecture. Case B is a later slice.

Accept only via :func:`~omnibias.core.proof.engine.prove` on ``identity``
or ``residual``. A float residual is never an accept gate.
``jacobian_conjecture_proof_claim`` stays False. A complete miss is
``BLOCKED``, not parent-false.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import make_certificate
from omnibias.core.proof.discovery import Statement
from omnibias.core.proof.engine import prove
from omnibias.holonomic._core.poly_n import PolyN
from omnibias.holonomic.jacobian_n2 import (
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    JACOBIAN_N2_PARENT,
    jacobian_n2_honesty,
    seal_jacobian_honesty,
)
from omnibias.holonomic.jacobian_n2_case_a import (
    _CASE_A_FIXES,
    QCQ_NAMES,
    _as_coeff,
    _as_q,
    _eliminate,
    _extract_eqs,
    _restrict_free,
    _restrict_names,
    _subst,
    _subst_const,
    _zero_identity_payload,
)

CASE_A_B02_KIND = "jacobian_n2_case_a_b02"
CASE_A_B02_SCHEMA = "omnibias.holonomic.jacobian_n2.case_a_b02.v1"
_B31_ZERO = ("b11", "b21", "b31")
_AXIS_NAMES = ("b02", "b03", "b04")
_FORCED_MIXED = ("b22", "b12", "b13")
_DOMAIN = [-1.0, 1.0]

Status = Literal["PROVED", "DISPROVED", "BLOCKED"]

_CACHE: tuple[PolyN, PolyN, PolyN] | None = None


def case_a_b02_statement() -> Statement:
    """Universal finite statement. Parent stays ``open``."""

    return Statement(
        name=CASE_A_B02_KIND,
        obligation=(
            "after b20=b30=b40=0, the (b11,b21,b31) origin, and forcing "
            "b22=b12=b13=0, the (b02,b03,b04) leftover vanishes over Q "
            "only at the origin and the Case A chart is then empty; "
            f"a miss is not {JACOBIAN_N2_PARENT}; Case B is a later slice"
        ),
        parent=JACOBIAN_N2_PARENT,
        parent_status="open",
        existential=False,
    )


def named_case_a_b02_generators() -> tuple[PolyN, PolyN, PolyN]:
    """The three leftover generators in ``Q[b02, b03, b04]``."""

    b02, b03, b04 = PolyN.var(3, 0), PolyN.var(3, 1), PolyN.var(3, 2)
    leftover_0 = (
        -(b02**4) * 16 + (b02**2) * b03 * 36 - (b03**2) * 9 - b02 * b04 * 16
    )
    leftover_1 = (
        -(b02**3) * b03 * 24
        + b02 * (b03**2) * 36
        + (b02**2) * b04 * 16
        - b03 * b04 * 24
    )
    leftover_2 = -(b02**3) * b04 * 32 + b02 * b03 * b04 * 48 - (b04**2) * 16
    return leftover_0, leftover_1, leftover_2


def case_a_b02_generators() -> tuple[PolyN, PolyN, PolyN]:
    """Extract the Case A axis leftover and replay the named gens."""

    global _CACHE
    if _CACHE is not None:
        return _CACHE
    extracted, emptied = _extract_case_a_b02()
    named = named_case_a_b02_generators()
    if extracted != named:
        raise ValueError("Case A (b02,b03,b04) leftover missed the named generators")
    if not emptied:
        raise ValueError("Case A chart was not empty after the axis origin")
    _CACHE = named
    return named


def case_a_b02_identity_payloads(
    generators: tuple[PolyN, PolyN, PolyN] | None = None,
) -> tuple[dict[str, object], ...]:
    """Exact ``Q`` laws for :func:`~omnibias.core.proof.engine.prove` ``identity``."""

    leftover_0, leftover_1, leftover_2 = generators or named_case_a_b02_generators()
    b02, b03, b04 = PolyN.var(3, 0), PolyN.var(3, 1), PolyN.var(3, 2)
    zero = PolyN.zero(3)
    leftover_0_z = _subst(leftover_0, 2, zero)
    leftover_1_z = _subst(leftover_1, 2, zero)
    graph = b02 * b03 * 3 - (b02**3) * 2
    quad = (b03**2) * 9 + (b02**2) * b03 * 12 - (b02**4) * 16
    leftover_0_graph = _subst(leftover_0, 2, graph)
    leftover_1_graph = _subst(leftover_1, 2, graph)
    identities = (
        leftover_2 + b04 * ((b02**3) * 2 - b02 * b03 * 3 + b04) * 16,
        leftover_1_z - b02 * b03 * (b03 * 3 - (b02**2) * 2) * 12,
        _subst(leftover_0_z, 0, zero) + (b03**2) * 9,
        _subst(leftover_0_z, 1, zero) + (b02**4) * 16,
        _subst(leftover_0_z, 1, (b02**2) * Fraction(2, 3)) - (b02**4) * 4,
        leftover_0_graph + quad,
        leftover_1_graph
        + b02 * quad * 4
        - (b02**3) * (b03 * 5 - (b02**2) * 4) * 24,
        _subst(quad, 1, (b02**2) * Fraction(4, 5)) * 25 + (b02**4) * 16,
    )
    return tuple(_zero_identity_payload(item) for item in identities)


@dataclass(frozen=True)
class CaseAAxisVerdict:
    """Finite engine verdict for the Case A ``(b02, b03, b04)`` slice."""

    status: Status
    detail: str
    certificate: dict[str, object]

    @property
    def proved(self) -> bool:
        return self.status == "PROVED"

    @property
    def disproved(self) -> bool:
        return self.status == "DISPROVED"

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"


def seal_case_a_b02(
    data: Mapping[str, Any] | None = None,
) -> CaseAAxisVerdict:
    """Adjudicate the Case A ``(b02, b03, b04)`` leftover over ``Q``."""

    if JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED:
        raise RuntimeError("parent-proof claim is not wired")
    payload = dict(data or {})
    try:
        generators = _generators_from_data(payload)
        witness = _witness_from_data(payload)
    except (TypeError, ValueError) as exc:
        return _verdict("BLOCKED", str(exc), payload, generators=None)
    if witness is not None:
        return _adjudicate_witness(generators, witness, payload)
    return _adjudicate_laws(generators, payload)


def replay_case_a_b02(certificate: Mapping[str, object]) -> bool:
    """Replay a sealed Case A axis verdict. Forged proof claims are rejected."""

    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping):
        return False
    seal_jacobian_honesty(honesty)
    if honesty.get("jacobian_conjecture_proof_claim"):
        return False
    if honesty.get("continuum_parent_inferred"):
        return False
    inner = certificate.get("payload")
    payload = inner if isinstance(inner, Mapping) else certificate
    if payload.get("schema") != CASE_A_B02_SCHEMA:
        return False
    if payload.get("parent_status") != "open":
        return False
    status = payload.get("status")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    fresh = seal_case_a_b02(inputs)
    return fresh.status == status


def _adjudicate_laws(
    generators: tuple[PolyN, PolyN, PolyN],
    inputs: Mapping[str, Any],
) -> CaseAAxisVerdict:
    for payload in case_a_b02_identity_payloads(generators):
        result = prove("identity", payload)
        if result.blocked:
            return _verdict("BLOCKED", result.verdict.detail, inputs, generators)
        if result.disproved:
            return _verdict(
                "DISPROVED",
                "Case A (b02,b03,b04) identity failed over Q",
                inputs,
                generators,
            )
        if not result.proved:
            return _verdict("BLOCKED", result.verdict.detail, inputs, generators)
    origin = prove(
        "identity",
        {
            "left": [_as_coeff(generators[0].eval((0, 0, 0)))],
            "right": [0],
            "domain": _DOMAIN,
        },
    )
    if not origin.proved:
        return _verdict("BLOCKED", "origin leftover is not a Q zero", inputs, generators)
    return _verdict(
        "PROVED",
        "Case A (b02,b03,b04) leftover vanishes over Q only at the origin",
        inputs,
        generators,
        surviving="origin",
    )


def _adjudicate_witness(
    generators: tuple[PolyN, PolyN, PolyN],
    witness: tuple[Fraction, Fraction, Fraction],
    inputs: Mapping[str, Any],
) -> CaseAAxisVerdict:
    residuals: list[str] = []
    for poly in generators:
        value = poly.eval(witness)
        result = prove(
            "identity",
            {"left": [_as_coeff(value)], "right": [0], "domain": _DOMAIN},
        )
        if result.blocked:
            return _verdict("BLOCKED", result.verdict.detail, inputs, generators)
        residuals.append(str(value))
        if result.disproved:
            return _adjudicate_laws(generators, inputs)
        if not result.proved:
            return _verdict("BLOCKED", result.verdict.detail, inputs, generators)
    if witness == (Fraction(0), Fraction(0), Fraction(0)):
        return _verdict(
            "PROVED",
            "origin is a leftover zero; Case A chart empties; not a violator",
            inputs,
            generators,
            surviving="origin",
            extra={"witness": ["0", "0", "0"], "evals": residuals},
        )
    return _verdict(
        "DISPROVED",
        "nonzero Q-point of Case A (b02,b03,b04) leftover",
        inputs,
        generators,
        extra={"witness": [str(item) for item in witness], "evals": residuals},
    )


def _verdict(
    status: Status,
    detail: str,
    inputs: Mapping[str, Any],
    generators: tuple[PolyN, PolyN, PolyN] | None,
    *,
    surviving: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> CaseAAxisVerdict:
    honesty = dict(jacobian_n2_honesty(discovered=False, n2_counterexample=False))
    honesty["continuum_parent_inferred"] = False
    honesty["float_residual_is_proof"] = False
    honesty["unproven_claim"] = False
    honesty["identity_collapse"] = True
    honesty = dict(seal_jacobian_honesty(honesty))
    body: dict[str, object] = {
        "schema": CASE_A_B02_SCHEMA,
        "engine_kind": CASE_A_B02_KIND,
        "status": status,
        "spec_name": CASE_A_B02_KIND,
        "surviving": surviving,
        "detail": detail,
        "parent": JACOBIAN_N2_PARENT,
        "parent_status": "open",
        "inputs": dict(inputs),
        "generators": None
        if generators is None
        else [_terms_payload(item) for item in generators],
    }
    if extra is not None:
        body.update(dict(extra))
    certificate = dict(
        make_certificate(
            claim=f"finite leftover chart {CASE_A_B02_KIND}",
            payload=body,
            honesty=honesty,
        )
    )
    return CaseAAxisVerdict(status, detail, certificate)


def _generators_from_data(
    data: Mapping[str, Any],
) -> tuple[PolyN, PolyN, PolyN]:
    raw = data.get("generators")
    if raw is None:
        try:
            return case_a_b02_generators()
        except ValueError as exc:
            raise ValueError(f"complete miss: {exc}") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != 3:
        raise TypeError("generators must be three exact Q polynomials")
    return tuple(_poly_from_terms(item) for item in raw)


def _witness_from_data(
    data: Mapping[str, Any],
) -> tuple[Fraction, Fraction, Fraction] | None:
    raw = data.get("witness")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != 3:
        raise TypeError("witness must be a length-3 exact Q point")
    return (_as_q(raw[0], "witness"), _as_q(raw[1], "witness"), _as_q(raw[2], "witness"))


def _terms_payload(poly: PolyN) -> list[list[object]]:
    return [[list(mon), str(coeff)] for mon, coeff in sorted(poly.terms.items())]


def _poly_from_terms(raw: object) -> PolyN:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise TypeError("generator terms must be a sequence")
    terms: dict[tuple[int, ...], Fraction] = {}
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
            raise TypeError("generator term must be [monomial, coeff]")
        mon_raw, coeff_raw = item
        if not isinstance(mon_raw, Sequence) or isinstance(mon_raw, str | bytes):
            raise TypeError("monomial must be an exponent sequence")
        mon = tuple(int(power) for power in mon_raw)
        if len(mon) != 3:
            raise ValueError("Case A (b02,b03,b04) generators are ternary")
        terms[mon] = _as_q(coeff_raw, "coeff")
    return PolyN(3, terms)


def _support_names(poly: PolyN, names: Sequence[str]) -> set[str]:
    return {
        names[index]
        for index in range(poly.nvars)
        if any(mon[index] for mon in poly.terms)
    }


def _force_univariate_zeros(
    eqs: Sequence[PolyN],
    names: Sequence[str],
) -> tuple[list[PolyN], list[str]]:
    current = [item for item in eqs if not item.is_zero()]
    forced: list[str] = []
    changed = True
    while changed:
        changed = False
        for item in current:
            used = [
                index
                for index in range(item.nvars)
                if any(mon[index] for mon in item.terms)
            ]
            if len(used) != 1 or item.constant_value() is not None:
                continue
            name = names[used[0]]
            forced.append(name)
            current = [_subst_const(poly, used[0], Fraction(0)) for poly in current]
            current = [poly for poly in current if not poly.is_zero()]
            changed = True
            break
    return current, forced


def _extract_case_a_b02() -> tuple[tuple[PolyN, PolyN, PolyN], bool]:
    eqs = _extract_eqs()
    solved, leftover = _eliminate(eqs, len(QCQ_NAMES))
    free = [index for index in range(len(QCQ_NAMES)) if index not in solved]
    free_names = [QCQ_NAMES[index] for index in free]
    current = [_restrict_free(item, free) for item in leftover]
    assigns = dict(_CASE_A_FIXES)
    assigns.update(dict.fromkeys(_B31_ZERO, Fraction(0)))
    for name, value in assigns.items():
        index = free_names.index(name)
        current = [_subst_const(item, index, value) for item in current]
        current = [item for item in current if not item.is_zero()]
    current, forced = _force_univariate_zeros(current, free_names)
    if set(forced) != set(_FORCED_MIXED):
        raise ValueError(f"expected force {_FORCED_MIXED}, got {tuple(forced)}")
    keep = set(_AXIS_NAMES)
    selected: list[PolyN] = []
    for item in current:
        support = _support_names(item, free_names)
        if not support:
            const = item.constant_value()
            if const not in (None, Fraction(0)):
                raise ValueError("const-nonzero leftover after force")
            continue
        if not support <= keep:
            raise ValueError(f"leftover escaped onto {sorted(support)}")
        selected.append(_restrict_names(item, free_names, _AXIS_NAMES))
    if len(selected) != 3:
        raise ValueError(f"expected 3 (b02,b03,b04) leftover gens, got {len(selected)}")
    emptied = list(current)
    for name in _AXIS_NAMES:
        index = free_names.index(name)
        emptied = [_subst_const(item, index, Fraction(0)) for item in emptied]
        emptied = [item for item in emptied if not item.is_zero()]
    emptied, _ = _force_univariate_zeros(emptied, free_names)
    return (selected[0], selected[1], selected[2]), not emptied


__all__ = [
    "CASE_A_B02_KIND",
    "CASE_A_B02_SCHEMA",
    "CaseAAxisVerdict",
    "case_a_b02_generators",
    "case_a_b02_identity_payloads",
    "case_a_b02_statement",
    "named_case_a_b02_generators",
    "replay_case_a_b02",
    "seal_case_a_b02",
]

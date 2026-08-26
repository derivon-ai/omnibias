# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Case A leftover: the ``(b11, b21, b31)`` subsystem over ``Q``.

On the I+Q+C+Qu Keller leftover, Case A is the slice ``b20=b30=b40=0``.
The three leftover generators that live in ``Q[b11, b21, b31]`` vanish
over ``Q`` only at the origin. That is a local chart seal, not the
Jacobian conjecture. The remaining Case A leftover in
``(b02, b03, b04)`` is a later slice.

Accept only via :func:`~omnibias.core.proof.engine.prove` on ``identity``
(exact ``Q`` laws) or ``residual`` (a sound point enclosure). A float
residual, SVD, or anneal→0 is never an accept gate.
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

CASE_A_B31_KIND = "jacobian_n2_case_a_b31"
CASE_A_B31_SCHEMA = "omnibias.holonomic.jacobian_n2.case_a_b31.v1"
_CASE_A_FIXES: dict[str, Fraction] = {
    "b20": Fraction(0),
    "b30": Fraction(0),
    "b40": Fraction(0),
}
_B31_NAMES = ("b11", "b21", "b31")
_DOMAIN = [-1.0, 1.0]
_QUADS = ((2, 0), (1, 1), (0, 2))
_CUBICS = ((3, 0), (2, 1), (1, 2), (0, 3))
_QUARTS = ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4))

Status = Literal["PROVED", "DISPROVED", "BLOCKED"]


def _layer_names(prefix: str, mons: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}{i}{j}" for i, j in mons)


QCQ_NAMES: tuple[str, ...] = (
    _layer_names("a", _QUADS)
    + _layer_names("b", _QUADS)
    + _layer_names("a", _CUBICS)
    + _layer_names("b", _CUBICS)
    + _layer_names("a", _QUARTS)
    + _layer_names("b", _QUARTS)
)
QCQ_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (_QUADS, _CUBICS, _QUARTS)

_CACHE: tuple[PolyN, PolyN, PolyN] | None = None


def case_a_b31_statement() -> Statement:
    """Universal finite statement. Parent stays ``open``."""

    return Statement(
        name=CASE_A_B31_KIND,
        obligation=(
            "after b20=b30=b40=0 on the I+Q+C+Qu leftover, the "
            "(b11,b21,b31) subsystem vanishes over Q only at the origin; "
            f"a miss is not {JACOBIAN_N2_PARENT}; leftover in "
            "(b02,b03,b04) is a later slice"
        ),
        parent=JACOBIAN_N2_PARENT,
        parent_status="open",
        existential=False,
    )


def named_case_a_b31_generators() -> tuple[PolyN, PolyN, PolyN]:
    """The three leftover generators in ``Q[b11, b21, b31]``."""

    b11, b21, b31 = PolyN.var(3, 0), PolyN.var(3, 1), PolyN.var(3, 2)
    leftover_0 = (
        -(b11**4) + (b11**2) * b21 * 3 - (b21**2) - b11 * b31 * 2
    )
    leftover_5 = (
        -(b11**3) * b21
        + b11 * (b21**2) * 2
        + (b11**2) * b31
        - b21 * b31 * 2
    )
    leftover_11 = -(b11**3) * b31 + b11 * b21 * b31 * 2 - (b31**2)
    return leftover_0, leftover_5, leftover_11


def case_a_b31_generators() -> tuple[PolyN, PolyN, PolyN]:
    """Extract the Case A ``(b11, b21, b31)`` leftover and replay the named gens."""

    global _CACHE
    if _CACHE is not None:
        return _CACHE
    extracted = _extract_case_a_b31()
    named = named_case_a_b31_generators()
    if extracted != named:
        raise ValueError("Case A (b11,b21,b31) leftover missed the named generators")
    _CACHE = named
    return named


def case_a_b31_identity_payloads(
    generators: tuple[PolyN, PolyN, PolyN] | None = None,
) -> tuple[dict[str, object], ...]:
    """Exact ``Q`` laws for :func:`~omnibias.core.proof.engine.prove` ``identity``."""

    leftover_0, leftover_5, leftover_11 = generators or named_case_a_b31_generators()
    b11, b21, b31 = PolyN.var(3, 0), PolyN.var(3, 1), PolyN.var(3, 2)
    zero_b31 = PolyN.zero(3)
    leftover_0_z = _subst(leftover_0, 2, zero_b31)
    leftover_5_z = _subst(leftover_5, 2, zero_b31)
    graph = b11 * b21 * 2 - b11**3
    quad = b21**2 + (b11**2) * b21 - b11**4
    leftover_0_graph = _subst(leftover_0, 2, graph)
    leftover_5_graph = _subst(leftover_5, 2, graph)
    identities = (
        leftover_11 + b31 * (b11**3 - b11 * b21 * 2 + b31),
        leftover_5_z - b11 * b21 * (b21 * 2 - b11**2),
        _subst(leftover_0_z, 0, zero_b31) + b21**2,
        _subst(leftover_0_z, 1, zero_b31) + b11**4,
        _subst(leftover_0_z, 1, (b11**2) * Fraction(1, 2))
        - (b11**4) * Fraction(1, 4),
        leftover_0_graph + quad,
        leftover_5_graph + b11 * quad * 2 - (b11**3) * (b21 * 5 - (b11**2) * 3),
        _subst(quad, 1, (b11**2) * Fraction(3, 5)) * 25 + b11**4,
    )
    return tuple(_zero_identity_payload(item) for item in identities)


@dataclass(frozen=True)
class CaseAVerdict:
    """Finite engine verdict for the Case A ``(b11, b21, b31)`` slice."""

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


def seal_case_a_b31(
    data: Mapping[str, Any] | None = None,
) -> CaseAVerdict:
    """Adjudicate the Case A ``(b11, b21, b31)`` leftover over ``Q``."""

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


def replay_case_a_b31(certificate: Mapping[str, object]) -> bool:
    """Replay a sealed Case A verdict. Forged proof claims are rejected."""

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
    if payload.get("schema") != CASE_A_B31_SCHEMA:
        return False
    if payload.get("parent_status") != "open":
        return False
    status = payload.get("status")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    fresh = seal_case_a_b31(inputs)
    return fresh.status == status


def _adjudicate_laws(
    generators: tuple[PolyN, PolyN, PolyN],
    inputs: Mapping[str, Any],
) -> CaseAVerdict:
    for payload in case_a_b31_identity_payloads(generators):
        result = prove("identity", payload)
        if result.blocked:
            return _verdict("BLOCKED", result.verdict.detail, inputs, generators)
        if result.disproved:
            return _verdict(
                "DISPROVED",
                "Case A (b11,b21,b31) identity failed over Q",
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
        "Case A (b11,b21,b31) leftover vanishes over Q only at the origin",
        inputs,
        generators,
        surviving="origin",
    )


def _adjudicate_witness(
    generators: tuple[PolyN, PolyN, PolyN],
    witness: tuple[Fraction, Fraction, Fraction],
    inputs: Mapping[str, Any],
) -> CaseAVerdict:
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
            "origin is a leftover zero; not a Jacobian n=2 violator",
            inputs,
            generators,
            surviving="origin",
            extra={"witness": ["0", "0", "0"], "evals": residuals},
        )
    return _verdict(
        "DISPROVED",
        "nonzero Q-point of Case A (b11,b21,b31) leftover",
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
) -> CaseAVerdict:
    honesty = dict(jacobian_n2_honesty(discovered=False, n2_counterexample=False))
    honesty["continuum_parent_inferred"] = False
    honesty["float_residual_is_proof"] = False
    honesty["unproven_claim"] = False
    honesty["identity_collapse"] = True
    honesty = dict(seal_jacobian_honesty(honesty))
    body: dict[str, object] = {
        "schema": CASE_A_B31_SCHEMA,
        "engine_kind": CASE_A_B31_KIND,
        "status": status,
        "spec_name": CASE_A_B31_KIND,
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
        make_certificate(claim=f"finite leftover chart {CASE_A_B31_KIND}", payload=body, honesty=honesty)
    )
    return CaseAVerdict(status, detail, certificate)


def _generators_from_data(
    data: Mapping[str, Any],
) -> tuple[PolyN, PolyN, PolyN]:
    raw = data.get("generators")
    if raw is None:
        try:
            return case_a_b31_generators()
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


def _as_q(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{name} must be exact Q; a float residual is not a proof")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"{name} must be exact Q; a float residual is not a proof")


def _as_coeff(value: Fraction) -> int | Fraction:
    return int(value) if value.denominator == 1 else value


def _zero_identity_payload(poly: PolyN) -> dict[str, object]:
    coeffs = [_as_coeff(coeff) for _, coeff in sorted(poly.terms.items())]
    if not coeffs:
        coeffs = [0]
    return {"left": coeffs, "right": [0] * len(coeffs), "domain": list(_DOMAIN)}


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
            raise ValueError("Case A (b11,b21,b31) generators are ternary")
        terms[mon] = _as_q(coeff_raw, "coeff")
    return PolyN(3, terms)


def _subst(poly: PolyN, index: int, expr: PolyN) -> PolyN:
    subs = [PolyN.var(poly.nvars, i) for i in range(poly.nvars)]
    subs[index] = expr
    return poly.compose(subs)


def _extract_eqs() -> list[PolyN]:
    n_coeff = len(QCQ_NAMES)
    nvars = 2 + n_coeff
    x, y = PolyN.var(nvars, 0), PolyN.var(nvars, 1)
    coeffs = [PolyN.var(nvars, 2 + i) for i in range(n_coeff)]
    first, second = x, y
    cursor = 0
    for layer in QCQ_LAYERS:
        width = len(layer)
        for (i, j), coeff in zip(layer, coeffs[cursor : cursor + width], strict=True):
            first = first + coeff * (x**i) * (y**j)
        cursor += width
        for (i, j), coeff in zip(layer, coeffs[cursor : cursor + width], strict=True):
            second = second + coeff * (x**i) * (y**j)
        cursor += width
    det = first.partial(0) * second.partial(1) - first.partial(1) * second.partial(0)
    det = det - PolyN.const(nvars, 1)
    grouped: dict[tuple[int, int], dict[tuple[int, ...], Fraction]] = {}
    for mon, coeff in det.terms.items():
        bucket = grouped.setdefault((mon[0], mon[1]), {})
        tail = mon[2:]
        bucket[tail] = bucket.get(tail, Fraction(0)) + coeff
    return [
        PolyN(n_coeff, {mon: value for mon, value in terms.items() if value != 0})
        for terms in grouped.values()
        if any(terms.values())
    ]


def _degree_in(poly: PolyN, index: int) -> int:
    if not poly.terms:
        return -1
    return max(mon[index] for mon in poly.terms)


def _solve_linear(poly: PolyN, index: int) -> PolyN | None:
    if _degree_in(poly, index) != 1:
        return None
    lead: dict[tuple[int, ...], Fraction] = {}
    rest: dict[tuple[int, ...], Fraction] = {}
    for mon, coeff in poly.terms.items():
        if mon[index] == 1:
            new = list(mon)
            new[index] = 0
            lead[tuple(new)] = coeff
        elif mon[index] == 0:
            rest[mon] = coeff
        else:
            return None
    slope = PolyN(poly.nvars, lead).constant_value()
    if slope is None or slope == 0:
        return None
    return PolyN(poly.nvars, rest) * Fraction(-1, slope)


def _subst_poly(poly: PolyN, index: int, expr: PolyN) -> PolyN:
    return _subst(poly, index, expr)


def _subst_const(poly: PolyN, index: int, value: Fraction) -> PolyN:
    out: dict[tuple[int, ...], Fraction] = {}
    for mon, coeff in poly.terms.items():
        power = mon[index]
        new = list(mon)
        new[index] = 0
        key = tuple(new)
        out[key] = out.get(key, Fraction(0)) + coeff * (value**power)
    return PolyN(poly.nvars, {key: coeff for key, coeff in out.items() if coeff != 0})


def _eliminate(eqs: Sequence[PolyN], n_coeff: int) -> tuple[dict[int, PolyN], list[PolyN]]:
    remaining = [item for item in eqs if not item.is_zero()]
    solved: dict[int, PolyN] = {}
    progress = True
    while progress:
        progress = False
        for eq in list(remaining):
            for index in range(n_coeff):
                if index in solved:
                    continue
                expr = _solve_linear(eq, index)
                if expr is None or _degree_in(expr, index) > 0:
                    continue
                solved[index] = expr
                remaining = [
                    item
                    for item in (_subst_poly(eq_item, index, expr) for eq_item in remaining)
                    if not item.is_zero()
                ]
                for other in list(solved):
                    if other != index:
                        solved[other] = _subst_poly(solved[other], index, expr)
                progress = True
                break
            if progress:
                break
    return solved, remaining


def _restrict_free(poly: PolyN, free: Sequence[int]) -> PolyN:
    inverse = {old: new for new, old in enumerate(free)}
    terms: dict[tuple[int, ...], Fraction] = {}
    for mon, coeff in poly.terms.items():
        newmon = [0] * len(free)
        for index, power in enumerate(mon):
            if not power:
                continue
            if index not in inverse:
                raise ValueError(f"solved var {index} appears in leftover")
            newmon[inverse[index]] = power
        key = tuple(newmon)
        terms[key] = terms.get(key, Fraction(0)) + coeff
    return PolyN(len(free), {key: coeff for key, coeff in terms.items() if coeff != 0})


def _restrict_names(poly: PolyN, names: Sequence[str], keep: tuple[str, ...]) -> PolyN:
    terms: dict[tuple[int, ...], Fraction] = {}
    for mon, coeff in poly.terms.items():
        new = [0] * len(keep)
        for index, power in enumerate(mon):
            if not power:
                continue
            name = names[index]
            if name not in keep:
                raise ValueError(f"support escaped onto {name}")
            new[keep.index(name)] = power
        key = tuple(new)
        terms[key] = terms.get(key, Fraction(0)) + coeff
    return PolyN(len(keep), {key: coeff for key, coeff in terms.items() if coeff != 0})


def _extract_case_a_b31() -> tuple[PolyN, PolyN, PolyN]:
    eqs = _extract_eqs()
    solved, leftover = _eliminate(eqs, len(QCQ_NAMES))
    free = [index for index in range(len(QCQ_NAMES)) if index not in solved]
    free_names = [QCQ_NAMES[index] for index in free]
    left_free = [_restrict_free(item, free) for item in leftover]
    current = left_free
    for name, value in _CASE_A_FIXES.items():
        index = free_names.index(name)
        current = [_subst_const(item, index, value) for item in current]
        current = [item for item in current if not item.is_zero()]
    keep_set = set(_B31_NAMES)
    selected: list[PolyN] = []
    for item in current:
        support = {
            free_names[index]
            for index in range(item.nvars)
            if any(mon[index] for mon in item.terms)
        }
        if support <= keep_set and support:
            selected.append(_restrict_names(item, free_names, _B31_NAMES))
    if len(selected) != 3:
        raise ValueError(f"expected 3 (b11,b21,b31) leftover gens, got {len(selected)}")
    return selected[0], selected[1], selected[2]


__all__ = [
    "CASE_A_B31_KIND",
    "CASE_A_B31_SCHEMA",
    "CaseAVerdict",
    "QCQ_LAYERS",
    "QCQ_NAMES",
    "case_a_b31_generators",
    "case_a_b31_identity_payloads",
    "case_a_b31_statement",
    "named_case_a_b31_generators",
    "replay_case_a_b31",
    "seal_case_a_b31",
]

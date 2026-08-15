# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Planted rational enclosure-trace certificates.

An ``enclosure_trace`` payload names one locked family (``tower``, ``nk``,
``bernoulli``, ``ldlt``) and the finite ``+ − × abs recip`` DAG Lean replays.
The Mathlib bridge re-derives the DAG over :class:`~fractions.Fraction` and
emits Lean that applies the matching ``OmnibiasAnalytic.Check`` plant theorem.

This is finite rational interval arithmetic on a planted DAG. It is not a
continuum PDE claim, not analytic continuation of a Dirichlet series, and
not a general SOS engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_TRACE_FAMILIES: tuple[str, ...] = ("tower", "nk", "bernoulli", "ldlt")

TraceFamily = Literal["tower", "nk", "bernoulli", "ldlt"]

Op = dict[str, Any]
Box = tuple[Fraction, Fraction]


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _as_frac(value: Any) -> Fraction | None:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) == 2
    ):
        return None
    num, den = value
    if (
        isinstance(num, int)
        and not isinstance(num, bool)
        and isinstance(den, int)
        and not isinstance(den, bool)
        and den != 0
    ):
        return Fraction(num, den)
    return None


def _const(num: int, den: int = 1) -> Op:
    return {"op": "const", "value": [num, den]}


def _binop(op: str, i: int, j: int) -> Op:
    return {"op": op, "i": i, "j": j}


def _unop(op: str, i: int) -> Op:
    return {"op": op, "i": i}


# A. Horner of sigmoidPoly 2 = [0, 1, -3, 2] at 2/3 → -2/27.
TOWER_HORNER_OPS: tuple[Op, ...] = (
    _const(2, 3),
    _const(2),
    _const(-3),
    _const(1),
    _const(0),
    _binop("mul", 1, 0),
    _binop("add", 5, 2),
    _binop("mul", 6, 0),
    _binop("add", 7, 3),
    _binop("mul", 8, 0),
    _binop("add", 9, 4),
)
TOWER_HORNER_RESULT = (Fraction(-2, 27), Fraction(-2, 27))

# B. |A(c^2-2)|, kappa = 2 Z2 r, p(r) = Y0 + Z2 r^2 - r.
NK_BOUND_OPS: tuple[Op, ...] = (
    _const(3, 2),
    _const(1, 3),
    _const(2),
    _const(1, 4),
    _const(2, 3),
    _binop("mul", 0, 0),
    _binop("sub", 5, 2),
    _binop("mul", 1, 6),
    _unop("abs", 7),
    _binop("mul", 4, 3),
    _binop("mul", 2, 9),
    _binop("mul", 3, 3),
    _binop("mul", 4, 11),
    _binop("add", 8, 12),
    _binop("sub", 13, 3),
)
NK_Y0 = (Fraction(1, 12), Fraction(1, 12))
NK_KAPPA = (Fraction(1, 3), Fraction(1, 3))
NK_P = (Fraction(-1, 8), Fraction(-1, 8))

# C. B2 = 1*2/(4*3) = 1/6, zetaNeg1 = B2 * (-1/2) = -1/12.
BERNOULLI_OPS: tuple[Op, ...] = (
    _const(1),
    _const(2),
    _const(4),
    _const(3),
    _binop("mul", 2, 3),
    _unop("recip", 4),
    _binop("mul", 0, 1),
    _binop("mul", 6, 5),
    _const(-2),
    _unop("recip", 8),
    _binop("mul", 7, 9),
)
BERNOULLI_B2 = (Fraction(1, 6), Fraction(1, 6))
BERNOULLI_ZETA_NEG1 = (Fraction(-1, 12), Fraction(-1, 12))

# D. LDLT of [[2, 1], [1, 2]]: d0 = 2, d1 = 3/2.
LDLT_OPS: tuple[Op, ...] = (
    _const(2),
    _const(1),
    _unop("recip", 0),
    _binop("mul", 1, 2),
    _binop("mul", 3, 3),
    _binop("mul", 4, 0),
    _binop("sub", 0, 5),
)
LDLT_D0 = (Fraction(2), Fraction(2))
LDLT_D1 = (Fraction(3, 2), Fraction(3, 2))

_LEAN_THMS: dict[str, str] = {
    "tower": "tower_horner_result",
    "nk": "nk_trace_unique_zero",
    "bernoulli": "bernoulli_b2_zetaNeg1",
    "ldlt": "ldlt_plant_pivots_pos",
}

_FAMILY_OPS: dict[str, tuple[Op, ...]] = {
    "tower": TOWER_HORNER_OPS,
    "nk": NK_BOUND_OPS,
    "bernoulli": BERNOULLI_OPS,
    "ldlt": LDLT_OPS,
}

_FAMILY_RESULT: dict[str, Box] = {
    "tower": TOWER_HORNER_RESULT,
    "nk": NK_P,
    "bernoulli": BERNOULLI_ZETA_NEG1,
    "ldlt": LDLT_D1,
}


def eval_ops(ops: Sequence[Mapping[str, Any]]) -> list[Box] | None:
    """Replay a rational enclosure DAG. ``None`` on a bad index or recip through 0."""
    acc: list[Box] = []
    for raw in ops:
        if not isinstance(raw, Mapping):
            return None
        kind = raw.get("op")
        if kind == "const":
            value = _as_frac(raw.get("value"))
            if value is None:
                return None
            acc.append((value, value))
            continue
        if kind in {"add", "sub", "mul"}:
            i, j = raw.get("i"), raw.get("j")
            if not (
                isinstance(i, int)
                and not isinstance(i, bool)
                and isinstance(j, int)
                and not isinstance(j, bool)
                and 0 <= i < len(acc)
                and 0 <= j < len(acc)
            ):
                return None
            a_lo, a_hi = acc[i]
            b_lo, b_hi = acc[j]
            if kind == "add":
                acc.append((a_lo + b_lo, a_hi + b_hi))
            elif kind == "sub":
                acc.append((a_lo - b_hi, a_hi - b_lo))
            else:
                corners = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
                acc.append((min(corners), max(corners)))
            continue
        if kind in {"abs", "recip"}:
            i = raw.get("i")
            if not (
                isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(acc)
            ):
                return None
            lo, hi = acc[i]
            if kind == "abs":
                if lo >= 0:
                    acc.append((lo, hi))
                elif hi <= 0:
                    acc.append((-hi, -lo))
                else:
                    acc.append((Fraction(0), max(-lo, hi)))
            elif lo > 0 or hi < 0:
                acc.append((1 / hi, 1 / lo))
            else:
                return None
            continue
        return None
    return acc


def _encode_box(box: Box) -> dict[str, list[int]]:
    return {"lo": _pair(box[0]), "hi": _pair(box[1])}


def _decode_box(value: Any) -> Box | None:
    if not isinstance(value, Mapping):
        return None
    lo, hi = _as_frac(value.get("lo")), _as_frac(value.get("hi"))
    if lo is None or hi is None:
        return None
    return lo, hi


def _ops_equal(got: Any, expected: Sequence[Op]) -> bool:
    if not (isinstance(got, Sequence) and not isinstance(got, str | bytes)):
        return False
    if len(got) != len(expected):
        return False
    for raw, want in zip(got, expected, strict=True):
        if not isinstance(raw, Mapping) or raw.get("op") != want["op"]:
            return False
        if want["op"] == "const":
            if _as_frac(raw.get("value")) != _as_frac(want["value"]):
                return False
        elif want["op"] in {"add", "sub", "mul"}:
            if raw.get("i") != want["i"] or raw.get("j") != want["j"]:
                return False
        elif raw.get("i") != want["i"]:
            return False
    return True


def family_nodes_hold(family: str, nodes: Sequence[Box]) -> bool:
    """Re-derived intermediate boxes for a locked family."""
    if family == "tower":
        return bool(nodes) and nodes[-1] == TOWER_HORNER_RESULT
    if family == "nk":
        return (
            len(nodes) > 14
            and nodes[8] == NK_Y0
            and nodes[10] == NK_KAPPA
            and nodes[14] == NK_P
        )
    if family == "bernoulli":
        return (
            len(nodes) > 10
            and nodes[7] == BERNOULLI_B2
            and nodes[10] == BERNOULLI_ZETA_NEG1
        )
    if family == "ldlt":
        return (
            len(nodes) > 6
            and nodes[0] == LDLT_D0
            and nodes[6] == LDLT_D1
            and nodes[0][0] > 0
            and nodes[6][0] > 0
        )
    return False


def locked_trace_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries a locked family DAG and matching result."""
    family = payload.get("family")
    if family not in _FAMILY_OPS:
        return False
    if not _ops_equal(payload.get("ops"), _FAMILY_OPS[family]):
        return False
    nodes = eval_ops(_FAMILY_OPS[family])
    if nodes is None or not family_nodes_hold(family, nodes):
        return False
    result = _decode_box(payload.get("result"))
    return result == nodes[-1] == _FAMILY_RESULT[family]


def lean_trace_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``enclosure_trace`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown enclosure-trace family {family!r}") from exc


def enclosure_trace_certificate(
    family: TraceFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``enclosure_trace`` certificate for the Mathlib bridge."""
    if family not in LEGAL_TRACE_FAMILIES:
        raise ValueError(
            f"unknown enclosure-trace family {family!r}; expected one of {LEGAL_TRACE_FAMILIES}"
        )
    ops = _FAMILY_OPS[family]
    nodes = eval_ops(ops)
    if nodes is None:
        raise RuntimeError(f"locked {family} trace failed to evaluate")
    return make_certificate(
        claim=f"replay locked {family} enclosure trace",
        payload={
            "type": "enclosure_trace",
            "family": family,
            "ops": [dict(op) for op in ops],
            "result": _encode_box(nodes[-1]),
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "BERNOULLI_B2",
    "BERNOULLI_OPS",
    "BERNOULLI_ZETA_NEG1",
    "LDLT_D0",
    "LDLT_D1",
    "LDLT_OPS",
    "LEGAL_TRACE_FAMILIES",
    "NK_BOUND_OPS",
    "NK_KAPPA",
    "NK_P",
    "NK_Y0",
    "TOWER_HORNER_OPS",
    "TOWER_HORNER_RESULT",
    "enclosure_trace_certificate",
    "eval_ops",
    "family_nodes_hold",
    "lean_trace_theorem",
    "locked_trace_matches",
]

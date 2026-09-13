# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite rational replay whose source operands occur in the Lean proposition.

The architecture-to-coefficient compiler and analytic enclosure producers are
outside this finite theorem. An expected source digest binds a replay to the
caller's original claim; a re-sealed different source is a different claim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import (
    Cert,
    canonical_json,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.realization.polynomial import (
    AlgebraBudgetExceeded,
    Rational,
    SparsePolynomial,
    rational,
)
from omnibias.core.realization.rank import ExactRankWitness, Matrix, determinant, matrix

Relation = Literal["eq", "ge", "gt"]
RationalInterval = tuple[Fraction, Fraction]


def _q(q: Fraction) -> list[int]:
    return [q.numerator, q.denominator]


def source_digest(source: object) -> str:
    return hashlib.sha256(canonical_json({"source": source}).encode()).hexdigest()


def _seal(kind: str, source: object, **data: object) -> Cert:
    return make_certificate(
        claim="Finite rational operand replay; no continuum or architecture-compiler theorem.",
        payload={
            "type": "realization_replay",
            "kind": kind,
            "source": source,
            "source_digest": source_digest(source),
            **data,
        },
        meta={"transcend_backend": "not_used"},
    )


def polynomial_evaluation_certificate(
    polynomials: Sequence[SparsePolynomial],
    point: Sequence[Rational],
    expected: Sequence[Rational],
) -> Cert:
    """Bind explicit coefficient polynomials, rational parameters and readout."""
    p = tuple(rational(x) for x in point)
    target = tuple(rational(x) for x in expected)
    if (
        not polynomials
        or len(polynomials) != len(target)
        or any(f.nvars != len(p) for f in polynomials)
    ):
        raise ValueError("polynomial replay dimensions differ")
    if tuple(f.evaluate(p) for f in polynomials) != target:
        raise ValueError("polynomial evaluation does not equal target")
    return _seal(
        "polynomial_evaluation",
        [f.to_payload() for f in polynomials],
        point=[_q(x) for x in p],
        expected=[_q(x) for x in target],
    )


def polynomial_inequality_certificate(
    polynomials: Sequence[SparsePolynomial],
    point: Sequence[Rational],
    relations: Sequence[Relation],
) -> Cert:
    """Prove explicit rational expressions are zero/nonnegative/positive."""
    p = tuple(rational(x) for x in point)
    if (
        not polynomials
        or len(polynomials) != len(relations)
        or any(f.nvars != len(p) for f in polynomials)
    ):
        raise ValueError("polynomial inequality dimensions differ")
    if any(rel not in ("eq", "ge", "gt") for rel in relations):
        raise ValueError("unknown relation")
    for f, rel in zip(polynomials, relations, strict=True):
        value = f.evaluate(p)
        if not (value == 0 if rel == "eq" else value >= 0 if rel == "ge" else value > 0):
            raise ValueError("false polynomial inequality")
    return _seal(
        "polynomial_inequality",
        [f.to_payload() for f in polynomials],
        point=[_q(x) for x in p],
        relations=list(relations),
    )


def rank_replay_certificate(witness: ExactRankWitness) -> Cert:
    """Prove A=B C and a recomputed nonzero source minor, with full operands.

    These finite facts support the standard rank sandwich; the emitted theorem
    itself asserts the factorization and minor, not an abstract rank theorem.
    """
    if not witness.verify():
        raise ValueError("invalid rank witness")
    return _seal(
        "matrix_rank",
        [[_q(x) for x in row] for row in witness.source],
        rank=witness.rank,
        left=[[_q(x) for x in row] for row in witness.left],
        right=[[_q(x) for x in row] for row in witness.right],
        minor_rows=list(witness.minor_rows),
        minor_cols=list(witness.minor_cols),
        minor_value=_q(witness.minor_value),
    )


def _interval(value: Sequence[Rational]) -> RationalInterval:
    if len(value) != 2:
        raise ValueError("interval needs two endpoints")
    lo, hi = rational(value[0]), rational(value[1])
    if lo > hi:
        raise ValueError("reversed interval")
    return lo, hi


def _iv_payload(iv: RationalInterval) -> list[list[int]]:
    return [_q(iv[0]), _q(iv[1])]


def _decode_iv(value: Any) -> RationalInterval:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("malformed rational interval")
    return _interval((_fraction(value[0]), _fraction(value[1])))


def _imul(a: RationalInterval, b: RationalInterval) -> RationalInterval:
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def _isub(a: RationalInterval, b: RationalInterval) -> RationalInterval:
    return a[0] - b[1], a[1] - b[0]


def _positive_pivots(a: Sequence[Sequence[RationalInterval]]) -> bool:
    n = len(a)
    if (
        not n
        or any(len(row) != n for row in a)
        or any(a[i][j] != a[j][i] for i in range(n) for j in range(n))
    ):
        return False
    current = [list(row) for row in a]
    while current:
        lo, hi = current[0][0]
        if lo <= 0 or lo > hi:
            return False
        reciprocal = (1 / hi, 1 / lo)
        current = [
            [
                _isub(current[i][j], _imul(_imul(current[i][0], current[0][j]), reciprocal))
                for j in range(1, len(current))
            ]
            for i in range(1, len(current))
        ]
    return True


def interval_ldlt_replay_certificate(matrix: Sequence[Sequence[Sequence[Rational]]]) -> Cert:
    """Recompute positive rational interval Schur pivots from original operands.

    The finite Lean theorem evaluates this arithmetic; real enclosure provenance
    and the general LDL positive-definiteness implication are upstream facts.
    """
    a = [[_interval(x) for x in row] for row in matrix]
    if not _positive_pivots(a):
        raise ValueError("interval LDL did not prove strictly positive pivots")
    return _seal("interval_ldlt", [[_iv_payload(x) for x in row] for row in a])


def strict_box_inclusion_certificate(
    inner: Sequence[Sequence[Rational]], outer: Sequence[Sequence[Rational]]
) -> Cert:
    a, b = tuple(map(_interval, inner)), tuple(map(_interval, outer))
    if (
        not a
        or len(a) != len(b)
        or any(not (y[0] < x[0] <= x[1] < y[1]) for x, y in zip(a, b, strict=True))
    ):
        raise ValueError("box is not strictly included")
    return _seal(
        "strict_box_inclusion",
        {"inner": [_iv_payload(x) for x in a], "outer": [_iv_payload(x) for x in b]},
    )


def interval_error_budget_certificate(error: Sequence[Rational], budget: Rational) -> Cert:
    iv, bound = _interval(error), rational(budget)
    if bound < 0 or iv[0] < -bound or iv[1] > bound:
        raise ValueError("error exceeds budget")
    return _seal("interval_error_budget", {"error": _iv_payload(iv), "budget": _q(bound)})


def _iadd(a: RationalInterval, b: RationalInterval) -> RationalInterval:
    return a[0] + b[0], a[1] + b[1]


def _isum(values: Sequence[RationalInterval]) -> RationalInterval:
    result = (Fraction(0), Fraction(0))
    for value in values:
        result = _iadd(result, value)
    return result


def _krawczyk(
    center: tuple[Fraction, ...],
    gradient: tuple[RationalInterval, ...],
    hessian: tuple[tuple[RationalInterval, ...], ...],
    preconditioner: Matrix,
    box: tuple[RationalInterval, ...],
) -> tuple[tuple[RationalInterval, ...], bool]:
    n = len(center)
    if (
        not n
        or len(gradient) != n
        or len(hessian) != n
        or len(preconditioner) != n
        or len(box) != n
    ):
        raise ValueError("Krawczyk dimensions differ")
    if any(len(row) != n for row in hessian) or any(len(row) != n for row in preconditioner):
        raise ValueError("Krawczyk matrix dimensions differ")
    if any(not lo <= c <= hi for c, (lo, hi) in zip(center, box, strict=True)):
        raise ValueError("Krawczyk center lies outside box")
    residual = tuple(
        tuple(
            _isub(
                (Fraction(int(i == j)), Fraction(int(i == j))),
                _isum([_imul((preconditioner[i][k],) * 2, hessian[k][j]) for k in range(n)]),
            )
            for j in range(n)
        )
        for i in range(n)
    )
    image = tuple(
        _iadd(
            _isub(
                (center[i],) * 2,
                _isum([_imul((preconditioner[i][j],) * 2, gradient[j]) for j in range(n)]),
            ),
            _isum([_imul(residual[i][j], _isub(box[j], (center[j],) * 2)) for j in range(n)]),
        )
        for i in range(n)
    )
    contraction = all(sum(max(abs(lo), abs(hi)) for lo, hi in row) < 1 for row in residual)
    return image, contraction


def krawczyk_replay_certificate(
    center: Sequence[Rational],
    gradient: Sequence[Sequence[Rational]],
    hessian: Sequence[Sequence[Sequence[Rational]]],
    preconditioner: Sequence[Sequence[Rational]],
    box: Sequence[Sequence[Rational]],
    *,
    require_contraction: bool = True,
    stationary_box: Sequence[Sequence[Rational]] | None = None,
) -> Cert:
    """Replay K=X0-R*g+(I-R*H)(X-X0), strict inclusion and invertibility.

    Gradient/Hessian enclosure provenance and differentiability are explicit
    analytic dependencies; Lean recomputes the complete finite interval image.
    """
    c = tuple(map(rational, center))
    g = tuple(map(_interval, gradient))
    h = tuple(tuple(map(_interval, row)) for row in hessian)
    r, bounds = matrix(preconditioner), tuple(map(_interval, box))
    image, contraction = _krawczyk(c, g, h, r, bounds)
    if not determinant(r) or any(
        not b[0] < a[0] <= a[1] < b[1] for a, b in zip(image, bounds, strict=True)
    ):
        raise ValueError("Krawczyk image is not strictly included or preconditioner is singular")
    if require_contraction and not contraction:
        raise ValueError("Krawczyk row contraction was not proved")
    root_box = None if stationary_box is None else tuple(map(_interval, stationary_box))
    if root_box is not None and (
        len(root_box) != len(image)
        or any(not b[0] <= a[0] <= a[1] <= b[1] for a, b in zip(image, root_box, strict=True))
    ):
        raise ValueError("claimed stationary box does not contain the Krawczyk image")
    return _seal(
        "krawczyk_replay",
        {
            "center": [_q(x) for x in c],
            "gradient": [_iv_payload(x) for x in g],
            "hessian": [[_iv_payload(x) for x in row] for row in h],
            "preconditioner": [[_q(x) for x in row] for row in r],
            "box": [_iv_payload(x) for x in bounds],
            "require_contraction": require_contraction,
            "stationary_box": None if root_box is None else [_iv_payload(x) for x in root_box],
        },
    )


def _fraction(value: Any) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(x) is not int for x in value)
        or value[1] <= 0
    ):
        raise ValueError("malformed rational")
    return Fraction(value[0], value[1])


def _matrix(value: Any) -> Matrix:
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value):
        raise ValueError("malformed matrix")
    return tuple(tuple(_fraction(x) for x in row) for row in value)


def _indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("malformed indices")
    return tuple(value)


def _polynomial(value: Any) -> SparsePolynomial:
    if (
        not isinstance(value, dict)
        or type(value.get("nvars")) is not int
        or not isinstance(value.get("terms"), list)
    ):
        raise ValueError("malformed polynomial")
    terms: dict[tuple[int, ...], Fraction] = {}
    for term in value["terms"]:
        if not isinstance(term, list) or len(term) != 2:
            raise ValueError("malformed monomial")
        index = _indices(term[0])
        if index in terms:
            raise ValueError("duplicate monomial")
        terms[index] = _fraction(term[1])
    return SparsePolynomial(value["nvars"], terms)


def _payload(cert: Mapping[str, Any], expected_source_digest: str | None) -> Mapping[str, Any]:
    if not verify_certificate_digest(cert):
        raise ValueError("invalid certificate digest")
    p = cert.get("payload")
    if not isinstance(p, dict) or p.get("type") != "realization_replay":
        raise ValueError("unsupported replay")
    digest = source_digest(p.get("source"))
    if p.get("source_digest") != digest or (
        expected_source_digest is not None and expected_source_digest != digest
    ):
        raise ValueError("source digest mismatch")
    return p


def verify_replay_certificate(
    cert: Mapping[str, Any], *, expected_source_digest: str | None = None
) -> bool:
    """Independently recompute a sealed finite replay in exact Python arithmetic."""
    try:
        p = _payload(cert, expected_source_digest)
        from omnibias.core.proof.realization_algebra_replay import (
            ADVANCED_KINDS,
            verify_algebra_replay,
        )

        if p["kind"] in ADVANCED_KINDS:
            return verify_algebra_replay(cert)
        if p["kind"] == "krawczyk_replay":
            s = p["source"]
            c = tuple(_fraction(x) for x in s["center"])
            g = tuple(_decode_iv(x) for x in s["gradient"])
            h = tuple(tuple(_decode_iv(x) for x in row) for row in s["hessian"])
            r = _matrix(s["preconditioner"])
            bounds = tuple(_decode_iv(x) for x in s["box"])
            image, contraction = _krawczyk(c, g, h, r, bounds)
            if s.get("stationary_box") is not None:
                root_box = tuple(_decode_iv(x) for x in s["stationary_box"])
                if len(root_box) != len(image) or any(
                    not b[0] <= a[0] <= a[1] <= b[1] for a, b in zip(image, root_box, strict=True)
                ):
                    return False
            return (
                type(s["require_contraction"]) is bool
                and bool(determinant(r))
                and all(b[0] < a[0] <= a[1] < b[1] for a, b in zip(image, bounds, strict=True))
                and (not s["require_contraction"] or contraction)
            )
        if p["kind"] == "interval_ldlt":
            return _positive_pivots([[_decode_iv(x) for x in row] for row in p["source"]])
        if p["kind"] == "strict_box_inclusion":
            a = tuple(_decode_iv(x) for x in p["source"]["inner"])
            b = tuple(_decode_iv(x) for x in p["source"]["outer"])
            return (
                bool(a)
                and len(a) == len(b)
                and all(y[0] < x[0] <= x[1] < y[1] for x, y in zip(a, b, strict=True))
            )
        if p["kind"] == "interval_error_budget":
            lo, hi = _decode_iv(p["source"]["error"])
            bound = _fraction(p["source"]["budget"])
            return bound >= 0 and -bound <= lo <= hi <= bound
        if p["kind"] == "matrix_rank":
            if type(p["rank"]) is not int:
                return False
            return ExactRankWitness(
                _matrix(p["source"]),
                p["rank"],
                _matrix(p["left"]),
                _matrix(p["right"]),
                _indices(p["minor_rows"]),
                _indices(p["minor_cols"]),
                _fraction(p["minor_value"]),
            ).verify()
        polys = tuple(_polynomial(f) for f in p["source"])
        point = tuple(_fraction(x) for x in p["point"])
        values = tuple(f.evaluate(point) for f in polys)
        if p["kind"] == "polynomial_evaluation":
            return values == tuple(_fraction(x) for x in p["expected"])
        if p["kind"] == "polynomial_inequality":
            rels = p["relations"]
            return len(values) == len(rels) and all(
                value == 0
                if rel == "eq"
                else value >= 0
                if rel == "ge"
                else value > 0
                if rel == "gt"
                else False
                for value, rel in zip(values, rels, strict=True)
            )
        return False
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, AlgebraBudgetExceeded):
        return False


def _literal(q: Fraction) -> str:
    return f"(({q.numerator} : Rat) / {q.denominator})"


def _list(items: Sequence[str]) -> str:
    return "[" + ", ".join(items) + "]"


def _matrix_literal(a: Matrix) -> str:
    return _list([_list([_literal(x) for x in row]) for row in a])


def _iv_literal(iv: RationalInterval) -> str:
    return f"({_literal(iv[0])}, {_literal(iv[1])})"


def generate_replay_obligation(cert: Mapping[str, Any], *, mathlib: bool = False) -> str | None:
    """Emit even a false, structurally valid claim so Lean rejects its arithmetic.

    Only exact integers parsed by this module can enter generated source.
    Malformed, unsealed or operand-substituted payloads are refused beforehand.
    """
    try:
        p = _payload(cert, None)
        from omnibias.core.proof.realization_algebra_replay import (
            ADVANCED_KINDS,
            generate_algebra_obligation,
        )

        if p["kind"] in ADVANCED_KINDS:
            return generate_algebra_obligation(cert, mathlib=mathlib)
        namespace = "OmnibiasAnalytic" if mathlib else "Omnibias"
        body: str
        if p["kind"] == "krawczyk_replay":
            s = p["source"]
            center = tuple(_fraction(x) for x in s["center"])
            if not center or type(s["require_contraction"]) is not bool:
                return None
            c_lit = _list([_literal(x) for x in center])
            g_lit = _list([_iv_literal(_decode_iv(x)) for x in s["gradient"]])
            h_lit = _list(
                [_list([_iv_literal(_decode_iv(x)) for x in row]) for row in s["hessian"]]
            )
            r_lit = _matrix_literal(_matrix(s["preconditioner"]))
            b_lit = _list([_iv_literal(_decode_iv(x)) for x in s["box"]])
            body = (
                f"def center : List Rat := {c_lit}\ndef gradient : List RI := {g_lit}\n"
                f"def hessian : RIMatrix := {h_lit}\ndef R : Matrix := {r_lit}\ndef box : List RI := {b_lit}\n"
                "theorem obligation : krawczykDataValid center gradient hessian R box = true ∧ "
                f"determinant {len(center)} R ≠ 0 ∧ "
                "strictIncluded (krawczykImage center gradient hessian R box) box = true"
            )
            if s["require_contraction"]:
                body += f" ∧ rowContraction (krawczykResidual hessian R {len(center)}) = true"
            if s.get("stationary_box") is not None:
                root_box_literal = _list([_iv_literal(_decode_iv(x)) for x in s["stationary_box"]])
                body += f" ∧ included (krawczykImage center gradient hessian R box) {root_box_literal} = true"
            body += " := by decide +kernel\n"
        elif p["kind"] == "interval_ldlt":
            intervals = [[_decode_iv(x) for x in row] for row in p["source"]]
            if not intervals:
                return None
            literal = _list([_list([_iv_literal(x) for x in row]) for row in intervals])
            body = (
                f"def A : RIMatrix := {literal}\ntheorem obligation : "
                f"validSymmetric A {len(intervals)} = true ∧ positivePivots {len(intervals)} A = true := by decide +kernel\n"
            )
        elif p["kind"] == "strict_box_inclusion":
            inner = _list([_iv_literal(_decode_iv(x)) for x in p["source"]["inner"]])
            outer = _list([_iv_literal(_decode_iv(x)) for x in p["source"]["outer"]])
            if not p["source"]["inner"]:
                return None
            body = (
                f"theorem obligation : strictIncluded {inner} {outer} = true := by decide +kernel\n"
            )
        elif p["kind"] == "interval_error_budget":
            error = _iv_literal(_decode_iv(p["source"]["error"]))
            budget = _literal(_fraction(p["source"]["budget"]))
            body = (
                f"theorem obligation : errorBudget {error} {budget} = true := by decide +kernel\n"
            )
        elif p["kind"] == "matrix_rank":
            a, b, c = _matrix(p["source"]), _matrix(p["left"]), _matrix(p["right"])
            rows, cols, r = _indices(p["minor_rows"]), _indices(p["minor_cols"]), p["rank"]
            if not a or not a[0] or type(r) is not int or r < 0 or r > min(len(a), len(a[0])):
                return None
            if len(rows) != r or len(cols) != r or len(set(rows)) != r or len(set(cols)) != r:
                return None
            if any(i >= len(a) for i in rows) or any(j >= len(a[0]) for j in cols):
                return None
            ri, ci = _list([str(i) for i in rows]), _list([str(i) for i in cols])
            value = _literal(_fraction(p["minor_value"]))
            body = (
                f"def A : Matrix := {_matrix_literal(a)}\ndef B : Matrix := {_matrix_literal(b)}\n"
                f"def C : Matrix := {_matrix_literal(c)}\n"
                f"theorem obligation : shape A {len(a)} {len(a[0])} = true ∧ "
                f"shape B {len(a)} {r} = true ∧ shape C {r} {len(a[0])} = true ∧ "
                f"A = multiply B C {len(a[0])} ∧ "
                f"determinant {r} (minor A {ri} {ci}) = {value} ∧ {value} ≠ 0 := by decide +kernel\n"
            )
        elif p["kind"] in ("polynomial_evaluation", "polynomial_inequality"):
            polys = tuple(_polynomial(f) for f in p["source"])
            point = tuple(_fraction(x) for x in p["point"])
            if not polys or any(f.nvars != len(point) for f in polys):
                return None
            definitions = [f"def point : List Rat := {_list([_literal(x) for x in point])}"]
            for i, f in enumerate(polys):
                terms = _list(
                    [f"({_list([str(k) for k in idx])}, {_literal(q)})" for idx, q in f.terms]
                )
                definitions.append(f"def p{i} : Polynomial := {terms}")
            claims: list[str] = []
            if p["kind"] == "polynomial_evaluation":
                expected = tuple(_fraction(x) for x in p["expected"])
                if len(polys) != len(expected):
                    return None
                claims = [f"evaluate p{i} point = {_literal(q)}" for i, q in enumerate(expected)]
            else:
                rels = p["relations"]
                if len(polys) != len(rels) or any(rel not in ("eq", "ge", "gt") for rel in rels):
                    return None
                symbols = {"eq": "=", "ge": "≥", "gt": ">"}
                claims = [f"evaluate p{i} point {symbols[rel]} 0" for i, rel in enumerate(rels)]
            body = (
                "\n".join(definitions)
                + "\ntheorem obligation : "
                + " ∧ ".join(claims)
                + " := by decide +kernel\n"
            )
        else:
            return None
        return (
            f"import {namespace}.RealizationReplay\nset_option maxRecDepth 100000\n"
            f"set_option maxHeartbeats 0\nopen {namespace}.RealizationReplay\n"
            f"namespace {namespace}.Generated\n{body}end {namespace}.Generated\n"
        )
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, AlgebraBudgetExceeded):
        return None


__all__ = [
    "generate_replay_obligation",
    "interval_error_budget_certificate",
    "interval_ldlt_replay_certificate",
    "krawczyk_replay_certificate",
    "polynomial_evaluation_certificate",
    "polynomial_inequality_certificate",
    "rank_replay_certificate",
    "source_digest",
    "strict_box_inclusion_certificate",
    "verify_replay_certificate",
]

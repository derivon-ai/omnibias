# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generic finite replay of algebraic witnesses, Laurent paths and box trees.

Sturm chain arithmetic is checked, while Sturm's theorem connecting variation
counts to real roots is an explicit mathematical dependency. Likewise the
finite box computation does not formalize the general real interval theorem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import Cert
from omnibias.core.proof.realization_replay import (
    _decode_iv,
    _fraction,
    _interval,
    _iv_literal,
    _iv_payload,
    _list,
    _literal,
    _payload,
    _polynomial,
    _q,
    _seal,
)
from omnibias.core.realization.algebraic import (
    RealAlgebraicField,
    UPoly,
    add,
    divide,
    multiply,
    sturm_sequence,
    trim,
)
from omnibias.core.realization.polynomial import (
    AlgebraBudgetExceeded,
    PolynomialRealizationMap,
    Rational,
    SparsePolynomial,
    rational,
)
from omnibias.core.realization.witness import AlgebraicRealizationWitness, LaurentClosureWitness

ADVANCED_KINDS = frozenset(
    {"sturm_arithmetic", "algebraic_substitution", "laurent_cancellation", "polynomial_box"}
)


def _sturm(field: RealAlgebraicField) -> dict[str, object]:
    chain = sturm_sequence(field.polynomial)
    quotients = [divide(a, b)[0] for a, b in zip(chain[:-1], chain[1:], strict=True)]
    return {
        "primitive": [_q(rational(x)) for x in field.polynomial],
        "interval": [_q(rational(field.lo)), _q(rational(field.hi))],
        "chain": [[_q(x) for x in p] for p in chain],
        "quotients": [[_q(x) for x in p] for p in quotients],
    }


def sturm_isolation_replay_certificate(field: RealAlgebraicField) -> Cert:
    """Replay derivative, Euclidean Sturm chain, endpoints and variation counts."""
    return _seal("sturm_arithmetic", _sturm(field))


def moment_identity_replay_certificate(
    offsets: Sequence[Rational],
    coefficients: Sequence[Rational],
    *,
    center: Rational,
    expected_moments: Sequence[Rational],
) -> Cert:
    """Replay normalized collision moments from every original rational operand."""
    from math import factorial

    from omnibias.core.proof.realization_replay import polynomial_evaluation_certificate

    n = len(offsets)
    if not n or len(coefficients) != n or not expected_moments:
        raise ValueError("nonempty matching offsets/coefficients and moments required")
    point = tuple(map(rational, coefficients)) + tuple(map(rational, offsets)) + (rational(center),)
    variables = tuple(SparsePolynomial.variable(2 * n + 1, i) for i in range(2 * n + 1))
    polys = []
    for order in range(len(expected_moments)):
        value = SparsePolynomial.constant(2 * n + 1, 0)
        for i in range(n):
            value = value + variables[i] * (variables[n + i] - variables[-1]) ** order * Fraction(
                1, factorial(order)
            )
        polys.append(value)
    return polynomial_evaluation_certificate(polys, point, expected_moments)


def _compose(poly: SparsePolynomial, parameters: Sequence[UPoly]) -> UPoly:
    result: UPoly = ()
    for powers, coefficient in poly.terms:
        term: UPoly = (coefficient,)
        for parameter, n in zip(parameters, powers, strict=True):
            for _ in range(n):
                term = multiply(term, parameter)
        result = add(result, term)
    return result


def algebraic_realization_replay_certificate(
    source: PolynomialRealizationMap, witness: AlgebraicRealizationWitness
) -> Cert:
    """Substitute algebraic parameters and verify divisibility by their primitive.

    A reducible primitive for which equality holds only at its selected root is
    unsupported by this finite divisibility family; refine the primitive first.
    """
    if not witness.verify(source):
        raise ValueError("invalid algebraic realization witness")
    primitive = trim(witness.field.polynomial)
    parameters = tuple(p.coefficients for p in witness.parameters)
    quotients = []
    for p, target in zip(source.polynomials, witness.target, strict=True):
        quotient, remainder = divide(add(_compose(p, parameters), (-target,)), primitive)
        if remainder:
            raise NotImplementedError(
                "refine reducible primitive: selected-root equality is not polynomial divisibility"
            )
        quotients.append(quotient)
    return _seal(
        "algebraic_substitution",
        {
            "fingerprint": source.fingerprint,
            "root": _sturm(witness.field),
            "polynomials": [p.to_payload() for p in source.polynomials],
            "parameters": [[_q(x) for x in p] for p in parameters],
            "target": [_q(x) for x in witness.target],
        },
        quotients=[[_q(x) for x in p] for p in quotients],
    )


def laurent_closure_replay_certificate(
    source: PolynomialRealizationMap,
    witness: LaurentClosureWitness,
    *,
    radius: Rational | None = None,
) -> Cert:
    """Replay cancellation of every pole, target coefficients and tail budgets."""
    if not witness.verify(source):
        raise ValueError("invalid Laurent closure witness")
    rho = witness.radius if radius is None else rational(radius)
    errors = witness.coefficient_error_bounds(source, rho)
    return _seal(
        "laurent_cancellation",
        {
            "fingerprint": source.fingerprint,
            "polynomials": [p.to_payload() for p in source.polynomials],
            "parameters": [[[k, _q(c)] for k, c in p.terms] for p in witness.parameters],
            "target": [_q(x) for x in witness.target],
            "radius": _q(rho),
        },
        errors=[_q(x) for x in errors],
    )


def polynomial_box_replay_certificate(
    polynomials: Sequence[SparsePolynomial],
    box: Sequence[Sequence[Rational]],
    tree: Mapping[str, Any],
) -> Cert:
    """Replay a finite binary cover and an excluded polynomial at every leaf.

    Tree format is ``{equation, enclosure}`` or ``{axis, midpoint, left, right}``;
    all numeric leaves use rational ``[numerator, denominator]`` pairs.
    """
    bounds = tuple(_interval(x) for x in box)
    if not polynomials or any(p.nvars != len(bounds) for p in polynomials):
        raise ValueError("polynomial box dimensions differ")
    certificate = _seal(
        "polynomial_box",
        {
            "polynomials": [p.to_payload() for p in polynomials],
            "box": [_iv_payload(x) for x in bounds],
        },
        tree=dict(tree),
    )
    if not verify_algebra_replay(certificate):
        raise ValueError("invalid polynomial exclusion tree")
    return certificate


def _up(value: Any) -> UPoly:
    if not isinstance(value, list):
        raise ValueError("malformed univariate polynomial")
    return tuple(_fraction(x) for x in value)


def _root(value: Any) -> RealAlgebraicField:
    field = RealAlgebraicField(
        _up(value["primitive"]), _fraction(value["interval"][0]), _fraction(value["interval"][1])
    )
    expected = _sturm(field)
    if value != expected:
        raise ValueError("Sturm chain differs from exact Euclidean replay")
    return field


def _interval_evaluate(
    poly: SparsePolynomial, box: tuple[tuple[Fraction, Fraction], ...]
) -> tuple[Fraction, Fraction]:
    from omnibias.core.proof.realization_replay import _iadd, _imul

    result = (Fraction(0), Fraction(0))
    for powers, c in poly.terms:
        term = (c, c)
        for (lo, hi), n in zip(box, powers, strict=True):
            a, b = lo**n, hi**n
            power = (
                ((Fraction(0) if n and lo <= 0 <= hi else min(a, b)), max(a, b))
                if n % 2 == 0
                else (a, b)
            )
            term = _imul(term, power)
        result = _iadd(result, term)
    return result


def _tree_check(
    polys: tuple[SparsePolynomial, ...], box: tuple[tuple[Fraction, Fraction], ...], tree: Any
) -> bool:
    if not isinstance(tree, dict):
        return False
    if "equation" in tree:
        i = tree["equation"]
        if type(i) is not int or not 0 <= i < len(polys):
            return False
        enclosure = _interval_evaluate(polys[i], box)
        return _decode_iv(tree["enclosure"]) == enclosure and (enclosure[1] < 0 or enclosure[0] > 0)
    axis, mid = tree["axis"], _fraction(tree["midpoint"])
    if type(axis) is not int or not 0 <= axis < len(box) or not box[axis][0] < mid < box[axis][1]:
        return False
    left = box[:axis] + ((box[axis][0], mid),) + box[axis + 1 :]
    right = box[:axis] + ((mid, box[axis][1]),) + box[axis + 1 :]
    return _tree_check(polys, left, tree["left"]) and _tree_check(polys, right, tree["right"])


def verify_algebra_replay(cert: Mapping[str, Any]) -> bool:
    try:
        p = _payload(cert, None)
        s = p["source"]
        if p["kind"] == "sturm_arithmetic":
            _root(s)
            return True
        polys = tuple(_polynomial(x) for x in s["polynomials"])
        if p["kind"] == "polynomial_box":
            box = tuple(_decode_iv(x) for x in s["box"])
            return (
                bool(polys)
                and all(f.nvars == len(box) for f in polys)
                and _tree_check(polys, box, p["tree"])
            )
        target = tuple(_fraction(x) for x in s["target"])
        if not polys or len(target) != len(polys):
            return False
        if p["kind"] == "algebraic_substitution":
            field = _root(s["root"])
            parameters = tuple(_up(x) for x in s["parameters"])
            quotients = tuple(_up(x) for x in p["quotients"])
            return (
                len(quotients) == len(polys)
                and all(f.nvars == len(parameters) for f in polys)
                and all(
                    add(_compose(f, parameters), (-t,)) == multiply(q, trim(field.polynomial))
                    for f, t, q in zip(polys, target, quotients, strict=True)
                )
            )
        if p["kind"] == "laurent_cancellation":
            from omnibias.core.realization.witness import LaurentPolynomial

            rows = s["parameters"]
            if any(
                any(type(term[0]) is not int for term in row)
                or len({term[0] for term in row}) != len(row)
                for row in rows
            ):
                return False
            params = tuple(LaurentPolynomial({k: _fraction(c) for k, c in row}) for row in rows)
            rho, errors = _fraction(s["radius"]), tuple(_fraction(x) for x in p["errors"])
            if rho <= 0 or len(errors) != len(polys) or any(f.nvars != len(params) for f in polys):
                return False
            for f, t, error in zip(polys, target, errors, strict=True):
                path = LaurentPolynomial({})
                for powers, c in f.terms:
                    term = LaurentPolynomial({0: c})
                    for x, n in zip(params, powers, strict=True):
                        term = term * x**n
                    path = path + term
                if (
                    any(k < 0 for k, _ in path.terms)
                    or dict(path.terms).get(0, Fraction(0)) != t
                    or sum(abs(c) * rho**k for k, c in path.terms if k > 0) > error
                ):
                    return False
            return True
        return False
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
        RecursionError,
        AlgebraBudgetExceeded,
    ):
        return False


def _up_literal(p: UPoly) -> str:
    return _list([_literal(x) for x in p])


def _poly_literal(p: SparsePolynomial) -> str:
    return _list([f"({_list([str(n) for n in idx])}, {_literal(c)})" for idx, c in p.terms])


def _root_claim(root: Any) -> str:
    primitive = _up_literal(_up(root["primitive"]))
    chain = _list([_up_literal(_up(p)) for p in root["chain"]])
    quotients = _list([_up_literal(_up(p)) for p in root["quotients"]])
    lo, hi = _fraction(root["interval"][0]), _fraction(root["interval"][1])
    return f"sturmArithmetic {primitive} {chain} {quotients} {_literal(lo)} {_literal(hi)} = true"


def _tree_literal(tree: Any) -> str:
    if "equation" in tree:
        if type(tree["equation"]) is not int or tree["equation"] < 0:
            raise ValueError("invalid equation index")
        return f"(.leaf {tree['equation']} {_iv_literal(_decode_iv(tree['enclosure']))})"
    if type(tree["axis"]) is not int or tree["axis"] < 0:
        raise ValueError("invalid split axis")
    return f"(.split {tree['axis']} {_literal(_fraction(tree['midpoint']))} {_tree_literal(tree['left'])} {_tree_literal(tree['right'])})"


def generate_algebra_obligation(cert: Mapping[str, Any], *, mathlib: bool = False) -> str | None:
    """Generate exact source operands; structurally valid false claims fail Lean."""
    try:
        p = _payload(cert, None)
        s, namespace = p["source"], "OmnibiasAnalytic" if mathlib else "Omnibias"
        claims: list[str] = []
        definitions: list[str] = []
        if p["kind"] == "sturm_arithmetic":
            claims.append(_root_claim(s))
        else:
            polys = tuple(_polynomial(x) for x in s["polynomials"])
            if not polys:
                return None
            definitions = [
                f"def p{i} : Polynomial := {_poly_literal(f)}" for i, f in enumerate(polys)
            ]
            if p["kind"] == "polynomial_box":
                box = tuple(_decode_iv(x) for x in s["box"])
                if any(f.nvars != len(box) for f in polys):
                    return None
                claims.append(
                    f"boxReplay {_list([f'p{i}' for i in range(len(polys))])} {_list([_iv_literal(x) for x in box])} {_tree_literal(p['tree'])} = true"
                )
            elif p["kind"] == "algebraic_substitution":
                parameters = tuple(_up(x) for x in s["parameters"])
                target, quotients = (
                    tuple(_fraction(x) for x in s["target"]),
                    tuple(_up(x) for x in p["quotients"]),
                )
                if (
                    len(target) != len(polys)
                    or len(quotients) != len(polys)
                    or any(f.nvars != len(parameters) for f in polys)
                ):
                    return None
                definitions.append(
                    f"def parameters : List UP := {_list([_up_literal(x) for x in parameters])}"
                )
                primitive = _up_literal(_up(s["root"]["primitive"]))
                claims.append(_root_claim(s["root"]))
                claims.extend(
                    f"uadd (ucompose p{i} parameters) [{_literal(-t)}] = umul {_up_literal(q)} {primitive}"
                    for i, (t, q) in enumerate(zip(target, quotients, strict=True))
                )
            elif p["kind"] == "laurent_cancellation":
                parameters_lit = []
                for row in s["parameters"]:
                    if any(type(term[0]) is not int for term in row):
                        return None
                    parameters_lit.append(
                        _list([f"({term[0]}, {_literal(_fraction(term[1]))})" for term in row])
                    )
                target = tuple(_fraction(x) for x in s["target"])
                errors = tuple(_fraction(x) for x in p["errors"])
                if (
                    len(target) != len(polys)
                    or len(errors) != len(polys)
                    or any(f.nvars != len(parameters_lit) for f in polys)
                ):
                    return None
                definitions.append(f"def parameters : List LP := {_list(parameters_lit)}")
                rho = _literal(_fraction(s["radius"]))
                claims.extend(
                    f"laurentCheck p{i} parameters {_literal(t)} {rho} {_literal(e)} = true"
                    for i, (t, e) in enumerate(zip(target, errors, strict=True))
                )
            else:
                return None
        return (
            f"import {namespace}.RealizationAlgebra\nset_option maxRecDepth 100000\nset_option maxHeartbeats 0\n"
            f"open {namespace}.RealizationReplay {namespace}.RealizationAlgebra\nnamespace {namespace}.Generated\n"
            + "\n".join(definitions)
            + "\ntheorem obligation : "
            + " ∧ ".join(claims)
            + f" := by decide +kernel\nend {namespace}.Generated\n"
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
        RecursionError,
        AlgebraBudgetExceeded,
    ):
        return None


__all__ = [
    "algebraic_realization_replay_certificate",
    "laurent_closure_replay_certificate",
    "moment_identity_replay_certificate",
    "polynomial_box_replay_certificate",
    "sturm_isolation_replay_certificate",
]

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact finite-box coverage for the Hilbert XVI research program.

The implemented theorem is finite polynomial-displacement coverage. This is
useful for checking finite atlas computations but does not assert coverage of
physical cycles, infinity, singular charts, or all polynomial vector fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.realization.polynomial import Rational, SparsePolynomial, rational
from omnibias.dynamics.cyclicity import (
    CyclicityCertificate,
    RationalBox,
    certify_polynomial_cyclicity,
    rational_box,
    verify_cyclicity_certificate,
)


@dataclass(frozen=True)
class CyclicityLeaf:
    certificate: CyclicityCertificate


@dataclass(frozen=True)
class CyclicitySplit:
    """Closed children meet exactly at cut; no missing boundary is permitted."""

    axis: int
    cut: Rational
    left: CyclicityTree
    right: CyclicityTree


CyclicityTree: TypeAlias = CyclicityLeaf | CyclicitySplit


@dataclass(frozen=True)
class PolynomialCoverCertificate:
    domain: RationalBox
    upper_bound: int
    leaves: int
    tree: CyclicityTree
    seal: Cert


def _check_tree(
    polynomial: SparsePolynomial, domain: RationalBox, tree: CyclicityTree
) -> tuple[int, int, object]:
    if isinstance(tree, CyclicityLeaf):
        if not verify_cyclicity_certificate(polynomial, tree.certificate, expected_domain=domain):
            raise ValueError("leaf source, domain, or zero count does not replay")
        return tree.certificate.upper_bound, 1, {"leaf": tree.certificate.seal}
    if not isinstance(tree, CyclicitySplit):
        raise TypeError("coverage requires a binary split tree with certified leaves")
    if type(tree.axis) is not int or not 0 <= tree.axis < len(domain):
        raise ValueError("invalid split axis")
    cut = rational(tree.cut)
    lo, hi = domain[tree.axis]
    if not lo < cut < hi:
        raise ValueError("split must be strictly inside the parent interval")
    left, right = list(domain), list(domain)
    left[tree.axis], right[tree.axis] = (lo, cut), (cut, hi)
    bl, nl, pl = _check_tree(polynomial, tuple(left), tree.left)
    br, nr, pr = _check_tree(polynomial, tuple(right), tree.right)
    # Height pieces contain different zeros; parameters choose a fiber. Closed
    # height boundaries can be counted twice, harmless for an upper bound.
    bound = bl + br if tree.axis == 0 else max(bl, br)
    return bound, nl + nr, {
        "axis": tree.axis,
        "cut": [cut.numerator, cut.denominator],
        "left": pl,
        "right": pr,
    }


def certify_polynomial_cover(
    polynomial: SparsePolynomial,
    box: Sequence[Sequence[Rational]],
    tree: CyclicityTree,
) -> PolynomialCoverCertificate:
    """Replay every leaf and exact split, then assemble the finite zero bound."""
    domain = rational_box(box, polynomial.nvars)
    # Validate the root and cap overcounting by the independently valid degree
    # or Rolle/Sturm bound. This does not repair an invalid or missing leaf.
    root = certify_polynomial_cyclicity(polynomial, domain)
    bound, leaves, payload = _check_tree(polynomial, domain, tree)
    bound = min(bound, root.upper_bound)
    seal = make_certificate(
        claim="Finite polynomial-displacement box cover; physical cycle capture remains separate.",
        payload={
            "type": "polynomial_cyclicity_cover",
            "root": root.seal,
            "tree": payload,
            "upper_bound": bound,
            "leaves": leaves,
        },
        meta={"transcend_backend": "not_used"},
    )
    return PolynomialCoverCertificate(domain, bound, leaves, tree, seal)


def verify_polynomial_cover(
    polynomial: SparsePolynomial, certificate: PolynomialCoverCertificate
) -> bool:
    try:
        return verify_certificate_digest(certificate.seal) and certificate == certify_polynomial_cover(
            polynomial, certificate.domain, certificate.tree
        )
    except (TypeError, ValueError, ArithmeticError, KeyError):
        return False


__all__ = [
    "CyclicityLeaf",
    "CyclicitySplit",
    "CyclicityTree",
    "PolynomialCoverCertificate",
    "certify_polynomial_cover",
    "verify_polynomial_cover",
]

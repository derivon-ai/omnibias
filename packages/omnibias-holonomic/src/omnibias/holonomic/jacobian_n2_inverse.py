# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gabber inverse-degree test for polynomial maps ``Q^2 -> Q^2``.

Gabber's theorem: if ``F`` is a polynomial automorphism of ``A^n``, then
``deg(F^{-1}) <= (deg F)^{n-1}``. For ``n=2`` the bound is ``deg F``.

The unique formal inverse at the origin exists whenever ``F(0)=0`` and
``DF(0)`` is invertible. Truncating that series at Gabber's bound and
testing the polynomial identities ``F ∘ G = id`` and ``G ∘ F = id`` is
therefore a finite decision procedure on one map:

* if the identities hold, ``F`` is an automorphism (not a Jacobian
  counterexample);
* if ``det JF`` is a nonzero constant and the identities fail, ``F`` is
  not an automorphism, so it is a genuine ``n=2`` counterexample.

The parent conjecture is not a theorem of this module. A miss on any
finite family is not the parent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.holonomic._core.poly_n import (
    PolyN,
    eval_map,
    identical_jacobian_constant,
)

Rational = Fraction | int


def map_degree(components: Sequence[PolyN]) -> int:
    """Maximum total degree of a square polynomial map."""

    if not components:
        raise ValueError("map_degree expects a non-empty map")
    return max(component.total_degree() for component in components)


def gabber_inverse_degree_bound(degree: int, nvars: int = 2) -> int:
    """``(deg F)^{n-1}``; ``n=2`` returns ``degree`` itself."""

    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    if nvars < 1:
        raise ValueError(f"nvars must be >= 1, got {nvars}")
    return int(degree ** (nvars - 1))


def shift_to_origin(components: Sequence[PolyN]) -> tuple[PolyN, ...]:
    """Translate the target so the map sends the origin to the origin."""

    if not components:
        raise ValueError("shift_to_origin expects a non-empty map")
    nvars = components[0].nvars
    origin = (0,) * nvars
    values = eval_map(components, origin)
    return tuple(
        component - PolyN.const(nvars, value)
        for component, value in zip(components, values, strict=True)
    )


def compose_map(
    left: Sequence[PolyN], right: Sequence[PolyN]
) -> tuple[PolyN, ...]:
    """``left ∘ right``."""

    if len(left) != len(right):
        raise ValueError("compose_map expects square maps of the same size")
    return tuple(component.compose(tuple(right)) for component in left)


def is_identity_map(components: Sequence[PolyN]) -> bool:
    nvars = len(components)
    if nvars == 0 or any(component.nvars != nvars for component in components):
        return False
    return all(
        component == PolyN.var(nvars, index)
        for index, component in enumerate(components)
    )


def linear_part_matrix(components: Sequence[PolyN]) -> list[list[Fraction]]:
    """Matrix of the degree-1 homogeneous part (``DF(0)`` after a shift)."""

    nvars = len(components)
    matrix = [[Fraction(0)] * nvars for _ in range(nvars)]
    for i, component in enumerate(components):
        for j in range(nvars):
            exp = [0] * nvars
            exp[j] = 1
            matrix[i][j] = component.terms.get(tuple(exp), Fraction(0))
    return matrix


def _invert_2x2(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if det == 0:
        raise ValueError("linear part is singular")
    return [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det],
    ]


def _apply_linear(
    matrix: list[list[Fraction]], vec: Sequence[PolyN]
) -> list[PolyN]:
    nvars = len(vec)
    out: list[PolyN] = []
    for i in range(nvars):
        acc = PolyN.zero(nvars)
        for j in range(nvars):
            coeff = matrix[i][j]
            if coeff:
                acc = acc + vec[j] * coeff
        out.append(acc)
    return out


def formal_inverse(
    components: Sequence[PolyN],
    *,
    max_degree: int,
) -> tuple[PolyN, ...]:
    """Unique origin-centered inverse jet through ``max_degree``.

    ``components`` must already send the origin to the origin and have
    an invertible linear part. The result is a polynomial of degree at
    most ``max_degree``; it is the inverse iff that inverse is polynomial
    of degree ``<= max_degree``.
    """

    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")
    if len(components) != 2:
        raise ValueError("formal_inverse is implemented only for n=2")
    if any(component.nvars != 2 for component in components):
        raise ValueError("formal_inverse expects two bivariate components")
    shifted = shift_to_origin(components)
    linear = linear_part_matrix(shifted)
    inverse_linear = _invert_2x2(linear)
    x, y = PolyN.var(2, 0), PolyN.var(2, 1)
    current = _apply_linear(inverse_linear, (x, y))
    for order in range(2, max_degree + 1):
        composed = compose_map(shifted, current)
        error = [part.homogeneous_part(order) for part in composed]
        current = [
            current[i] + correction
            for i, correction in enumerate(
                _apply_linear(inverse_linear, (-error[0], -error[1]))
            )
        ]
    return tuple(current)


@dataclass(frozen=True)
class GabberN2Result:
    """Finite Gabber test on one ``n=2`` map."""

    keller: bool
    jacobian_constant: Fraction | None
    degree: int
    bound: int
    inverse_ok: bool
    fails: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "gabber_bound": self.bound,
            "gabber_degree": self.degree,
            "gabber_inverse_ok": self.inverse_ok,
            "gabber_fails": self.fails,
            "gabber_keller": self.keller,
            "gabber_jacobian_constant": (
                "" if self.jacobian_constant is None else str(self.jacobian_constant)
            ),
        }


def gabber_n2_test(components: Sequence[PolyN]) -> GabberN2Result:
    """Run the Gabber identities on a bivariate polynomial map."""

    if len(components) != 2 or any(component.nvars != 2 for component in components):
        raise ValueError("gabber_n2_test expects a map Q^2 -> Q^2")
    constant = identical_jacobian_constant(components)
    keller = constant is not None and constant != 0
    degree = max(map_degree(components), 0)
    if not keller or degree < 1:
        return GabberN2Result(
            keller=False,
            jacobian_constant=constant,
            degree=degree,
            bound=0,
            inverse_ok=False,
            fails=False,
        )
    bound = gabber_inverse_degree_bound(degree, nvars=2)
    shifted = shift_to_origin(components)
    inverse = formal_inverse(shifted, max_degree=bound)
    forward = compose_map(shifted, inverse)
    backward = compose_map(inverse, shifted)
    inverse_ok = is_identity_map(forward) and is_identity_map(backward)
    return GabberN2Result(
        keller=True,
        jacobian_constant=constant,
        degree=degree,
        bound=bound,
        inverse_ok=inverse_ok,
        fails=not inverse_ok,
    )


__all__ = [
    "GabberN2Result",
    "compose_map",
    "formal_inverse",
    "gabber_inverse_degree_bound",
    "gabber_n2_test",
    "is_identity_map",
    "linear_part_matrix",
    "map_degree",
    "shift_to_origin",
]

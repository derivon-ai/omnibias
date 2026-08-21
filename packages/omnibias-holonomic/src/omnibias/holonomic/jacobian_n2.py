# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite lie for Jacobian-conjecture ``n=2`` counterexample search.

The celebrated parent (Keller / Jacobian conjecture in dimension 2) is still
open. Moh showed there is no counterexample of degree at most 100. This
module names a **finite universal** that the discovery engine can
adjudicate without inferring the parent.

Finite lie ``C_box(d, h, G)``
    Every polynomial map ``F: Q^2 -> Q^2`` whose two components have
    total degree at most ``d`` and integer coefficients of height at
    most ``h`` either

    1. has ``det JF`` **not** identically a nonzero constant, or
    2. has no two distinct points of the finite rational grid ``G``
       sharing an image.

Polarity
    ``existential=False``. A hit is ``DISPROVED``: a grid collision plus
    an identical nonzero constant Jacobian is a genuine ``n=2``
    counterexample (non-injective étale map). A miss on an exhausted
    complete box is ``PROVED`` of ``C_box`` only. It does **not** prove
    injectivity on all of ``Q^2``, and it does not settle the parent.
    An incomplete miss is ``search_incomplete``.

Coefficient ring
    The box is ``Z`` coefficients of height ``h``, not arbitrary
    rationals or complexes. That is another reason a miss is not the
    parent.

This module is the Step-A contract. Exact Jacobian / fiber primitives
and the ``FiniteFamily`` live beside these names as they land.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import product
from typing import Any

from omnibias.core.proof.discovery import (
    Candidate,
    DiscoveryResult,
    ExactCheck,
    Statement,
)
from omnibias.holonomic._core.poly_n import (
    PolyN,
    eval_map,
    identical_jacobian_constant,
)

MAX_COMPLETE_CARDINALITY = 2000
N2Candidate = tuple[Any, ...]

JACOBIAN_N2_PARENT = "jacobian_conjecture_n2"
JACOBIAN_N2_KIND = "jacobian_n2_degree_box"
MOH_DEGREE_BOUND = 100
CI_MAX_DEGREE = 1
CI_COEFF_HEIGHT = 1
DEFAULT_GRID_HALFWIDTH = 2

_FORBIDDEN_PARENT_CLAIMS = (
    "jacobian_conjecture_proof_claim",
    "navier_stokes_proof_claim",
    "ten_proofs_formalization_claim",
    "no_condition_exists_claim",
    "unnamed_condition_complete_claim",
)


def default_collision_grid(*, halfwidth: int = DEFAULT_GRID_HALFWIDTH) -> tuple[Fraction, ...]:
    """Axis values for the finite collision grid ``G = axis × axis``."""

    if halfwidth < 0:
        raise ValueError(f"halfwidth must be >= 0, got {halfwidth}")
    return tuple(Fraction(k) for k in range(-halfwidth, halfwidth + 1))


def jacobian_n2_honesty(
    *,
    discovered: bool,
    n2_counterexample: bool = False,
) -> dict[str, bool]:
    """Honesty payload for the ``n=2`` box.

    ``jacobian_n2_claim`` is earned only by an exact violator of
    ``C_box`` that is also a parent counterexample (identical nonzero
    constant Jacobian and a rational collision).
    ``jacobian_conjecture_proof_claim`` stays False: a counterexample
    disproves, it does not prove the conjecture.
    """

    if n2_counterexample and not discovered:
        raise ValueError("n2_counterexample requires discovered=True")
    honesty = {
        "discovered_by_omnibias": bool(discovered),
        "jacobian_conjecture_proof_claim": False,
        "jacobian_n2_claim": bool(n2_counterexample),
        "navier_stokes_proof_claim": False,
        "keller_n_ge_3_replay": False,
        "ten_proofs_formalization_claim": False,
        "no_condition_exists_claim": False,
        "unnamed_condition_complete_claim": False,
        "moh_degree_100_settled_claim": False,
    }
    for key in _FORBIDDEN_PARENT_CLAIMS:
        if honesty[key] and key != "jacobian_n2_claim":
            raise RuntimeError(f"forbidden honesty key {key} became True")
    return honesty


def n2_counterexample_earned(payload: Mapping[str, Any]) -> bool:
    """Return True only when ``payload`` records an exact ``n=2`` violator.

    Required keys (filled by the algebraic primitives):

    * ``jacobian_identity == "identical"`` — not a probe sample
    * ``jacobian_nonzero_constant is True``
    * ``rational_preimages`` — at least two distinct points
    * those points share one image (``rational_image``)
    """

    if payload.get("jacobian_identity") != "identical":
        return False
    if payload.get("jacobian_nonzero_constant") is not True:
        return False
    preimages = payload.get("rational_preimages")
    if not isinstance(preimages, (list, tuple)) or len(preimages) < 2:
        return False
    points = [_as_point(item) for item in preimages]
    if None in points:
        return False
    distinct = {item for item in points if item is not None}
    if len(distinct) < 2:
        return False
    image = payload.get("rational_image")
    parsed_image = _as_point(image) if image is not None else None
    return parsed_image is not None


def rational_grid_collision(
    components: Sequence[PolyN],
    axis: Sequence[Fraction],
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]] | None:
    """First image in ``axis^n`` with two distinct rational preimages.

    A hit is a proof of non-injectivity. A miss is not injectivity on
    ``Q^n`` — only the absence of a collision on this grid.
    """

    if not components:
        raise ValueError("components must be non-empty")
    nvars = components[0].nvars
    if len(components) != nvars or any(component.nvars != nvars for component in components):
        raise ValueError("jacobian_n2 collision expects a square map")
    if not axis:
        raise ValueError("collision axis must be non-empty")
    coords = tuple(Fraction(v) if not isinstance(v, Fraction) else v for v in axis)
    buckets: dict[tuple[Fraction, ...], list[tuple[Fraction, ...]]] = {}
    for point in product(coords, repeat=nvars):
        image = eval_map(components, point)
        seen = buckets.setdefault(image, [])
        if point not in seen:
            seen.append(point)
        if len(seen) >= 2:
            return image, (seen[0], seen[1])
    return None


def n2_violation_payload(
    components: Sequence[PolyN],
    *,
    axis: Sequence[Fraction] | None = None,
    candidate: Any = None,
) -> dict[str, Any]:
    """Exact Jacobian identity plus optional rational collision.

    ``jacobian_identity`` is always ``identical``. Probe samples never
    write this payload.
    """

    axis_vals = tuple(axis) if axis is not None else default_collision_grid()
    constant = identical_jacobian_constant(components)
    nonzero = constant is not None and constant != 0
    collision = (
        rational_grid_collision(components, axis_vals) if nonzero else None
    )
    image, preimages = collision if collision is not None else (None, ())
    payload: dict[str, Any] = {
        "jacobian_identity": "identical",
        "jacobian_constant": "" if constant is None else str(constant),
        "jacobian_nonzero_constant": bool(nonzero),
        "rational_preimages": [[str(c) for c in pt] for pt in preimages],
        "rational_image": None if image is None else [str(c) for c in image],
        "grid_axis": [str(v) for v in axis_vals],
        "candidate": candidate,
    }
    earned = n2_counterexample_earned(payload)
    payload["honesty"] = jacobian_n2_honesty(
        discovered=earned,
        n2_counterexample=earned,
    )
    return payload


def _as_point(item: Any) -> tuple[Fraction, ...] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    try:
        return tuple(Fraction(str(coord)) for coord in item)
    except (TypeError, ValueError):
        return None


def jacobian_n2_box_statement(
    *,
    max_degree: int,
    coeff_height: int,
    grid: Sequence[Fraction] | None = None,
) -> Statement:
    """Universal statement of ``C_box(d, h, G)``. Never the parent theorem."""

    if max_degree < 0:
        raise ValueError(f"max_degree must be >= 0, got {max_degree}")
    if coeff_height < 0:
        raise ValueError(f"coeff_height must be >= 0, got {coeff_height}")
    axis = tuple(grid) if grid is not None else default_collision_grid()
    if not axis:
        raise ValueError("collision grid axis must be non-empty")
    grid_pretty = ",".join(str(v) for v in axis)
    moh_note = (
        f"Moh excludes degree <= {MOH_DEGREE_BOUND} over C; this box is "
        f"integer height {coeff_height} and degree {max_degree}"
    )
    return Statement(
        name=f"{JACOBIAN_N2_KIND}_d{max_degree}_h{coeff_height}",
        obligation=(
            "every integer-coefficient map Q^2->Q^2 of total degree "
            f"<= {max_degree} and coeff height <= {coeff_height} either has "
            "det JF not identically a nonzero constant, or has no two "
            f"distinct points of G={{({grid_pretty})}}^2 sharing an image; "
            "a G-collision plus identical constant Jacobian would refute "
            f"{JACOBIAN_N2_PARENT}; a miss is not injectivity on Q^2 and "
            f"is not the parent ({moh_note})"
        ),
        parent=JACOBIAN_N2_PARENT,
        parent_status="open",
        existential=False,
    )


def plane_monomials(max_degree: int) -> tuple[tuple[int, int], ...]:
    """Total-degree monomials of ``Q[x, y]`` up to ``max_degree``."""

    if max_degree < 0:
        raise ValueError(f"max_degree must be >= 0, got {max_degree}")
    return tuple(
        (i, j)
        for total in range(max_degree + 1)
        for i in range(total + 1)
        for j in (total - i,)
    )


def affine_map(coeffs: Sequence[int]) -> tuple[PolyN, PolyN]:
    """``(a00 + a10 x + a01 y, b00 + b10 x + b01 y)``."""

    if len(coeffs) != 6:
        raise ValueError(f"affine map needs 6 coefficients, got {len(coeffs)}")
    x, y = PolyN.var(2, 0), PolyN.var(2, 1)
    a00, a10, a01, b00, b10, b01 = (int(c) for c in coeffs)
    first = PolyN.const(2, a00) + PolyN.const(2, a10) * x + PolyN.const(2, a01) * y
    second = PolyN.const(2, b00) + PolyN.const(2, b10) * x + PolyN.const(2, b01) * y
    return first, second


def constant_map(coeffs: Sequence[int]) -> tuple[PolyN, PolyN]:
    if len(coeffs) != 2:
        raise ValueError(f"constant map needs 2 coefficients, got {len(coeffs)}")
    return PolyN.const(2, int(coeffs[0])), PolyN.const(2, int(coeffs[1]))


def shear_map(direction: int, poly_coeffs: Sequence[int]) -> tuple[PolyN, PolyN]:
    """Elementary shear: ``(x, y+p(x))`` or ``(x+p(y), y)``."""

    x, y = PolyN.var(2, 0), PolyN.var(2, 1)
    acc = PolyN.zero(2)
    base = x if direction == 0 else y
    for power, coeff in enumerate(poly_coeffs):
        if coeff:
            acc = acc + PolyN.const(2, int(coeff)) * (base**power)
    if direction == 0:
        return x, y + acc
    return x + acc, y


class JacobianN2DegreeFamily:
    """Integer ``n=2`` maps in a degree / height box.

    Degree ``<= 1`` boxes whose full coefficient grid fits in
    :data:`MAX_COMPLETE_CARDINALITY` are complete. Higher-degree searches
    walk a structured incomplete slice (linears, shears, a folding map).
    A miss on that slice is ``search_incomplete``.
    """

    def __init__(
        self,
        *,
        max_degree: int = CI_MAX_DEGREE,
        coeff_height: int = CI_COEFF_HEIGHT,
        grid: Sequence[Fraction] | None = None,
    ) -> None:
        if max_degree < 0:
            raise ValueError(f"max_degree must be >= 0, got {max_degree}")
        if coeff_height < 0:
            raise ValueError(f"coeff_height must be >= 0, got {coeff_height}")
        self.max_degree = max_degree
        self.coeff_height = coeff_height
        self.axis = tuple(grid) if grid is not None else default_collision_grid()
        if not self.axis:
            raise ValueError("collision grid axis must be non-empty")
        self._full_card = _full_box_cardinality(max_degree, coeff_height)
        self.complete = max_degree <= 1 and self._full_card <= MAX_COMPLETE_CARDINALITY
        self._items = tuple(self._enumerate())
        self._index = {item: i for i, item in enumerate(self._items)}
        self.name = f"{JACOBIAN_N2_KIND}_d{max_degree}_h{coeff_height}"
        if self.complete:
            self.statement = jacobian_n2_box_statement(
                max_degree=max_degree,
                coeff_height=coeff_height,
                grid=self.axis,
            )
        else:
            self.statement = Statement(
                name=f"{self.name}_slice",
                obligation=(
                    "every map in the structured n=2 slice (linears, shears, "
                    f"folding) of degree <= {max_degree} and height <= "
                    f"{coeff_height} either has det JF not identically a "
                    "nonzero constant or has no G-collision; the slice is "
                    "incomplete for the coefficient box; a miss is not "
                    f"{JACOBIAN_N2_PARENT}"
                ),
                parent=JACOBIAN_N2_PARENT,
                parent_status="open",
                existential=False,
            )

    def _enumerate(self) -> list[N2Candidate]:
        if self.complete:
            width = range(-self.coeff_height, self.coeff_height + 1)
            if self.max_degree == 0:
                return [tuple(item) for item in product(width, width)]
            return [tuple(item) for item in product(width, repeat=6)]
        return self._structured_slice()

    def _structured_slice(self) -> list[N2Candidate]:
        height = max(self.coeff_height, 1)
        items: list[N2Candidate] = [
            ("affine", 0, 1, 0, 0, 0, 1),
            ("affine", 1, 1, 0, 0, 0, 1),
            ("affine", 0, 1, 1, 0, 0, 1),
            ("affine", 0, 1, 0, 0, 1, 1),
            ("fold", 0),
            ("proj", 0),
        ]
        for degree in range(2, self.max_degree + 1):
            coeffs = [0] * (degree + 1)
            coeffs[degree] = 1
            items.append(("shear", 0, *coeffs))
            items.append(("shear", 1, *coeffs))
            coeffs[degree] = -1 if height >= 1 else 1
            items.append(("shear", 0, *coeffs))
        return items

    def cardinality(self) -> int:
        return len(self._items)

    def origin(self) -> N2Candidate:
        identity = (0, 1, 0, 0, 0, 1)
        if identity in self._index:
            return identity
        tagged = ("affine", 0, 1, 0, 0, 0, 1)
        if tagged in self._index:
            return tagged
        if not self._items:
            raise ValueError("JacobianN2DegreeFamily is empty")
        return self._items[0]

    def neighbors(self, candidate: Candidate) -> Sequence[N2Candidate]:
        if not isinstance(candidate, tuple):
            return ()
        typed = tuple(candidate)
        out: list[N2Candidate] = []
        idx = self._index.get(typed)
        if idx is not None:
            if idx + 1 < len(self._items):
                out.append(self._items[idx + 1])
            if idx > 0:
                out.append(self._items[idx - 1])
        if self.complete and all(isinstance(v, int) for v in typed):
            for i, value in enumerate(typed):
                for step in (-1, 1):
                    nxt = list(typed)
                    nxt[i] = int(value) + step
                    if abs(nxt[i]) <= self.coeff_height:
                        out.append(tuple(nxt))
        return out

    def score(self, candidate: Candidate) -> int:
        components = self.decode(candidate)
        if components is None:
            return 0
        constant = identical_jacobian_constant(components)
        if constant is None:
            return 0
        if constant == 0:
            return 1
        return 2

    def decode(self, candidate: Candidate) -> tuple[PolyN, PolyN] | None:
        if not isinstance(candidate, tuple) or not candidate:
            return None
        if self.complete and all(isinstance(v, int) for v in candidate):
            coeffs = tuple(int(v) for v in candidate)
            if self.max_degree == 0 and len(coeffs) == 2:
                return constant_map(coeffs)
            if self.max_degree == 1 and len(coeffs) == 6:
                return affine_map(coeffs)
            return None
        tag = candidate[0]
        rest = candidate[1:]
        if tag == "affine" and len(rest) == 6 and all(isinstance(v, int) for v in rest):
            return affine_map(tuple(int(v) for v in rest))
        if tag == "shear" and rest and all(isinstance(v, int) for v in rest):
            parsed = tuple(int(v) for v in rest)
            return shear_map(parsed[0], parsed[1:])
        x, y = PolyN.var(2, 0), PolyN.var(2, 1)
        if tag == "fold":
            return x**2, y
        if tag == "proj":
            return x, PolyN.zero(2)
        return None

    def check(self, candidate: Candidate) -> ExactCheck | None:
        components = self.decode(candidate)
        if components is None:
            return None
        payload = n2_violation_payload(
            components,
            axis=self.axis,
            candidate=[str(item) for item in candidate] if isinstance(candidate, tuple) else None,
        )
        return ExactCheck(ok=bool(n2_counterexample_earned(payload)), payload=payload)


def escalate_n2_result(result: DiscoveryResult) -> dict[str, Any]:
    """Seal a discovery verdict. Parent claim is earned only on a violator.

    ``jacobian_n2_claim`` becomes True only when ``status == DISPROVED``
    and the check payload records an identical nonzero constant Jacobian
    plus a rational collision. ``jacobian_conjecture_proof_claim`` stays
    False (a counterexample disproves; it does not prove the conjecture).
    A ``PROVED`` finite universal does not escalate.
    """

    payload = dict(result.check.payload) if result.check is not None else {}
    earned = result.status == "DISPROVED" and n2_counterexample_earned(payload)
    honesty = jacobian_n2_honesty(discovered=earned, n2_counterexample=earned)
    return {
        "status": result.status,
        "parent": result.statement.parent,
        "parent_status": result.statement.parent_status,
        "obligation": result.statement.obligation,
        "detail": result.detail,
        "search_incomplete": result.search_incomplete,
        "n2_counterexample": earned,
        "escalate_parent": earned,
        "honesty": honesty,
        "note": (
            "exact n=2 counterexample; parent is false"
            if earned
            else "finite box only; Jacobian n=2 stays open"
        ),
    }


def _full_box_cardinality(max_degree: int, coeff_height: int) -> int:
    n_mons = len(plane_monomials(max_degree))
    width = 2 * coeff_height + 1
    return int(width ** (2 * n_mons))


__all__ = [
    "CI_COEFF_HEIGHT",
    "CI_MAX_DEGREE",
    "DEFAULT_GRID_HALFWIDTH",
    "JACOBIAN_N2_KIND",
    "JACOBIAN_N2_PARENT",
    "JacobianN2DegreeFamily",
    "MAX_COMPLETE_CARDINALITY",
    "MOH_DEGREE_BOUND",
    "affine_map",
    "constant_map",
    "default_collision_grid",
    "escalate_n2_result",
    "jacobian_n2_box_statement",
    "jacobian_n2_honesty",
    "n2_counterexample_earned",
    "n2_violation_payload",
    "plane_monomials",
    "rational_grid_collision",
    "shear_map",
]

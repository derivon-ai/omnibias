# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Blind n=3 tangent-sweep search (Gao §3.1–3.3).

This module enumerates degree-2 curves ``p`` and affine parameters, solves the
twist side conditions, and keeps maps whose Jacobian is a nonzero constant and
that hit some rational target three-to-one. It does **not** seed a published
counterexample: classification against a known replay map happens in
:mod:`omnibias.holonomic.keller`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.proof.discovery import ExactCheck, Statement, run_discovery
from omnibias.holonomic._core.factor import rational_roots
from omnibias.holonomic._core.linalg import solve_exact
from omnibias.holonomic._core.poly_n import PolyN, identical_jacobian_constant, q_from_p
from omnibias.holonomic._core.rational_poly import Poly, peval, to_poly

ComponentOrder = Literal["alpoge", "sweep"]


def _frac(value: Fraction | int) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def divide_by_var(poly: PolyN, index: int) -> PolyN | None:
    """Exact division by a coordinate, or ``None`` if not divisible."""
    out: dict[tuple[int, ...], Fraction] = {}
    for mon, coeff in poly.terms.items():
        if mon[index] < 1:
            return None
        new = list(mon)
        new[index] -= 1
        out[tuple(new)] = coeff
    return PolyN(poly.nvars, out)


def affine_gamma(gamma0: Fraction | int, a: Fraction | int, b: Fraction | int) -> PolyN:
    """``γ = γ0 + a x y + b x² z`` in ``(x, y, z)``."""
    x = PolyN.var(3, 0)
    y = PolyN.var(3, 1)
    z = PolyN.var(3, 2)
    return PolyN.const(3, gamma0) + PolyN.const(3, a) * x * y + PolyN.const(3, b) * x * x * z


def eval_univariate_at_poly(p: Poly, w: PolyN) -> PolyN:
    """Horner evaluation of a univariate polynomial at a :class:`PolyN`."""
    acc = PolyN.zero(w.nvars)
    for coeff in reversed(to_poly(p)):
        acc = acc * w + PolyN.const(w.nvars, coeff)
    return acc


def side_condition_matrix(p: Poly, gamma0: Fraction | int) -> tuple[list[list[Fraction]], list[Fraction]]:
    r"""Linear residuals that must vanish for ``C | P`` and ``C² | Q`` at ``x=0``.

    After setting the constant terms of ``p`` and ``q`` to zero, the remaining
    conditions at ``x=0`` (where ``γ=γ0`` and ``u=1``) are
    ``p(γ0)+2 = 0`` wait no: ``R|_{x=0} = 0`` and ``S|_{x=0} = 0``.
    For a general ``p`` they are affine in the coefficients. The searcher
    treats ``(p_0, q`` constant) as already zero and asks ``solve_exact``
    whether the two residuals vanish.
    """
    p = to_poly(p)
    q = q_from_p(p)
    g0 = _frac(gamma0)
    # R = sum_{k>=1} p_k γ0^{k-1} + 2
    residual_r = Fraction(2)
    for k, coeff in enumerate(p):
        if k == 0:
            residual_r += coeff
            continue
        residual_r += coeff * (g0 ** (k - 1))
    # S = sum_{k>=2} q_k γ0^{k-2} + (q_1 terms handled) + 1  (the +u at u=1)
    residual_s = Fraction(1)
    for k, coeff in enumerate(q):
        if k == 0:
            residual_s += coeff
        elif k == 1:
            residual_s += coeff  # leftover γ^1 / γ^2 is not polynomial unless q1=0
        else:
            residual_s += coeff * (g0 ** (k - 2))
    # Encoded as a 2×1 dummy system so callers use solve_exact uniformly:
    # [residual_r] [t] = [0]  is consistent iff residuals are already zero
    # [residual_s]
    # We return the residuals as a 2x1 "A t = -residuals" with A = I.
    return (
        [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
        [-residual_r, -residual_s],
    )


def side_conditions_hold(p: Poly, gamma0: Fraction | int) -> bool:
    """Whether the twist divisibilities hold for this ``(p, γ0)`` (``p0=q0=q1=0``)."""
    matrix, rhs = side_condition_matrix(p, gamma0)
    # A t = rhs with A = I is solvable iff we accept t = rhs; we instead
    # require the residuals themselves to vanish (rhs = 0).
    return rhs[0] == 0 and rhs[1] == 0


def solve_side_conditions(p: Poly) -> Fraction | None:
    """A rational ``γ0`` making ``R|_{x=0}=S|_{x=0}=0``, or ``None``.

    Degree 2 is a 2×1 linear system in ``γ0``. Degree 3 is quadratic in
    ``γ0`` and is solved over a small rational box.
    """
    p = to_poly(p)
    deg = len(p) - 1
    if deg < 2 or (p and p[0] != 0):
        return None
    if deg == 2:
        p1, p2 = p[1], p[2]
        if p2 == 0:
            return None
        rhs = [-(p1 + 2), -(p1 / 4 + 1)]
        sol = solve_exact([[p2], [p2 / 3]], rhs)
        return None if sol is None else sol[0]
    if deg == 3:
        return _gamma0_from_box(p)
    return None


def _gamma0_from_box(p: Poly) -> Fraction | None:
    """First rational ``γ0`` in a small box that kills both residuals."""
    for num in range(-6, 7):
        if num == 0:
            continue
        for den in (1, 2, 3, 4):
            gamma0 = Fraction(num, den)
            if side_conditions_hold(p, gamma0):
                return gamma0
    return None


def solve_twist_a(p: Poly, gamma0: Fraction | int) -> Fraction | None:
    r"""Rational ``a`` making the ``x^1`` coefficient of ``S`` vanish.

    For degree 3, ``C² | Q`` is not implied by the ``x=0`` residuals alone.
    The linear term in ``x`` of ``S`` is proportional to ``y`` and solves
    for ``a``. Not a closed-form Hessian; just one exact linear identity.
    """
    p = to_poly(p)
    q = q_from_p(p)
    g0 = _frac(gamma0)
    q2 = q[2] if len(q) > 2 else Fraction(0)
    q3 = q[3] if len(q) > 3 else Fraction(0)
    q4 = q[4] if len(q) > 4 else Fraction(0)
    denom = q3 + 2 * q4 * g0
    if denom == 0:
        return None
    rhs = -1 - 2 * q2 - 3 * q3 * g0 - 4 * q4 * (g0**2)
    return rhs / denom


def build_sweep_map(
    p: Poly,
    gamma0: Fraction | int,
    a: Fraction | int,
    b: Fraction | int,
    *,
    component_order: ComponentOrder = "alpoge",
) -> tuple[PolyN, PolyN, PolyN] | None:
    """Twisted padded sweep, or ``None`` if the result is not polynomial."""
    p = to_poly(p)
    q = to_poly(q_from_p(p))
    if p and p[0] != 0:
        return None
    if q and q[0] != 0:
        return None
    if len(q) > 1 and q[1] != 0:
        return None
    if not side_conditions_hold(p, gamma0):
        return None

    gamma = affine_gamma(gamma0, a, b)
    u = PolyN.const(3, 1) + PolyN.var(3, 0) * PolyN.var(3, 1)
    # R = sum_{k>=1} p_k γ^{k-1} u^k + 2
    r_poly = PolyN.const(3, 2)
    for k, coeff in enumerate(p):
        if k == 0:
            continue
        r_poly = r_poly + PolyN.const(3, coeff) * (gamma ** (k - 1)) * (u**k)
    # S = u + sum_{k>=2} q_k γ^{k-2} u^k
    s_poly = u
    for k, coeff in enumerate(q):
        if k < 2:
            continue
        s_poly = s_poly + PolyN.const(3, coeff) * (gamma ** (k - 2)) * (u**k)

    p_over_c = divide_by_var(r_poly, 0)
    q_over_c2 = divide_by_var(s_poly, 0)
    if p_over_c is None or q_over_c2 is None:
        return None
    q_over_c2 = divide_by_var(q_over_c2, 0)
    if q_over_c2 is None:
        return None
    c_poly = gamma * PolyN.var(3, 0)
    if component_order == "alpoge":
        return q_over_c2, p_over_c, c_poly
    return c_poly, p_over_c, q_over_c2


def _eval_jacobian_constant(components: Sequence[PolyN]) -> Fraction | None:
    """Return ``det JF`` if it is an identically nonzero constant polynomial.

    Probe samples are not a certificate. A zero constant is rejected the
    same way a non-constant determinant is.
    """
    value = identical_jacobian_constant(components)
    if value is None or value == 0:
        return None
    return value


def _lift_sweep_preimages(
    p: Poly,
    gamma0: Fraction,
    a: Fraction,
    b: Fraction,
    x_coord: Fraction,
    y_coord: Fraction,
) -> list[tuple[Fraction, Fraction, Fraction]] | None:
    """Lift three rational tangency roots to ``(x,y,z)`` with a common ``C=1``."""
    try:
        roots = [r for r in rational_roots(tangency_polynomial(p, x_coord, y_coord))]
    except ValueError:
        return None
    distinct = []
    for root in roots:
        if root not in distinct:
            distinct.append(root)
    if len(distinct) < 3:
        return None
    preimages: list[tuple[Fraction, Fraction, Fraction]] = []
    for w in distinct[:3]:
        gamma = (x_coord - peval(to_poly(p), w)) / 2
        if gamma == 0 or b == 0:
            return None
        x = Fraction(1) / gamma
        u = w / gamma
        y = (u - 1) / x
        z = (gamma - gamma0 - a * x * y) / (b * x * x)
        preimages.append((x, y, z))
    if len({pt for pt in preimages}) < 3:
        return None
    return preimages


def find_three_to_one_witness(
    components: Sequence[PolyN],
    *,
    grid: Sequence[Fraction | int] | None = None,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...] | None:
    """Three distinct rational preimages of one common image, or ``None``."""
    if grid is None:
        xy = [Fraction(k, 2) for k in range(-4, 5)]
        z = [Fraction(k, 2) for k in range(-8, 9)]
        grid_xyz = (xy, xy, z)
    else:
        coords = [_frac(v) for v in grid]
        grid_xyz = (coords, coords, coords)
    buckets: dict[tuple[Fraction, Fraction, Fraction], list[tuple[Fraction, Fraction, Fraction]]] = defaultdict(list)
    f1, f2, f3 = components
    xs, ys, zs = grid_xyz
    for x in xs:
        for y in ys:
            for z in zs:
                point = (x, y, z)
                image = (f1.eval(point), f2.eval(point), f3.eval(point))
                if point not in buckets[image]:
                    buckets[image].append(point)
                if len(buckets[image]) >= 3:
                    return tuple(buckets[image][:3])
    return None


@dataclass(frozen=True)
class SweepSearchHit:
    """One constant-Jacobian multi-to-one map found by the blind sweep."""

    p: Poly
    gamma0: Fraction
    a: Fraction
    b: Fraction
    jacobian_constant: Fraction
    witness: tuple[tuple[Fraction, Fraction, Fraction], ...]
    search: str = "tangent_sweep_side_conditions"
    generic_fiber: int = 0


def _p_from_candidate(deg_p: int, candidate: tuple[int, ...]) -> Poly:
    if deg_p == 2:
        p1, p2, _a, _b = candidate
        return to_poly([0, p1, p2])
    p1, p2, p3, _b = candidate
    return to_poly([0, p1, p2, p3])


def _ab_from_candidate(deg_p: int, candidate: tuple[int, ...], p: Poly, gamma0: Fraction) -> tuple[Fraction, Fraction] | None:
    if deg_p == 2:
        return Fraction(candidate[2]), Fraction(candidate[3])
    solved = solve_twist_a(p, gamma0)
    if solved is None:
        return None
    return solved, Fraction(candidate[3])


def _try_witness(
    p: Poly,
    gamma0: Fraction,
    a: Fraction,
    b: Fraction,
    built: Sequence[PolyN],
    *,
    require: bool,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...] | None:
    for x_coord in (Fraction(1), Fraction(-1), Fraction(3), Fraction(0)):
        for y_coord in (Fraction(1), Fraction(-1), Fraction(2), Fraction(0)):
            lifted = _lift_sweep_preimages(p, gamma0, a, b, x_coord, y_coord)
            if lifted is None:
                continue
            images = {tuple(f.eval(pt) for f in built) for pt in lifted}
            if len(images) == 1:
                return tuple(lifted)
    if not require:
        return ()
    return find_three_to_one_witness(built)


class SweepFamily:
    """Named tangent-sweep box. Incomplete: not every Keller map is in-family."""

    complete = False

    def __init__(self, *, deg_p: int, height: int) -> None:
        if deg_p not in (2, 3):
            raise ValueError("deg_p must be 2 or 3")
        if height < 1:
            raise ValueError("height must be >= 1")
        self.deg_p = deg_p
        self.height = height
        self.name = f"sweep_deg{deg_p}"
        self.statement = Statement(
            name=f"keller_{self.name}",
            obligation=(
                f"a deg-{deg_p} sweep map with constant nonzero Jacobian "
                "and a multi-to-one tangency fiber"
            ),
            parent="jacobian_conjecture_n_ge_3",
            parent_status="already_false",
        )
        self._items = tuple(self._enumerate())
        self._index = {item: i for i, item in enumerate(self._items)}

    def _enumerate(self) -> list[tuple[int, ...]]:
        height = self.height
        items: list[tuple[int, ...]] = []
        if self.deg_p == 2:
            p2_values = [v for v in range(-height, height + 1) if v and height * abs(v) >= 6]
            for radius in range(1, height + 1):
                for p2 in p2_values:
                    for p1 in range(-height, height + 1):
                        for a in range(-radius, radius + 1):
                            if a == 0:
                                continue
                            for b in range(-radius, radius + 1):
                                if b == 0 or max(abs(a), abs(b)) != radius:
                                    continue
                                items.append((p1, p2, a, b))
            return items
        for p1 in range(-height, height + 1):
            for p2 in range(-height, height + 1):
                for p3 in range(-height, height + 1):
                    if p3 == 0:
                        continue
                    if solve_side_conditions(to_poly([0, p1, p2, p3])) is None:
                        continue
                    for b in range(-height, height + 1):
                        if b == 0:
                            continue
                        items.append((p1, p2, p3, b))
        return items

    def origin(self) -> tuple[int, ...]:
        if not self._items:
            return (0, 1, 1, 1) if self.deg_p == 2 else (0, 1, 1, 1)
        return self._items[0]

    def neighbors(self, candidate: tuple[int, ...]) -> Sequence[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        idx = self._index.get(candidate)
        if idx is not None and idx + 1 < len(self._items):
            out.append(self._items[idx + 1])
        if idx is not None and idx > 0:
            out.append(self._items[idx - 1])
        coords = list(candidate)
        for i, value in enumerate(coords):
            for step in (-1, 1):
                nxt = list(coords)
                nxt[i] = value + step
                if all(abs(v) <= self.height for v in nxt):
                    if self.deg_p == 2 and (nxt[2] == 0 or nxt[3] == 0):
                        continue
                    if self.deg_p == 3 and (nxt[2] == 0 or nxt[3] == 0):
                        continue
                    out.append(tuple(nxt))
        return out

    def score(self, candidate: tuple[int, ...]) -> int:
        """0 fail side conditions, 1 constant Jac, 2 + multi-to-one fiber.

        The fiber bonus uses the tangency degree only — the accept gate is
        still :meth:`check` (sampled Jac + lift / generic fiber).
        """
        if not isinstance(candidate, tuple) or len(candidate) != 4:
            return 0
        p = _p_from_candidate(self.deg_p, candidate)
        gamma0 = solve_side_conditions(p)
        if gamma0 is None:
            return 0
        ab = _ab_from_candidate(self.deg_p, candidate, p, gamma0)
        if ab is None:
            return 0
        a, b = ab
        built = build_sweep_map(p, gamma0, a, b, component_order="alpoge")
        if built is None or _eval_jacobian_constant(built) is None:
            return 0
        if generic_fiber_degree(p) >= 3:
            return 2
        return 1

    def check(self, candidate: tuple[int, ...]) -> ExactCheck | None:
        if not isinstance(candidate, tuple) or len(candidate) != 4:
            return None
        if not all(isinstance(v, int) for v in candidate):
            return None
        p = _p_from_candidate(self.deg_p, candidate)
        gamma0 = solve_side_conditions(p)
        honesty = honesty_search()
        if gamma0 is None:
            return ExactCheck(ok=False, payload={"side_conditions": False, "honesty": honesty})
        ab = _ab_from_candidate(self.deg_p, candidate, p, gamma0)
        if ab is None:
            return ExactCheck(ok=False, payload={"side_conditions": True, "honesty": honesty})
        a, b = ab
        built = build_sweep_map(p, gamma0, a, b, component_order="alpoge")
        if built is None:
            return ExactCheck(ok=False, payload={"side_conditions": True, "honesty": honesty})
        constant = _eval_jacobian_constant(built)
        fiber = generic_fiber_degree(p)
        require_witness = self.deg_p == 2
        witness = None
        if constant is not None and fiber >= 3:
            witness = _try_witness(p, gamma0, a, b, built, require=require_witness)
        ok = constant is not None and fiber >= 3 and (not require_witness or witness)
        payload = {
            "side_conditions": True,
            "p": [str(c) for c in p],
            "gamma0": str(gamma0),
            "a": str(a),
            "b": str(b),
            "jacobian_identity": "identical",
            "jacobian_constant": "" if constant is None else str(constant),
            "generic_fiber": fiber,
            "witness": [] if not witness else [[str(c) for c in pt] for pt in witness],
            "honesty": honesty,
        }
        return ExactCheck(ok=bool(ok), payload=payload)


def search_tangent_sweep(
    *,
    deg_p: int = 2,
    height: int = 6,
    max_hits: int = 1,
    proposer: str = "score_guided",
    budget: int | None = None,
) -> list[SweepSearchHit]:
    """Propose candidates in the sweep family and accept exact hits."""
    if deg_p not in (2, 3):
        raise ValueError("deg_p must be 2 or 3")
    family = SweepFamily(deg_p=deg_p, height=height)
    limit = len(family._items) + 8 if budget is None else budget
    hits: list[SweepSearchHit] = []
    result = run_discovery(family.statement, family, proposer, budget=limit)
    if result.status != "PROVED" or result.check is None or result.candidate is None:
        return hits
    payload = result.check.payload
    p = to_poly([Fraction(c) for c in payload["p"]])
    witness_raw = payload.get("witness") or []
    witness = tuple(
        tuple(Fraction(c) for c in pt) for pt in witness_raw  # type: ignore[misc]
    )
    hits.append(
        SweepSearchHit(
            p=p,
            gamma0=Fraction(payload["gamma0"]),
            a=Fraction(payload["a"]),
            b=Fraction(payload["b"]),
            jacobian_constant=Fraction(payload["jacobian_constant"]),
            witness=witness,
            search=f"tangent_sweep_deg{deg_p}",
            generic_fiber=int(payload.get("generic_fiber") or 0),
        )
    )
    if len(hits) >= max_hits:
        return hits
    return hits


def tangency_polynomial(p: Poly, x_coord: Fraction | int, y_coord: Fraction | int) -> Poly:
    r"""``W_{X,Y}(w) = q(w) + (w/2)(X - p(w)) - Y`` (Gao (2))."""
    p = to_poly(p)
    q = q_from_p(p)
    x_c, y_c = _frac(x_coord), _frac(y_coord)
    # q(w) - Y + (w/2) X - (w/2) p(w)
    from omnibias.holonomic._core.rational_poly import padd, pscale, psub

    w_over_2 = to_poly([0, Fraction(1, 2)])
    half_w_x = pscale(w_over_2, x_c)
    half_w_p = _mul_univariate(w_over_2, p)
    body = padd(q, half_w_x)
    body = psub(body, half_w_p)
    body = padd(body, to_poly([-y_c]))
    return to_poly(body)


def _mul_univariate(a: Poly, b: Poly) -> Poly:
    from omnibias.holonomic._core.rational_poly import pmul

    return pmul(to_poly(a), to_poly(b))


def generic_fiber_degree(p: Poly) -> int:
    """Degree of the tangency polynomial (generic sweep fiber size)."""
    # Leading term is independent of (X, Y) under Gao's normalization.
    sample = tangency_polynomial(p, 0, 0)
    return len(to_poly(sample)) - 1


def tangency_leading_coeff(p: Poly) -> Fraction:
    """Leading coefficient of ``W_{X,Y}``; constant in ``(X,Y)`` for the family."""
    w0 = tangency_polynomial(p, 0, 0)
    w1 = tangency_polynomial(p, 1, 0)
    w2 = tangency_polynomial(p, 0, 1)
    lead = to_poly(w0)[-1]
    if to_poly(w1)[-1] != lead or to_poly(w2)[-1] != lead:
        raise ValueError("tangency leading coefficient depends on the target")
    return lead


def honesty_search() -> dict[str, bool]:
    return {
        "discovered_by_omnibias": False,
        "jacobian_conjecture_proof_claim": False,
        "jacobian_n2_claim": False,
        "navier_stokes_proof_claim": False,
        "keller_n_ge_3_replay": False,
        "ten_proofs_formalization_claim": False,
    }


__all__ = [
    "SweepFamily",
    "SweepSearchHit",
    "affine_gamma",
    "build_sweep_map",
    "divide_by_var",
    "find_three_to_one_witness",
    "generic_fiber_degree",
    "honesty_search",
    "q_from_p",
    "search_tangent_sweep",
    "side_condition_matrix",
    "side_conditions_hold",
    "solve_side_conditions",
    "solve_twist_a",
    "tangency_leading_coeff",
    "tangency_polynomial",
]

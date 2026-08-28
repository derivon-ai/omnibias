# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Neural quadrature / cubature from pack moments (theory 03-06).

A pack's moments are closed form under the founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. Do not conflate the two. The moment residual is
solved, not table-looked-up; Peano error is a sound enclosure when
a bound on ``f^(d+1)`` is supplied and is refused otherwise.
Non-product cubature in high dimension is out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from omnibias.core.mollifier import MollifierSpec
from omnibias.core.multipack import MultiPackSpec, PackSpec, _birkhoff_vandermonde
from omnibias.core.verified.interval import Interval

MeasureName = Literal["lebesgue", "gaussian", "custom"]
FunctionalName = Literal["point", "pack"]


def honesty_payload() -> dict[str, bool]:
    return {
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "non_product_cubature": False,
        "smoothing_bias_cancelled": True,
        "refuses_without_deriv_bound": True,
    }


def target_moments(
    *,
    degree: int,
    measure: MeasureName = "lebesgue",
    interval: tuple[float, float] = (-1.0, 1.0),
    custom: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Moments ``M_j = integral x^j dmu`` for ``j = 0 .. degree``."""
    if int(degree) < 0:
        raise ValueError("degree must be >= 0")
    if measure == "custom":
        if custom is None or len(custom) < int(degree) + 1:
            raise ValueError("custom moments must cover 0 .. degree")
        return tuple(float(m) for m in custom[: int(degree) + 1])
    lo, hi = float(interval[0]), float(interval[1])
    if not (lo < hi):
        raise ValueError("interval needs lo < hi")
    out: list[float] = []
    for j in range(int(degree) + 1):
        if measure == "lebesgue":
            out.append((hi ** (j + 1) - lo ** (j + 1)) / float(j + 1))
        elif measure == "gaussian":
            # Standard normal moments on R (interval is ignored).
            if j % 2 == 1:
                out.append(0.0)
            else:
                k = j // 2
                out.append(float(math.factorial(j) // (2**k * math.factorial(k))))
        else:
            raise ValueError(f"unknown measure {measure!r}")
    return tuple(out)


@dataclass(frozen=True)
class MomentSystem:
    degree: int
    measure: MeasureName
    target_moments: tuple[float, ...]
    interval: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        if int(self.degree) < 0:
            raise ValueError("degree must be >= 0")
        if len(self.target_moments) < int(self.degree) + 1:
            raise ValueError("target_moments must cover 0 .. degree")
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "target_moments", tuple(float(m) for m in self.target_moments[: self.degree + 1]))

    @classmethod
    def lebesgue(cls, degree: int, *, interval: tuple[float, float] = (-1.0, 1.0)) -> MomentSystem:
        return cls(degree, "lebesgue", target_moments(degree=degree, measure="lebesgue", interval=interval), interval)


@dataclass(frozen=True)
class QuadratureRule:
    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    degree: int
    functional: FunctionalName
    pack_scale: float | None
    bias_cancelled: bool
    measure: str
    interval: tuple[float, float]
    pack_base: str = "gaussian"

    def __post_init__(self) -> None:
        nodes = tuple(float(x) for x in self.nodes)
        weights = tuple(float(w) for w in self.weights)
        if len(nodes) != len(weights) or not nodes:
            raise ValueError("nodes and weights must be nonempty and aligned")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def apply_monomial(self, power: int) -> float:
        """Rule applied to ``x^power`` (pack moments when ``functional='pack'``)."""
        acc = 0.0
        for x, w in zip(self.nodes, self.weights, strict=True):
            if self.functional == "point":
                acc += w * (x**power)
            else:
                acc += w * pack_moment(
                    PackSpec(order=0, mean=x),
                    power,
                    scale=float(self.pack_scale or 1.0),
                    base=self.pack_base,
                )
        return acc

    def apply(self, values: Sequence[float]) -> float:
        if len(values) != self.n_nodes:
            raise ValueError("values must match nodes")
        return float(sum(w * float(v) for w, v in zip(self.weights, values, strict=True)))


def pack_moment(pack: PackSpec, j: int, *, scale: float, base: str = "gaussian") -> float:
    """Closed-form raw moment of a width-``scale`` pack. Reuses 01-05."""
    spec = MollifierSpec(base=base, scale=float(scale), packs=(pack,))
    return float(spec.moment(int(j)))


def _as_decimal(value: object) -> Decimal:
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, int):
        return Decimal(value)
    return Decimal(repr(float(value)))  # type: ignore[arg-type]


def _lebesgue_moments_exact(degree: int, lo: float, hi: float) -> list[Fraction]:
    out: list[Fraction] = []
    for j in range(int(degree) + 1):
        if lo == -1.0 and hi == 1.0:
            out.append(Fraction(0) if j % 2 else Fraction(2, j + 1))
        else:
            num = Fraction(hi) ** (j + 1) - Fraction(lo) ** (j + 1)
            out.append(num / Fraction(j + 1))
    return out


def _chebyshev_jacobi(
    moments: Sequence[object], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gautschi Chebyshev algorithm in ``Decimal`` so Jacobi entries stay accurate."""
    with localcontext() as ctx:
        ctx.prec = 80
        m = [_as_decimal(moments[j]) for j in range(2 * n)]
        if m[0] <= 0:
            raise ValueError("M_0 must be positive")
        sigma = [[Decimal(0) for _ in range(2 * n)] for _ in range(n + 1)]
        sigma[0] = list(m)
        alpha = [Decimal(0)] * n
        beta = [Decimal(0)] * n
        alpha[0] = m[1] / m[0]
        beta[0] = m[0]
        for k in range(1, n):
            for ell in range(k, 2 * n - k):
                prev = sigma[k - 2][ell] if k >= 2 else Decimal(0)
                sigma[k][ell] = (
                    sigma[k - 1][ell + 1] - alpha[k - 1] * sigma[k - 1][ell] - beta[k - 1] * prev
                )
            if abs(sigma[k][k]) < Decimal("1e-50") or abs(sigma[k - 1][k - 1]) < Decimal("1e-50"):
                raise ValueError("moment system is degenerate (no real Gauss rule)")
            alpha[k] = sigma[k][k + 1] / sigma[k][k] - sigma[k - 1][k] / sigma[k - 1][k - 1]
            beta[k] = sigma[k][k] / sigma[k - 1][k - 1]
        return (
            np.array([float(a) for a in alpha], dtype=np.float64),
            np.array([float(b) for b in beta], dtype=np.float64),
        )


def _golub_welsch(moments: Sequence[object], n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    alpha, beta = _chebyshev_jacobi(moments, n)
    off = np.sqrt(np.maximum(beta[1:], 0.0))
    jacobi = np.diag(alpha) + np.diag(off, k=1) + np.diag(off, k=-1)
    evals, evecs = np.linalg.eigh(jacobi)
    order = np.argsort(evals)
    nodes = evals[order]
    weights = beta[0] * (evecs[0, order] ** 2)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _fixed_node_weights(
    nodes: Sequence[float] | NDArray[np.float64], moments: Sequence[object]
) -> NDArray[np.float64]:
    """Newton–Cotes-type linear solve via the 01-01/01-04 confluent Vandermonde."""
    packs = MultiPackSpec.from_packs(tuple(PackSpec(order=0, mean=float(x)) for x in nodes))
    vand = _birkhoff_vandermonde(packs)
    rhs = np.asarray(moments[: len(nodes)], dtype=np.float64)
    return np.linalg.solve(vand.T, rhs)


def solve_rule(
    system: MomentSystem,
    *,
    nodes: int,
    free_nodes: bool = True,
    functional: FunctionalName = "point",
    pack_scale: float | None = None,
    pack_base: str = "gaussian",
    fixed_nodes: Sequence[float] | None = None,
) -> QuadratureRule:
    """Solve a quadrature rule from the moment system.

    Point + free nodes is Gauss-type (Golub–Welsch). Point + fixed nodes
    is Newton–Cotes via the Birkhoff Vandermonde. Pack functionals
    re-solve so the ``O(scale^2)`` bump moment is cancelled.
    """
    n = int(nodes)
    if n < 1:
        raise ValueError("nodes must be >= 1")
    lo, hi = system.interval
    if functional == "pack":
        if pack_scale is None or float(pack_scale) <= 0.0:
            raise ValueError("pack functionals need pack_scale > 0")
        return _solve_pack_rule(system, n=n, scale=float(pack_scale), base=pack_base, free_nodes=free_nodes)
    need = 2 * n if free_nodes else n
    moms = target_moments(
        degree=max(system.degree, need - 1),
        measure=system.measure,
        interval=system.interval,
        custom=system.target_moments if system.measure == "custom" else None,
    )
    if free_nodes:
        exact = (
            _lebesgue_moments_exact(2 * n - 1, lo, hi)
            if system.measure == "lebesgue"
            else moms
        )
        xs, ws = _golub_welsch(exact, n)
        deg = 2 * n - 1
    else:
        if fixed_nodes is None:
            # Chebyshev extrema on the interval, Newton–Cotes weights.
            if n == 1:
                xs = np.array([0.5 * (lo + hi)], dtype=np.float64)
            else:
                k = np.arange(n, dtype=np.float64)
                t = np.cos(math.pi * k / (n - 1))
                xs = 0.5 * (hi + lo) + 0.5 * (hi - lo) * t
        else:
            xs = np.asarray(fixed_nodes, dtype=np.float64)
            if xs.size != n:
                raise ValueError("fixed_nodes length must match nodes")
        ws = _fixed_node_weights(xs, moms)
        deg = n - 1
    return QuadratureRule(
        nodes=tuple(float(x) for x in xs),
        weights=tuple(float(w) for w in ws),
        degree=deg,
        functional="point",
        pack_scale=None,
        bias_cancelled=False,
        measure=system.measure,
        interval=(lo, hi),
    )


def _solve_pack_rule(
    system: MomentSystem,
    *,
    n: int,
    scale: float,
    base: str,
    free_nodes: bool,
) -> QuadratureRule:
    """Symmetric pack rule: cancel bump moments against the target measure."""
    if n != 2:
        raise ValueError("pack-functional solve is implemented for nodes=2")
    if not free_nodes:
        raise ValueError("pack-functional solve needs free_nodes=True")
    m0, m2 = system.target_moments[0], system.target_moments[2]
    # Two equal-weight packs at ±x0. <p, 1> = 1, <p, x^2> = x0^2 + c2 scale^2.
    mass = pack_moment(PackSpec(order=0, mean=0.0), 0, scale=scale, base=base)
    var = pack_moment(PackSpec(order=0, mean=0.0), 2, scale=scale, base=base)
    # 2 w mass = m0  =>  w = m0 / (2 mass)
    w = m0 / (2.0 * mass)
    # 2 w (x0^2 + var) = m2  =>  x0^2 = m2 / (2w) - var
    inner = m2 / (2.0 * w) - var
    if inner <= 0.0:
        raise ValueError("pack scale is too wide to cancel the second moment")
    x0 = math.sqrt(inner)
    return QuadratureRule(
        nodes=(-x0, x0),
        weights=(w, w),
        degree=3,
        functional="pack",
        pack_scale=scale,
        bias_cancelled=True,
        measure=system.measure,
        interval=system.interval,
        pack_base=base,
    )


def _plus_power(x: float, t: float, p: int) -> float:
    d = x - t
    if d <= 0.0:
        return 0.0
    return d**p


def peano_kernel(rule: QuadratureRule, *, degree: int) -> Callable[[float], float]:
    """``K_d(t)`` with ``E(f) = integral K_d(t) f^{(d+1)}(t) dt``."""
    if int(degree) < 0:
        raise ValueError("degree must be >= 0")
    lo, hi = rule.interval
    d = int(degree)
    fact = math.factorial(d)

    def kernel(t: float) -> float:
        # ∫_lo^hi (x-t)_+^d dx
        if t >= hi:
            integral = 0.0
        elif t <= lo:
            integral = ((hi - t) ** (d + 1) - (lo - t) ** (d + 1)) / float(d + 1)
        else:
            integral = ((hi - t) ** (d + 1)) / float(d + 1)
        sample = 0.0
        for x, w in zip(rule.nodes, rule.weights, strict=True):
            sample += w * _plus_power(x, t, d)
        return (integral - sample) / fact

    return kernel


def peano_l1_bound(rule: QuadratureRule, *, degree: int, panels: int = 2048) -> Interval:
    """Sound enclosure of ``||K_d||_1`` via Lipschitz panels."""
    kernel = peano_kernel(rule, degree=degree)
    lo, hi = rule.interval
    d = int(degree)
    span = hi - lo
    # |K'| <= (span^d + sum |w| d span^{d-1}) / d!   (crude, sound)
    fact = math.factorial(max(d, 1))
    wabs = sum(abs(w) for w in rule.weights)
    if d == 0:
        lip = 1.0 + wabs
    else:
        lip = (span**d + wabs * float(d) * (span ** (d - 1))) / fact
    h = span / float(panels)
    acc = Interval.point(0.0)
    for i in range(panels):
        a = lo + i * h
        mid = a + 0.5 * h
        km = abs(kernel(mid))
        kmax = km + lip * (0.5 * h)
        acc = acc + Interval.from_value(kmax) * Interval.from_value(h)
    return acc


def certified_error(rule: QuadratureRule, *, deriv_bound: Interval | None, degree: int) -> Interval:
    """Sound enclosure of the quadrature error. Refuses without ``deriv_bound``."""
    if deriv_bound is None:
        raise ValueError("certified_error refuses without a bound on f^(d+1)")
    l1 = peano_l1_bound(rule, degree=int(degree))
    mag = Interval.from_value(deriv_bound.mag)
    radius = l1 * mag
    return Interval(-radius.hi, radius.hi)


def apply_rule(rule: QuadratureRule, fn: Callable[[float], float]) -> float:
    if rule.functional != "point":
        raise ValueError("apply_rule on pack functionals is only closed-form on monomials")
    return float(sum(w * float(fn(x)) for x, w in zip(rule.nodes, rule.weights, strict=True)))


def design_rule(
    integrands: Sequence[Callable[[float], float]],
    exacts: Sequence[float],
    *,
    nodes: int = 2,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> QuadratureRule:
    """Symmetric 2-node design: keep mass exact, fit the family (not generic GL)."""
    if int(nodes) != 2:
        raise ValueError("design_rule smoke path is the symmetric 2-node rule")
    if len(integrands) != len(exacts) or not integrands:
        raise ValueError("integrands and exacts must align")
    lo, hi = interval
    mass = hi - lo
    # Grid-search x0 in (0, 1) to minimize L2 error on the family.
    best_x = math.sqrt(1.0 / 3.0)
    best_err = float("inf")
    for x0 in np.linspace(0.35, 0.95, 121):
        w = 0.5 * mass
        trial = QuadratureRule(
            nodes=(-float(x0), float(x0)),
            weights=(w, w),
            degree=1,
            functional="point",
            pack_scale=None,
            bias_cancelled=False,
            measure="designed",
            interval=interval,
        )
        err = 0.0
        for fn, exact in zip(integrands, exacts, strict=True):
            err += (apply_rule(trial, fn) - float(exact)) ** 2
        if err < best_err:
            best_err = err
            best_x = float(x0)
    w = 0.5 * mass
    return QuadratureRule(
        nodes=(-best_x, best_x),
        weights=(w, w),
        degree=1,
        functional="point",
        pack_scale=None,
        bias_cancelled=False,
        measure="designed",
        interval=interval,
    )


def tensor_product_cost(*, nodes_per_axis: int, dim: int) -> int:
    """``n^D`` node count. Non-product cubature is out of scope."""
    if int(nodes_per_axis) < 1 or int(dim) < 1:
        raise ValueError("nodes_per_axis and dim must be >= 1")
    return int(int(nodes_per_axis) ** int(dim))


def dimension_scaling_table(*, nodes_per_axis: int = 8, max_dim: int = 8) -> tuple[tuple[int, int], ...]:
    return tuple((d, tensor_product_cost(nodes_per_axis=nodes_per_axis, dim=d)) for d in range(1, int(max_dim) + 1))


__all__ = [
    "MomentSystem",
    "QuadratureRule",
    "apply_rule",
    "certified_error",
    "design_rule",
    "dimension_scaling_table",
    "honesty_payload",
    "pack_moment",
    "peano_kernel",
    "peano_l1_bound",
    "solve_rule",
    "target_moments",
    "tensor_product_cost",
]

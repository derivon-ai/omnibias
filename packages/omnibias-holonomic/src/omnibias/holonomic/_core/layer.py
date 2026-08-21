# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Holonomic layer (theory 09-12).

The block **is** an Ore annihilator. Forward prolongs the unique
D-finite jet of ``L u = 0`` from an initial jet. Representable maps
are D-finite by construction; discovery reads the operator.

This block is not founding bias collapse. A ``sigma`` net may still
supply the data jet via founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not appear.
Do not conflate the two.

D-finite class only. Not a general PINN. Nonlinear PDEs are not
D-finite here. Lean flags are 09-26, not asserted. Not CCF stretch.
Not NS.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.holonomic._core.guess import guess_dfinite
from omnibias.holonomic._core.ore import OrePolynomial, diff_algebra
from omnibias.holonomic._core.rational_poly import Poly, pderiv, peval

DISCLAIMER = (
    "Holonomic layer: D-finite Ore annihilator only; not a general PINN, "
    "not a nonlinear PDE, and not CCF stretch"
)

Rational = Fraction | int


def honesty_payload() -> dict[str, bool]:
    return {
        "nonlinear_pde_claimed_dfinite": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class HolonomicLayerSpec:
    max_order: int = 2
    max_degree: int = 1


DEFAULT_SPEC = HolonomicLayerSpec()


def _as_frac(value: Rational | float) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(value).limit_denominator()


def _nth_deriv(poly: Poly, n: int) -> Poly:
    cur = poly
    for _ in range(n):
        cur = pderiv(cur)
        if not cur:
            return ()
    return cur


def holonomic_jet(
    op: OrePolynomial,
    init_jet: Sequence[Rational | float],
    x0: Rational | float,
    order: int,
) -> tuple[Fraction, ...]:
    """Prolong the jet of ``u`` at ``x0`` using ``L u = 0``.

    ``init_jet`` is ``(u, u', ..., u^{(m-1)})`` with ``m = order(L)``.
    Higher derivatives follow by Leibniz expansion of ``L u = 0``.
    Must not import torch or jax.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    m = op.order
    if m < 1:
        raise ValueError("annihilator must have order >= 1")
    if len(init_jet) < m:
        raise ValueError(f"need >= {m} initial jet entries, got {len(init_jet)}")
    point = _as_frac(x0)
    jet = [_as_frac(v) for v in init_jet[:m]]
    lead_poly = op.coeffs[m] if m < len(op.coeffs) else ()
    lead0 = peval(lead_poly, point)
    if lead0 == 0:
        raise ValueError("leading coefficient vanishes at x0")
    while len(jet) <= order:
        n = len(jet) - m
        rest = Fraction(0)
        for k in range(m + 1):
            pk = op.coeffs[k] if k < len(op.coeffs) else ()
            for j in range(n + 1):
                if k == m and j == 0:
                    continue
                idx = k + n - j
                if idx < 0 or idx >= len(jet):
                    continue
                weight = math.comb(n, j) * peval(_nth_deriv(pk, j), point)
                rest += weight * jet[idx]
        jet.append(-rest / lead0)
    return tuple(jet[: order + 1])


def apply_operator_to_jet(
    op: OrePolynomial,
    x: Rational | float,
    jet: Sequence[Rational | float],
) -> Fraction:
    """``(L u)(x)`` from a jet long enough for ``order(L)``."""
    m = op.order
    if len(jet) <= m:
        raise ValueError(f"jet must include derivative {m}")
    point = _as_frac(x)
    total = Fraction(0)
    for k in range(m + 1):
        pk = op.coeffs[k] if k < len(op.coeffs) else ()
        total += peval(pk, point) * _as_frac(jet[k])
    return total


def content_clear_operator(op: OrePolynomial) -> OrePolynomial:
    """Clear denominators and a content integer; leading coefficient ``> 0``."""
    fracs: list[Fraction] = []
    for poly in op.coeffs:
        fracs.extend(poly)
    if not fracs:
        return op
    den_lcm = 1
    for coeff in fracs:
        den_lcm = math.lcm(den_lcm, coeff.denominator)
    scaled: list[tuple[Fraction, ...]] = []
    nums: list[int] = []
    for poly in op.coeffs:
        row = tuple(coeff * den_lcm for coeff in poly)
        scaled.append(row)
        nums.extend(int(coeff.numerator) for coeff in row)
    content = 0
    for num in nums:
        content = math.gcd(content, abs(num))
    if content == 0:
        content = 1
    lead = scaled[op.order]
    lead_c = lead[-1] if lead else Fraction(0)
    sign = -1 if lead_c < 0 else 1
    cleared = tuple(tuple(coeff / content * sign for coeff in row) for row in scaled)
    return OrePolynomial(op.algebra, cleared)


def is_rational_multiple_of_d2_plus_1(op: OrePolynomial) -> bool:
    """True when ``op`` is ``c(D^2 + 1)`` for a nonzero rational ``c``."""
    cleared = content_clear_operator(op)
    if cleared.order != 2:
        return False
    p0 = cleared.coeffs[0]
    p1 = cleared.coeffs[1] if len(cleared.coeffs) > 1 else ()
    p2 = cleared.coeffs[2]
    if p1 or len(p0) != 1 or len(p2) != 1:
        return False
    return p0[0] == p2[0] != 0


def fit_holonomic_layer(
    series: Sequence[Rational],
    *,
    spec: HolonomicLayerSpec | None = None,
) -> OrePolynomial:
    """Guess a differential annihilator of a Taylor prefix (heuristic)."""
    cfg = DEFAULT_SPEC if spec is None else spec
    if cfg.max_order < 1:
        raise ValueError("max_order must be >= 1")
    if cfg.max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    op = guess_dfinite(series, max_order=cfg.max_order, max_degree=cfg.max_degree)
    if op is None:
        raise ValueError("no D-finite annihilator within the given bounds")
    return op


def d_minus_1() -> OrePolynomial:
    return diff_algebra().operator(((-1,), (1,)))


def d2_plus_1() -> OrePolynomial:
    return diff_algebra().operator(((1,), (), (1,)))


def worked_example() -> dict[str, object]:
    """Spec 09-12: ``D-1`` on ``exp`` and ``D^2+1`` on ``sin`` at ``0``."""
    exp_jet = holonomic_jet(d_minus_1(), (1,), 0, 5)
    exp_err = max(abs(float(v) - 1.0) for v in exp_jet)
    sin_jet = holonomic_jet(d2_plus_1(), (0, 1), 0, 3)
    return {
        "exp_jet": exp_jet,
        "exp_err": exp_err,
        "sin_jet": sin_jet,
        "sin_u2": sin_jet[2],
        "sin_u3": sin_jet[3],
    }


def _sin_taylor(count: int) -> list[Fraction]:
    out: list[Fraction] = []
    for k in range(count):
        if k % 2 == 0:
            out.append(Fraction(0))
        else:
            sign = 1 if (k // 2) % 2 == 0 else -1
            out.append(Fraction(sign, math.factorial(k)))
    return out


def sin_skill(*, n_samples: int = 16) -> dict[str, object]:
    """G2: recover ``D^2+1`` from a 16-term sine prefix; residual on ``[0,1]``."""
    if n_samples < 2:
        raise ValueError("need at least two samples")
    series = _sin_taylor(n_samples)
    op = fit_holonomic_layer(series, spec=HolonomicLayerSpec(max_order=2, max_degree=1))
    multiple = is_rational_multiple_of_d2_plus_1(op)
    residuals: list[float] = []
    for i in range(n_samples):
        x = 0.0 if n_samples == 1 else i / (n_samples - 1)
        jet = (math.sin(x), math.cos(x), -math.sin(x))
        residuals.append(abs(float(apply_operator_to_jet(op, x, jet))))
    max_residual = max(residuals)
    return {
        "multiple_of_d2_plus_1": multiple,
        "max_residual": max_residual,
        "residual_ok": max_residual < 1e-8,
        "g2_earned": multiple or max_residual < 1e-8,
    }


__all__ = [
    "DEFAULT_SPEC",
    "DISCLAIMER",
    "HolonomicLayerSpec",
    "apply_operator_to_jet",
    "content_clear_operator",
    "d2_plus_1",
    "d_minus_1",
    "fit_holonomic_layer",
    "holonomic_jet",
    "honesty_payload",
    "is_rational_multiple_of_d2_plus_1",
    "sin_skill",
    "worked_example",
]

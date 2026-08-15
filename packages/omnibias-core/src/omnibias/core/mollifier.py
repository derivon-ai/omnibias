# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Mollifier / distribution calculus for OMBU packs (theory 01-05).

A collapsing pack is a mollifier: the family ``phi_eps(u) = (1/eps) phi(u/eps)``
converges weakly to a Dirac mass as ``eps -> 0``, and a ``K``-pack converges to
the ``(K-1)``-st distributional derivative of that mass. This is the founding
``delta -> 0`` register, restated as a test-function generator for weak forms.

Analytic bases are **not** compactly supported. Boundary terms in a weak form
are bounded by a certified exponentially small tail (an :class:`Interval`),
never assumed zero. Higher-order (moment-annihilating) mollifiers take
**negative** values and are no longer densities.

Pure Python: no tensor imports. Tensor-side test-function assembly lives in
``omnibias.fields.weak`` (theory 02-04).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial, pi

from omnibias.core.multipack import PackSpec
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import gauss_cdf_iv, sigmoid_iv, tanh_iv

#: Bases whose unit-scale moments have a closed form.
_CLOSED_FORM_BASES: frozenset[str] = frozenset(
    {"gaussian", "logistic", "sigmoid", "sech", "tanh"}
)

#: Alias map onto the three moment families.
_FAMILY: dict[str, str] = {
    "gaussian": "gaussian",
    "logistic": "logistic",
    "sigmoid": "logistic",
    "sech": "sech",
    "tanh": "sech",
}

_MOMENT_TOL = 1e-10


def _canonical_base(base: str) -> str:
    name = str(base).lower().strip()
    if name not in _FAMILY:
        raise ValueError(
            f"unsupported mollifier base {base!r}; "
            f"expected one of {sorted(_CLOSED_FORM_BASES)}"
        )
    return name


def _family(base: str) -> str:
    return _FAMILY[_canonical_base(base)]


def _unit_even_moment(family: str, order: int) -> float:
    """Raw even moment ``M_{2k}`` of the unit-scale density (odd vanish)."""
    if order < 0:
        raise ValueError(f"moment order must be >= 0, got {order}")
    if order == 0:
        return 1.0
    if order % 2 == 1:
        return 0.0
    if family == "gaussian":
        # Standard normal: M_{2k} = (2k-1)!!
        k = order // 2
        return float(factorial(order) // (2**k * factorial(k)))
    if family == "logistic":
        # Logistic(0, 1): Var = pi^2/3, kurtosis 4.2 => M_4 = 7 pi^4 / 15.
        # Higher even moments: Euler polynomials / Bernoulli via |E_{2k}|.
        return _logistic_even_moment(order)
    # sech-type: density (1/2) sech^2 = Logistic(0, 1/2). M_{2k} = 2^{-2k} logistic.
    return _logistic_even_moment(order) / float(2**order)


def _logistic_even_moment(order: int) -> float:
    """Even raw moment of Logistic(0, 1). ``M_2 = pi^2/3``, ``M_4 = 7 pi^4 / 15``.

    The moment generating function is ``pi t / sin(pi t)`` for ``|t| < 1``,
    equivalently ``M_{2k} = 2 (2^{2k-1} - 1) |B_{2k}| pi^{2k}``.
    """
    if order == 0:
        return 1.0
    if order % 2 == 1:
        return 0.0
    b = abs(_bernoulli_even(order))
    return 2.0 * (2.0 ** (order - 1) - 1.0) * b * (pi**order)


def _bernoulli_even(n: int) -> float:
    """Bernoulli ``B_n`` for even ``n >= 2``, via Akiyama–Tanigawa (exact over Q)."""
    if n < 2 or n % 2 == 1:
        raise ValueError(f"expected even n >= 2, got {n}")
    a: list[Fraction] = [Fraction(0)] * (n + 1)
    for m in range(n + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = Fraction(j) * (a[j - 1] - a[j])
    return float(a[0])


def _gauss_elim(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Solve ``A w = b`` over ``Q``. ``A`` is square and must be nonsingular."""
    n = len(rhs)
    if any(len(row) != n for row in matrix) or len(matrix) != n:
        raise ValueError("Vandermonde system is not square")
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[pivot][col]):
                pivot = r
        if a[pivot][col] == 0:
            raise ValueError("singular moment-annihilation system")
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        for c in range(col, n + 1):
            a[col][c] /= div
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [row[n] for row in a]


def _relative_scales(n_scales: int) -> tuple[int, ...]:
    return tuple(2**i for i in range(n_scales))


@dataclass(frozen=True)
class MollifierSpec:
    """A (possibly multi-scale) mollifier built from CDF-type packs.

    Parameters
    ----------
    base:
        Activation name. ``sigmoid`` / ``logistic`` share the logistic density;
        ``tanh`` / ``sech`` share ``(1/2) sech^2``; ``gaussian`` is the standard
        normal density. The founding limit is ``eps -> 0`` (equivalently
        ``delta -> 0``), not temperature collapse.
    scale:
        Reference width ``eps > 0``. Each pack is evaluated at
        ``pack_scales[g] * scale``.
    packs:
        :class:`PackSpec` entries. ``order`` is unused for the density itself
        (keep ``0``); ``mean`` is the physical offset; ``weight`` is the outer
        coefficient. Negative weights are required past order 2 and mean the
        kernel is no longer a density.
    pack_scales:
        Relative scale per pack (default all ``1``). Richardson order-4 uses
        ``(1, 2)`` with weights ``(4/3, -1/3)``.
    """

    base: str
    scale: float
    packs: tuple[PackSpec, ...]
    pack_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        name = _canonical_base(self.base)
        if not (self.scale > 0.0) or self.scale != self.scale:
            raise ValueError(f"scale must be finite and positive, got {self.scale}")
        packs = tuple(self.packs)
        if not packs:
            raise ValueError("MollifierSpec requires at least one pack")
        rel = self.pack_scales
        if not rel:
            rel = tuple(1.0 for _ in packs)
        if len(rel) != len(packs):
            raise ValueError("pack_scales must match packs")
        if any(s <= 0.0 or s != s for s in rel):
            raise ValueError("pack_scales must be finite and positive")
        object.__setattr__(self, "base", name)
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "packs", packs)
        object.__setattr__(self, "pack_scales", tuple(float(s) for s in rel))

    @property
    def family(self) -> str:
        """Moment family: ``gaussian``, ``logistic``, or ``sech``."""
        return _family(self.base)

    @property
    def order(self) -> int:
        """First non-vanishing moment index ``>= 1`` (``0`` if only mass remains)."""
        # Even kernels: scan even indices. Cap at 16 so the property is cheap.
        for j in range(1, 17):
            if abs(self.moment(j)) > _MOMENT_TOL:
                return j
        return 0

    @property
    def is_positive(self) -> bool:
        """``False`` once any outer weight is negative (order ``> 2``)."""
        return all(p.weight >= 0.0 for p in self.packs)

    def moment(self, index: int) -> float:
        """Raw moment ``integral u^{index} psi(u) du`` of this spec."""
        return moments(self, up_to=index)[index]


def moments(spec: MollifierSpec, up_to: int) -> tuple[float, ...]:
    """Raw moments ``(M_0, ..., M_{up_to})``.

    Closed form for gaussian / logistic / sech-type families. Odd unit
    moments vanish; a non-zero mean expands by the binomial theorem.
    """
    if int(up_to) < 0:
        raise ValueError(f"up_to must be >= 0, got {up_to}")
    family = spec.family
    unit = [_unit_even_moment(family, k) for k in range(up_to + 1)]
    out = [0.0] * (up_to + 1)
    for pack, rel in zip(spec.packs, spec.pack_scales, strict=True):
        s = spec.scale * rel
        mu = pack.mean
        w = pack.weight
        for j in range(up_to + 1):
            acc = 0.0
            for k in range(j + 1):
                acc += float(comb(j, k)) * (s**k) * unit[k] * (mu ** (j - k))
            out[j] += w * acc
    return tuple(out)


def is_admissible(spec: MollifierSpec, *, form_order: int) -> bool:
    """Petrov-Galerkin admissibility: smoothness, decay, known integrals.

    Analytic bases are ``C^infty``, so ``form_order`` only rejects negatives.
    Rapid decay and closed-form antiderivatives are supplied by the family.
    This is not a compact-support test.
    """
    if int(form_order) < 0:
        raise ValueError(f"form_order must be >= 0, got {form_order}")
    try:
        _canonical_base(spec.base)
    except ValueError:
        return False
    return spec.scale > 0.0 and len(spec.packs) > 0


def design_order(base: str, order: int, *, scale: float = 1.0) -> MollifierSpec:
    """Solve pack weights that annihilate moments ``1 .. order-1``.

    Even kernels: geometric scales ``1, 2, 4, ...`` and a Vandermonde system
    over ``Q`` (the same exact-rational discipline as irregular Birkhoff
    stencils, implemented here so ``omnibias.core`` never imports
    ``omnibias.difference``). Order 4 recovers the Richardson pair
    ``(4/3) phi_eps - (1/3) phi_{2 eps}``.
    """
    _canonical_base(base)
    m = int(order)
    if m < 2 or m % 2 == 1:
        raise ValueError(f"design_order requires even order >= 2, got {order}")
    if not (scale > 0.0) or scale != scale:
        raise ValueError(f"scale must be finite and positive, got {scale}")
    n = m // 2
    rel = _relative_scales(n)
    # Rows: mass, then even powers 2, 4, ..., 2(n-1).
    matrix: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for row in range(n):
        power = 2 * row
        matrix.append([Fraction(rel[i]) ** power for i in range(n)])
        rhs.append(Fraction(1) if row == 0 else Fraction(0))
    weights = _gauss_elim(matrix, rhs)
    packs = tuple(
        PackSpec(order=0, mean=0.0, weight=float(w)) for w in weights
    )
    return MollifierSpec(
        base=base,
        scale=float(scale),
        packs=packs,
        pack_scales=tuple(float(s) for s in rel),
    )


def _unit_survival(family: str, z: Interval) -> Interval:
    """Enclosure of ``1 - F(z)`` for the unit-scale CDF ``F`` (``z`` may be negative)."""
    one = Interval.from_rational(1)
    half = Interval.from_rational(Fraction(1, 2))
    if family == "gaussian":
        return one - gauss_cdf_iv(z)
    if family == "logistic":
        return one - sigmoid_iv(z)
    # sech-type CDF F = (1 + tanh)/2, so 1-F = (1 - tanh)/2.
    return (one - tanh_iv(z)) * half


def _component_outside(
    family: str, *, mean: float, scale: float, half_width: float
) -> Interval:
    """Enclosure of mass of one scaled density outside ``[-W, W]``."""
    s = Interval.point(float(scale))
    w = Interval.point(float(half_width))
    mu = Interval.point(float(mean))
    left_arg = ((-w) - mu) / s
    right_arg = (w - mu) / s
    # P(X < -W) + P(X > W) = F((-W-mu)/s) + 1 - F((W-mu)/s)
    # = (1 - survival(left)) wait: F(x) = 1 - survival(x), so
    # F(left) + 1 - F(right) = 1 - survival(left) + survival(right)?
    # F(left) = 1 - surv(left); 1 - F(right) = surv(right)
    # total = 1 - surv(left) + surv(right). For mu=0, left=-W/s, surv(-z)=F(z),
    # 1 - F(W/s) + (1-F(W/s)) = 2 surv(W/s). Use CDFs directly:
    one = Interval.from_rational(1)
    f_left = one - _unit_survival(family, left_arg)
    f_right = one - _unit_survival(family, right_arg)
    return f_left + (one - f_right)


def tail_bound(spec: MollifierSpec, *, half_width: float) -> Interval:
    """Outward-rounded enclosure of mass outside ``[-half_width, half_width]``.

    Analytic bases have exponentially small tails, not compact support. The
    returned interval is a sound enclosure of the *signed* outside integral
    (higher-order kernels can be negative). Zero violations means every true
    value lies inside the interval.
    """
    if not (half_width > 0.0) or half_width != half_width:
        raise ValueError(f"half_width must be finite and positive, got {half_width}")
    family = spec.family
    total = Interval.from_rational(0)
    for pack, rel in zip(spec.packs, spec.pack_scales, strict=True):
        component = _component_outside(
            family,
            mean=pack.mean,
            scale=spec.scale * rel,
            half_width=half_width,
        )
        total = total + component * Interval.point(pack.weight)
    return total


def true_outside_mass(spec: MollifierSpec, *, half_width: float) -> float:
    """Closed-form signed outside mass (float64 evaluation of the same CDFs).

    Used as the truth oracle for G3. Not a replacement for :func:`tail_bound`.
    """
    from math import erf, exp, tanh

    family = spec.family
    w = float(half_width)
    acc = 0.0
    for pack, rel in zip(spec.packs, spec.pack_scales, strict=True):
        s = spec.scale * rel
        mu = pack.mean
        left = (-w - mu) / s
        right = (w - mu) / s
        if family == "gaussian":
            # Phi(x) = 0.5 (1 + erf(x / sqrt(2)))
            inv = 0.7071067811865476  # 1/sqrt(2)
            f_left = 0.5 * (1.0 + erf(left * inv))
            f_right = 0.5 * (1.0 + erf(right * inv))
        elif family == "logistic":
            f_left = 1.0 / (1.0 + exp(-left))
            f_right = 1.0 / (1.0 + exp(-right))
        else:
            f_left = 0.5 * (1.0 + tanh(left))
            f_right = 0.5 * (1.0 + tanh(right))
        acc += pack.weight * (f_left + (1.0 - f_right))
    return acc


__all__ = [
    "MollifierSpec",
    "design_order",
    "is_admissible",
    "moments",
    "tail_bound",
    "true_outside_mass",
]

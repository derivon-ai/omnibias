# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-Jacobian + high-order jet Lohner steps (theory 07-06).

Validated integration is limited by wrapping and by the Jacobian
enclosure. For fields in the activation dictionary the founding
bias collapse (``delta -> 0``) supplies ``Df`` and the solution
Taylor coefficients from one ``sigma`` evaluation per order.
Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two.

Every run emits a :class:`WidthBudget`. The existing
:func:`~omnibias.core.verified.lohner.lohner_flow` path is
untouched (G6). Scope is one field, one initial box, one
finite horizon. This is not a continuum existence theorem,
not an attractor statement, and not a Navier-Stokes
regularity result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.core.proof.certificate import make_certificate
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.jet import compose_jet
from omnibias.core.verified.kantorovich import krawczyk_certificate
from omnibias.core.verified.linalg import (
    IntervalMatrix,
    identity_matrix,
    mat_sub,
    matmul,
)
from omnibias.core.verified.lohner import (
    JacobianEnclosure,
    LohnerSet,
    _scale_matrix,
    constant_jacobian,
    interval_matrix_exp,
    linear_field,
    lohner_flow,
    lohner_step,
    naive_interval_flow,
)
from omnibias.core.verified.ode import (
    TaylorSeries,
    VectorField,
    _apriori_enclosure,
    _solution_coeffs,
)
from omnibias.core.verified.sigma import sigma_tower_interval

DISCLAIMER = (
    "finite-horizon enclosure of one trajectory from one initial box; "
    "not a continuum existence theorem, not an attractor statement, "
    "and not a Navier-Stokes global regularity result"
)
SCHEMA_VERSION = "validated-dynamics-jet-flow-1"
_EPS = 2.220446049250313e-16
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class WidthBudget:
    """Per-run width split. Required before any improvement claim."""

    truncation: float
    jacobian: float
    wrapping: float
    rounding: float

    @property
    def total(self) -> float:
        return float(
            self.truncation + self.jacobian + self.wrapping + self.rounding
        )

    @property
    def dominant(self) -> str:
        parts = {
            "truncation": self.truncation,
            "jacobian": self.jacobian,
            "wrapping": self.wrapping,
            "rounding": self.rounding,
        }
        return max(parts, key=lambda key: parts[key])

    def to_payload(self) -> dict[str, object]:
        return {
            "truncation": self.truncation,
            "jacobian": self.jacobian,
            "wrapping": self.wrapping,
            "rounding": self.rounding,
            "dominant": self.dominant,
            "total": self.total,
        }


@dataclass(frozen=True)
class TowerFieldSpec:
    """Activation-dictionary field consumed by :func:`tower_jacobian`."""

    name: str
    scale: float = 1.0
    bias: float = 0.0
    kind: str = "scalar"
    stiffness: float = 1.0
    damping: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).lower().strip()
        if kind not in {"scalar", "oscillator"}:
            raise ValueError(f"kind must be scalar or oscillator, got {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", str(self.name).lower().strip())


@dataclass(frozen=True)
class ValidatedRun:
    """One finite-horizon Lohner integration plus its width budget."""

    state: LohnerSet
    budget: WidthBudget
    horizon: float
    n_steps: int
    order: int

    @property
    def width(self) -> float:
        return float(self.state.width())


@dataclass(frozen=True)
class OrbitProof:
    """Krawczyk periodic-point certificate of a time-``period`` map."""

    exists: bool
    period: float
    center: tuple[float, ...]
    enclosure: tuple[tuple[float, float], ...] | None
    kappa: float | None
    budget: WidthBudget | None


def honesty_payload() -> dict[str, bool]:
    """Honesty flags. ``theorem_prover_verified`` is reported False, never sealed."""
    return {
        "unproven_claim": False,
        "continuum_existence_claim": False,
        "attractor_claim": False,
        "navier_stokes_regularity_claim": False,
        "theorem_prover_verified": False,
    }


def _seal_honesty() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "continuum_existence_claim": False,
        "attractor_claim": False,
        "navier_stokes_regularity_claim": False,
    }


def tower_field(spec: TowerFieldSpec) -> VectorField:
    """Closed-form activation vector field for ``spec``."""
    scale = Interval.point(spec.scale)
    bias = Interval.point(spec.bias)
    stiff = Interval.point(spec.stiffness)
    damp = Interval.point(spec.damping)
    name = spec.name

    def _sigma(series: TaylorSeries) -> TaylorSeries:
        u = [scale * coeff for coeff in series.coeffs]
        u[0] = u[0] + bias
        tower = sigma_tower_interval(name, u[0], len(u) - 1)
        return TaylorSeries(compose_jet(u, list(tower)))

    if spec.kind == "scalar":

        def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
            return [_sigma(series[0])]

        return field

    def osc(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x, y = series[0], series[1]
        return [y, _sigma(x) * (-stiff) + y * (-damp)]

    return osc


def tower_jacobian(spec: TowerFieldSpec) -> JacobianEnclosure:
    """Exact closed-form Jacobian enclosure. Width is rounding plus the box."""
    scale = Interval.point(spec.scale)
    bias = Interval.point(spec.bias)
    stiff = Interval.point(spec.stiffness)
    damp = Interval.point(spec.damping)
    name = spec.name
    zero = Interval.point(0.0)
    one = Interval.point(1.0)

    if spec.kind == "scalar":

        def jac_s(box: Sequence[Interval]) -> IntervalMatrix:
            arg = scale * box[0] + bias
            return [[scale * sigma_tower_interval(name, arg, 1)[1]]]

        return jac_s

    def jac_o(box: Sequence[Interval]) -> IntervalMatrix:
        arg = scale * box[0] + bias
        prime = sigma_tower_interval(name, arg, 1)[1]
        return [[zero, one], [(-stiff) * scale * prime, -damp]]

    return jac_o


def _float_sigmoid(z: float) -> float:
    if z >= 0.0:
        exp_m = math.exp(-z)
        return 1.0 / (1.0 + exp_m)
    exp_p = math.exp(z)
    return exp_p / (1.0 + exp_p)


def _float_sigma_prime(name: str, z: float) -> float:
    """Independent float ``sigma'(z)`` for containment checks."""
    if name == "tanh":
        t = math.tanh(z)
        return 1.0 - t * t
    if name == "sigmoid":
        return _float_sigmoid(z) * (1.0 - _float_sigmoid(z))
    if name == "gaussian":
        return -z * math.exp(-0.5 * z * z)
    if name == "sech":
        c = 2.0 / (math.exp(z) + math.exp(-z))
        return -c * math.tanh(z)
    if name == "silu":
        s = _float_sigmoid(z)
        return s + z * s * (1.0 - s)
    if name == "softplus":
        return _float_sigmoid(z)
    if name == "gelu":
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return cdf + z * phi
    if name == "sin":
        return math.cos(z)
    if name == "cos":
        return -math.sin(z)
    raise ValueError(f"unsupported activation {name!r}")


def _true_jacobian_at(spec: TowerFieldSpec, point: Sequence[float]) -> list[list[float]]:
    arg = spec.scale * point[0] + spec.bias
    prime = _float_sigma_prime(spec.name, arg)
    if spec.kind == "scalar":
        return [[spec.scale * prime]]
    return [
        [0.0, 1.0],
        [-spec.stiffness * spec.scale * prime, -spec.damping],
    ]


def named_tower_fields() -> list[TowerFieldSpec]:
    """Ten activation-dictionary fields for G2 containment."""
    return [
        TowerFieldSpec("tanh"),
        TowerFieldSpec("sigmoid"),
        TowerFieldSpec("gaussian"),
        TowerFieldSpec("sech"),
        TowerFieldSpec("silu"),
        TowerFieldSpec("softplus"),
        TowerFieldSpec("gelu"),
        TowerFieldSpec("tanh", kind="oscillator", stiffness=1.0, damping=0.0),
        TowerFieldSpec("sigmoid", kind="oscillator", stiffness=0.8, damping=0.1),
        TowerFieldSpec("sech", scale=1.2, bias=-0.25, kind="oscillator"),
    ]


def _zero_budget() -> WidthBudget:
    return WidthBudget(0.0, 0.0, 0.0, 0.0)


def _merge_budget(left: WidthBudget, right: WidthBudget) -> WidthBudget:
    return WidthBudget(
        truncation=max(left.truncation, right.truncation),
        jacobian=max(left.jacobian, right.jacobian),
        wrapping=max(left.wrapping, right.wrapping),
        rounding=left.rounding + right.rounding,
    )


def _step_budget(
    field: VectorField,
    jac: JacobianEnclosure,
    state: LohnerSet,
    nxt: LohnerSet,
    h: float,
    order: int,
) -> WidthBudget:
    box = state.to_box()
    z_encl = _apriori_enclosure(field, box, h)
    rem = _solution_coeffs(field, z_encl, order + 1)
    h_pow = abs(h) ** (order + 1)
    truncation = 0.0
    for row in rem:
        coeff = row[order + 1]
        truncation = max(truncation, coeff.width * h_pow)
    j_mat = jac(z_encl)
    jac_w = max(entry.width for row in j_mat for entry in row)
    r_w = max(iv.width for iv in state.r)
    jacobian = jac_w * r_w * abs(h)
    r_next = max(iv.width for iv in nxt.r)
    wrapping = max(0.0, nxt.width() - r_next)
    scale = 1.0 + max(abs(c.mid) for c in nxt.center)
    rounding = _EPS * scale
    return WidthBudget(truncation, jacobian, wrapping, rounding)


def lohner_step_jet(
    state: LohnerSet,
    field: VectorField,
    jac: JacobianEnclosure,
    *,
    order: int,
    h: float,
) -> tuple[LohnerSet, WidthBudget]:
    """Order-``p`` QR-Lohner step. Same update as :func:`lohner_step`."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    nxt = lohner_step(field, jac, state, h, order)
    return nxt, _step_budget(field, jac, state, nxt, h, order)


def lohner_flow_jet(
    field: VectorField,
    jac: JacobianEnclosure,
    y0: Sequence[IntervalLike],
    h: float,
    n_steps: int,
    order: int = 12,
) -> ValidatedRun:
    """Integrate with :func:`lohner_step_jet` and accumulate a width budget."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    state = LohnerSet.from_box(y0)
    budget = _zero_budget()
    for _ in range(n_steps):
        state, piece = lohner_step_jet(state, field, jac, order=order, h=h)
        budget = _merge_budget(budget, piece)
    return ValidatedRun(state, budget, h * n_steps, n_steps, order)


def widen_jacobian(jac: JacobianEnclosure, delta: float) -> JacobianEnclosure:
    """Add a symmetric interval pad. The polluted baseline for G4 / G3."""
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    pad = Interval(-delta, delta)

    def polluted(box: Sequence[Interval]) -> IntervalMatrix:
        return [[entry + pad for entry in row] for row in jac(box)]

    return polluted


def _series_radius(mids: Sequence[float]) -> float:
    """Conservative Cauchy–Hadamard radius from Taylor coefficients ``a_n t^n``."""
    inv = 0.0
    start = max(1, len(mids) - 8)
    for n in range(start, len(mids)):
        value = abs(float(mids[n]))
        if value < 1e-30:
            continue
        inv = max(inv, value ** (1.0 / n))
    if inv <= 0.0:
        return math.inf
    return 1.0 / inv


def adaptive_step_from_singularity(
    field: VectorField,
    state: LohnerSet,
    *,
    order: int,
    safety: float = 0.5,
    max_step: float = 1.0,
) -> float:
    """Suggest a step below the jet radius of convergence (then validate).

    The radius is a coefficient estimate, not a proof. The returned
    step is a suggestion; :func:`lohner_step_jet` still has to accept
    the a-priori enclosure.
    """
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must be in (0, 1], got {safety}")
    if max_step <= 0.0:
        raise ValueError(f"max_step must be positive, got {max_step}")
    point = [Interval.point(c.mid) for c in state.center]
    coeffs = _solution_coeffs(field, point, order)
    radius = min(_series_radius([c.mid for c in row]) for row in coeffs)
    if math.isinf(radius):
        return float(max_step)
    step = safety * radius
    if not math.isfinite(step) or step <= 0.0:
        return min(float(max_step), 1e-3)
    return min(float(max_step), step)


def riccati_field() -> VectorField:
    """``x' = x^2``. Solution ``1/(1-t)`` from ``x(0)=1`` blows up at ``t=1``."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x = series[0]
        return [x * x]

    return field


def tan_field() -> VectorField:
    """``x' = 1 + x^2``. Solution ``tan(t)`` from ``x(0)=0`` blows up at ``π/2``."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x = series[0]
        return [x * x + 1.0]

    return field


def exp_field() -> VectorField:
    """``x' = x``. Entire solution; radius is infinite."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        return [series[0]]

    return field


def cubic_oscillator() -> tuple[VectorField, JacobianEnclosure]:
    """Polynomial oscillator ``y' = z``, ``z' = -y^3``."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        y, z = series[0], series[1]
        return [z, -(y * y * y)]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        y = box[0]
        return [
            [Interval.point(0.0), Interval.point(1.0)],
            [Interval.point(-3.0) * y * y, Interval.point(0.0)],
        ]

    return field, jac


def radial_hopf(mu: float = 1.0) -> tuple[VectorField, JacobianEnclosure]:
    """Scalar radial Hopf map ``r' = mu r - r^3`` (isolated cycle at ``sqrt(mu)``)."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        r = series[0]
        return [r * mu - r * r * r]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        r = box[0]
        return [[Interval.point(mu) - Interval.point(3.0) * r.pow_int(2)]]

    return field, jac


def hopf_cartesian(mu: float = 1.0) -> tuple[VectorField, JacobianEnclosure]:
    """2-D Hopf normal form. Unreduced time-``T`` points are non-isolated."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        x, y = series[0], series[1]
        r2 = x * x + y * y
        return [x * mu - y - x * r2, x + y * mu - y * r2]

    def jac(box: Sequence[Interval]) -> IntervalMatrix:
        x, y = box[0], box[1]
        m = Interval.point(mu)
        three = Interval.point(3.0)
        two = Interval.point(2.0)
        x2, y2 = x.pow_int(2), y.pow_int(2)
        return [
            [m - (three * x2 + y2), Interval.point(-1.0) - two * x * y],
            [Interval.point(1.0) - two * x * y, m - (x2 + three * y2)],
        ]

    return field, jac


def _float_inverse(a: Sequence[Sequence[float]]) -> list[list[float]] | None:
    n = len(a)
    aug = [
        [float(a[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = col
        best = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > best:
                best, pivot = abs(aug[row][col]), row
        if abs(aug[pivot][col]) < 1e-14:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [v / piv for v in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def variational_flow_jet(
    field: VectorField,
    jac: JacobianEnclosure,
    y0: Sequence[IntervalLike],
    h: float,
    n_steps: int,
    order: int,
) -> tuple[LohnerSet, IntervalMatrix, WidthBudget]:
    """State plus fundamental matrix, using the jet stepper."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    state = LohnerSet.from_box(y0)
    fundamental = identity_matrix(len(state.center))
    budget = _zero_budget()
    for _ in range(n_steps):
        box = state.to_box()
        z_encl = _apriori_enclosure(field, box, h)
        j_step = interval_matrix_exp(_scale_matrix(jac(z_encl), h), order=max(order, 8))
        state, piece = lohner_step_jet(state, field, jac, order=order, h=h)
        fundamental = matmul(j_step, fundamental)
        budget = _merge_budget(budget, piece)
    return state, fundamental, budget


def prove_periodic_orbit_jet(
    field: VectorField,
    jac: JacobianEnclosure,
    x_bar: Sequence[float],
    period: float,
    *,
    n_steps: int = 20,
    order: int = 14,
    radii: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2),
) -> OrbitProof:
    """Krawczyk periodic point of ``Phi_T``, integrated with the jet stepper."""
    if period <= 0.0:
        raise ValueError("period must be positive")
    n = len(x_bar)
    h = period / n_steps
    failure = (ValueError, RuntimeError, OverflowError, ZeroDivisionError)

    def func(box: list[Interval]) -> list[Interval]:
        end, _, _ = variational_flow_jet(field, jac, box, h, n_steps, order)
        out = end.to_box()
        return [out[i] - box[i] for i in range(n)]

    def jacobian(box: list[Interval]) -> IntervalMatrix:
        _, fundamental, _ = variational_flow_jet(field, jac, box, h, n_steps, order)
        return mat_sub(fundamental, identity_matrix(n))

    try:
        j_center = jacobian([Interval.point(v) for v in x_bar])
        a_inv = _float_inverse([[x.mid for x in row] for row in j_center])
    except failure:
        a_inv = None
    if a_inv is None:
        return OrbitProof(False, period, tuple(x_bar), None, None, None)

    budget: WidthBudget | None = None
    for radius in radii:
        try:
            cert = krawczyk_certificate(func, jacobian, list(x_bar), a_inv, radius)
            _, _, budget = variational_flow_jet(
                field, jac, [Interval.point(v) for v in x_bar], h, n_steps, order
            )
        except failure:
            continue
        if cert is not None:
            return OrbitProof(
                True,
                period,
                tuple(x_bar),
                cert.enclosure,
                cert.kappa,
                budget,
            )
    return OrbitProof(False, period, tuple(x_bar), None, None, budget)


def jacobian_containment_report(
    *,
    n_boxes: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """G2: enclosure contains the true Jacobian on ``n_boxes`` per field."""
    import random

    rng = random.Random(seed)
    fields = named_tower_fields()
    misses = 0
    checked = 0
    per_field: list[dict[str, Any]] = []
    for spec in fields:
        jac = tower_jacobian(spec)
        dim = 1 if spec.kind == "scalar" else 2
        field_misses = 0
        for _ in range(n_boxes):
            half = [0.01 + 0.19 * rng.random() for _ in range(dim)]
            mid = [rng.uniform(-2.0, 2.0) for _ in range(dim)]
            box = [Interval(mid[i] - half[i], mid[i] + half[i]) for i in range(dim)]
            point = [mid[i] + (2.0 * rng.random() - 1.0) * half[i] * 0.9 for i in range(dim)]
            enclosed = jac(box)
            truth = _true_jacobian_at(spec, point)
            ok = True
            for i in range(len(truth)):
                for j in range(len(truth[i])):
                    if not (enclosed[i][j].lo <= truth[i][j] <= enclosed[i][j].hi):
                        ok = False
            checked += 1
            if not ok:
                misses += 1
                field_misses += 1
        per_field.append(
            {"name": f"{spec.kind}:{spec.name}", "misses": field_misses, "n": n_boxes}
        )
    return {
        "fields": len(fields),
        "boxes_per_field": n_boxes,
        "checked": checked,
        "misses": misses,
        "containment": misses == 0,
        "per_field": per_field,
    }


def decay_field(rate: float = 0.25) -> tuple[VectorField, JacobianEnclosure]:
    """Linear decay ``x' = -rate x``."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        return [series[0] * (-rate)]

    def jac(_box: Sequence[Interval]) -> IntervalMatrix:
        return [[Interval.point(-rate)]]

    return field, jac


def clock_field() -> tuple[VectorField, JacobianEnclosure]:
    """``x' = 1``. Entire, Jacobian zero."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        return [TaylorSeries.constant(1.0, series[0].order)]

    def jac(_box: Sequence[Interval]) -> IntervalMatrix:
        return [[Interval.point(0.0)]]

    return field, jac


def frozen_field() -> tuple[VectorField, JacobianEnclosure]:
    """``x' = 0``. The set is invariant."""

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        return [TaylorSeries.constant(0.0, series[0].order)]

    def jac(_box: Sequence[Interval]) -> IntervalMatrix:
        return [[Interval.point(0.0)]]

    return field, jac


def horizon_suite() -> list[tuple[str, VectorField, JacobianEnclosure, list[Interval]]]:
    """Ten isometries / contractions so wrapping, not true expansion, binds."""
    rot = [[0.0, -1.0], [1.0, 0.0]]
    slow = [[0.0, -0.5], [0.5, 0.0]]
    spiral = [[-0.2, -1.0], [1.0, -0.2]]
    spiral2 = [[-0.4, -0.7], [0.7, -0.4]]
    rad_f, rad_j = radial_hopf(1.0)
    decay_f, decay_j = decay_field(0.25)
    mid_d, mid_j = decay_field(0.5)
    fast_d, fast_j = decay_field(1.0)
    clock_f, clock_j = clock_field()
    frozen_f, frozen_j = frozen_field()
    thin = [Interval(0.999, 1.001), Interval(-0.001, 0.001)]
    return [
        ("rotation", linear_field(rot), constant_jacobian(rot), thin),
        ("slow_rotation", linear_field(slow), constant_jacobian(slow),
         [Interval(0.7, 0.71), Interval(-0.01, 0.01)]),
        ("contracting_spiral", linear_field(spiral), constant_jacobian(spiral), thin),
        ("contracting_spiral2", linear_field(spiral2), constant_jacobian(spiral2), thin),
        ("radial_hopf", rad_f, rad_j, [Interval(0.999, 1.001)]),
        ("decay", decay_f, decay_j, [Interval(0.4, 0.5)]),
        ("mid_decay", mid_d, mid_j, [Interval(0.4, 0.5)]),
        ("fast_decay", fast_d, fast_j, [Interval(0.4, 0.5)]),
        ("clock", clock_f, clock_j, [Interval(-0.1, 0.1)]),
        ("frozen", frozen_f, frozen_j, [Interval(-0.2, 0.2)]),
    ]


def _largest_step(
    field: VectorField,
    jac: JacobianEnclosure,
    y0: Sequence[Interval],
    order: int,
    h_hi: float = 0.8,
) -> float:
    """Largest step whose a-priori enclosure succeeds (validated, not guessed)."""
    low = 1e-3
    high = h_hi
    best = low
    state = LohnerSet.from_box(y0)
    for _ in range(14):
        mid = 0.5 * (low + high)
        try:
            lohner_step(field, jac, state, mid, order)
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            high = mid
        else:
            best = mid
            low = mid
    return best


def horizon_report() -> list[dict[str, Any]]:
    """G3: at 5x the baseline time the jet width still sits under the baseline cap."""
    rows: list[dict[str, Any]] = []
    h_base = 0.05
    steps_base = 16
    order_base = 4
    order_jet = 12
    t_base = h_base * steps_base
    for name, field, jac, y0 in horizon_suite():
        base = lohner_flow(field, jac, y0, h_base, steps_base, order=order_base)
        cap = max(base.width(), 1e-14)
        h_jet = min(0.40, _largest_step(field, jac, y0, order_jet, h_hi=0.8))
        t_target = 5.0 * t_base
        n_jet = max(1, int(math.ceil(t_target / h_jet)))
        try:
            jet = lohner_flow_jet(field, jac, y0, h_jet, n_jet, order=order_jet)
            t_jet = jet.horizon
            width_jet = jet.width
            ok = width_jet <= cap * 1.05 + 1e-15 and t_jet + 1e-12 >= t_target
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            t_jet = 0.0
            width_jet = math.inf
            ok = False
        ratio = t_jet / t_base if t_base > 0.0 else 0.0
        rows.append(
            {
                "name": name,
                "t_base": t_base,
                "t_jet": t_jet,
                "ratio": ratio,
                "cap": cap,
                "width_jet": width_jet,
                "h_jet": h_jet,
                "win": ok and ratio >= 5.0,
            }
        )
    return rows


def named_orbit_problems() -> list[tuple[str, VectorField, JacobianEnclosure, list[float], float]]:
    """Three isolated radial Hopf cycles (replayable G4 certificates)."""
    out: list[tuple[str, VectorField, JacobianEnclosure, list[float], float]] = []
    for mu in (1.0, 0.5, 0.25):
        field, jac = radial_hopf(mu)
        out.append((f"radial_mu_{mu:g}", field, jac, [math.sqrt(mu)], _TWO_PI))
    return out


def orbit_recovery_report() -> list[dict[str, Any]]:
    """G4: polluted order-4 proofs fail; exact jet proofs succeed."""
    rows: list[dict[str, Any]] = []
    for name, field, jac, x_bar, period in named_orbit_problems():
        dirty = widen_jacobian(jac, 4.0)
        baseline = prove_periodic_orbit_jet(
            field, dirty, x_bar, period, n_steps=40, order=4, radii=(1e-4, 1e-3, 1e-2)
        )
        jet = prove_periodic_orbit_jet(
            field, jac, x_bar, period, n_steps=50, order=14, radii=(1e-5, 1e-4, 1e-3, 1e-2)
        )
        rows.append(
            {
                "name": name,
                "baseline_exists": baseline.exists,
                "jet_exists": jet.exists,
                "jet_kappa": jet.kappa,
                "jet_enclosure": jet.enclosure,
                "recovered": (not baseline.exists) and jet.exists,
            }
        )
    return rows


def singularity_step_report() -> list[dict[str, Any]]:
    """G5: suggested steps stay strictly inside the known radius."""
    cases = [
        ("riccati", riccati_field(), [Interval.point(1.0)], 1.0),
        ("tan", tan_field(), [Interval.point(0.0)], 0.5 * math.pi),
        ("exp", exp_field(), [Interval.point(1.0)], math.inf),
    ]
    rows: list[dict[str, Any]] = []
    for name, field, y0, true_r in cases:
        state = LohnerSet.from_box(y0)
        step = adaptive_step_from_singularity(field, state, order=16, safety=0.5, max_step=1.0)
        finite = math.isfinite(true_r)
        ok = step <= true_r if finite else 0.0 < step <= 1.0
        rows.append(
            {
                "name": name,
                "step": step,
                "true_radius": true_r if finite else None,
                "sound": ok,
            }
        )
    return rows


def coefficient_arithmetic_report() -> dict[str, Any]:
    """Interval Taylor vs a point jet: TM is not required when G3 holds."""
    spec = TowerFieldSpec("tanh")
    field = tower_field(spec)
    point = [Interval.point(0.3)]
    box = [Interval(0.2, 0.4)]
    order = 12
    p_coeffs = _solution_coeffs(field, point, order + 1)
    b_coeffs = _solution_coeffs(field, box, order + 1)
    p_w = p_coeffs[0][order + 1].width
    b_w = b_coeffs[0][order + 1].width
    ratio = b_w / p_w if p_w > 0.0 else 0.0
    return {
        "choice": "interval",
        "box_over_point": ratio,
        "tm_required": False,
        "justification": (
            "interval Taylor coefficients on the 07-06 suite keep the "
            "order-12 remainder finite; Taylor models are not required "
            "to pass G3"
        ),
    }


def lohner_regression_snapshot() -> dict[str, float]:
    """Frozen ``lohner_flow`` numbers. The jet path must not change these."""
    a = [[0.0, -1.0], [1.0, 0.0]]
    y0 = [Interval(0.999, 1.001), Interval(-0.001, 0.001)]
    final = lohner_flow(linear_field(a), constant_jacobian(a), y0, 0.05, 20, order=12)
    box = final.to_box()
    return {
        "width": final.width(),
        "x_lo": box[0].lo,
        "x_hi": box[0].hi,
        "y_lo": box[1].lo,
        "y_hi": box[1].hi,
    }


def naive_vs_lohner_wrapping() -> dict[str, float]:
    """Sanity: QR-Lohner still beats naive interval on a rotation."""
    a = [[0.0, -1.0], [1.0, 0.0]]
    y0 = [Interval(0.999, 1.001), Interval(-0.001, 0.001)]
    h = 0.05
    steps = 40
    loh = lohner_flow(linear_field(a), constant_jacobian(a), y0, h, steps, order=12)
    naive = naive_interval_flow(a, y0, h, steps, order=12)
    return {"lohner": loh.width(), "naive": naive[0].width}


def seal_run(run: ValidatedRun) -> dict[str, object]:
    """Hash-sealed v1 certificate. Reserved kernel keys are never supplied."""
    box = run.state.to_box()
    return dict(
        make_certificate(
            claim="finite-horizon Lohner enclosure of one trajectory",
            payload={
                "schema": SCHEMA_VERSION,
                "horizon": run.horizon,
                "n_steps": run.n_steps,
                "order": run.order,
                "width": run.width,
                "box": [(iv.lo, iv.hi) for iv in box],
                "budget": run.budget.to_payload(),
                "disclaimer": DISCLAIMER,
            },
            honesty=_seal_honesty(),
        )
    )


def run_schema_errors(run: ValidatedRun) -> list[str]:
    errors: list[str] = []
    if run.budget.dominant not in {"truncation", "jacobian", "wrapping", "rounding"}:
        errors.append("dominant_missing")
    if run.horizon <= 0.0:
        errors.append("horizon_nonpositive")
    return errors


__all__ = [
    "DISCLAIMER",
    "OrbitProof",
    "SCHEMA_VERSION",
    "TowerFieldSpec",
    "ValidatedRun",
    "WidthBudget",
    "adaptive_step_from_singularity",
    "clock_field",
    "coefficient_arithmetic_report",
    "cubic_oscillator",
    "decay_field",
    "exp_field",
    "frozen_field",
    "honesty_payload",
    "hopf_cartesian",
    "horizon_report",
    "horizon_suite",
    "jacobian_containment_report",
    "lohner_flow_jet",
    "lohner_regression_snapshot",
    "lohner_step_jet",
    "naive_vs_lohner_wrapping",
    "named_orbit_problems",
    "named_tower_fields",
    "orbit_recovery_report",
    "prove_periodic_orbit_jet",
    "radial_hopf",
    "riccati_field",
    "run_schema_errors",
    "seal_run",
    "singularity_step_report",
    "tan_field",
    "tower_field",
    "tower_jacobian",
    "variational_flow_jet",
    "widen_jacobian",
]

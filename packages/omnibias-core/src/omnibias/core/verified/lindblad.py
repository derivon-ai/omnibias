# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified GKSL / Lindblad dynamics (theory 09-32).

The rigorous twin of :mod:`omnibias.core.lindblad`. ``ComplexInterval`` is
scalar-only, so every matrix path realifies once: a complex ``n x n``
matrix ``M = A + i B`` acting on ``v = x + i y`` becomes the real
``2n x 2n`` block ``[[A, -B], [B, A]]``. A Hermitian density matrix
``rho = A + i B`` is positive semidefinite iff the real symmetric
``[[A, -B], [B, A]]`` is, which unlocks the shipped
:func:`~omnibias.core.verified.eig_operator.interval_ldlt_pivots`
certificate with no new Hermitian machinery.

``propagator_enclosure`` reuses
:func:`~omnibias.core.verified.lohner.interval_matrix_exp` and *must*
subdivide ``[0, t]`` so each step satisfies
``||M||_inf / (order+2) < 1``. No admissible step is a first-class
``None`` refusal, never a wrong bound. ``lohner_flow`` is used only
when the generator is an exact float matrix
(:func:`~omnibias.core.verified.lohner.linear_field` takes floats);
interval-valued rates take the matrix-exponential composition path.

Two collapse senses are named (mirroring :mod:`omnibias.core.lindblad`)
and must not be conflated: the founding bias collapse (``delta -> 0``)
never appears here, and the ``beta -> inf`` founding **temperature
collapse** (feasibility sense) is evaluated only as an external
reference via the occupancy bridge, never as a registry-seeking claim
of this module. Do not conflate the two.

The GKSL form is a caller input. There is no Born–Markov–secular
derivation here. General ``d`` is a certified numerical propagator,
not a closed form.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.collapse.schema import CollapseOutcome, default_honesty
from omnibias.core.collapse.verdict import ObligationVerdict
from omnibias.core.lindblad import LindbladModel, dissipative_gap, liouvillian, steady_state
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.eig_operator import interval_ldlt_inertia
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import (
    IntervalMatrix,
    identity_matrix,
    inf_norm_matrix,
    interval_solve,
    matmul,
    matvec,
)
from omnibias.core.verified.lohner import (
    constant_jacobian,
    interval_matrix_exp,
    linear_field,
    lohner_flow,
    naive_interval_flow,
)

ComplexIntervalMatrix = tuple[tuple[ComplexInterval, ...], ...]


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="relaxation")
    payload["relaxation_collapse"] = True
    payload["wave_function_collapse_claim"] = False
    payload["measurement_problem_resolved"] = False
    payload["single_outcome_claim"] = False
    payload["born_rule_derived"] = False
    payload["markovian_model_declared_not_derived"] = True
    payload["non_markovian_claim"] = False
    payload["general_closed_form_claim"] = False
    payload["thermodynamic_limit_taken"] = False
    payload["continuum_limit_taken"] = False
    payload["quantum_advantage_claim"] = False
    payload["float_residual_is_proof"] = False
    return payload


def _scale_matrix(matrix: IntervalMatrix, factor: IntervalLike) -> IntervalMatrix:
    scale = Interval.from_value(factor)
    return [[entry * scale for entry in row] for row in matrix]


def _complex_interval_to_realified(
    matrix: Sequence[Sequence[ComplexInterval]],
) -> IntervalMatrix:
    """Realify a complex interval matrix ``M = A + i B`` to ``[[A, -B], [B, A]]``."""
    n = len(matrix)
    out: IntervalMatrix = []
    for i in range(n):
        row = [matrix[i][j].re for j in range(n)]
        row.extend(-matrix[i][j].im for j in range(n))
        out.append(row)
    for i in range(n):
        row = [matrix[i][j].im for j in range(n)]
        row.extend(matrix[i][j].re for j in range(n))
        out.append(row)
    return out


def _complex_matrix_to_realified(matrix: Sequence[Sequence[complex]]) -> IntervalMatrix:
    """Realify a point complex matrix to a point interval matrix of size ``2n``."""
    boxed = [
        [ComplexInterval.point(complex(matrix[i][j])) for j in range(len(matrix))]
        for i in range(len(matrix))
    ]
    return _complex_interval_to_realified(boxed)


def _interval_matrix_pow(matrix: IntervalMatrix, exponent: int) -> IntervalMatrix:
    """Binary exponentiation of an interval matrix (``exponent >= 1``)."""
    if exponent < 1:
        raise ValueError(f"exponent must be >= 1, got {exponent}")
    acc: IntervalMatrix | None = None
    base = matrix
    n = exponent
    while n:
        if n & 1:
            acc = base if acc is None else matmul(acc, base)
        n >>= 1
        if n:
            base = matmul(base, base)
    assert acc is not None
    return acc


def _to_complex_interval_matrix(
    rho: Sequence[Sequence[ComplexLike]],
) -> ComplexIntervalMatrix:
    rows = [tuple(ComplexInterval.from_value(entry) for entry in row) for row in rho]
    dim = len(rows)
    if dim < 1 or any(len(row) != dim for row in rows):
        raise ValueError("rho must be a non-empty square matrix")
    return tuple(rows)


def _rho_to_real_vec(rho: ComplexIntervalMatrix) -> list[Interval]:
    dim = len(rho)
    re_part: list[Interval] = []
    im_part: list[Interval] = []
    for j in range(dim):
        for i in range(dim):
            re_part.append(rho[i][j].re)
            im_part.append(rho[i][j].im)
    return re_part + im_part


def _real_vec_to_rho(vector: Sequence[Interval], dim: int) -> ComplexIntervalMatrix:
    dim2 = dim * dim
    if len(vector) != 2 * dim2:
        raise ValueError(f"realified vector length {len(vector)} != {2 * dim2}")
    rows: list[list[ComplexInterval]] = [
        [ComplexInterval.zero() for _ in range(dim)] for _ in range(dim)
    ]
    for j in range(dim):
        for i in range(dim):
            idx = i + j * dim
            rows[i][j] = ComplexInterval.from_parts(vector[idx], vector[dim2 + idx])
    return tuple(tuple(row) for row in rows)


def _inf_norm_enclosure(matrix: IntervalMatrix) -> Interval:
    """Sound enclosure of ``||M||_inf`` over an interval matrix box."""
    lower = 0.0
    upper = 0.0
    for row in matrix:
        lower = max(lower, sum(entry.mig for entry in row))
        upper = max(upper, sum(entry.mag for entry in row))
    if upper < lower:
        upper = lower
    return Interval(lower, upper)


def liouvillian_enclosure(model: LindbladModel) -> IntervalMatrix:
    """Realified point-interval enclosure of the complex superoperator."""
    return _complex_matrix_to_realified(liouvillian(model).tolist())


def propagator_enclosure(
    model: LindbladModel,
    time: IntervalLike,
    *,
    order: int = 12,
) -> IntervalMatrix | None:
    """Certified realified ``exp(t L)``. ``None`` if no admissible step exists."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    time_iv = Interval.from_value(time)
    if time_iv.lo < 0.0:
        raise ValueError(f"time must be nonnegative, got lo={time_iv.lo!r}")
    generator = liouvillian_enclosure(model)
    size = len(generator)
    if time_iv.hi == 0.0:
        return identity_matrix(size)
    norm = inf_norm_matrix(generator)
    if norm == 0.0:
        return identity_matrix(size)
    max_h = 0.5 * (order + 2) / norm
    n_steps = max(1, int(math.ceil(time_iv.hi / max_h)))
    k = (n_steps - 1).bit_length()
    n_pow = 1 << k
    step_time = Interval(time_iv.lo / n_pow, time_iv.hi / n_pow)
    scaled = _scale_matrix(generator, step_time)
    try:
        step = interval_matrix_exp(scaled, order=order)
    except ValueError:
        return None
    return _interval_matrix_pow(step, n_pow)


def density_matrix_enclosure(
    model: LindbladModel,
    rho0: Sequence[Sequence[ComplexLike]],
    time: IntervalLike,
    *,
    order: int = 12,
) -> ComplexIntervalMatrix | None:
    """Sound enclosure of ``rho(t) = exp(t L) rho(0)``."""
    rho = _to_complex_interval_matrix(rho0)
    if len(rho) != model.dim:
        raise ValueError(f"rho0 dim {len(rho)} != model dim {model.dim}")
    prop = propagator_enclosure(model, time, order=order)
    if prop is None:
        return None
    vec = matvec(prop, _rho_to_real_vec(rho))
    return _real_vec_to_rho(vec, model.dim)


@dataclass(frozen=True)
class TrajectoryEnclosure:
    """Final-time enclosure of a linear GKSL flow."""

    rho: ComplexIntervalMatrix
    width: float
    method: Literal["lohner", "naive"]


def trajectory_enclosure(
    model: LindbladModel,
    rho0: Sequence[Sequence[ComplexLike]],
    *,
    h: float,
    n_steps: int,
    order: int = 12,
    method: Literal["lohner", "naive"] = "lohner",
) -> TrajectoryEnclosure:
    """QR-Lohner (or wrapping-prone naive) flow of the realified generator.

    ``linear_field`` takes a float matrix, so this path requires an exact
    (point) generator -- the usual case for a ``LindbladModel``.
    """
    if h <= 0.0 or n_steps < 1:
        raise ValueError("h must be > 0 and n_steps >= 1")
    rho = _to_complex_interval_matrix(rho0)
    if len(rho) != model.dim:
        raise ValueError(f"rho0 dim {len(rho)} != model dim {model.dim}")
    generator = liouvillian_enclosure(model)
    float_gen = [[entry.mid for entry in row] for row in generator]
    y0 = _rho_to_real_vec(rho)
    if method == "naive":
        box = naive_interval_flow(float_gen, y0, h, n_steps, order=order)
        width = max(iv.width for iv in box)
        return TrajectoryEnclosure(
            rho=_real_vec_to_rho(box, model.dim),
            width=width,
            method="naive",
        )
    final = lohner_flow(
        linear_field(float_gen),
        constant_jacobian(float_gen),
        y0,
        h,
        n_steps,
        order=order,
    )
    box = final.to_box()
    return TrajectoryEnclosure(
        rho=_real_vec_to_rho(box, model.dim),
        width=final.width(),
        method="lohner",
    )


def trace_enclosure(rho: Sequence[Sequence[ComplexLike]]) -> Interval:
    """Sound enclosure of ``Tr(rho)``."""
    boxed = _to_complex_interval_matrix(rho)
    total = Interval.point(0.0)
    for i in range(len(boxed)):
        total = total + boxed[i][i].re
    return total


def hermiticity_residual_enclosure(rho: Sequence[Sequence[ComplexLike]]) -> Interval:
    """Sound enclosure of ``max_{ij} |rho_ij - conj(rho_ji)|``."""
    boxed = _to_complex_interval_matrix(rho)
    dim = len(boxed)
    best = Interval.point(0.0)
    for i in range(dim):
        for j in range(dim):
            diff = boxed[i][j] - boxed[j][i].conj()
            mag = diff.modulus()
            best = Interval(max(best.lo, mag.lo), max(best.hi, mag.hi))
    return best


def _hermitian_realification(rho: ComplexIntervalMatrix) -> IntervalMatrix:
    dim = len(rho)
    a = [[rho[i][j].re for j in range(dim)] for i in range(dim)]
    b = [[rho[i][j].im for j in range(dim)] for i in range(dim)]
    out: IntervalMatrix = []
    for i in range(dim):
        row = list(a[i])
        row.extend(-b[i][j] for j in range(dim))
        out.append(row)
    for i in range(dim):
        row = list(b[i])
        row.extend(a[i][j] for j in range(dim))
        out.append(row)
    return out


def positivity_verdict(rho: Sequence[Sequence[ComplexLike]]) -> ObligationVerdict:
    """Decide whether a Hermitian interval matrix is certified positive definite.

    A rank-1 (pure) state has a zero eigenvalue, so ``PROVED`` is
    structurally unreachable and the honest outcome is ``BLOCKED``.
    """
    boxed = _to_complex_interval_matrix(rho)
    realified = _hermitian_realification(boxed)
    inertia = interval_ldlt_inertia(realified)
    residual = Interval.point(0.0)
    if inertia is None:
        outcome = CollapseOutcome(
            status="inconclusive",
            spec_name="relaxation",
            surviving=None,
            residual=residual,
            detail=(
                "LDL^T pivot straddles 0; positivity Inconclusive, not falsity "
                "(a pure state is structurally BLOCKED)"
            ),
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="BLOCKED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=False,
            detail=outcome.detail,
        )
    if inertia.negative > 0:
        outcome = CollapseOutcome(
            status="excluded",
            spec_name="relaxation",
            surviving="DISPROVED",
            residual=residual,
            detail=f"certified {inertia.negative} negative pivot(s); not PSD",
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="DISPROVED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    outcome = CollapseOutcome(
        status="collapsed",
        spec_name="relaxation",
        surviving="positive_definite_state",
        residual=residual,
        detail="realified LDL^T pivots are all strictly positive",
        honesty=_honesty(),
    )
    return ObligationVerdict(
        status="PROVED",
        outcome=outcome,
        existential=True,
        evaluated=1,
        complete=True,
        detail=outcome.detail,
    )


def _augmented_complex_liouvillian(model: LindbladModel) -> list[list[complex]]:
    """Complex ``L`` with the last row replaced by the trace covector."""
    gen = liouvillian(model)
    dim = model.dim
    dim2 = dim * dim
    augmented = [list(row) for row in gen.tolist()]
    augmented[-1] = [0j] * dim2
    for i in range(dim):
        augmented[-1][i + i * dim] = 1.0 + 0j
    return augmented


def steady_state_enclosure(model: LindbladModel) -> ComplexIntervalMatrix | None:
    """Krawczyk enclosure of the unique trace-1 kernel vector, or ``None``.

    The realified unaugmented generator has a two-dimensional real kernel
    (the complex line ``C rho_ss``). Augment the *complex* ``d^2 x d^2``
    system first (trace-1), then realify, so both real constraints
    ``Re Tr = 1`` and ``Im Tr = 0`` are present.
    """
    try:
        steady_state(model)
    except ValueError:
        return None
    dim = model.dim
    dim2 = dim * dim
    realified = _complex_matrix_to_realified(_augmented_complex_liouvillian(model))
    rhs = [Interval.point(0.0) for _ in range(2 * dim2)]
    rhs[dim2 - 1] = Interval.point(1.0)
    try:
        vec = interval_solve(realified, rhs)
    except ValueError:
        return None
    return _real_vec_to_rho(vec, dim)


def _projector_realified(ss: ComplexIntervalMatrix) -> IntervalMatrix:
    """Realification of ``Pi x = rho_ss Tr(x)``."""
    dim = len(ss)
    dim2 = dim * dim
    ss_vec: list[ComplexInterval] = []
    for j in range(dim):
        for i in range(dim):
            ss_vec.append(ss[i][j])
    tau = [0.0] * dim2
    for i in range(dim):
        tau[i + i * dim] = 1.0
    projector_c = [
        [ss_vec[row] * tau[col] for col in range(dim2)] for row in range(dim2)
    ]
    return _complex_interval_to_realified(projector_c)


def _direct_contraction_enclosure(
    model: LindbladModel,
    time: IntervalLike,
    *,
    order: int,
    ss: ComplexIntervalMatrix,
) -> Interval | None:
    prop = propagator_enclosure(model, time, order=order)
    if prop is None:
        return None
    projector = _projector_realified(ss)
    residual = [
        [prop[i][j] - projector[i][j] for j in range(len(prop))]
        for i in range(len(prop))
    ]
    return _inf_norm_enclosure(residual)


def _contractive_base(
    model: LindbladModel,
    *,
    order: int,
    ss: ComplexIntervalMatrix,
) -> tuple[float, float] | None:
    """A time ``t0`` at which the direct contraction upper bound is in ``(0, 1)``.

    Prefers the fastest certified decay ``log(q)/t0``. Used to lift a tight
    short-time bound to long times via ``(exp(t0 L) - Pi)^n`` without
    wrapping the long-horizon propagator.
    """
    best: tuple[float, float, float] | None = None
    for t0 in (0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        enc = _direct_contraction_enclosure(model, t0, order=order, ss=ss)
        if enc is None or enc.hi <= 0.0 or enc.hi >= 1.0:
            continue
        rate = math.log(enc.hi) / t0
        if best is None or rate < best[0]:
            best = (rate, t0, enc.hi)
    if best is None:
        return None
    return best[1], best[2]


def _product_contraction_upper(
    model: LindbladModel,
    time: float,
    *,
    order: int,
    ss: ComplexIntervalMatrix,
) -> float | None:
    """Sound upper bound on ``||exp(t L) - Pi||_inf`` from a contractive step.

    Uses ``(exp(s L) - Pi)(exp(u L) - Pi) = exp((s+u) L) - Pi``.
    """
    if time <= 0.0:
        return None
    base = _contractive_base(model, order=order, ss=ss)
    if base is None:
        return None
    t0, q = base
    n = int(math.floor(time / t0 + 1e-15))
    rem_t = time - n * t0
    upper = q**n if n > 0 else 1.0
    if rem_t > 1e-15:
        rem = _direct_contraction_enclosure(model, rem_t, order=order, ss=ss)
        if rem is None:
            return None
        upper *= rem.hi
    elif n == 0:
        return None
    if not math.isfinite(upper):
        return None
    return float(upper)


def contraction_enclosure(
    model: LindbladModel,
    time: IntervalLike,
    *,
    order: int = 12,
) -> Interval | None:
    """Sound enclosure of ``||exp(t L) - Pi||_inf`` toward the unique ``rho_ss``.

    ``Pi x = rho_ss * Tr(x)``. Returns ``None`` when uniqueness or the
    propagator cannot be certified. A short-time product bound tightens
    the upper endpoint when the long-horizon propagator wraps.
    """
    ss = steady_state_enclosure(model)
    if ss is None:
        return None
    try:
        steady_state(model)
    except ValueError:
        return None
    direct = _direct_contraction_enclosure(model, time, order=order, ss=ss)
    time_iv = Interval.from_value(time)
    product_hi: float | None = None
    if time_iv.width <= 1e-14:
        product_hi = _product_contraction_upper(
            model, time_iv.mid, order=order, ss=ss
        )
    if direct is None and product_hi is None:
        return None
    lower = 0.0 if direct is None else direct.lo
    upper = math.inf if direct is None else direct.hi
    if product_hi is not None:
        upper = min(upper, product_hi)
    if not math.isfinite(upper):
        return None
    if upper < lower:
        upper = lower
    return Interval(lower, upper)


@dataclass(frozen=True)
class RelaxationTime:
    """Certified relaxation-time report. An empty ``time`` is a halt."""

    time: Interval | None
    reason: Literal["certified", "not_unique", "not_contracting", "horizon"]
    contraction: Interval | None


def certified_relaxation_time(
    model: LindbladModel,
    *,
    distance_budget: float,
    t_max: float,
    order: int = 12,
    n_search: int = 24,
) -> RelaxationTime:
    """Smallest probed ``T`` with contraction upper bound ``< eps``, or a halt.

    Conservative: the returned ``T`` is never smaller than a float spectral
    abscissa oracle would require. A non-unique kernel or a nonnegative
    dissipative gap is a named halt, not a number.
    """
    eps = float(distance_budget)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"distance_budget must be a positive finite number, got {eps!r}")
    tmax = float(t_max)
    if not math.isfinite(tmax) or tmax <= 0.0:
        raise ValueError(f"t_max must be a positive finite number, got {t_max!r}")

    try:
        steady_state(model)
    except ValueError:
        return RelaxationTime(time=None, reason="not_unique", contraction=None)
    gap = dissipative_gap(model)
    if gap >= -1e-14:
        return RelaxationTime(time=None, reason="not_contracting", contraction=None)

    lo, hi = 0.0, tmax
    best: Interval | None = None
    best_t: float | None = None
    for _ in range(n_search):
        mid = 0.5 * (lo + hi)
        enc = contraction_enclosure(model, mid, order=order)
        if enc is None:
            hi = mid
            continue
        if enc.hi < eps:
            best = enc
            best_t = mid
            hi = mid
        else:
            lo = mid
    if best_t is None:
        final = contraction_enclosure(model, tmax, order=order)
        return RelaxationTime(time=None, reason="horizon", contraction=final)
    return RelaxationTime(
        time=Interval.point(best_t),
        reason="certified",
        contraction=best,
    )


__all__ = [
    "RelaxationTime",
    "TrajectoryEnclosure",
    "certified_relaxation_time",
    "contraction_enclosure",
    "density_matrix_enclosure",
    "hermiticity_residual_enclosure",
    "liouvillian_enclosure",
    "positivity_verdict",
    "propagator_enclosure",
    "steady_state_enclosure",
    "trace_enclosure",
    "trajectory_enclosure",
]

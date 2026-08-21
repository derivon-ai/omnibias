# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact jet line-search algebra (theory 03-12).

A directional Taylor jet of a scalar ``phi(s) = L(theta + s d)`` turns the
line-search subproblem into root-finding on a known polynomial. This module is
the shared, backend-free half: Taylor coefficients, certified Lagrange
remainder radii, Wolfe-as-interval, and model-step selection. Tensor
evaluation of the jet lives in the torch / jax twins.

The jet comes from the founding bias collapse (``delta -> 0``) tower. No
temperature collapse appears. The polynomial is a model, not the loss:
truncation is real, the trust radius bounds it when a sound
``|phi^(N+1)|`` bound is supplied, and ``verify=True`` is the never-worse
backstop.

``omnibias.core`` cannot import ``omnibias.difference`` (that package depends
on core). The remainder used here is the same Lagrange identity that
``certified_fd_error_general`` applies to stencils: if
``|phi^(N+1)| <= M`` on ``[0, r]`` then
``|R_N(s)| <= M |s|^(N+1) / (N+1)!``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.rootfind import certified_sign_change, interval_newton

#: Default bracket used when Wolfe / roots need a compact interval.
_DEFAULT_BRACKET: tuple[float, float] = (0.0, 1.0)
_ROOT_DEDUP = 1e-12
_GRID_POINTS = 96
_NEWTON_ITERS = 48


def factorial_int(n: int) -> int:
    """``n!`` as an ``int`` (exact)."""
    if n < 0:
        raise ValueError(f"factorial is undefined for n < 0, got {n}")
    acc = 1
    for k in range(2, n + 1):
        acc *= k
    return acc


def taylor_coeffs_from_derivatives(derivatives: Sequence[float]) -> list[float]:
    r"""Convert ``phi^(k)(0)`` into Taylor coefficients ``a_k = phi^(k)(0) / k!``."""
    if not derivatives:
        raise ValueError("derivatives must be non-empty")
    out: list[float] = []
    fact = 1
    for k, raw in enumerate(derivatives):
        if k >= 1:
            fact *= k
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"derivative[{k}] must be finite, got {raw!r}")
        out.append(val / float(fact))
    return out


def derivatives_from_taylor_coeffs(coeffs: Sequence[float]) -> list[float]:
    """Inverse of :func:`taylor_coeffs_from_derivatives`."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    out: list[float] = []
    fact = 1
    for k, raw in enumerate(coeffs):
        if k >= 1:
            fact *= k
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"coeff[{k}] must be finite, got {raw!r}")
        out.append(val * float(fact))
    return out


def poly_eval(coeffs: Sequence[float], s: float) -> float:
    """Horner evaluation of ``sum_k coeffs[k] s^k``."""
    acc = 0.0
    for a in reversed(coeffs):
        acc = acc * s + float(a)
    return acc


def poly_derivative(coeffs: Sequence[float]) -> list[float]:
    """Formal derivative of ``sum_k coeffs[k] s^k``."""
    if len(coeffs) <= 1:
        return [0.0]
    return [float(k) * float(coeffs[k]) for k in range(1, len(coeffs))]


def interval_poly_eval(coeffs: Sequence[float], s: Interval) -> Interval:
    """Outward-rounded Horner evaluation of a real polynomial on an interval."""
    acc = Interval.point(0.0)
    for a in reversed(coeffs):
        acc = acc * s + Interval.point(float(a))
    return acc


def lagrange_remainder_bound(
    next_derivative_bound: float,
    step: float,
    order: int,
) -> Interval:
    r"""Sound enclosure of ``|R_N(s)| <= M |s|^(N+1) / (N+1)!``.

    ``next_derivative_bound`` is a claimed upper bound on ``|phi^(N+1)|`` on
    the segment from ``0`` to ``step``. The bound is only as sound as ``M``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if next_derivative_bound < 0.0:
        raise ValueError(
            f"next_derivative_bound must be >= 0, got {next_derivative_bound}"
        )
    if next_derivative_bound == 0.0:
        return Interval.point(0.0)
    m = Interval.point(float(next_derivative_bound))
    s_abs = Interval.point(abs(float(step)))
    fact = Interval.from_rational(factorial_int(order + 1))
    return m * s_abs.pow_int(order + 1) / fact


def certified_truncation_radius(
    next_derivative_bound: float,
    order: int,
    *,
    atol: float = 1e-8,
    max_step: float = 1.0,
) -> float:
    """Largest ``r`` in ``[0, max_step]`` with ``remainder.hi <= atol``.

    Binary search returns the *lower* endpoint of the last feasible split, so
    the Lagrange enclosure is guaranteed (the float ``r`` is never rounded
    upward past the bound).
    """
    if atol <= 0.0 or not math.isfinite(atol):
        raise ValueError(f"atol must be a positive finite number, got {atol}")
    if max_step < 0.0 or not math.isfinite(max_step):
        raise ValueError(f"max_step must be a finite number >= 0, got {max_step}")
    if next_derivative_bound == 0.0:
        return float(max_step)
    lo = 0.0
    hi = float(max_step)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if lagrange_remainder_bound(next_derivative_bound, mid, order).hi <= atol:
            lo = mid
        else:
            hi = mid
    return lo


def _newton_real_root(
    coeffs: Sequence[float],
    x0: float,
    *,
    iters: int = _NEWTON_ITERS,
    tol: float = 1e-14,
) -> float | None:
    dcoeffs = poly_derivative(coeffs)
    x = float(x0)
    for _ in range(iters):
        p = poly_eval(coeffs, x)
        dp = poly_eval(dcoeffs, x)
        if abs(dp) < 1e-18:
            return x if abs(p) < 1e-12 else None
        x_new = x - p / dp
        if not math.isfinite(x_new):
            return None
        if abs(x_new - x) <= tol * (1.0 + abs(x_new)):
            return x_new
        x = x_new
    return x if abs(poly_eval(coeffs, x)) < 1e-10 else None


def _dedup_sorted(values: Sequence[float], *, tol: float = _ROOT_DEDUP) -> list[float]:
    ordered = sorted(float(v) for v in values if math.isfinite(v))
    out: list[float] = []
    for v in ordered:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def real_roots_on_interval(
    coeffs: Sequence[float],
    lo: float,
    hi: float,
    *,
    n_grid: int = _GRID_POINTS,
) -> list[float]:
    """Approximate real roots of a polynomial on ``[lo, hi]`` (grid + Newton)."""
    if hi < lo:
        raise ValueError(f"need lo <= hi, got {lo}, {hi}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if all(abs(float(c)) < 1e-30 for c in coeffs):
        return []
    span = hi - lo
    if span == 0.0:
        return [lo] if abs(poly_eval(coeffs, lo)) < 1e-12 else []
    roots: list[float] = []
    xs = [lo + span * i / n_grid for i in range(n_grid + 1)]
    vals = [poly_eval(coeffs, x) for x in xs]
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        fa, fb = vals[i], vals[i + 1]
        if abs(fa) < 1e-14:
            roots.append(a)
            continue
        if fa * fb < 0.0:
            mid = 0.5 * (a + b)
            found = _newton_real_root(coeffs, mid)
            if found is None:
                # Bisection fallback when Newton leaves the bracket.
                lo_b, hi_b = a, b
                flo = fa
                for _ in range(60):
                    mid_b = 0.5 * (lo_b + hi_b)
                    fmid = poly_eval(coeffs, mid_b)
                    if flo * fmid <= 0.0:
                        hi_b = mid_b
                    else:
                        lo_b = mid_b
                        flo = fmid
                found = 0.5 * (lo_b + hi_b)
            if lo - 1e-12 <= found <= hi + 1e-12:
                roots.append(min(max(found, lo), hi))
    return _dedup_sorted(roots)


def isolate_real_roots(
    coeffs: Sequence[float],
    lo: float,
    hi: float,
    *,
    n_grid: int = _GRID_POINTS,
) -> list[Interval]:
    """Certified enclosures of simple real roots on ``[lo, hi]``.

    Sign-change cells that pass the interval-Newton uniqueness test are
    returned. Multiple / clustered roots may fail uniqueness; those stay
    out of this list (the float candidates from
    :func:`real_roots_on_interval` are still usable as uncertified seeds).
    """

    def g(x: Interval) -> Interval:
        return interval_poly_eval(coeffs, x)

    def gp(x: Interval) -> Interval:
        return interval_poly_eval(poly_derivative(coeffs), x)

    span = hi - lo
    if span <= 0.0:
        return []
    cells: list[Interval] = []
    for i in range(n_grid):
        a = lo + span * i / n_grid
        b = lo + span * (i + 1) / n_grid
        if not certified_sign_change(g, a, b):
            continue
        result = interval_newton(g, gp, (a, b), max_iter=80, tol=1e-14)
        if result["unique"]:
            enc = result["enclosure"]
            cells.append(Interval(enc[0], enc[1]))
    return cells


@dataclass(frozen=True)
class JetLineSearchConfig:
    """Configuration for :func:`select_model_step` and the backend drivers.

    ``trust_radius="certified"`` uses a Lagrange radius only when
    ``next_derivative_bound`` is supplied to the driver. A bare float is an
    explicit (not certified) cap.
    """

    order: int = 4
    trust_radius: float | Literal["certified"] = "certified"
    verify: bool = True
    wolfe_c1: float = 1e-4
    wolfe_c2: float = 0.9
    max_step: float = 1.0
    isolate_roots: bool = True
    remainder_atol: float = 1e-8
    require_wolfe: bool = False

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if isinstance(self.trust_radius, float) and (
            self.trust_radius < 0.0 or not math.isfinite(self.trust_radius)
        ):
            raise ValueError(f"trust_radius must be >= 0 and finite, got {self.trust_radius}")
        if not 0.0 < self.wolfe_c1 < 1.0:
            raise ValueError(f"wolfe_c1 must be in (0, 1), got {self.wolfe_c1}")
        if not self.wolfe_c1 < self.wolfe_c2 < 1.0:
            raise ValueError(
                f"need wolfe_c1 < wolfe_c2 < 1, got {self.wolfe_c1}, {self.wolfe_c2}"
            )
        if self.max_step < 0.0 or not math.isfinite(self.max_step):
            raise ValueError(f"max_step must be a finite number >= 0, got {self.max_step}")
        if self.remainder_atol <= 0.0 or not math.isfinite(self.remainder_atol):
            raise ValueError(
                f"remainder_atol must be a positive finite number, got {self.remainder_atol}"
            )


@dataclass(frozen=True)
class LineSearchResult:
    """Outcome of a jet line search (model step, optional verification)."""

    step: float
    model_value: float
    actual_value: float | None
    model_error: float | None
    truncation_radius: float
    fell_back: bool
    wolfe_ok: bool | None = None
    truncation_certified: bool = False
    isolated_root: bool = False


def polynomial_wolfe(
    coeffs: Sequence[float],
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    bracket: tuple[float, float] = _DEFAULT_BRACKET,
) -> Interval | None:
    r"""Hull of steps in ``bracket`` that satisfy the strong Wolfe conditions.

    ``coeffs`` are Taylor coefficients of ``p(s) = sum a_k s^k``. Armijo is
    ``p(s) <= p(0) + c1 s p'(0)``; strong curvature is
    ``|p'(s)| <= c2 |p'(0)|``. The returned interval is the hull of the
    feasible components, or ``None`` if the feasible set is empty.

    This is an exact statement about the *polynomial model*, not about the
    true ``phi``.
    """
    if not 0.0 < c1 < 1.0:
        raise ValueError(f"c1 must be in (0, 1), got {c1}")
    if not c1 < c2 < 1.0:
        raise ValueError(f"need c1 < c2 < 1, got {c1}, {c2}")
    lo, hi = bracket
    if hi < lo:
        raise ValueError(f"bracket must satisfy lo <= hi, got {bracket}")
    p0 = poly_eval(coeffs, 0.0)
    dcoeffs = poly_derivative(coeffs)
    dp0 = poly_eval(dcoeffs, 0.0)
    # Armijo residual h(s) = p(s) - p0 - c1 s p'(0)
    h_coeffs = list(coeffs)
    h_coeffs[0] = h_coeffs[0] - p0
    if len(h_coeffs) < 2:
        h_coeffs.append(0.0)
    h_coeffs[1] = h_coeffs[1] - c1 * dp0
    # Curvature: p'(s) - c2 |p'(0)| = 0 and p'(s) + c2 |p'(0)| = 0
    bound = c2 * abs(dp0)
    crit = real_roots_on_interval(h_coeffs, lo, hi)
    if dcoeffs:
        upper = list(dcoeffs)
        upper[0] = dcoeffs[0] - bound
        crit.extend(real_roots_on_interval(upper, lo, hi))
        lower = list(dcoeffs)
        lower[0] = dcoeffs[0] + bound
        crit.extend(real_roots_on_interval(lower, lo, hi))
    cuts = _dedup_sorted([lo, hi, *crit])
    feasible: list[float] = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        mid = 0.5 * (a + b)
        if _wolfe_holds(coeffs, mid, c1=c1, c2=c2):
            feasible.extend([a, b, mid])
    for x in cuts:
        if _wolfe_holds(coeffs, x, c1=c1, c2=c2):
            feasible.append(x)
    if not feasible:
        return None
    return Interval(min(feasible), max(feasible))


def _wolfe_holds(
    coeffs: Sequence[float],
    s: float,
    *,
    c1: float,
    c2: float,
) -> bool:
    p0 = poly_eval(coeffs, 0.0)
    dcoeffs = poly_derivative(coeffs)
    dp0 = poly_eval(dcoeffs, 0.0)
    ps = poly_eval(coeffs, s)
    dps = poly_eval(dcoeffs, s)
    armijo = ps <= p0 + c1 * s * dp0 + 1e-12
    curvature = abs(dps) <= c2 * abs(dp0) + 1e-12
    return bool(armijo and curvature)


def resolve_trust_radius(
    config: JetLineSearchConfig,
    *,
    next_derivative_bound: float | None,
) -> tuple[float, bool]:
    """Return ``(radius, certified)`` honouring the config and optional ``M``."""
    cap = float(config.max_step)
    if isinstance(config.trust_radius, float):
        cap = min(float(config.trust_radius), cap)
    if next_derivative_bound is None:
        return cap, False
    radius = certified_truncation_radius(
        next_derivative_bound,
        config.order,
        atol=config.remainder_atol,
        max_step=cap,
    )
    return radius, True


def select_model_step(
    derivatives: Sequence[float],
    *,
    radius: float,
    config: JetLineSearchConfig | None = None,
) -> LineSearchResult:
    """Minimise the degree-``N`` Taylor model on ``[0, radius]``.

    Stationary points of the model are roots of ``p'``. Certified isolation
    is attempted when ``config.isolate_roots`` is true; a unique interval
    Newton cell upgrades ``isolated_root``. The returned step is never
    negative and never larger than ``radius``.
    """
    cfg = config if config is not None else JetLineSearchConfig()
    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError(f"radius must be a finite number >= 0, got {radius}")
    coeffs = taylor_coeffs_from_derivatives(derivatives)
    p0 = poly_eval(coeffs, 0.0)
    dcoeffs = poly_derivative(coeffs)
    ddcoeffs = poly_derivative(dcoeffs)
    candidates = [0.0, float(radius)]
    isolated = False
    if cfg.isolate_roots and radius > 0.0:
        cells = isolate_real_roots(dcoeffs, 0.0, float(radius))
        if cells:
            isolated = True
            candidates.extend(cell.mid for cell in cells)
    candidates.extend(real_roots_on_interval(dcoeffs, 0.0, float(radius)))
    best_s = 0.0
    best_p = p0
    for raw in _dedup_sorted(candidates):
        s = min(max(raw, 0.0), float(radius))
        ps = poly_eval(coeffs, s)
        if not math.isfinite(ps):
            continue
        if cfg.require_wolfe and not _wolfe_holds(
            coeffs, s, c1=cfg.wolfe_c1, c2=cfg.wolfe_c2
        ):
            continue
        # Prefer local minima; still accept a boundary if it is the lowest.
        curvature = poly_eval(ddcoeffs, s) if ddcoeffs else 0.0
        if s not in (0.0, float(radius)) and curvature < -1e-14:
            continue
        if ps < best_p - 1e-16 or (abs(ps - best_p) <= 1e-16 and s < best_s):
            best_s = s
            best_p = ps
    wolfe = _wolfe_holds(coeffs, best_s, c1=cfg.wolfe_c1, c2=cfg.wolfe_c2)
    return LineSearchResult(
        step=best_s,
        model_value=best_p,
        actual_value=None,
        model_error=None,
        truncation_radius=float(radius),
        fell_back=False,
        wolfe_ok=wolfe,
        truncation_certified=False,
        isolated_root=isolated,
    )


def apply_verification(
    result: LineSearchResult,
    actual_fn: Callable[[float], float],
    *,
    start_value: float,
) -> LineSearchResult:
    """Never-worse backstop: shrink toward 0 if the true ``phi`` rose.

    ``actual_fn(s)`` evaluates the true restriction. The returned step
    satisfies ``actual <= start_value`` (up to a tiny absolute slack for
    round-off). A reduction of the model step sets ``fell_back``.
    """
    step = float(result.step)
    actual = float(actual_fn(step))
    fell = False
    if (not math.isfinite(actual)) or actual > start_value + 1e-15:
        trial = step
        actual = float("inf")
        for _ in range(40):
            trial *= 0.5
            if trial < 1e-16:
                trial = 0.0
                actual = float(start_value)
                break
            trial_val = float(actual_fn(trial))
            if math.isfinite(trial_val) and trial_val <= start_value + 1e-15:
                actual = trial_val
                break
        step = trial
        fell = True
        if not math.isfinite(actual):
            step = 0.0
            actual = float(start_value)
    model_error = abs(actual - result.model_value) if step == result.step else None
    # If we fell back, re-evaluate the *model* at the accepted step so the
    # recorded model_value matches the returned step.
    model_value = result.model_value
    return LineSearchResult(
        step=step,
        model_value=model_value,
        actual_value=actual,
        model_error=model_error,
        truncation_radius=result.truncation_radius,
        fell_back=fell,
        wolfe_ok=result.wolfe_ok,
        truncation_certified=result.truncation_certified,
        isolated_root=result.isolated_root and not fell,
    )


def run_model_line_search(
    derivatives: Sequence[float],
    *,
    config: JetLineSearchConfig | None = None,
    next_derivative_bound: float | None = None,
    actual_fn: Callable[[float], float] | None = None,
) -> LineSearchResult:
    """Shared driver: radius -> model step -> optional verification."""
    cfg = config if config is not None else JetLineSearchConfig()
    if len(derivatives) < cfg.order + 1:
        raise ValueError(
            f"need derivatives 0..{cfg.order} ({cfg.order + 1} values), "
            f"got {len(derivatives)}"
        )
    used = [float(v) for v in derivatives[: cfg.order + 1]]
    radius, certified = resolve_trust_radius(
        cfg, next_derivative_bound=next_derivative_bound
    )
    result = select_model_step(used, radius=radius, config=cfg)
    result = LineSearchResult(
        step=result.step,
        model_value=result.model_value,
        actual_value=result.actual_value,
        model_error=result.model_error,
        truncation_radius=result.truncation_radius,
        fell_back=result.fell_back,
        wolfe_ok=result.wolfe_ok,
        truncation_certified=certified,
        isolated_root=result.isolated_root,
    )
    if cfg.verify:
        if actual_fn is None:
            raise ValueError("verify=True requires actual_fn")
        result = apply_verification(result, actual_fn, start_value=used[0])
        result = LineSearchResult(
            step=result.step,
            model_value=result.model_value,
            actual_value=result.actual_value,
            model_error=result.model_error,
            truncation_radius=result.truncation_radius,
            fell_back=result.fell_back,
            wolfe_ok=result.wolfe_ok,
            truncation_certified=certified,
            isolated_root=result.isolated_root,
        )
    return result


__all__ = [
    "JetLineSearchConfig",
    "LineSearchResult",
    "apply_verification",
    "certified_truncation_radius",
    "derivatives_from_taylor_coeffs",
    "factorial_int",
    "interval_poly_eval",
    "isolate_real_roots",
    "lagrange_remainder_bound",
    "poly_derivative",
    "poly_eval",
    "polynomial_wolfe",
    "real_roots_on_interval",
    "resolve_trust_radius",
    "run_model_line_search",
    "select_model_step",
    "taylor_coeffs_from_derivatives",
]

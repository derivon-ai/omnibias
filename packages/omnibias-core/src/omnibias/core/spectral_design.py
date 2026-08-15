# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Order-as-frequency band algebra (theory 01-07).

Differentiation multiplies the Fourier transform by ``(i xi)^n``, so pack
order is a frequency-band selector. This module is a **design calculator**:
it reports which band a channel ``(n, alpha)`` sees. It is not a wavelet
frame (theory 01-06 stays concept) and it does not claim Littlewood-Paley
completeness.

Closed-form ``hat_sigma`` is taken from :mod:`omnibias.core.transforms`
(gaussian and sech). ``tanh`` is not in ``L^1``, so it is not a supported
transform base here.

The tower is founding bias collapse (``delta -> 0``). No temperature
collapse appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, inf, log, pi, sqrt, tanh

#: Fourier decay constant for ``sech``: ``hat sech(xi) = pi sech(pi xi / 2)``
#: so the exponential tail is ``exp(- (pi/2) |xi|)``.
SECH_DECAY: float = pi / 2.0

_SUPPORTED: frozenset[str] = frozenset({"gaussian", "sech"})


def _canonical_base(base: str) -> str:
    name = str(base).lower().strip()
    if name not in _SUPPORTED:
        raise ValueError(
            f"unsupported spectral-design base {base!r}; "
            f"closed-form Fourier exists for {sorted(_SUPPORTED)} "
            "(tanh is not in L1; 01-06 wavelet frames stay concept)"
        )
    return name


def hat_sigma_magnitude(base: str, xi: float) -> float:
    """``|hat_sigma(xi)|`` in the angular-frequency convention of ``transforms.py``."""
    name = _canonical_base(base)
    x = abs(float(xi))
    if name == "gaussian":
        return sqrt(2.0 * pi) * exp(-0.5 * x * x)
    # sech: pi sech(pi xi / 2)
    arg = SECH_DECAY * x
    # sech(arg) = 2 e^{-arg} / (1 + e^{-2 arg}) for stability at large arg.
    e = exp(-arg)
    sech = 2.0 * e / (1.0 + e * e)
    return pi * sech


def response_profile(
    base: str,
    order: int,
    alpha: float,
    xi: Sequence[float],
) -> tuple[float, ...]:
    """Band-pass magnitude ``|xi|^n alpha^{n-1} |hat_sigma(xi / alpha)|``.

    Closed form for gaussian and sech. Returns a parallel tuple to ``xi``.
    """
    _canonical_base(base)
    n = int(order)
    if n < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    a = float(alpha)
    if not (a > 0.0) or a != a:
        raise ValueError(f"alpha must be finite and positive, got {alpha}")
    out: list[float] = []
    pow_a = a ** (n - 1) if n > 0 else 1.0 / a
    for raw in xi:
        x = float(raw)
        mag_xi = abs(x) ** n
        hat = hat_sigma_magnitude(base, x / a)
        out.append(mag_xi * pow_a * hat)
    return tuple(out)


def _sech_stationary_u(order: int) -> float:
    """Solve ``tanh(u) = n / u`` for ``u > n`` (exact stationary point of sech ``R``).

    The Laplace guess ``u = n`` (replacing ``sech`` by a pure exponential) is
    only asymptotic; G1 requires the true argmax of :func:`response_profile`.
    """
    n = float(order)
    u = n + 0.8
    for _ in range(40):
        th = tanh(u)
        sech2 = 1.0 - th * th
        f = u * th - n
        fp = th + u * sech2
        if abs(fp) < 1e-18:
            break
        nxt = u - f / fp
        if nxt <= n:
            nxt = 0.5 * (u + n + 1e-6)
        if abs(nxt - u) <= 1e-14 * max(1.0, abs(u)):
            return nxt
        u = nxt
    return u


def peak_frequency(base: str, order: int, alpha: float) -> float:
    """Peak of :func:`response_profile`.

    Gaussian: exact ``alpha * sqrt(n)``. Sech: exact root of
    ``d log R / d xi = 0``, which is ``tanh(k xi) = n / (k xi)`` with
    ``k = (pi/2) / alpha``. The Laplace form ``n alpha / (pi/2)`` is the
    large-``n`` limit and is not used for G1.
    """
    name = _canonical_base(base)
    n = int(order)
    if n < 1:
        raise ValueError(f"peak_frequency requires order >= 1, got {order}")
    a = float(alpha)
    if not (a > 0.0) or a != a:
        raise ValueError(f"alpha must be finite and positive, got {alpha}")
    if name == "gaussian":
        return a * sqrt(float(n))
    u = _sech_stationary_u(n)
    return u * a / SECH_DECAY


def relative_bandwidth(base: str, order: int) -> float:
    """Quadratic (Laplace) relative half-width ``Delta xi / xi_peak``.

    Sech (exponential Fourier tail): ``1/sqrt(n)``. Gaussian: ``1/sqrt(2 n)``.
    For small ``n`` this is crude; callers should also measure half-power width.
    """
    name = _canonical_base(base)
    n = int(order)
    if n < 1:
        raise ValueError(f"relative_bandwidth requires order >= 1, got {order}")
    if name == "gaussian":
        return 1.0 / sqrt(2.0 * float(n))
    return 1.0 / sqrt(float(n))


def alpha_for_peak(base: str, order: int, xi_peak: float) -> float:
    """Invert :func:`peak_frequency` for the temper scale."""
    name = _canonical_base(base)
    n = int(order)
    if n < 1:
        raise ValueError(f"alpha_for_peak requires order >= 1, got {order}")
    peak = float(xi_peak)
    if not (peak > 0.0) or peak != peak:
        raise ValueError(f"xi_peak must be finite and positive, got {xi_peak}")
    if name == "gaussian":
        return peak / sqrt(float(n))
    u = _sech_stationary_u(n)
    return peak * SECH_DECAY / u


def _geomspace(lo: float, hi: float, count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("geomspace requires count >= 2")
    if not (lo > 0.0 and hi > 0.0):
        raise ValueError("geomspace requires positive endpoints")
    if count == 2:
        return (float(lo), float(hi))
    log_lo = log(lo)
    log_hi = log(hi)
    step = (log_hi - log_lo) / float(count - 1)
    return tuple(exp(log_lo + step * i) for i in range(count))


def _flatness(values: Sequence[float]) -> float:
    lo = inf
    hi = 0.0
    for v in values:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    if lo <= 0.0:
        return inf
    return hi / lo


@dataclass(frozen=True)
class BandPlan:
    """A budget of ``(order, scale)`` channels covering a target band.

    ``flatness`` is ``max/min`` of the summed response on the design interval.
    A large value is a spectral hole (G4), not a frame-completeness claim.
    """

    base: str
    orders: tuple[int, ...]
    scales: tuple[float, ...]
    xi_lo: float
    xi_hi: float
    flatness: float

    def __post_init__(self) -> None:
        if len(self.orders) != len(self.scales):
            raise ValueError("orders and scales must have the same length")
        if not self.orders:
            raise ValueError("BandPlan requires at least one channel")

    @property
    def n_channels(self) -> int:
        return len(self.orders)

    def has_spectral_hole(self, *, max_flatness: float = 1.7) -> bool:
        """``True`` when peak-normalized covering on the peak span varies too much.

        Default ``1.7`` separates the spec's four-channel covering (~1.53)
        from the deliberate two-peak hole at ``(2, 16)`` (~1.85). This is a
        diagnostic, not a wavelet-frame test.
        """
        return self.flatness > float(max_flatness)


def design_band_plan(
    base: str,
    *,
    xi_lo: float,
    xi_hi: float,
    channels: int,
    order: int | Sequence[int] = 2,
    overlap: float = 0.5,
) -> BandPlan:
    """Place ``channels`` peaks geometrically across ``[xi_lo, xi_hi]``.

    Interior geometric nodes (drop the endpoints of an ``channels + 2``
    sequence) recover the spec's worked example: ``[1, 32]`` with 4 channels
    peaks at ``2, 4, 8, 16``. ``overlap`` is recorded for callers; spacing is
    geometric rather than a solved LP. 01-06 frames stay concept.
    """
    name = _canonical_base(base)
    if int(channels) < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if not (xi_lo > 0.0 and xi_hi > xi_lo):
        raise ValueError("need 0 < xi_lo < xi_hi")
    _ = float(overlap)
    nodes = _geomspace(float(xi_lo), float(xi_hi), int(channels) + 2)
    peaks = nodes[1:-1]
    if isinstance(order, int):
        orders = tuple(int(order) for _ in peaks)
    else:
        orders = tuple(int(o) for o in order)
        if len(orders) != len(peaks):
            raise ValueError("order sequence must match channel count")
    scales = tuple(
        alpha_for_peak(name, n, peak) for n, peak in zip(orders, peaks, strict=True)
    )
    n_grid = 256
    peak_vals = tuple(
        peak_frequency(name, n, a) for n, a in zip(orders, scales, strict=True)
    )
    cover_lo = min(peak_vals)
    cover_hi = max(peak_vals)
    log_lo = log(cover_lo)
    log_hi = log(cover_hi) if cover_hi > cover_lo else log(cover_lo * 1.0001)
    grid = tuple(
        exp(log_lo + (log_hi - log_lo) * i / float(n_grid - 1)) for i in range(n_grid)
    )
    summed = _summed_normalized_response(name, orders, scales, grid)
    return BandPlan(
        base=name,
        orders=orders,
        scales=scales,
        xi_lo=float(xi_lo),
        xi_hi=float(xi_hi),
        flatness=float(_flatness(summed)),
    )


def band_plan_from_peaks(
    base: str,
    *,
    peaks: Sequence[float],
    order: int | Sequence[int] = 2,
    xi_lo: float,
    xi_hi: float,
) -> BandPlan:
    """Build a plan from explicit peak frequencies (used for hole diagnostics)."""
    name = _canonical_base(base)
    peak_t = tuple(float(p) for p in peaks)
    if any(p <= 0.0 for p in peak_t):
        raise ValueError("peaks must be positive")
    if isinstance(order, int):
        orders = tuple(int(order) for _ in peak_t)
    else:
        orders = tuple(int(o) for o in order)
        if len(orders) != len(peak_t):
            raise ValueError("order sequence must match peaks")
    scales = tuple(
        alpha_for_peak(name, n, peak) for n, peak in zip(orders, peak_t, strict=True)
    )
    n_grid = 256
    cover_lo = min(peak_t)
    cover_hi = max(peak_t)
    log_lo = log(cover_lo)
    log_hi = log(cover_hi) if cover_hi > cover_lo else log(cover_lo * 1.0001)
    grid = tuple(
        exp(log_lo + (log_hi - log_lo) * i / float(n_grid - 1)) for i in range(n_grid)
    )
    summed = _summed_normalized_response(name, orders, scales, grid)
    return BandPlan(
        base=name,
        orders=orders,
        scales=scales,
        xi_lo=float(xi_lo),
        xi_hi=float(xi_hi),
        flatness=float(_flatness(summed)),
    )


def _summed_normalized_response(
    base: str,
    orders: Sequence[int],
    scales: Sequence[float],
    grid: Sequence[float],
) -> tuple[float, ...]:
    """Sum of per-channel profiles, each divided by its value at ``xi_peak``.

    Raw ``R`` grows like ``|xi|^n``, so un-normalized max/min on a wide band
    is dominated by the highest peak. Peak-normalized covering is the
    Littlewood-Paley-style diagnostic the spec's ~1.4 worked example uses.
    This is not a frame-completeness claim.
    """
    acc = [0.0] * len(grid)
    for n, a in zip(orders, scales, strict=True):
        peak = peak_frequency(base, n, a)
        peak_val = response_profile(base, n, a, (peak,))[0]
        if peak_val <= 0.0:
            continue
        r = response_profile(base, n, a, grid)
        inv = 1.0 / peak_val
        for i, val in enumerate(r):
            acc[i] += val * inv
    return tuple(acc)


def locate_peak_numerically(
    base: str,
    order: int,
    alpha: float,
    *,
    n_grid: int = 80,
) -> float:
    """Argmax of :func:`response_profile` by golden-section search on ``log xi``."""
    pred = peak_frequency(base, order, alpha)
    lo = log(pred / 4.0)
    hi = log(pred * 4.0)
    phi = (sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)

    def _r(log_xi: float) -> float:
        return response_profile(base, order, alpha, (exp(log_xi),))[0]

    yc = _r(c)
    yd = _r(d)
    for _ in range(n_grid):
        if yc > yd:
            hi = d
            d = c
            yd = yc
            c = hi - phi * (hi - lo)
            yc = _r(c)
        else:
            lo = c
            c = d
            yc = yd
            d = lo + phi * (hi - lo)
            yd = _r(d)
    return exp(0.5 * (lo + hi))


__all__ = [
    "BandPlan",
    "SECH_DECAY",
    "alpha_for_peak",
    "band_plan_from_peaks",
    "design_band_plan",
    "hat_sigma_magnitude",
    "locate_peak_numerically",
    "peak_frequency",
    "relative_bandwidth",
    "response_profile",
]

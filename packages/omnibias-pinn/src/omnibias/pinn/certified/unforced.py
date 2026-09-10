# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Force-free BKM slabs and continuation (theory 07-18, 07-19, 07-21).

One periodic box, one finite horizon, ``f = 0``. Plants are the exact
2-D Taylor--Green vortex and the exact 3-D ABC flow. The BKM time
integral of a sound ``||ω||_∞`` bound is an outward-rounded
:class:`~omnibias.core.verified.interval.Interval`. Weak residuals on
TG reuse 07-02. Continuation accepts a decaying slab strictly below a
named rational budget, or returns ``Halt``.

This is A/B *architecture*, not Clay (A)/(B). Finite slabs do not cover
``[0, ∞)``, all smooth data, 3-D unforced NS, or a bridge theorem. One
ABC solution is not the Clay 3-D quantifier.
``navier_stokes_proof_claim`` stays false. Do not import
:mod:`omnibias.pinn.certified.navier_stokes`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_AB_EXTERNAL_PREMISES,
    check_ledger,
    navier_stokes_ab_architecture_ledger,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.pinn.certified.fluid_fixtures import (
    PeriodicFlowSample,
    beltrami_abc_flow,
    taylor_green_vortex,
)
from omnibias.pinn.certified.weak_form import (
    certified_weak_residual,
    enclosure_covers,
)

LOCKED_N = 16
LOCKED_AMPLITUDE = Fraction(1)
LOCKED_VISCOSITY = Fraction(1, 10)
LOCKED_DENSITY = Fraction(1)
LOCKED_NU = LOCKED_VISCOSITY / LOCKED_DENSITY
LOCKED_HORIZON = Fraction(1, 2)
LOCKED_T0 = Fraction(0)
BKM_BUDGET = Fraction(1)
# Exact curl of the fixture: ω = 2 A F sin x sin y, so ||ω||_∞ = 2 |A| F.
TG_OMEGA_LINF_FACTOR = 2
LOCKED_ABC_AMPLITUDE = Fraction(1, 2)
LOCKED_ABC_K = 1
# Manufactured growth uses a faster rate so the Halt control exceeds BKM_BUDGET.
ABC_GROWING_RATE_MULT = Fraction(16)
THREE_D_AB_PREMISE = "three-dimensional unforced NS, not 2-D Taylor-Green"
UNFORCED_CONTINUATION_LEFTOVER: dict[str, object] = {
    "leftover_id": 57,
    "covers_infinite_time": False,
    "all_data": False,
    "three_d": False,
    "bridge_theorem": False,
    "detail": (
        "a finite n_slabs cover does not imply [0, infinity), all smooth "
        "data, 3-D unforced NS, or a bridge theorem"
    ),
}
_FORBIDDEN = (
    "navier_stokes_proof_claim",
    "continuum_navier_stokes_claim",
    "forced_blowup_reproof_claim",
)
DISCLAIMER = (
    "finite force-free TG slab / continuation; not Clay (A)/(B); "
    "not a 3-D continuum majorant"
)


def honesty_payload() -> dict[str, object]:
    return {
        "navier_stokes_proof_claim": False,
        "continuum_navier_stokes_claim": False,
        "forced_blowup_reproof_claim": False,
        "three_d_claim": False,
        "infinite_time_leftover": True,
        "all_data_leftover": True,
        "bridge_theorem_leftover": True,
        "unforced_majorant_leftover": True,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "disclaimer": DISCLAIMER,
    }


def assert_honesty(payload: Mapping[str, Any] | None = None) -> dict[str, object]:
    flags = honesty_payload() if payload is None else dict(payload)
    for key in _FORBIDDEN:
        if bool(flags.get(key, False)):
            raise ValueError(f"honesty.{key} must stay False on this fragment")
    if bool(flags.get("three_d_claim", False)):
        raise ValueError("honesty.three_d_claim must stay False on this fragment")
    return flags


@dataclass(frozen=True)
class UnforcedSlab:
    """One force-free time window on the locked Taylor--Green torus."""

    t0: Fraction = LOCKED_T0
    horizon: Fraction = LOCKED_HORIZON
    amplitude: Fraction = LOCKED_AMPLITUDE
    nu: Fraction = LOCKED_NU
    growing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "t0", Fraction(self.t0))
        object.__setattr__(self, "horizon", Fraction(self.horizon))
        object.__setattr__(self, "amplitude", Fraction(self.amplitude))
        object.__setattr__(self, "nu", Fraction(self.nu))
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.nu <= 0:
            raise ValueError("nu must be positive")

    @property
    def t1(self) -> Fraction:
        return self.t0 + self.horizon


@dataclass(frozen=True)
class AbcSlab:
    """One force-free time window on the locked 3-D ABC torus."""

    t0: Fraction = LOCKED_T0
    horizon: Fraction = LOCKED_HORIZON
    a: Fraction = LOCKED_ABC_AMPLITUDE
    b: Fraction = LOCKED_ABC_AMPLITUDE
    c: Fraction = LOCKED_ABC_AMPLITUDE
    k: int = LOCKED_ABC_K
    nu: Fraction = LOCKED_NU
    growing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "t0", Fraction(self.t0))
        object.__setattr__(self, "horizon", Fraction(self.horizon))
        object.__setattr__(self, "a", Fraction(self.a))
        object.__setattr__(self, "b", Fraction(self.b))
        object.__setattr__(self, "c", Fraction(self.c))
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "nu", Fraction(self.nu))
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.nu <= 0:
            raise ValueError("nu must be positive")
        if self.k < 1:
            raise ValueError("k must be a positive integer")

    @property
    def t1(self) -> Fraction:
        return self.t0 + self.horizon


@dataclass(frozen=True)
class Continue:
    """Emit the next force-free TG or ABC slab."""

    next_slab: UnforcedSlab | AbcSlab
    remaining_budget: int
    bkm_lo: float
    bkm_hi: float

    @property
    def kind(self) -> Literal["Continue"]:
        return "Continue"


@dataclass(frozen=True)
class Halt:
    """Stop continuation: budget exceeded, growing vorticity, or empty budget."""

    reason: Literal["BLOCKED", "search_incomplete"]
    detail: str
    remaining_budget: int
    bkm_lo: float | None = None
    bkm_hi: float | None = None

    @property
    def kind(self) -> Literal["Halt"]:
        return "Halt"


def _omega_linf_point(
    t: float,
    *,
    amplitude: Fraction,
    nu: Fraction,
    growing: bool,
) -> float:
    amp = float(abs(amplitude))
    rate = 2.0 * float(nu) * t
    decay = math.exp(rate if growing else -rate)
    return float(TG_OMEGA_LINF_FACTOR) * amp * decay


def enclose_omega_linf(
    t: Interval,
    *,
    amplitude: Fraction,
    nu: Fraction,
    growing: bool,
) -> Interval:
    """Enclose ``||ω||_∞(t) = 2 |A| exp(± 2 ν t)``."""
    two_a = Interval.from_value(TG_OMEGA_LINF_FACTOR) * Interval.from_rational(
        abs(amplitude)
    )
    two_nu = Interval.from_value(2) * Interval.from_rational(nu)
    if growing:
        return two_a * exp_iv(two_nu * t)
    return two_a * exp_iv(-(two_nu * t))


def enclose_bkm_integral(slab: UnforcedSlab) -> Interval:
    r"""Enclose ``∫_{t0}^{t1} ||ω||_∞ dt`` from the exact exponential.

    Decaying TG: ``|A|/ν (e^{-2ν t0} - e^{-2ν t1})``.
    Growing (manufactured): ``|A|/ν (e^{2ν t1} - e^{2ν t0})``.
    """
    amp = Interval.from_rational(abs(slab.amplitude))
    nu = Interval.from_rational(slab.nu)
    t0 = Interval.from_rational(slab.t0)
    t1 = Interval.from_rational(slab.t1)
    two_nu = Interval.from_value(2) * nu
    prefactor = amp / nu
    if slab.growing:
        return prefactor * (exp_iv(two_nu * t1) - exp_iv(two_nu * t0))
    return prefactor * (exp_iv(-(two_nu * t0)) - exp_iv(-(two_nu * t1)))


def _exact_bkm_float(slab: UnforcedSlab) -> float:
    a = float(abs(slab.amplitude))
    nu = float(slab.nu)
    t0 = float(slab.t0)
    t1 = float(slab.t1)
    if slab.growing:
        return (a / nu) * (math.exp(2.0 * nu * t1) - math.exp(2.0 * nu * t0))
    return (a / nu) * (math.exp(-2.0 * nu * t0) - math.exp(-2.0 * nu * t1))


def _integrand_box(slab: UnforcedSlab) -> Interval:
    t0 = Interval.from_rational(slab.t0)
    t1 = Interval.from_rational(slab.t1)
    return Interval.hull(
        enclose_omega_linf(
            t0, amplitude=slab.amplitude, nu=slab.nu, growing=slab.growing
        ),
        enclose_omega_linf(
            t1, amplitude=slab.amplitude, nu=slab.nu, growing=slab.growing
        ),
    )


def integrand_contains_grid_and_sample(
    slab: UnforcedSlab,
    *,
    n_grid: int = 8,
    n_sample: int = 8,
    seed: int = 0,
) -> bool:
    box = _integrand_box(slab)
    t0 = float(slab.t0)
    t1 = float(slab.t1)
    grid = np.linspace(t0, t1, n_grid, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.uniform(t0, t1, size=n_sample)
    for t in (*grid.tolist(), *samples.tolist()):
        val = _omega_linf_point(
            float(t),
            amplitude=slab.amplitude,
            nu=slab.nu,
            growing=slab.growing,
        )
        if not box.contains(val):
            return False
    return True


def _tg_omega_on_grid(sample: PeriodicFlowSample) -> np.ndarray:
    n = int(sample.grid_shape[0])
    length = float(sample.lengths[0])
    axis = length * np.arange(n, dtype=float) / n
    x, y = np.meshgrid(axis, axis, indexing="ij")
    nu = float(sample.viscosity) / float(sample.density)
    time = float(sample.descriptor["time"])
    amp = float(sample.descriptor["amplitude"])
    decay = math.exp(-2.0 * nu * time)
    return (
        float(TG_OMEGA_LINF_FACTOR)
        * amp
        * decay
        * np.sin(x)
        * np.sin(y)
    )


def _grid_omega_matches_factor(sample: PeriodicFlowSample) -> bool:
    omega = _tg_omega_on_grid(sample)
    nu = float(sample.viscosity) / float(sample.density)
    time = float(sample.descriptor["time"])
    amp = float(sample.descriptor["amplitude"])
    expected = float(TG_OMEGA_LINF_FACTOR) * abs(amp) * math.exp(-2.0 * nu * time)
    observed = float(np.max(np.abs(omega)))
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)


def _strictly_below(enclosure: Interval, threshold: Fraction) -> bool:
    bound = Interval.from_rational(threshold)
    return enclosure.hi < bound.lo


def locked_force_free_slab(
    *,
    t0: Fraction = LOCKED_T0,
    growing: bool = False,
) -> UnforcedSlab:
    return UnforcedSlab(
        t0=t0,
        horizon=LOCKED_HORIZON,
        amplitude=LOCKED_AMPLITUDE,
        nu=LOCKED_NU,
        growing=growing,
    )


def force_free_bkm_slab(
    *,
    t0: Fraction = LOCKED_T0,
    growing: bool = False,
) -> dict[str, Any]:
    """Build the locked 07-18 force-free TG slab plus honesty."""
    slab = locked_force_free_slab(t0=t0, growing=growing)
    sample = taylor_green_vortex(
        LOCKED_N,
        viscosity=float(LOCKED_VISCOSITY),
        density=float(LOCKED_DENSITY),
        time=float(slab.t0),
        amplitude=float(slab.amplitude),
    )
    force_zero = bool(np.all(sample.forcing == 0.0)) and (
        sample.descriptor.get("forced") is False
    )
    integral = enclose_bkm_integral(slab)
    exact = _exact_bkm_float(slab)
    weak = certified_weak_residual(
        n_cells=LOCKED_N,
        viscosity=float(LOCKED_VISCOSITY),
        form="weak",
    )
    payload = weak.get("payload", weak)
    residual = payload["residual_enclosure"]
    misses = enclosure_covers(form="weak")
    weak_covers = residual[0] <= 0.0 <= residual[1] and misses == 0
    ledger = navier_stokes_ab_architecture_ledger()
    ledger_report = check_ledger(ledger)
    flags = assert_honesty()
    bounded = math.isfinite(integral.lo) and math.isfinite(integral.hi)
    return {
        "slab": slab,
        "force_zero": force_zero,
        "plant_unforced": force_zero,
        "omega_factor_locked": _grid_omega_matches_factor(sample),
        "bkm_integral": [integral.lo, integral.hi],
        "bkm_bounded": bounded,
        "bkm_contains_exact": integral.contains(exact),
        "integrand_grid_and_sample": integrand_contains_grid_and_sample(slab),
        "weak_covers": weak_covers,
        "weak_misses": misses,
        "horizon": float(LOCKED_HORIZON),
        "honesty": flags,
        "ledger_strength": ledger_report.strength,
        "ledger_holds": ledger_report.holds,
        "ledger_premises": list(ledger.external_premises),
        "ab_premises_nonempty": bool(NS_AB_EXTERNAL_PREMISES),
    }


def try_continue_slab(
    slab: UnforcedSlab,
    *,
    remaining_budget: int,
    next_horizon: Fraction,
    threshold: Fraction = BKM_BUDGET,
) -> Continue | Halt:
    """Accept the next TG window, or halt.

    Empty ``remaining_budget`` is ``search_incomplete``, never A/B.
    Growing vorticity, or a BKM integral that is not strictly below
    ``threshold``, is ``BLOCKED``.
    """
    horizon = Fraction(next_horizon)
    if remaining_budget <= 0:
        return Halt(
            reason="search_incomplete",
            detail="empty remaining budget",
            remaining_budget=int(remaining_budget),
        )
    integral = enclose_bkm_integral(slab)
    if not _strictly_below(integral, threshold):
        detail = "growing_vorticity" if slab.growing else "bkm_budget_exceeded"
        return Halt(
            reason="BLOCKED",
            detail=detail,
            remaining_budget=int(remaining_budget),
            bkm_lo=integral.lo,
            bkm_hi=integral.hi,
        )
    nxt = UnforcedSlab(
        t0=slab.t1,
        horizon=horizon,
        amplitude=slab.amplitude,
        nu=slab.nu,
        growing=slab.growing,
    )
    return Continue(
        next_slab=nxt,
        remaining_budget=int(remaining_budget) - 1,
        bkm_lo=integral.lo,
        bkm_hi=integral.hi,
    )


def locked_two_slab_continuation() -> dict[str, Any]:
    """Locked ``[0, 1/2]`` then ``[1/2, 1]`` on decaying TG, plus Halt controls."""
    first = locked_force_free_slab()
    accepted = try_continue_slab(
        first,
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    empty: Halt | None = None
    growing = try_continue_slab(
        locked_force_free_slab(growing=True),
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    n_slabs = 1
    second_t0: Fraction | None = None
    if isinstance(accepted, Continue):
        n_slabs = 2
        second_t0 = accepted.next_slab.t0
        empty_decision = try_continue_slab(
            accepted.next_slab,
            remaining_budget=0,
            next_horizon=LOCKED_HORIZON,
        )
        empty = empty_decision if isinstance(empty_decision, Halt) else None
    flags = assert_honesty()
    leftover = dict(UNFORCED_CONTINUATION_LEFTOVER)
    return {
        "accepted": isinstance(accepted, Continue),
        "n_slabs": n_slabs,
        "second_t0": str(second_t0) if second_t0 is not None else None,
        "covers_infinite_time": False,
        "empty_budget_reason": None if empty is None else empty.reason,
        "growing_reason": growing.reason if isinstance(growing, Halt) else None,
        "growing_detail": growing.detail if isinstance(growing, Halt) else None,
        "leftover": leftover,
        "leftover_id": leftover["leftover_id"],
        "honesty": flags,
    }


def abc_honesty_payload() -> dict[str, object]:
    flags = honesty_payload()
    flags["exact_3d_abc_plant"] = True
    flags["disclaimer"] = (
        "finite force-free ABC slab / continuation; not Clay (A)/(B); "
        "one exact 3-D plant is not the Clay 3-D quantifier"
    )
    return flags


def enclose_abc_u0_linf(slab: AbcSlab) -> Interval:
    """Sound Euclidean hull of the ABC component bounds."""
    ux = Interval.from_rational(abs(slab.a) + abs(slab.c))
    uy = Interval.from_rational(abs(slab.b) + abs(slab.a))
    uz = Interval.from_rational(abs(slab.c) + abs(slab.b))
    return ((ux * ux) + (uy * uy) + (uz * uz)).sqrt()


def _abc_decay_lambda(slab: AbcSlab) -> Interval:
    k2 = Interval.from_value(slab.k * slab.k)
    nu = Interval.from_rational(slab.nu)
    if slab.growing:
        return Interval.from_rational(ABC_GROWING_RATE_MULT) * nu * k2
    return nu * k2


def enclose_abc_omega_linf(t: Interval, slab: AbcSlab) -> Interval:
    """Enclose ``k ‖u_0‖_∞ exp(± λ t)`` with a sound ``‖u_0‖_∞`` hull."""
    k_iv = Interval.from_value(slab.k)
    u0 = enclose_abc_u0_linf(slab)
    lam = _abc_decay_lambda(slab)
    if slab.growing:
        return k_iv * u0 * exp_iv(lam * t)
    return k_iv * u0 * exp_iv(-(lam * t))


def enclose_abc_bkm_integral(slab: AbcSlab) -> Interval:
    r"""Enclose ``∫_{t0}^{t1} k ‖u_0‖_∞ e^{±λ t} dt``.

    Decaying Beltrami: ``λ = ν k²``. Manufactured growth uses
    ``λ = 16 ν k²`` so the Halt control exceeds ``BKM_BUDGET``.
    """
    k_iv = Interval.from_value(slab.k)
    u0 = enclose_abc_u0_linf(slab)
    lam = _abc_decay_lambda(slab)
    t0 = Interval.from_rational(slab.t0)
    t1 = Interval.from_rational(slab.t1)
    prefactor = (k_iv * u0) / lam
    if slab.growing:
        return prefactor * (exp_iv(lam * t1) - exp_iv(lam * t0))
    return prefactor * (exp_iv(-(lam * t0)) - exp_iv(-(lam * t1)))


def _abc_omega_linf_point(t: float, slab: AbcSlab) -> float:
    amp = math.sqrt(
        (abs(float(slab.a)) + abs(float(slab.c))) ** 2
        + (abs(float(slab.b)) + abs(float(slab.a))) ** 2
        + (abs(float(slab.c)) + abs(float(slab.b))) ** 2
    )
    lam = float(slab.nu) * float(slab.k * slab.k)
    if slab.growing:
        lam *= float(ABC_GROWING_RATE_MULT)
        decay = math.exp(lam * t)
    else:
        decay = math.exp(-lam * t)
    return float(slab.k) * amp * decay


def _abc_exact_bkm_float(slab: AbcSlab) -> float:
    t0 = float(slab.t0)
    t1 = float(slab.t1)
    amp = math.sqrt(
        (abs(float(slab.a)) + abs(float(slab.c))) ** 2
        + (abs(float(slab.b)) + abs(float(slab.a))) ** 2
        + (abs(float(slab.c)) + abs(float(slab.b))) ** 2
    )
    lam = float(slab.nu) * float(slab.k * slab.k)
    k = float(slab.k)
    if slab.growing:
        lam *= float(ABC_GROWING_RATE_MULT)
        return (k * amp / lam) * (math.exp(lam * t1) - math.exp(lam * t0))
    return (k * amp / lam) * (math.exp(-lam * t0) - math.exp(-lam * t1))


def _abc_integrand_box(slab: AbcSlab) -> Interval:
    t0 = Interval.from_rational(slab.t0)
    t1 = Interval.from_rational(slab.t1)
    return Interval.hull(
        enclose_abc_omega_linf(t0, slab),
        enclose_abc_omega_linf(t1, slab),
    )


def abc_integrand_contains_grid_and_sample(
    slab: AbcSlab,
    *,
    n_grid: int = 8,
    n_sample: int = 8,
    seed: int = 0,
) -> bool:
    box = _abc_integrand_box(slab)
    t0 = float(slab.t0)
    t1 = float(slab.t1)
    grid = np.linspace(t0, t1, n_grid, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.uniform(t0, t1, size=n_sample)
    for t in (*grid.tolist(), *samples.tolist()):
        val = _abc_omega_linf_point(float(t), slab)
        if not box.contains(val):
            return False
    return True


def _abc_grid_omega_linf(sample: PeriodicFlowSample) -> float:
    velocity = sample.velocity
    k = int(sample.descriptor["wavenumber"])
    speed = np.sqrt(np.sum(velocity * velocity, axis=0))
    return float(k) * float(np.max(speed))


def locked_force_free_abc_slab(
    *,
    t0: Fraction = LOCKED_T0,
    growing: bool = False,
) -> AbcSlab:
    return AbcSlab(
        t0=t0,
        horizon=LOCKED_HORIZON,
        a=LOCKED_ABC_AMPLITUDE,
        b=LOCKED_ABC_AMPLITUDE,
        c=LOCKED_ABC_AMPLITUDE,
        k=LOCKED_ABC_K,
        nu=LOCKED_NU,
        growing=growing,
    )


def force_free_abc_bkm_slab(
    *,
    t0: Fraction = LOCKED_T0,
    growing: bool = False,
) -> dict[str, Any]:
    """Build the locked 07-21 force-free ABC slab plus honesty."""
    slab = locked_force_free_abc_slab(t0=t0, growing=growing)
    sample = beltrami_abc_flow(
        LOCKED_N,
        viscosity=float(LOCKED_VISCOSITY),
        density=float(LOCKED_DENSITY),
        a=float(slab.a),
        b=float(slab.b),
        c=float(slab.c),
        wavenumber=int(slab.k),
        time=float(slab.t0),
    )
    force_zero = bool(np.all(sample.forcing == 0.0)) and (
        sample.descriptor.get("forced") is False
    )
    dimension = int(sample.descriptor.get("dimension", 0))
    integral = enclose_abc_bkm_integral(slab)
    exact = _abc_exact_bkm_float(slab)
    bound = enclose_abc_u0_linf(slab)
    grid_omega = _abc_grid_omega_linf(sample)
    grid_inside_bound = grid_omega <= float(slab.k) * bound.hi
    ledger = navier_stokes_ab_architecture_ledger()
    ledger_report = check_ledger(ledger)
    flags = assert_honesty(abc_honesty_payload())
    bounded = math.isfinite(integral.lo) and math.isfinite(integral.hi)
    return {
        "slab": slab,
        "force_zero": force_zero,
        "plant_unforced": force_zero,
        "dimension": dimension,
        "bkm_integral": [integral.lo, integral.hi],
        "bkm_bounded": bounded,
        "bkm_contains_exact": integral.contains(exact),
        "integrand_grid_and_sample": abc_integrand_contains_grid_and_sample(slab),
        "grid_omega_inside_bound": grid_inside_bound,
        "horizon": float(LOCKED_HORIZON),
        "honesty": flags,
        "ledger_strength": ledger_report.strength,
        "ledger_holds": ledger_report.holds,
        "ledger_premises": list(ledger.external_premises),
        "ab_premises_nonempty": bool(NS_AB_EXTERNAL_PREMISES),
        "three_d_premise_present": THREE_D_AB_PREMISE in NS_AB_EXTERNAL_PREMISES,
        "leftover": dict(UNFORCED_CONTINUATION_LEFTOVER),
        "leftover_id": UNFORCED_CONTINUATION_LEFTOVER["leftover_id"],
    }


def try_continue_abc_slab(
    slab: AbcSlab,
    *,
    remaining_budget: int,
    next_horizon: Fraction,
    threshold: Fraction = BKM_BUDGET,
) -> Continue | Halt:
    """Accept the next ABC window, or halt.

    Empty ``remaining_budget`` is ``search_incomplete``, never A/B.
    Growing vorticity, or a BKM integral that is not strictly below
    ``threshold``, is ``BLOCKED``.
    """
    horizon = Fraction(next_horizon)
    if remaining_budget <= 0:
        return Halt(
            reason="search_incomplete",
            detail="empty remaining budget",
            remaining_budget=int(remaining_budget),
        )
    integral = enclose_abc_bkm_integral(slab)
    if not _strictly_below(integral, threshold):
        detail = "growing_vorticity" if slab.growing else "bkm_budget_exceeded"
        return Halt(
            reason="BLOCKED",
            detail=detail,
            remaining_budget=int(remaining_budget),
            bkm_lo=integral.lo,
            bkm_hi=integral.hi,
        )
    nxt = AbcSlab(
        t0=slab.t1,
        horizon=horizon,
        a=slab.a,
        b=slab.b,
        c=slab.c,
        k=slab.k,
        nu=slab.nu,
        growing=slab.growing,
    )
    return Continue(
        next_slab=nxt,
        remaining_budget=int(remaining_budget) - 1,
        bkm_lo=integral.lo,
        bkm_hi=integral.hi,
    )


def locked_two_abc_slab_continuation() -> dict[str, Any]:
    """Locked ``[0, 1/2]`` then ``[1/2, 1]`` on decaying ABC, plus Halt controls."""
    first = locked_force_free_abc_slab()
    accepted = try_continue_abc_slab(
        first,
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    empty: Halt | None = None
    growing = try_continue_abc_slab(
        locked_force_free_abc_slab(growing=True),
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    n_slabs = 1
    second_t0: Fraction | None = None
    if isinstance(accepted, Continue) and isinstance(accepted.next_slab, AbcSlab):
        n_slabs = 2
        second_t0 = accepted.next_slab.t0
        empty_decision = try_continue_abc_slab(
            accepted.next_slab,
            remaining_budget=0,
            next_horizon=LOCKED_HORIZON,
        )
        empty = empty_decision if isinstance(empty_decision, Halt) else None
    flags = assert_honesty(abc_honesty_payload())
    leftover = dict(UNFORCED_CONTINUATION_LEFTOVER)
    return {
        "accepted": isinstance(accepted, Continue),
        "n_slabs": n_slabs,
        "second_t0": str(second_t0) if second_t0 is not None else None,
        "covers_infinite_time": False,
        "empty_budget_reason": None if empty is None else empty.reason,
        "growing_reason": growing.reason if isinstance(growing, Halt) else None,
        "growing_detail": growing.detail if isinstance(growing, Halt) else None,
        "leftover": leftover,
        "leftover_id": leftover["leftover_id"],
        "honesty": flags,
        "three_d_premise_present": THREE_D_AB_PREMISE in NS_AB_EXTERNAL_PREMISES,
    }


__all__ = [
    "ABC_GROWING_RATE_MULT",
    "AbcSlab",
    "BKM_BUDGET",
    "Continue",
    "Halt",
    "LOCKED_ABC_AMPLITUDE",
    "LOCKED_HORIZON",
    "LOCKED_N",
    "LOCKED_NU",
    "TG_OMEGA_LINF_FACTOR",
    "THREE_D_AB_PREMISE",
    "UNFORCED_CONTINUATION_LEFTOVER",
    "UnforcedSlab",
    "abc_honesty_payload",
    "abc_integrand_contains_grid_and_sample",
    "assert_honesty",
    "enclose_abc_bkm_integral",
    "enclose_abc_omega_linf",
    "enclose_abc_u0_linf",
    "enclose_bkm_integral",
    "enclose_omega_linf",
    "force_free_abc_bkm_slab",
    "force_free_bkm_slab",
    "honesty_payload",
    "integrand_contains_grid_and_sample",
    "locked_force_free_abc_slab",
    "locked_force_free_slab",
    "locked_two_abc_slab_continuation",
    "locked_two_slab_continuation",
    "try_continue_abc_slab",
    "try_continue_slab",
]

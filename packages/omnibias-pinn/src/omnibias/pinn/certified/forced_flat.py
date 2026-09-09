# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jet-flat forced concentrating field (theory 07-13).

An explicit axisymmetric polynomial swirl on the locked 07-09 scales whose
regularized leading stress ``T_0`` (paper (4.8)--(4.11)) vanishes at the
axis after one exact-``Q`` correction of the linear slope. Core
``||u||_infty ~ tau^{-A}`` and core energy ``O(tau^{1/2-3h}) -> 0`` are
exact monomials. Force is that residual times a spatial mollifier cutoff;
a from-rest occupancy ramp is ``0`` at ``t = 0``.

This is a **different, weaker** object than Clay (C)/(D). The OpenAI
construction keeps the force ``C^infty`` through ``t = 1``; that joining,
pulse, and ``sigma``-cycle work stays leftover. Not unforced (A)/(B).
``forced_blowup_reproof_claim`` stays false. Not a CCF residual and not
an extension of :mod:`omnibias.pinn.certified.navier_stokes`.

Raw ``T_0(0,0)`` is ``(0,0)`` by the factors of ``X`` in (4.11). The
load-bearing gate is the regularized axis jet
``(lim T_{r theta}/X, lim T_{rz}/sqrt(2X))``. The ``X^2`` coefficient
``b`` does not enter that jet; the named extra coefficient is the linear
slope ``a = (1+h)/4``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.pulse_envelope import (
    DECAY_RATE,
    GROWTH_RATE,
    PulseEnvelope,
    locked_decay_envelope,
    locked_growth_envelope,
    locked_pulse_grid,
    locked_pulse_grid_matches_tower,
    mollifier_tail_contains_truth,
    pulse_product_contains_grid_and_sample,
)
from omnibias.pinn.certified.anisotropic import (
    DX,
    LOCKED_H,
    AxisRegularProfile,
    SimilarityScales,
    apply_T_b,
)

LOCKED_C = Fraction(1)
LOCKED_A_UNCORRECTED = Fraction(1)
LOCKED_B_POLY = Fraction(0)
LOCKED_X_CORE = Fraction(1)
LOCKED_X_BOX = Fraction(1)
LOCKED_ETA_BOX = Fraction(1)
LOCKED_TAU_HI = Fraction(1)
LOCKED_TAU_LO = Fraction(1, 4)

_FORBIDDEN = (
    "navier_stokes_proof_claim",
    "continuum_navier_stokes_claim",
    "forced_blowup_reproof_claim",
    "unproven_claim",
)

def _as_frac(value: Fraction | int | str) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def honesty_payload() -> dict[str, bool]:
    """Parent flags stay false. Named leftovers stay named."""
    return {
        "unproven_claim": False,
        "navier_stokes_proof_claim": False,
        "continuum_navier_stokes_claim": False,
        "forced_blowup_reproof_claim": False,
        "continuum_pde_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "c_infinity_through_t1_leftover": True,
        "pulses_leftover": True,
        "joining_leftover": True,
        "uniqueness_leftover": True,
        "remaining_q_powers_leftover": True,
    }


def assert_honesty(payload: Mapping[str, Any] | None = None) -> dict[str, bool]:
    flags = honesty_payload() if payload is None else dict(payload)
    for key in _FORBIDDEN:
        if bool(flags.get(key, False)):
            raise ValueError(f"honesty.{key} must stay False on this fragment")
    return {str(k): bool(v) for k, v in flags.items()}


@dataclass(frozen=True)
class JetFlatProfile:
    """Polynomial swirl ``F = c (1 + a X + b X^2)`` on locked scales.

    ``U = eta X`` and ``Pi = int_0^X F^2`` match the 07-09 germ. The
    ``X^2`` term is kept so the family matches the paper; it does not
    enter the axis jet of ``T_0``.
    """

    h: Fraction
    c: Fraction
    a: Fraction
    b: Fraction = LOCKED_B_POLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "h", _as_frac(self.h))
        object.__setattr__(self, "c", _as_frac(self.c))
        object.__setattr__(self, "a", _as_frac(self.a))
        object.__setattr__(self, "b", _as_frac(self.b))

    @property
    def scales(self) -> SimilarityScales:
        return SimilarityScales(self.h)

    def F(self, X: Fraction | int | str) -> Fraction:
        xx = _as_frac(X)
        return self.c * (1 + self.a * xx + self.b * xx * xx)

    def F_X(self, X: Fraction | int | str) -> Fraction:
        xx = _as_frac(X)
        return self.c * (self.a + 2 * self.b * xx)

    def U(self, X: Fraction | int | str, eta: Fraction | int | str) -> Fraction:
        return _as_frac(eta) * _as_frac(X)

    def Pi(self, X: Fraction | int | str) -> Fraction:
        """Exact antiderivative of ``F^2`` with ``Pi(0) = 0``."""
        xx = _as_frac(X)
        c2 = self.c * self.c
        a = self.a
        b = self.b
        return c2 * (
            xx
            + a * xx * xx
            + (a * a + 2 * b) * xx**3 / 3
            + (a * b) * xx**4 / 2
            + (b * b) * xx**5 / 5
        )

    def to_axis_regular(self) -> AxisRegularProfile:
        if self.b != 0:
            raise ValueError("AxisRegularProfile has no X^2 coefficient")
        return AxisRegularProfile(h=self.h, c=self.c, a=self.a)


ProfileLike = AxisRegularProfile | JetFlatProfile


def _as_flat(profile: ProfileLike | None) -> JetFlatProfile:
    if profile is None:
        return uncorrected_jet_flat_profile()
    if isinstance(profile, JetFlatProfile):
        return profile
    return JetFlatProfile(h=profile.h, c=profile.c, a=profile.a, b=Fraction(0))


def uncorrected_jet_flat_profile() -> JetFlatProfile:
    """Locked 07-09 plant ``F = 1 + X`` as a negative control for the axis jet."""
    return JetFlatProfile(
        h=LOCKED_H,
        c=LOCKED_C,
        a=LOCKED_A_UNCORRECTED,
        b=LOCKED_B_POLY,
    )


def locked_jet_flat_profile() -> JetFlatProfile:
    """Axis-flat lock: ``a = (1+h)/4`` so the regularized ``T_0`` jet vanishes."""
    return correct_axis_stress(uncorrected_jet_flat_profile())[0]


def logarithmic_slope(profile: ProfileLike, X: Fraction | int | str) -> Fraction:
    """``l = D_X log H = 1 + X F_X / F`` (paper (4.8))."""
    prof = _as_flat(profile)
    xx = _as_frac(X)
    f = prof.F(xx)
    if f == 0:
        raise ZeroDivisionError("logarithmic slope: F vanishes")
    if xx == 0:
        return Fraction(1)
    return 1 + DX(xx, prof.F_X(xx)) / f


def linear_T_b_at(
    profile: ProfileLike,
    *,
    X: Fraction = LOCKED_X_CORE,
    eta: Fraction = Fraction(0),
    b: Fraction = Fraction(0),
) -> Fraction:
    """Lemma 4.1 ``T_b`` on the linear restriction (``X^2`` coeff must vanish)."""
    prof = _as_flat(profile)
    linear = prof.to_axis_regular()
    return apply_T_b(linear.h, b, X, eta, linear.jet_at(X, eta))


def axis_sources(profile: ProfileLike | None = None) -> dict[str, Fraction]:
    """Paper (4.9) sources at ``(X, eta) = (0, 0)`` with ``C_Q = C_N = 0``.

    ``W = 1``, ``l = 1``, ``H_c = 0``, ``(log E)_eta = 0`` give
    ``S_q = -(1+h)`` and ``S_n = 0``, hence ``Q_s(0) = S_q/2`` and
    ``N_s(0) = S_n``.
    """
    prof = _as_flat(profile)
    h = prof.h
    sq = -(1 + h)
    sn = Fraction(0)
    return {
        "S_q": sq,
        "S_n": sn,
        "Q_s": sq / 2,
        "N_s": sn,
        "L": Fraction(1),
        "l": Fraction(1),
        "W": Fraction(1),
    }


def axis_T0(profile: ProfileLike | None = None) -> tuple[Fraction, Fraction]:
    """Regularized axis jet of paper (4.11) ``T_0 = F(p_s - s)``.

    Returns ``(lim T_{r theta}/X, lim T_{rz}/sqrt(2X))`` at ``(0,0)``.
    The raw vector ``T_0(0,0)`` is ``(0,0)`` by the factors of ``X``.
    """
    prof = _as_flat(profile)
    src = axis_sources(prof)
    f0 = prof.F(Fraction(0))
    fx0 = prof.F_X(Fraction(0))
    t_rtheta = f0 * src["Q_s"] / src["L"] + 2 * fx0
    t_rz = src["N_s"] / (2 * src["L"])
    return (t_rtheta, t_rz)


def leading_stress_T0(
    profile: ProfileLike,
    X: Fraction | int | str,
    eta: Fraction | int | str,
) -> tuple[Fraction, Fraction]:
    """Paper (4.11) ``T_0``. This fragment evaluates the axis jet only."""
    if _as_frac(X) != 0 or _as_frac(eta) != 0:
        raise ValueError(
            "leading T0 on this fragment is the regularized axis jet; "
            "off-axis integral (4.10) stays leftover"
        )
    return axis_T0(profile)


def correct_axis_stress(
    profile: ProfileLike | None = None,
) -> tuple[JetFlatProfile, Fraction]:
    """Exact-``Q`` correction of the linear slope so the axis jet vanishes.

    ``T_{r theta}/X -> c(2a - (1+h)/2)``. The ``X^2`` coefficient ``b``
    does not enter, so the named extra coefficient is ``a = (1+h)/4``.
    """
    prof = _as_flat(profile)
    a_star = (1 + prof.h) / 4
    corrected = JetFlatProfile(h=prof.h, c=prof.c, a=a_star, b=prof.b)
    return corrected, a_star


@dataclass(frozen=True)
class ScaleMonomial:
    """``prefactor * tau^{exponent}`` with both pieces exact over ``Q``.

    Irrational values of ``tau^{exponent}`` are not evaluated; gates
    compare prefactors, exponents, and the sign of the exponent.
    """

    prefactor: Fraction
    exponent: Fraction
    tau: Fraction
    name: str = "scale"


def core_linfty_scale(
    tau: Fraction | int | str,
    profile: ProfileLike | None = None,
) -> ScaleMonomial:
    """``||u||_infty ~ tau^{-A}`` with tau-independent prefactor ``F(X_c)``."""
    prof = _as_flat(profile)
    tt = _as_frac(tau)
    if tt <= 0:
        raise ValueError("tau must be positive")
    return ScaleMonomial(
        prefactor=prof.F(LOCKED_X_CORE),
        exponent=-prof.scales.A,
        tau=tt,
        name="linfty",
    )


def energy_box_prefactor(profile: ProfileLike | None = None) -> Fraction:
    """``int_0^{X_c} int_{-eta_c}^{eta_c} 2 X F^2 d eta d X`` at ``X_c = eta_c = 1``."""
    prof = _as_flat(profile)
    a = prof.a
    b = prof.b
    c2 = prof.c * prof.c
    radial = (
        Fraction(1, 2)
        + (2 * a) / 3
        + (a * a + 2 * b) / 4
        + (2 * a * b) / 5
        + (b * b) / 6
    )
    return 4 * c2 * radial


def core_energy_scale(
    tau: Fraction | int | str,
    profile: ProfileLike | None = None,
) -> ScaleMonomial:
    """Core kinetic energy ``~ tau^{1/2 - 3h}`` times an exact rational prefactor."""
    prof = _as_flat(profile)
    tt = _as_frac(tau)
    if tt <= 0:
        raise ValueError("tau must be positive")
    return ScaleMonomial(
        prefactor=energy_box_prefactor(prof),
        exponent=Fraction(1, 2) - 3 * prof.h,
        tau=tt,
        name="energy",
    )


def from_rest_ramp() -> PulseEnvelope:
    """Occupancy ``(s, t) = (0, 1)``: ``P = 0`` at rest."""
    return PulseEnvelope(Fraction(0), Fraction(1), name="from_rest")


def on_window_ramp() -> PulseEnvelope:
    """Occupancy ``(s, t) = (1, 1)``: ``P = 1`` on the locked on-window."""
    return PulseEnvelope(Fraction(1), Fraction(1), name="on_window")


def forced_field(
    tau: Fraction | int | str = LOCKED_TAU_HI,
    *,
    cutoff: bool = True,
    ramp: bool = True,
    corrected: bool = True,
) -> dict[str, Any]:
    """Honesty payload plus locked scales, axis jet, cutoff, and ramp."""
    profile = (
        correct_axis_stress(uncorrected_jet_flat_profile())[0]
        if corrected
        else uncorrected_jet_flat_profile()
    )
    flags = assert_honesty()
    payload: dict[str, Any] = {
        "kind": "jet_flat_forced_blowup",
        "tau": _as_frac(tau),
        "profile": profile,
        "linfty": core_linfty_scale(tau, profile),
        "energy": core_energy_scale(tau, profile),
        "axis_T0": axis_T0(profile),
        "honesty": flags,
        "leftover": {
            "c_infinity_through_t1": True,
            "pulses": True,
            "joining": True,
            "uniqueness": True,
            "remaining_q_powers": True,
        },
    }
    if cutoff:
        payload["cutoff_ok"] = mollifier_tail_contains_truth()
    if ramp:
        payload["ramp_at_rest"] = from_rest_ramp().value()
        payload["ramp_on_window"] = on_window_ramp().value()
    return payload


def composition_honesty_payload(*, identity_holds: bool) -> dict[str, bool]:
    """07-17 honesty. Parent flags stay false. Named leftovers stay named.

    ``pulses_leftover`` flips only when the locked-pulse identity holds.
    Joining, uniqueness, and ``C^infty`` through ``t=1`` stay leftover.
    """
    flags = honesty_payload()
    flags["pulses_leftover"] = not identity_holds
    return flags


def compose_locked_pulse_family() -> dict[str, Any]:
    """Compose the locked 07-12 pulse grid into the 07-13 forced field.

    One named pulse family. Not cutoff summation, not joining, not a
    Clay (C)/(D) reproof. Does not trim ``NS_SCALE_EXTERNAL_PREMISES``.
    """
    tower_ok = locked_pulse_grid_matches_tower()
    corrected = locked_jet_flat_profile()
    uncorrected = uncorrected_jet_flat_profile()
    t0_c = axis_T0(corrected)
    t0_u = axis_T0(uncorrected)
    composed_zero = True
    grid_rows: list[dict[str, Any]] = []
    for env in locked_pulse_grid():
        p = env.value()
        p_prime = env.derivative()
        composed = (p * t0_c[0], p * t0_c[1])
        if composed != (0, 0):
            composed_zero = False
        grid_rows.append(
            {
                "name": env.name,
                "P": p,
                "P_prime": p_prime,
                "composed_corrected": composed,
                "composed_uncorrected": (p * t0_u[0], p * t0_u[1]),
            }
        )
    growth = locked_growth_envelope()
    decay = locked_decay_envelope()
    product_rule_ok = (
        growth.derivative() * t0_u[0] == GROWTH_RATE * growth.value() * t0_u[0]
        and decay.derivative() * t0_u[0] == DECAY_RATE * decay.value() * t0_u[0]
    )
    enclosure_ok = pulse_product_contains_grid_and_sample(t0_u[0])
    identity_holds = tower_ok and composed_zero and product_rule_ok and enclosure_ok
    leftover_id = None if identity_holds else 56
    flags = assert_honesty(composition_honesty_payload(identity_holds=identity_holds))
    return {
        "kind": "pulse_family_composition",
        "tower_ok": tower_ok,
        "composed_axis_zero": composed_zero,
        "product_rule_ok": product_rule_ok,
        "enclosure_ok": enclosure_ok,
        "identity_holds": identity_holds,
        "leftover_id": leftover_id,
        "grid": grid_rows,
        "uncorrected_axis_T0": t0_u,
        "corrected_axis_T0": t0_c,
        "honesty": flags,
    }


__all__ = [
    "JetFlatProfile",
    "LOCKED_A_UNCORRECTED",
    "LOCKED_B_POLY",
    "LOCKED_C",
    "LOCKED_ETA_BOX",
    "LOCKED_H",
    "LOCKED_TAU_HI",
    "LOCKED_TAU_LO",
    "LOCKED_X_BOX",
    "LOCKED_X_CORE",
    "ScaleMonomial",
    "assert_honesty",
    "axis_T0",
    "axis_sources",
    "compose_locked_pulse_family",
    "composition_honesty_payload",
    "core_energy_scale",
    "core_linfty_scale",
    "correct_axis_stress",
    "energy_box_prefactor",
    "forced_field",
    "from_rest_ramp",
    "honesty_payload",
    "leading_stress_T0",
    "linear_T_b_at",
    "locked_jet_flat_profile",
    "logarithmic_slope",
    "on_window_ramp",
    "uncorrected_jet_flat_profile",
]

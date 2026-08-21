# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact oscillator ladder inside a neural ansatz (theory 07-07).

The founding bias collapse (``delta -> 0``) supplies ``a`` and
``a^+`` as first-order closed-form operators. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.

``Normalization`` is required: the raw tower and the quantum
oscillator eigenfunctions differ by an explicit Rodrigues
reweight. This module is a **tool** for a variational ansatz. It is
not a solution of the many-body problem and not a physical
discovery.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.ladder import Normalization, hermite_function

DISCLAIMER = (
    "tooling: exact ladder operators inside a variational ansatz; "
    "not a many-body solution and not a physical discovery"
)
_SQRT2 = math.sqrt(2.0)
_PI_Q = math.pi ** -0.25
Op = Literal["raise", "lower"]


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "many_body_solution_claim": False,
        "physical_discovery": False,
        "theorem_prover_verified": False,
    }


def _parse_normalization(name: str) -> Normalization:
    key = str(name).lower().strip()
    if key in {"rodrigues", "tower"}:
        return Normalization.TOWER
    if key == "oscillator":
        return Normalization.OSCILLATOR
    raise ValueError(
        "normalization must be 'rodrigues'/'tower' or 'oscillator', "
        f"got {name!r}"
    )


def _l2_const(n: int) -> float:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return float(_PI_Q / math.sqrt((2.0 ** n) * float(math.factorial(n))))


def oscillator_phi(n: int, x: float) -> float:
    """L2-normalized oscillator eigenfunction ``φ_n``."""
    return _l2_const(n) * hermite_function(
        n, x, normalization=Normalization.OSCILLATOR
    )


def _physicist_H(n: int, x: float) -> float:
    gauss = math.exp(-0.5 * x * x)
    return hermite_function(n, x, normalization=Normalization.OSCILLATOR) / gauss


def _phi_deriv(n: int, x: float) -> float:
    gauss = math.exp(-0.5 * x * x)
    h = _physicist_H(n, x)
    hp = 0.0 if n == 0 else 2.0 * float(n) * _physicist_H(n - 1, x)
    return _l2_const(n) * (hp - x * h) * gauss


def apply_ladder(
    n: int,
    x: float,
    op: Op,
    *,
    normalization: str,
) -> float:
    """Exact ``a`` / ``a^+`` from the closed-form tower. Normalization required."""
    parsed = _parse_normalization(normalization)
    if op not in {"raise", "lower"}:
        raise ValueError(f"op must be 'raise' or 'lower', got {op!r}")
    if parsed is Normalization.TOWER:
        from omnibias.core.ladder import tower_lower, tower_raise

        return tower_raise(n, x) if op == "raise" else tower_lower(n, x)
    phi = oscillator_phi(n, x)
    dphi = _phi_deriv(n, x)
    if op == "lower":
        return (x * phi + dphi) / _SQRT2
    return (x * phi - dphi) / _SQRT2


@dataclass(frozen=True)
class LadderBasis:
    """Finite oscillator / Rodrigues basis. Normalization is stored, never defaulted."""

    n_modes: int
    normalization: str

    def __post_init__(self) -> None:
        if int(self.n_modes) < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        _parse_normalization(self.normalization)
        object.__setattr__(self, "n_modes", int(self.n_modes))

    def eval(self, n: int, x: float) -> float:
        if n < 0 or n >= self.n_modes:
            raise ValueError(f"n must be in [0, {self.n_modes}), got {n}")
        parsed = _parse_normalization(self.normalization)
        if parsed is Normalization.OSCILLATOR:
            return oscillator_phi(n, x)
        return hermite_function(n, x, normalization=Normalization.TOWER)

    def apply(self, n: int, x: float, op: Op) -> float:
        return apply_ladder(n, x, op, normalization=self.normalization)


def hermite_ladder_basis(
    n_modes: int,
    *,
    normalization: str,
) -> LadderBasis:
    """Build a basis. ``normalization`` is required (spec 02-10)."""
    return LadderBasis(n_modes, normalization)


def ladder_coefficient_errors(
    n_max: int = 8,
    xs: Sequence[float] = (0.3, 0.7, 1.1, -0.4),
) -> list[float]:
    """``|a φ_n / φ_{n-1} - sqrt(n)|`` and the raise twin, skipping nodes."""
    errors: list[float] = []
    for n in range(1, n_max + 1):
        expect_l = math.sqrt(float(n))
        expect_r = math.sqrt(float(n + 1))
        for x in xs:
            lower = apply_ladder(n, x, "lower", normalization="oscillator")
            phi_m = oscillator_phi(n - 1, x)
            if abs(phi_m) > 1e-8:
                errors.append(abs(lower / phi_m - expect_l))
            raised = apply_ladder(n, x, "raise", normalization="oscillator")
            phi_p = oscillator_phi(n + 1, x)
            if abs(phi_p) > 1e-8:
                errors.append(abs(raised / phi_p - expect_r))
    return errors


def qho_ground_energy() -> float:
    """Textbook 1-D oscillator ground energy (Schiff / any QM text): ``1/2``."""
    return 0.5


def qho_rayleigh(n: int, xs: Sequence[float]) -> float:
    """Discrete Rayleigh quotient of ``H φ_n`` / ``φ_n`` (mean over ``xs``)."""
    from omnibias.core.ladder import oscillator_hamiltonian_apply

    nums = 0.0
    dens = 0.0
    for x in xs:
        phi = oscillator_phi(n, x)
        hphi = _l2_const(n) * oscillator_hamiltonian_apply(
            n, x, normalization=Normalization.OSCILLATOR
        )
        nums += hphi * phi
        dens += phi * phi
    if dens <= 0.0:
        raise ZeroDivisionError("empty Rayleigh weight")
    return nums / dens


def gaussian_envelope_energy(alpha: float) -> float:
    """Exact variational energy of ``exp(-α x^2 / 2)`` on the 1-D QHO."""
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return 0.25 * alpha + 0.25 / alpha


def vmc_comparison_report(*, seeds: int = 5) -> dict[str, object]:
    """G2: 1-D QHO Gaussian envelope already contains the exact ground state.

    A full FermiNet many-body comparison is a heavy run under
    ``$OMNIBIAS_SCRATCH``, not a CI claim. Five seeds of a wrong-width
    Gaussian Newton-step to ``α = 1``; the ladder projection is already
    at ``E = 1/2``. Result: **no improvement** to report against this
    competent 1-D baseline.
    """
    ladder_e = qho_rayleigh(0, (-1.2, -0.4, 0.2, 0.8, 1.5))
    steps: list[int] = []
    for seed in range(seeds):
        alpha = 0.4 + 0.15 * float(seed)
        n_steps = 0
        for _ in range(8):
            if abs(gaussian_envelope_energy(alpha) - 0.5) <= 1.6e-3:
                break
            # Newton on E(α) = α/4 + 1/(4α).
            deriv = 0.25 - 0.25 / (alpha * alpha)
            hess = 0.5 / (alpha**3)
            alpha = alpha - deriv / hess
            n_steps += 1
        steps.append(n_steps)
    return {
        "ladder_energy": ladder_e,
        "reference": 0.5,
        "chemical_accuracy": 1.6e-3,
        "gaussian_newton_steps": steps,
        "ladder_steps": 0,
        "no_improvement": True,
        "reason": (
            "1-D QHO Gaussian envelope already contains the exact ground "
            "state; FermiNet many-body G2 is a heavy run, not CI"
        ),
        "disclaimer": DISCLAIMER,
    }


__all__ = [
    "DISCLAIMER",
    "LadderBasis",
    "apply_ladder",
    "gaussian_envelope_energy",
    "hermite_ladder_basis",
    "honesty_payload",
    "ladder_coefficient_errors",
    "oscillator_phi",
    "qho_ground_energy",
    "qho_rayleigh",
    "vmc_comparison_report",
]

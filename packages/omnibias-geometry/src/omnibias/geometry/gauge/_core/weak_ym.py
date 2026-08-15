# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Weak Yang-Mills residual against adjoint test 1-forms.

The integral jet that belongs to a 4-form theory is

    r(omega) = sum_b w_b eta_{nu nu'} omega^{nu a}(x_b) (D*F)^{nu' a}(x_b)

not a 1-D Fredholm column and not ``omnibias.fields.weak`` (1-D OMBU VPINN).
This module evaluates that residual on an existing
:class:`~omnibias.geometry.gauge._core.covariant_jet.GaugeCovariantJet`
(``ym_eom``). No new derivative kernel.

Honesty: classical local / weak identities on a smooth connection. A typical
lattice vacuum draw does not satisfy ``D*F = 0``. Not a mass-gap claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.geometry.gauge._core.covariant_jet import GaugeCovariantJet
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra

WEAK_YM_FLOOR = 1e-7


@dataclass(frozen=True)
class AdjointTestOneForm:
    """One adjoint-valued test 1-form sampled on the jet batch."""

    omega: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class AdjointTestBank:
    """Stacked test 1-forms. ``omega`` has shape ``(T, B, d, n)``."""

    omega: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.omega.shape[0])


def gaussian_adjoint_test_bank(
    points: np.ndarray,
    *,
    n_tests: int,
    algebra: LieAlgebra,
    rng: np.random.Generator,
    length_scale: float | None = None,
) -> AdjointTestBank:
    """Smooth Gaussian bumps times a unit adjoint polarization.

    Not OMBU and not :mod:`omnibias.fields.weak`.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"points must have shape (B, 4), got {x.shape}")
    batch, dim = x.shape
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1, got {n_tests}")
    adj = algebra.dim
    span = float(np.max(x) - np.min(x))
    scale = float(length_scale) if length_scale is not None else max(span / 4.0, 0.25)
    lo = np.min(x, axis=0)
    hi = np.max(x, axis=0)
    centers = rng.uniform(lo, hi, size=(n_tests, dim))
    delta = x[None, :, :] - centers[:, None, :]
    amp = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / (scale * scale))
    omega = np.zeros((n_tests, batch, dim, adj), dtype=np.float64)
    for index in range(n_tests):
        nu = int(index % dim)
        color = int((index // dim) % adj)
        omega[index, :, nu, color] = amp[index]
    weights = np.full(batch, 1.0 / float(batch), dtype=np.float64)
    return AdjointTestBank(omega=omega, weights=weights)


def weak_yang_mills_residuals(
    jet: GaugeCovariantJet, bank: AdjointTestBank
) -> np.ndarray:
    r"""``r_i = sum_b w_b eta_nu omega_i^{nu a} (D*F)^{nu a}``."""
    if bank.omega.shape[1:] != jet.ym_eom.shape:
        raise ValueError(
            f"test 1-form batch/shape {bank.omega.shape[1:]} != "
            f"ym_eom {jet.ym_eom.shape}"
        )
    if bank.weights.shape != (jet.batch,):
        raise ValueError(
            f"weights must have shape {(jet.batch,)}, got {bank.weights.shape}"
        )
    eta = np.asarray(jet.signature, dtype=np.float64)
    return np.einsum(
        "b,n,Tbna,bna->T", bank.weights, eta, bank.omega, jet.ym_eom
    )


def evaluate_weak_ym_identity(
    jet: GaugeCovariantJet,
    points: np.ndarray,
    *,
    atol: float = WEAK_YM_FLOOR,
    n_tests: int = 8,
    rng: np.random.Generator | None = None,
    bank: AdjointTestBank | None = None,
) -> dict[str, Any]:
    """Fail-closed weak EOM check. Not a continuum or mass-gap claim."""
    gen = rng if rng is not None else np.random.default_rng(0)
    tests = (
        bank
        if bank is not None
        else gaussian_adjoint_test_bank(
            points, n_tests=n_tests, algebra=jet.algebra, rng=gen
        )
    )
    residuals = weak_yang_mills_residuals(jet, tests)
    max_abs = float(np.max(np.abs(residuals)))
    return {
        "passed": bool(max_abs <= atol),
        "max_abs": max_abs,
        "n_tests": int(len(tests)),
        "atol": float(atol),
        "yang_mills_claim": False,
        "continuum_claim": False,
    }


__all__ = [
    "WEAK_YM_FLOOR",
    "AdjointTestBank",
    "AdjointTestOneForm",
    "evaluate_weak_ym_identity",
    "gaussian_adjoint_test_bank",
    "weak_yang_mills_residuals",
]

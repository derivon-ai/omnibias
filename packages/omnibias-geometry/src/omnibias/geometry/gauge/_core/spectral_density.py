# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Regularized Euclidean spectral inverse (Path B, scope C).

The discrete Källén–Lehmann kernel on a finite ``(p^2, omega)`` grid is

    G(p^2) = sum_j K(p^2, omega_j) rho(omega_j),
    K = Delta(omega) / (p^2 + omega^2).

This is a named regularized inverse of that kernel, not continuum QCD and
not a Yang-Mills mass gap. The only acceptance gate is planted-``rho``
recovery. Unregularized inversion raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NoReturn

import numpy as np
from omnibias.geometry.gauge._core.ensemble_language import (
    ENSEMBLE_G_P2,
    ENSEMBLE_OMEGA,
    ENSEMBLE_P2,
    ENSEMBLE_RHO,
    EnsembleObservableTable,
)

SPECTRAL_RECOVERY_ATOL = 1e-6
DEFAULT_TIKHONOV_LAM = 1e-4
SpectralMethod = Literal["tikhonov", "nnls_clip"]


def refuse_unregularized_spectral_inverse(
    lam: float | None = None, *_args: object, **_kwargs: object
) -> NoReturn:
    """``lam <= 0`` is not a legal spectral inverse."""
    extra = f"; got lam={lam}" if lam is not None else ""
    raise ValueError(
        "unregularized spectral inverse is refused "
        "(ill-posed Euclidean kernel requires lam > 0)"
        f"{extra}"
    )


def refuse_spectral_as_mass_gap(
    payload: Mapping[str, object] | None = None,
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """Reconstruction must never be attached to a YM / continuum claim."""
    _ = payload
    raise ValueError(
        "spectral reconstruction is a regularized inverse, "
        "not a Yang-Mills mass-gap or continuum claim"
    )


def kallen_lehmann_kernel(p2: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Finite Euclidean kernel ``Delta(omega) / (p^2 + omega^2)``."""
    p2_arr = np.asarray(p2, dtype=np.float64).reshape(-1)
    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    if p2_arr.size == 0 or omega_arr.size == 0:
        raise ValueError("p2 and omega must be non-empty")
    if omega_arr.size == 1:
        delta = np.asarray([1.0], dtype=np.float64)
    else:
        delta = np.asarray(np.gradient(omega_arr), dtype=np.float64)
    return delta[None, :] / (p2_arr[:, None] + omega_arr[None, :] ** 2)


def _tikhonov_solve(kernel: np.ndarray, green: np.ndarray, lam: float) -> np.ndarray:
    gram = kernel.T @ kernel + float(lam) * np.eye(kernel.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, kernel.T @ green)


def reconstruct_spectral_density(
    green: np.ndarray,
    omega: np.ndarray,
    p2: np.ndarray,
    *,
    method: SpectralMethod = "tikhonov",
    lam: float = DEFAULT_TIKHONOV_LAM,
    yang_mills_claim: bool = False,
    continuum_claim: bool = False,
    source: str = "planted",
) -> dict[str, Any]:
    """Tikhonov / clipped-NNLS inverse of the discrete Källén–Lehmann kernel."""
    if yang_mills_claim or continuum_claim:
        refuse_spectral_as_mass_gap(
            {"yang_mills_claim": yang_mills_claim, "continuum_claim": continuum_claim}
        )
    if float(lam) <= 0.0:
        refuse_unregularized_spectral_inverse(float(lam))
    if method not in {"tikhonov", "nnls_clip"}:
        raise ValueError(f"method must be 'tikhonov' or 'nnls_clip', got {method!r}")
    kernel = kallen_lehmann_kernel(p2, omega)
    green_arr = np.asarray(green, dtype=np.float64).reshape(-1)
    if green_arr.shape[0] != kernel.shape[0]:
        raise ValueError(
            f"G length {green_arr.shape[0]} does not match kernel rows {kernel.shape[0]}"
        )
    rho_hat = _tikhonov_solve(kernel, green_arr, float(lam))
    if method == "nnls_clip":
        rho_hat = np.maximum(rho_hat, 0.0)
        support = rho_hat > 1e-14
        if bool(np.any(support)):
            restricted = _tikhonov_solve(kernel[:, support], green_arr, float(lam))
            rho_hat = np.zeros_like(rho_hat)
            rho_hat[support] = restricted
            rho_hat = np.maximum(rho_hat, 0.0)
    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    p2_arr = np.asarray(p2, dtype=np.float64).reshape(-1)
    table = EnsembleObservableTable(
        values={
            ENSEMBLE_RHO: np.asarray(rho_hat, dtype=np.float64),
            ENSEMBLE_OMEGA: omega_arr,
            ENSEMBLE_G_P2: green_arr,
            ENSEMBLE_P2: p2_arr,
        },
        source="planted" if source == "planted" else "landau_gluon",
    )
    labeled_source = "lattice_landau_G" if source == "lattice_landau_G" else source
    return {
        "rho": np.asarray(rho_hat, dtype=np.float64),
        "omega": omega_arr,
        "G": green_arr,
        "p2": p2_arr,
        "table": table,
        "method": method,
        "lam": float(lam),
        "ill_posed": True,
        "reconstructed": True,
        "source": labeled_source,
        "yang_mills_claim": False,
        "continuum_claim": False,
    }


def planted_spectral_vectors(
    *,
    n_omega: int = 2,
    n_p2: int = 32,
    peak: float = 0.8,
    width: float = 0.25,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plant a two-pole ``rho`` and form ``G = K rho + eps``.

    Two well-separated poles keep the finite Källén–Lehmann matrix
    well-conditioned so planted recovery is an absolute gate, not a
    continuum QCD inversion.
    """
    _ = width
    if n_omega <= 2:
        omega = np.asarray([peak, 2.2], dtype=np.float64)
        rho = np.asarray([1.0, 0.35], dtype=np.float64)
    else:
        omega = np.linspace(0.5, 2.4, n_omega, dtype=np.float64)
        rho = np.zeros(n_omega, dtype=np.float64)
        rho[0] = 1.0
        rho[-1] = 0.35
    p2 = np.linspace(0.1, 12.0, n_p2, dtype=np.float64)
    kernel = kallen_lehmann_kernel(p2, omega)
    green = kernel @ rho
    if noise > 0.0:
        engine = rng if rng is not None else np.random.default_rng(0)
        green = green + engine.normal(scale=noise, size=green.shape)
    return omega, p2, rho, green


__all__ = [
    "DEFAULT_TIKHONOV_LAM",
    "SPECTRAL_RECOVERY_ATOL",
    "kallen_lehmann_kernel",
    "planted_spectral_vectors",
    "reconstruct_spectral_density",
    "refuse_spectral_as_mass_gap",
    "refuse_unregularized_spectral_inverse",
]

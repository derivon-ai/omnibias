# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Shared, pure math for the fractional / hyperdissipative Navier-Stokes track.

Spectral fractional operators on the periodic torus, exact solutions of the
fractional NS equations, the 3D criticality ladder (``alpha_c = 5/4``), the
learnable-order recovery core, and Tao's logarithmically supercritical
dissipation. Every routine here is honest about what it certifies: numerical
evidence and *external* analytic theorems (Lions 1969; Tao 2009), never a global-regularity
result. No function sets ``unproven_claim`` to anything but ``False``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi
CRITICAL_ALPHA_3D = 1.25  # (n + 2) / 4 with n = 3
LIONS_CITATION = "J.-L. Lions (1969); Katz-Pavlovic (2002)"
TAO_CITATION = "T. Tao (2009), 'Global regularity for a logarithmically supercritical hyperdissipative NS'"


# --------------------------------------------------------------------------- #
# Spectral operators (exact for band-limited periodic fields on a cubic grid). #
# --------------------------------------------------------------------------- #
def wavenumbers(n: int, length: float = TWO_PI) -> np.ndarray:
    return np.fft.fftfreq(n, d=length / n) * TWO_PI


def frac_laplacian(f: np.ndarray, alpha: float, length: float = TWO_PI) -> np.ndarray:
    r"""Isotropic fractional Laplacian ``(-\Delta)^alpha f`` (symbol ``|k|^{2 alpha}``)."""
    d = f.ndim
    n = f.shape[0]
    k = wavenumbers(n, length)
    ks = np.meshgrid(*([k] * d), indexing="ij")
    k2 = sum(kk**2 for kk in ks)
    sym = np.zeros_like(k2)
    nz = k2 > 0
    sym[nz] = k2[nz] ** alpha
    return np.real(np.fft.ifftn(sym * np.fft.fftn(f)))


def spectral_grad(f: np.ndarray, length: float = TWO_PI) -> list[np.ndarray]:
    d = f.ndim
    n = f.shape[0]
    k = wavenumbers(n, length)
    ks = np.meshgrid(*([k] * d), indexing="ij")
    fhat = np.fft.fftn(f)
    return [np.real(np.fft.ifftn(1j * ks[i] * fhat)) for i in range(d)]


def fractional_ns_residual(
    u: np.ndarray,
    p: np.ndarray,
    u_t: np.ndarray,
    *,
    alpha: float,
    nu: float,
    forcing: np.ndarray | None = None,
    length: float = TWO_PI,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generic spectral residual of ``u_t + (u.grad)u + grad p + nu(-Delta)^a u - f``."""
    d = u.shape[0]
    grads = [spectral_grad(u[i], length) for i in range(d)]
    adv = np.stack([sum(u[j] * grads[i][j] for j in range(d)) for i in range(d)])
    grad_p = np.stack(spectral_grad(p, length))
    frac = np.stack([frac_laplacian(u[i], alpha, length) for i in range(d)])
    f = np.zeros_like(u) if forcing is None else forcing
    residual = u_t + adv + grad_p + nu * frac - f
    div = sum(grads[i][i] for i in range(d))
    return residual, div


# --------------------------------------------------------------------------- #
# Exact solutions of the fractional NS equations (for every alpha).           #
# --------------------------------------------------------------------------- #
def _grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = TWO_PI * np.arange(n, dtype=float) / n
    return np.meshgrid(x, x, x, indexing="ij")


def exact_decaying_shear(n: int, t: float, *, m: int, nu: float, alpha: float):
    r"""``u = (exp(-nu m^{2a} t) sin(m y), 0, 0)``, ``p = 0``, ``f = 0``.

    Advection ``(u.grad)u`` and divergence vanish exactly; ``(-Delta)^a u =
    m^{2a} u``. Exact unforced solution for every ``alpha`` with an
    ``alpha``-dependent decay rate ``nu m^{2a}``.
    """
    _, yy, _ = _grid(n)
    rate = nu * (m ** (2.0 * alpha))
    decay = math.exp(-rate * t)
    ux = decay * np.sin(m * yy)
    zeros = np.zeros_like(ux)
    u = np.stack([ux, zeros, zeros])
    u_t = np.stack([-rate * ux, zeros, zeros])
    p = np.zeros((n, n, n))
    return u, p, u_t, rate


def exact_decaying_abc(n: int, t: float, *, nu: float):
    r"""Decaying Beltrami ABC field: a ``(-Delta)^a`` eigenfunction (``|k| = 1``).

    Exact for every ``alpha`` with the ``alpha``-independent rate ``nu``.
    """
    xx, yy, zz = _grid(n)
    decay = math.exp(-nu * t)
    big_u = np.stack([np.sin(zz) + np.cos(yy), np.sin(xx) + np.cos(zz), np.sin(yy) + np.cos(xx)])
    u = decay * big_u
    u_t = -nu * u
    p = -0.5 * (decay**2) * np.sum(big_u * big_u, axis=0)
    return u, p, u_t


def exact_beltrami_shell(n: int, t: float, *, wavenumber: int, nu: float, alpha: float):
    r"""Decaying Beltrami field on the wavenumber shell ``|k| = K`` (``K = wavenumber``).

    ``U_K(x) = U(K x)`` with the ABC generator ``U``; then ``curl U_K = K U_K``
    (Beltrami) and ``(-Delta)^a U_K = K^{2a} U_K``. Advection is a pure pressure
    gradient (``(U_K.grad)U_K = grad(|U_K|^2/2)``), so

        u(x,t) = exp(-nu K^{2a} t) U_K(x),   p = -1/2 exp(-2 nu K^{2a} t) |U_K|^2

    is an **exact** unforced fractional-NS solution whose decay rate
    ``nu K^{2a}`` genuinely depends on ``alpha`` (``K = 1`` recovers the
    ``alpha``-independent ABC flow).
    """
    xx, yy, zz = _grid(n)
    k = int(wavenumber)
    rate = nu * (k ** (2.0 * alpha))
    decay = math.exp(-rate * t)
    big_u = np.stack(
        [
            np.sin(k * zz) + np.cos(k * yy),
            np.sin(k * xx) + np.cos(k * zz),
            np.sin(k * yy) + np.cos(k * xx),
        ]
    )
    u = decay * big_u
    u_t = -rate * u
    p = -0.5 * (decay**2) * np.sum(big_u * big_u, axis=0)
    return u, p, u_t, rate


# --------------------------------------------------------------------------- #
# Criticality ladder (honest proven / open labels).                          #
# --------------------------------------------------------------------------- #
def classify_regime(alpha: float) -> dict:
    r"""Honest regime label for exponent ``alpha`` in 3D (``alpha_c = 5/4``)."""
    if alpha >= CRITICAL_ALPHA_3D:
        status = "proven_global_regularity_external"
        regime = "critical" if abs(alpha - CRITICAL_ALPHA_3D) < 1e-12 else "subcritical"
        source = LIONS_CITATION
    else:
        status = "open"
        regime = "supercritical"
        source = "unresolved"
    return {
        "alpha": alpha,
        "regime": regime,
        "global_regularity_status": status,
        "proof_source": source,
        "omnibias_verified": False,  # external theorem / open problem -- never omnibias-proven
        "is_classical_open_problem": bool(abs(alpha - 1.0) < 1e-12),
        "unproven_claim": False,
    }


# --------------------------------------------------------------------------- #
# Tao's logarithmically supercritical dissipation (the genuine research edge). #
# --------------------------------------------------------------------------- #
def tao_dissipation_symbol(k2: np.ndarray, *, beta: float = 0.25) -> np.ndarray:
    r"""Dissipation symbol ``|k|^{5/2} / g(|k|)^2`` with ``g(r) = (log(e + r^2))^beta``.

    This is *weaker* than the critical hyperdissipation ``|k|^{5/2}`` by a
    logarithmic factor -- "logarithmically supercritical". Tao (2009) proves
    global regularity in 3D iff ``\int^\infty dr /(r g(r)^4) = \infty``, i.e.
    ``4 beta <= 1`` for this ``g``.
    """
    g4 = np.log(np.e + k2) ** (4.0 * beta)  # g(r)^4 = (log(e + r^2))^{4 beta}
    g2 = np.sqrt(g4)
    sym = np.zeros_like(k2, dtype=float)
    nz = k2 > 0
    sym[nz] = (k2[nz] ** 1.25) / g2[nz]
    return sym


def tao_dissipation_symbol_torch(k2: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""Torch twin of :func:`tao_dissipation_symbol`, differentiable in ``beta``.

    Symbol ``|k|^{5/2} / (log(e + |k|^2))^{2 beta} = exp(1.25 log k2 - 2 beta
    log log(e + k2))``. A tensor ``beta`` backpropagates (learnable exponent).
    """
    nz = k2 > 0
    safe = torch.where(nz, k2, torch.ones_like(k2))
    logfac = torch.log(math.e + k2)
    sym = torch.where(
        nz,
        torch.exp(1.25 * torch.log(safe) - 2.0 * beta * torch.log(logfac)),
        torch.zeros_like(k2),
    )
    return sym


def log_supercritical_rate(m: int, *, nu: float, beta: float) -> float:
    r"""Dissipation rate ``nu |m|^{5/2} / (log(e + m^2))^{2 beta}`` for mode ``m``."""
    k2 = float(m * m)
    return nu * (k2**1.25) / (math.log(math.e + k2) ** (2.0 * beta))


def exact_decaying_shear_log(n: int, t: float, *, m: int, nu: float, beta: float):
    r"""Shear ``u = (exp(-r t) sin(m y), 0, 0)`` under log-supercritical dissipation.

    The rate ``r = nu m^{5/2} / (log(e + m^2))^{2 beta}`` is the Tao operator's
    eigenvalue on mode ``m``; advection and divergence vanish, so this is an
    exact unforced solution of the (linear) log-supercritical shear dynamics.
    """
    _, yy, _ = _grid(n)
    rate = log_supercritical_rate(m, nu=nu, beta=beta)
    decay = math.exp(-rate * t)
    ux = decay * np.sin(m * yy)
    zeros = np.zeros_like(ux)
    u = np.stack([ux, zeros, zeros])
    u_t = np.stack([-rate * ux, zeros, zeros])
    p = np.zeros((n, n, n))
    return u, p, u_t, rate


def tao_log_supercritical_diagnostic(beta: float, *, r_max: float = 1e8, samples: int = 200000) -> dict:
    r"""Evaluate Tao's divergence condition ``\int_1^\infty dr/(r g(r)^4)``.

    Returns the analytic verdict (``4 beta <= 1``) plus a numerical partial
    integral to ``r_max`` as corroborating evidence (the integral diverges like
    ``log log r`` on the borderline, so it grows slowly but without bound).
    """
    diverges_analytic = bool(4.0 * beta <= 1.0 + 1e-12)
    r = np.logspace(0.0, math.log10(r_max), samples)
    g4 = np.log(np.e + r**2) ** (4.0 * beta)
    integrand = 1.0 / (r * g4)
    partial = float(np.trapezoid(integrand, r)) if hasattr(np, "trapezoid") else float(np.trapz(integrand, r))
    return {
        "beta": beta,
        "g_exponent_note": "g(r) = (log(e + r^2))^beta ; condition int dr/(r g^4) = inf  <=>  4 beta <= 1",
        "diverges_analytic": diverges_analytic,
        "tao_global_regularity_applies": diverges_analytic,
        "partial_integral_to_r_max": partial,
        "r_max": float(r_max),
        "omnibias_verified": False,  # Tao's theorem is external; omnibias records it only
        "unproven_claim": False,
    }


def classify_log_supercritical(beta: float) -> dict:
    diag = tao_log_supercritical_diagnostic(beta)
    return {
        "family": "logarithmically_supercritical_hyperdissipation",
        "dissipation": "|k|^{5/2} / (log(e + |k|^2))^{2 beta}",
        "beta": beta,
        "just_below_critical": True,  # strictly weaker than |k|^{5/2} for any beta > 0
        "global_regularity_status": (
            "proven_global_regularity_external" if diag["tao_global_regularity_applies"] else "open"
        ),
        "proof_source": TAO_CITATION if diag["tao_global_regularity_applies"] else "unresolved",
        "divergence_condition_met": diag["tao_global_regularity_applies"],
        "omnibias_verified": False,
        "unproven_claim": False,
    }


# --------------------------------------------------------------------------- #
# Learnable fractional order (differentiable in the |k|^{2a} multiplier).      #
# --------------------------------------------------------------------------- #
def frac_laplacian_torch(f: torch.Tensor, alpha: torch.Tensor, length: float = TWO_PI) -> torch.Tensor:
    r"""1D ``(-\Delta)^alpha`` with a differentiable (tensor) order ``alpha``."""
    n = f.shape[0]
    k = torch.fft.fftfreq(n, d=length / n, dtype=torch.float64) * TWO_PI
    k2 = k * k
    nz = k2 > 0
    safe = torch.where(nz, k2, torch.ones_like(k2))
    sym = torch.where(nz, torch.exp(alpha * torch.log(safe)), torch.zeros_like(k2))
    fhat = torch.fft.fft(f.to(torch.complex128))
    return torch.real(torch.fft.ifft(sym.to(torch.complex128) * fhat))


# --------------------------------------------------------------------------- #
# Torch spectral operators on a full periodic grid (for the 3D fractional PINN).#
# The field u = curl(A) is band-limited, so the FFT (-Delta)^alpha is exact and #
# differentiable in both the field parameters and (a tensor) alpha.            #
# --------------------------------------------------------------------------- #
def _k2_grid_torch(n: int, d: int, device: torch.device, length: float = TWO_PI):
    k = torch.fft.fftfreq(n, d=length / n, device=device, dtype=torch.float64) * TWO_PI
    ks = torch.meshgrid(*([k] * d), indexing="ij")
    k2 = sum(kk * kk for kk in ks)
    return ks, k2


def frac_laplacian_nd_torch(
    f: torch.Tensor, alpha: torch.Tensor | float, length: float = TWO_PI
) -> torch.Tensor:
    r"""Isotropic ``(-\Delta)^alpha`` on a ``d``-dim periodic grid via the FFT."""
    d = f.ndim
    n = f.shape[-1]
    _, k2 = _k2_grid_torch(n, d, f.device, length)
    nz = k2 > 0
    safe = torch.where(nz, k2, torch.ones_like(k2))
    a = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(float(alpha), dtype=torch.float64)
    sym = torch.where(nz, torch.exp(a * torch.log(safe)), torch.zeros_like(k2))
    fhat = torch.fft.fftn(f.to(torch.complex128))
    return torch.real(torch.fft.ifftn(sym.to(torch.complex128) * fhat))


def spectral_grad_nd_torch(f: torch.Tensor, length: float = TWO_PI) -> list[torch.Tensor]:
    r"""Spectral gradient of a ``d``-dim periodic field via the FFT."""
    d = f.ndim
    n = f.shape[-1]
    ks, _ = _k2_grid_torch(n, d, f.device, length)
    fhat = torch.fft.fftn(f.to(torch.complex128))
    return [torch.real(torch.fft.ifftn(1j * ks[i].to(torch.complex128) * fhat)) for i in range(d)]


def fractional_ns_residual_torch(
    u: torch.Tensor,
    p: torch.Tensor,
    u_t: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    nu: float,
    forcing: torch.Tensor | None = None,
    length: float = TWO_PI,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Torch twin of :func:`fractional_ns_residual` on a periodic ``(d, N, ..., N)`` grid."""
    d = u.shape[0]
    grads = [spectral_grad_nd_torch(u[i], length) for i in range(d)]
    adv = torch.stack([sum(u[j] * grads[i][j] for j in range(d)) for i in range(d)])
    grad_p = torch.stack(spectral_grad_nd_torch(p, length))
    frac = torch.stack([frac_laplacian_nd_torch(u[i], alpha, length) for i in range(d)])
    f = torch.zeros_like(u) if forcing is None else forcing
    residual = u_t + adv + grad_p + nu * frac - f
    div = sum(grads[i][i] for i in range(d))
    return residual, div

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Dissipative CCF residual with fractional Laplacian (jax).

Adds a spectral fractional dissipation term ``(+(-Delta)^{alpha/2} Theta)`` to
the self-similar CCF residual. ``alpha`` may be a fixed float or a
differentiable jax scalar (learnable order). Honesty: spectral fractional
operator is numerical / non-local — **not** the closed-form sigma tower; this
is **not** a Clay Navier-Stokes claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.jax.equations.ccf_compactified import (
    ccf_compactified_residual_samples,
)
from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import ccf_residual_samples


def spectral_fractional_laplacian_1d(
    values: Array,
    *,
    alpha: Array | float,
    length: float,
) -> Array:
    r"""Periodic spectral ``(-Delta)^{alpha/2}`` on a uniform 1-D grid.

    Multiplier ``|2 pi k / L|^{alpha}`` on FFT modes (k integer).
    When ``alpha`` is (near) zero the operator is the zero map (inviscid limit),
    not the identity — matching the dissipative-CCF embedding
    ``R_inviscid + (-Delta)^{alpha/2} Theta``.
    """
    x = jnp.asarray(values)
    a = jnp.asarray(alpha, dtype=x.dtype)

    def _nonzero(_a: Array) -> Array:
        n = x.shape[-1]
        k = jnp.fft.fftfreq(n) * n
        omega = jnp.abs(2.0 * jnp.pi * k / float(length))
        mult = jnp.power(omega + 0.0, _a)
        mult = mult.at[0].set(0.0)
        return jnp.real(jnp.fft.ifft(jnp.fft.fft(x) * mult))

    return jnp.where(jnp.abs(a) < 1e-14, jnp.zeros_like(x), _nonzero(a))


def ccf_dissipative_residual_samples(
    y: Array,
    theta: Array,
    theta_y: Array,
    lam: Array | float,
    alpha: Array | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    domain: str = "periodic",
    length: float | None = None,
) -> Array:
    """CCF residual plus ``+ (-Delta)^{alpha/2} Theta``.

    Parameters
    ----------
    domain
        ``"periodic"`` uses periodic Hilbert + spectral fractional on the
        sample period; ``"line_compactified"`` uses truncated-line Hilbert and
        fractional on the truncated uniform span.
    """
    y = jnp.asarray(y)
    theta = jnp.asarray(theta)
    if domain == "periodic":
        base = ccf_residual_samples(
            y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign
        )
        L = float(length) if length is not None else float(y[-1] - y[0] + (y[1] - y[0]))
        diss = spectral_fractional_laplacian_1d(theta, alpha=alpha, length=L)
        return base + diss
    if domain == "line_compactified":
        eq, _factored, _w = ccf_compactified_residual_samples(
            y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign
        )
        # resample to uniform for spectral fractional
        order = jnp.argsort(y)
        y_s = y[order]
        th_s = theta[order]
        n = int(y_s.shape[0])
        y_u = jnp.linspace(y_s[0], y_s[-1], n)
        th_u = jnp.interp(y_u, y_s, th_s)
        L = float(y_u[-1] - y_u[0])
        diss_u = spectral_fractional_laplacian_1d(th_u, alpha=alpha, length=max(L, 1e-12))
        diss_s = jnp.interp(y_s, y_u, diss_u)
        diss = jnp.empty_like(theta).at[order].set(diss_s)
        return eq + diss
    raise ValueError(f"unknown domain {domain!r}")


@dataclass
class CordobaCordobaFontelosDissipative:
    """Callable residual helper for dissipative CCF on sample arrays."""

    lam: float = 0.6057
    alpha: float = 0.0
    form: str = "transport"
    velocity_sign: float = 1.0
    domain: str = "periodic"

    def residual(
        self,
        y: Array,
        theta: Array,
        theta_y: Array,
        *,
        alpha: Array | float | None = None,
    ) -> Array:
        a = self.alpha if alpha is None else alpha
        return ccf_dissipative_residual_samples(
            y,
            theta,
            theta_y,
            self.lam,
            a,
            form=self.form,
            velocity_sign=self.velocity_sign,
            domain=self.domain,
        )


__all__ = [
    "CordobaCordobaFontelosDissipative",
    "ccf_dissipative_residual_samples",
    "spectral_fractional_laplacian_1d",
]

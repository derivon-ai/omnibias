# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Periodic spectral Hilbert transform (jax twin).

The Hilbert transform :math:`H` is the nonlocal singular-integral operator at
the heart of the Córdoba-Córdoba-Fontelos (CCF) model and of the vorticity
formulation of the Euler / Navier-Stokes equations. On the line it is

.. math::

    (Hf)(x) = \frac{1}{\pi}\,\mathrm{p.v.}\!\int \frac{f(s)}{x-s}\,ds .

This module implements the **periodic** discrete Hilbert transform via the FFT.
It is a *numerical* nonlocal Fourier multiplier -- it is emphatically **not** an
omnibias closed-form activation derivative. (omnibias contributes the exact
*local* derivatives of the profile; the Hilbert transform is the genuinely
nonlocal piece that no closed-form derivative tower can supply.)

Convention
----------
On the Fourier modes :math:`m` of an equally-spaced sample over one period,

.. math::

    \widehat{Hf}(m) = -i\,\operatorname{sgn}(m)\,\hat f(m),

with the mean (``m = 0``) sent to zero and, for an even number of samples, the
Nyquist mode (``m = N/2``) sent to zero (its sign is ambiguous and its Hilbert
image is not representable on the grid). With this convention

.. math::

    H[\cos(k\,\cdot)] = \sin(k\,\cdot), \qquad
    H[\sin(k\,\cdot)] = -\cos(k\,\cdot), \qquad
    H[\text{const}] = 0 ,

and :math:`H` commutes with differentiation, :math:`H[f'] = (Hf)'`.

The transform is *exact* for trigonometric polynomials sampled on a uniform grid
over one period and is **independent of the period length** (the Hilbert
transform is invariant under positive dilations). Consequently the caller need
only supply the samples *in grid order*; the physical spacing is irrelevant.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def hilbert_transform(values: Array, *, axis: int = -1) -> Array:
    r"""Periodic discrete Hilbert transform of ``values`` along ``axis``.

    Parameters
    ----------
    values
        Real (or complex) samples of a periodic function on a uniform grid
        covering exactly one period, in grid order, along ``axis``.
    axis
        Axis holding the periodic samples. Defaults to the last axis.

    Returns
    -------
    Array
        ``H[values]`` with the same shape and (real) dtype as ``values``.

    Notes
    -----
    Uses the multiplier :math:`-i\,\operatorname{sgn}(m)` with mean and (even-N)
    Nyquist modes zeroed. Exact for band-limited periodic data; period-length
    independent.
    """
    x = jnp.asarray(values)
    n = x.shape[axis]
    if n < 2:
        raise ValueError(f"hilbert_transform needs at least 2 samples, got {n}")
    modes = jnp.fft.fftfreq(n) * n  # integer mode numbers as float
    mult = -1j * jnp.sign(modes)
    if n % 2 == 0:
        mult = mult.at[n // 2].set(0.0)  # Nyquist
    shape = [1] * x.ndim
    shape[axis] = n
    mult = mult.reshape(shape)
    spectrum = jnp.fft.fft(x, axis=axis)
    out = jnp.fft.ifft(spectrum * mult, axis=axis)
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        return out
    return jnp.real(out).astype(x.dtype)


__all__ = ["hilbert_transform"]

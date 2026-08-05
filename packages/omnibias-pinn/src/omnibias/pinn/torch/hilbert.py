# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Periodic spectral Hilbert transform (torch twin).

Bit-parity companion of :mod:`omnibias.pinn.jax.hilbert`. See that module for
the full mathematical contract. In short: this is the **periodic discrete
Hilbert transform** via the FFT, a numerical nonlocal Fourier multiplier
(*not* an omnibias closed-form activation derivative), with convention

.. math::

    \widehat{Hf}(m) = -i\,\operatorname{sgn}(m)\,\hat f(m),

so that :math:`H[\cos] = \sin`, :math:`H[\sin] = -\cos`, :math:`H[\text{const}]
= 0`, the mean and (even-N) Nyquist modes are zeroed, and the transform is exact
for band-limited periodic data and independent of the period length.
"""

from __future__ import annotations

import torch
from torch import Tensor


def hilbert_transform(values: Tensor, *, axis: int = -1) -> Tensor:
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
    Tensor
        ``H[values]`` with the same shape and (real) dtype as ``values``.
    """
    x = values
    n = x.shape[axis]
    if n < 2:
        raise ValueError(f"hilbert_transform needs at least 2 samples, got {n}")
    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    modes = torch.fft.fftfreq(n, dtype=real_dtype, device=x.device) * n
    mult = torch.complex(torch.zeros_like(modes), -torch.sign(modes))  # -i*sgn(m)
    if n % 2 == 0:
        mult[n // 2] = 0.0  # Nyquist
    shape = [1] * x.dim()
    shape[axis] = n
    mult = mult.reshape(shape)
    spectrum = torch.fft.fft(x, dim=axis)
    out = torch.fft.ifft(spectrum * mult, dim=axis)
    if x.is_complex():
        return out
    return out.real.to(x.dtype)


__all__ = ["hilbert_transform"]

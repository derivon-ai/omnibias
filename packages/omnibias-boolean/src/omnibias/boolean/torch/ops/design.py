# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Boolean-function design losses (torch).

Spectral objectives for *shaping* a Boolean function during optimization, all
built on the differentiable :mod:`~omnibias.boolean.torch.ops.spectrum` engine:

* :func:`degree_penalty` -- penalize high-order spectral energy (push toward
  low-degree / low-complexity functions, e.g. for noise-stable or
  easy-to-learn targets);
* :func:`influence_penalty` -- penalize total influence (average sensitivity);
* :func:`target_spectrum_loss` -- match a prescribed Walsh or Mobius spectrum
  (spectral design / function approximation).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.boolean.torch.ops.spectrum import influences_diff, walsh_coeffs
from torch import Tensor


def degree_penalty(values: Sequence[float] | Tensor) -> Tensor:
    """``sum_S |S| * hat f(S)^2`` -- spectral energy weighted by monomial order."""
    w = walsh_coeffs(values)
    n = w.shape[0].bit_length() - 1
    orders = torch.tensor(
        [bin(mask).count("1") for mask in range(1 << n)],
        dtype=w.dtype,
        device=w.device,
    )
    return (orders * w**2).sum()


def influence_penalty(values: Sequence[float] | Tensor) -> Tensor:
    """Total influence ``sum_i Inf_i`` (average sensitivity), differentiable."""
    return influences_diff(values).sum()


def target_spectrum_loss(
    values: Sequence[float] | Tensor,
    target: Sequence[float] | Tensor,
    basis: str = "walsh",
) -> Tensor:
    """Mean-squared error between the function's spectrum and a target spectrum.

    ``basis`` is ``"walsh"`` (``hat f(S)``) or ``"mobius"`` (``m_S``); ``target`` is
    indexed by subset mask, matching the spectrum-engine outputs.
    """
    if basis == "walsh":
        coeffs = walsh_coeffs(values)
    elif basis == "mobius":
        from omnibias.boolean.torch.ops.spectrum import mobius_coeffs

        coeffs = mobius_coeffs(values)
    else:
        raise ValueError(f"basis must be 'walsh' or 'mobius', got {basis!r}")
    target_t = torch.as_tensor(target, dtype=coeffs.dtype, device=coeffs.device)
    return ((coeffs - target_t) ** 2).mean()


__all__ = [
    "degree_penalty",
    "influence_penalty",
    "target_spectrum_loss",
]

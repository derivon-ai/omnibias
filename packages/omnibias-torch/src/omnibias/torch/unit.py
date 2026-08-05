# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""OperatorMultiBiasUnit (OMBU): the trainable scalar-operator primitive.

A K-bias multi-bias unit applied per channel:

    f_K(z; b, s) = sum_{k=1}^{K} s_k * sigma(z + b_k)

with K learnable biases ``b_1..b_K`` and K signs ``s_1..s_K`` per channel,
on top of a fixed base activation ``sigma``.

Two operating regimes:

1. **Identity-nested (Lemma 1).** With tied biases and signs summing to
   one, the unit reduces bit-identically to ``sigma(z + b)``. This is the
   default initialisation for any K, so a freshly-instantiated OMBU is a
   drop-in replacement for ``sigma`` at step zero.
2. **Bias-collapse derivative tower (Lemma collapse).** With biases on
   the forward-difference (or central-difference) stencil and rescaled
   signs, the unit converges to ``sigma^(K-1)(z + b_mean)`` as the bias
   spread tends to zero. The closed-form derivative is available via
   :meth:`analytic_derivative`, which uses the per-activation fast-path
   kernel (Eulerian polynomial for sigmoid, Hermite for Gaussian, etc.).

OMBU keeps biases and signs as free parameters; the operator-typed
:class:`omnibias.blocks.OperatorBlock` is the wrapper that picks K and
calls either the literal forward or :meth:`analytic_derivative` depending
on the operator role.

Terminology: this ``delta -> 0`` limit is *the* **founding bias collapse**
(many biases collapse onto one value, yielding a derivative). It is distinct
from **temperature collapse**, the ``beta -> inf`` feasibility penalty in
:mod:`omnibias.convex` (and ``omnibias.control`` / ``omnibias.routing``), which
sharpens one constraint into a 0/1 feasibility step -- an indicator, not a derivative. See
``docs/theory.md``.
"""

from __future__ import annotations

from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.fastpath.dispatch import (
    multibias_integral_forward,
    multibias_integral_window_forward,
    multibias_literal_forward,
)
from omnibias.torch.identity_init import identity_init_biases, identity_init_signs
from omnibias.torch.stencil import central_bias_offsets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OperatorMultiBiasUnit(nn.Module):
    """The trainable scalar-operator primitive.

    Parameters
    ----------
    num_channels : int
        Number of independent channels. Each channel has its own K biases
        (and optionally its own K signs).
    K : int, default 2
        Number of bias terms per channel. ``K=1`` recovers the standard
        single-bias activation.
    base : str or :class:`ActivationSpec`, default ``"sigmoid"``
        Base activation. May be a string from
        :func:`omnibias.activations.list_activations` or a custom spec.
    init_bias : float, default 0.0
        Initial value for every bias entry (Lemma 1: tied biases recover
        ``sigma(z + init_bias)`` bit-identically).
    init_delta : float, default 0.0
        If non-zero, initialises biases on the central-difference stencil
        ``b + (k - (K+1)/2) * init_delta``. Use ``0.0`` for strict Lemma 1
        identity-nesting; use a small positive value to break the
        symmetry at init for non-monotone shapes.
    learnable_biases : bool, default True
        If False, biases are registered as a buffer (frozen).
    learnable_signs : bool, default False
        If True, signs are a learnable parameter (mb-K-lrn-s in the paper).
        If False, signs are a frozen buffer initialised by
        :func:`identity_init_signs`.
    """

    spec: ActivationSpec[Tensor]
    biases: Tensor
    signs: Tensor

    def __init__(
        self,
        num_channels: int,
        K: int = 2,
        base: str | ActivationSpec[Tensor] = "sigmoid",
        init_bias: float = 0.0,
        init_delta: float = 0.0,
        learnable_biases: bool = True,
        learnable_signs: bool = False,
        biases_init: Tensor | None = None,
        signs_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if init_delta < 0:
            raise ValueError(f"init_delta must be >= 0, got {init_delta}")

        self.num_channels = num_channels
        self.K = K
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)

        if biases_init is None:
            biases_init = identity_init_biases(num_channels, K, bias_value=init_bias)
            if init_delta > 0 and K > 1:
                offsets = central_bias_offsets(K, init_delta).unsqueeze(0)  # (1, K)
                biases_init = biases_init + offsets
        else:
            biases_init = torch.as_tensor(biases_init, dtype=torch.get_default_dtype())
            if biases_init.shape != (num_channels, K):
                raise ValueError(
                    f"biases_init shape {tuple(biases_init.shape)} does not match "
                    f"({num_channels}, {K})."
                )

        if signs_init is None:
            signs_init = identity_init_signs(num_channels, K)
        else:
            signs_init = torch.as_tensor(signs_init, dtype=torch.get_default_dtype())
            if signs_init.shape == (K,):
                signs_init = signs_init.unsqueeze(0).expand(num_channels, K).contiguous()
            elif signs_init.shape != (num_channels, K):
                raise ValueError(
                    f"signs_init shape {tuple(signs_init.shape)} does not match "
                    f"({num_channels}, {K}) (or ({K},) for broadcast)."
                )

        if learnable_biases:
            self.biases = nn.Parameter(biases_init)
        else:
            self.register_buffer("biases", biases_init)

        if learnable_signs:
            self.signs = nn.Parameter(signs_init)
        else:
            self.register_buffer("signs", signs_init)

    # ----- forward passes -----

    def forward(self, z: Tensor) -> Tensor:
        """Literal multi-bias forward.

        Computes ``sum_k s_k * sigma(z + b_k)`` channel-wise.

        ``z`` must have shape ``(..., num_channels)``; the return value
        has the same shape.
        """
        return multibias_literal_forward(z, self.biases, self.signs, self.spec.forward)

    def analytic_derivative(self, z: Tensor, order: int | None = None) -> Tensor:
        """Closed-form ``sigma^(order)(z + bias_mean)`` for activations
        whose derivative tower has a known polynomial form (sigmoid,
        tanh, gaussian, ...).

        ``order`` defaults to ``K - 1`` (the standard bias-collapse limit
        order for this unit).

        Raises
        ------
        NotImplementedError
            If the spec has no fast-path kernel.
        """
        if order is None:
            order = self.K - 1
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if self.spec.fastpath is None:
            raise NotImplementedError(
                f"Activation {self.spec.name!r} has no closed-form derivative kernel; "
                "use forward() instead, or pick a base from the smooth/proximal families."
            )
        b_mean = self.biases.mean(dim=-1)  # (num_channels,)
        out: Tensor = self.spec.fastpath(z + b_mean, order)
        return out

    def analytic_integral(
        self,
        z: Tensor,
        *,
        strict_window: bool = False,
        normalize: bool = False,
        small_width_threshold: float = 1e-4,
    ) -> Tensor:
        """Closed-form sign-weighted antiderivative sum.

        If ``S' = sigma``, the default path evaluates
        ``sum_k s_k S(z + b_k)``. With ``strict_window=True`` and ``K=2``,
        the two stored bias parameters are interpreted as ``(center,
        raw_width)`` and converted to an ordered low-to-high bias window.
        """
        if strict_window:
            if self.K != 2:
                raise ValueError("strict integral windows require K=2.")
            center = self.biases[:, 0]
            width = F.softplus(self.biases[:, 1])
            use_taylor = not isinstance(self.signs, nn.Parameter)
            return multibias_integral_window_forward(
                z,
                center,
                width,
                self.signs,
                self.spec,
                normalize=normalize,
                small_width_threshold=small_width_threshold,
                use_small_width_taylor=use_taylor,
            )
        return multibias_integral_forward(z, self.biases, self.signs, self.spec)

    # ----- introspection helpers -----

    @property
    def bias_spread(self) -> Tensor:
        """Per-channel ``max(b_k) - min(b_k)``; near zero in the collapse regime."""
        with torch.no_grad():
            return self.biases.max(dim=-1).values - self.biases.min(dim=-1).values

    @property
    def is_identity_nested(self) -> bool:
        """True iff biases are tied and signs sum to one (Lemma 1)."""
        with torch.no_grad():
            tied = bool((self.bias_spread == 0).all().item())
            sums_to_one = bool(((self.signs.sum(dim=-1) - 1.0).abs() < 1e-6).all().item())
            return tied and sums_to_one

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, K={self.K}, base={self.spec.name!r}, "
            f"learnable_biases={isinstance(self.biases, nn.Parameter)}, "
            f"learnable_signs={isinstance(self.signs, nn.Parameter)}"
        )

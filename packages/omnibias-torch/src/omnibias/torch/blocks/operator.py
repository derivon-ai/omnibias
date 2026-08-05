# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed wrapper around :class:`OperatorMultiBiasUnit`.

An :class:`OperatorBlock` carries an explicit operator role tag --
``"identity"``, ``"grad"``, ``"laplacian"``, ``"derivative"``, ``"band"``, or
``"integral"`` -- that selects both the multi-bias arity ``K`` and the
forward-pass behaviour:

============  ===  ================================================  ===================================
op            K    forward                                           sign init / typical use
============  ===  ================================================  ===================================
``identity``  1    ``sigma(z + b)``                                  Lemma 1; drop-in for the base sigma
``grad``      2    closed-form ``sigma'(z + b_mean)``                analytic; ignored signs
``laplacian`` 3    closed-form ``sigma''(z + b_mean)``               analytic; ignored signs
``derivative`` n+1  closed-form ``sigma^(n)(z + b_mean)``            analytic; ignored signs
``band``      2    literal ``sigma(z + b_hi) - sigma(z + b_lo)``     fixed oriented signs
``integral``  2    closed-form ``S(z + b_hi) - S(z + b_lo)``, ``S'=sigma`` fixed oriented signs
============  ===  ================================================  ===================================

The ``"grad"``, ``"laplacian"``, and ``"derivative"`` paths use the
activation's fast-path kernel. The ``"integral"`` path uses the activation's
antiderivative kernel. These roles therefore require a base activation whose
:class:`ActivationSpec` provides the matching kernel. Combinations without
one raise a clear ``TypeError`` at construction.
"""

from __future__ import annotations

import math
from typing import Literal

from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.fastpath.dispatch import multibias_literal_forward
from omnibias.torch.identity_init import identity_init_biases
from omnibias.torch.stencil import central_bias_offsets
from omnibias.torch.unit import OperatorMultiBiasUnit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

OpName = Literal["identity", "grad", "laplacian", "derivative", "band", "integral"]


_OP_K: dict[str, int] = {
    "identity": 1,
    "grad": 2,
    "laplacian": 3,
    # "derivative" has dynamic K = derivative_order + 1.
    "band": 2,
    "integral": 2,
}

_VALID_OPS = (*_OP_K.keys(), "derivative")

_OP_DERIVATIVE_ORDER: dict[str, int] = {
    "identity": 0,
    "grad": 1,
    "laplacian": 2,
    # "derivative" has dynamic order.
    "band": 0,
    "integral": 0,
}


def _window_signs(num_channels: int, K: int, dtype: torch.dtype) -> Tensor:
    """Fixed low-to-high window signs.

    For K=2, ``(-1, +1)`` gives ``sigma(z + b_hi) - sigma(z + b_lo)``
    for ``band`` and ``S(z + b_hi) - S(z + b_lo)`` for ``integral``.
    """
    if K != 2:
        raise ValueError("window signs are defined for K=2 operator windows.")
    s = torch.tensor([-1.0, 1.0], dtype=dtype)
    return s.unsqueeze(0).expand(num_channels, K).contiguous()


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        return -20.0
    return math.log(math.expm1(value))


class OperatorBlock(nn.Module):
    """An OMBU with operator-type semantics.

    Parameters
    ----------
    op : ``"identity"`` | ``"grad"`` | ``"laplacian"`` | ``"derivative"`` | ``"band"`` | ``"integral"``
        Operator role; determines K and the forward behaviour.
    base : str or :class:`ActivationSpec`, default ``"gaussian"``
        Base activation. ``"grad"`` / ``"laplacian"`` / ``"derivative"``
        require a derivative fast-path kernel; ``"integral"`` requires an
        antiderivative kernel.
    channels : int, default 1
        Number of independent channels.
    init_delta : float, optional
        Spread between adjacent biases at init. Default is ``0.0`` for
        ``"identity"`` / ``"grad"`` / ``"laplacian"`` (Lemma 1) and
        ``1.0`` for ``"integral"`` (so the integral window is non-empty
        at init).
    learnable_biases : bool, default True
    learnable_signs : bool, default False
        Expert option for ``"band"`` / ``"integral"``. Fixed signs preserve
        clean window semantics; learnable signs turn the operator into a
        learned activation or antiderivative mixture.
    derivative_order : int, optional
        Required for ``op="derivative"``. The block uses ``K = order + 1``
        and evaluates ``sigma^(order)(z + b_mean)`` through the activation
        fast path. Use ``op="grad"`` / ``op="laplacian"`` as aliases for
        orders 1 and 2.
    normalize_integral : bool, default False
        For ``op="integral"``, divide the window area by its positive width.
    integral_small_width : float, default 1e-4
        For strict integral windows, use the midpoint Taylor limit below this
        width to avoid subtractive cancellation.
    """

    op: OpName
    ombu: OperatorMultiBiasUnit

    def __init__(
        self,
        op: OpName,
        base: str | ActivationSpec[Tensor] = "gaussian",
        channels: int = 1,
        init_bias: float = 0.0,
        init_delta: float | None = None,
        learnable_biases: bool = True,
        learnable_signs: bool | None = None,
        derivative_order: int | None = None,
        normalize_integral: bool = False,
        integral_small_width: float = 1e-4,
    ) -> None:
        super().__init__()
        if op not in _VALID_OPS:
            raise ValueError(f"op must be one of {list(_VALID_OPS)}, got {op!r}.")
        self.op = op
        self.normalize_integral = normalize_integral
        self.integral_small_width = integral_small_width
        if derivative_order is not None and op in ("grad", "laplacian"):
            raise ValueError(
                f"derivative_order is only valid with op='derivative'; use op={op!r} without it "
                f"or op='derivative' for arbitrary order."
            )
        if op == "derivative":
            if derivative_order is None:
                raise ValueError("OperatorBlock(op='derivative') requires derivative_order.")
            if derivative_order < 1:
                raise ValueError(f"derivative_order must be >= 1, got {derivative_order}.")
            K = derivative_order + 1
            self._derivative_order = derivative_order
        else:
            K = _OP_K[op]
            self._derivative_order = _OP_DERIVATIVE_ORDER[op]
        spec = base if isinstance(base, ActivationSpec) else get_activation(base)

        # Validate fast-path availability for the analytic-derivative ops.
        if op in ("grad", "laplacian", "derivative"):
            required_order = self._derivative_order
            if spec.fastpath is None:
                raise TypeError(
                    f"OperatorBlock(op={op!r}) requires a base activation with a "
                    f"closed-form derivative kernel; activation {spec.name!r} has none. "
                    "Choose one of: sigmoid, tanh, softplus, gaussian, exp, "
                    "huber, arctan, log1pu2."
                )
            try:
                spec.fastpath(torch.zeros(1), required_order)
            except NotImplementedError as e:
                raise TypeError(
                    f"OperatorBlock(op={op!r}) requires order-{required_order} "
                    f"derivative; activation {spec.name!r} only supports a partial "
                    f"fastpath. ({e})"
                ) from None
        if op == "integral" and spec.integral is None:
            raise TypeError(
                f"OperatorBlock(op='integral') requires a base activation with a "
                f"closed-form integral kernel; activation {spec.name!r} has none."
            )

        # Per-op defaults for init_delta and learnable_signs.
        if init_delta is None:
            init_delta = 1.0 if op in ("band", "integral") else 0.0
        if learnable_signs is None:
            learnable_signs = False

        # Custom init for bias windows. "band" stores literal low/high
        # endpoints; strict "integral" stores (center, raw_width) so the
        # ordered positive width cannot cross during training.
        if op == "band":
            signs_init = _window_signs(channels, K, dtype=torch.get_default_dtype())
            offsets = central_bias_offsets(K, init_delta).unsqueeze(0)
            biases_init = identity_init_biases(channels, K, bias_value=init_bias) + offsets
        elif op == "integral":
            signs_init = _window_signs(channels, K, dtype=torch.get_default_dtype())
            biases_init = torch.empty(channels, K, dtype=torch.get_default_dtype())
            biases_init[:, 0] = init_bias
            biases_init[:, 1] = _inverse_softplus(init_delta)
        else:
            signs_init = None
            biases_init = None  # let OMBU build Lemma-1 init internally

        self.ombu = OperatorMultiBiasUnit(
            num_channels=channels,
            K=K,
            base=spec,
            init_bias=init_bias,
            init_delta=init_delta if op not in ("band", "integral") else 0.0,
            learnable_biases=learnable_biases,
            learnable_signs=learnable_signs,
            biases_init=biases_init,
            signs_init=signs_init,
        )

    @property
    def channels(self) -> int:
        return self.ombu.num_channels

    @property
    def K(self) -> int:
        return self.ombu.K

    @property
    def base_name(self) -> str:
        return self.ombu.spec.name

    @property
    def derivative_order(self) -> int:
        return self._derivative_order

    def forward(self, z: Tensor) -> Tensor:
        """Dispatch to the appropriate forward path for this op type.

        - ``"identity"``: literal forward (== ``sigma(z + b)`` for K=1).
        - ``"grad"`` / ``"laplacian"`` / ``"derivative"``: closed-form analytic derivative.
        - ``"integral"``: closed-form sign-weighted antiderivative.
        """
        if self.op == "identity":
            out: Tensor = self.ombu(z)
            return out
        if self.op == "band":
            out = self.ombu(z)
            return out
        if self.op == "integral":
            return self.ombu.analytic_integral(
                z,
                strict_window=True,
                normalize=self.normalize_integral,
                small_width_threshold=self.integral_small_width,
            )
        return self.ombu.analytic_derivative(z, order=self._derivative_order)

    def literal_forward(self, z: Tensor) -> Tensor:
        """Literal activation-space counterpart for diagnostics.

        For strict ``"integral"`` blocks this returns the corresponding
        activation band over the ordered window endpoints.
        """
        if self.op == "integral":
            center = self.ombu.biases[:, 0]
            width = F.softplus(self.ombu.biases[:, 1])
            half_width = 0.5 * width
            endpoints = torch.stack((center - half_width, center + half_width), dim=-1)
            return multibias_literal_forward(z, endpoints, self.ombu.signs, self.ombu.spec.forward)
        out: Tensor = self.ombu(z)
        return out

    def extra_repr(self) -> str:
        return (
            f"op={self.op!r}, K={self.K}, derivative_order={self.derivative_order}, "
            f"base={self.base_name!r}, channels={self.channels}"
        )


__all__ = ["OpName", "OperatorBlock"]

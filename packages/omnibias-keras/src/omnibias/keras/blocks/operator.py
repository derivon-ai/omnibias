# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed wrapper around the Keras :class:`OperatorMultiBiasUnit`.

An :class:`OperatorBlock` carries an operator role tag --
``"identity"``, ``"grad"``, ``"laplacian"``, ``"derivative"``, ``"band"``,
or ``"integral"`` -- that selects both the multi-bias arity ``K`` and the
forward behaviour. Mirrors :class:`omnibias.torch.blocks.operator.OperatorBlock`.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from omnibias.core.spec import ActivationSpec
from omnibias.keras.activations.registry import get_activation
from omnibias.keras.fastpath.dispatch import multibias_literal_forward
from omnibias.keras.identity_init import identity_init_biases
from omnibias.keras.stencil import central_bias_offsets
from omnibias.keras.unit import OperatorMultiBiasUnit

from keras import layers, ops

OpName = Literal["identity", "grad", "laplacian", "derivative", "band", "integral"]

_OP_K: dict[str, int] = {
    "identity": 1,
    "grad": 2,
    "laplacian": 3,
    "band": 2,
    "integral": 2,
}

_VALID_OPS = (*_OP_K.keys(), "derivative")

_OP_DERIVATIVE_ORDER: dict[str, int] = {
    "identity": 0,
    "grad": 1,
    "laplacian": 2,
    "band": 0,
    "integral": 0,
}


def _window_signs(num_channels: int, K: int, dtype: str) -> np.ndarray:
    if K != 2:
        raise ValueError("window signs are defined for K=2 operator windows.")
    s = np.array([-1.0, 1.0], dtype=dtype)
    return np.broadcast_to(s, (num_channels, K)).copy()


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        return -20.0
    return math.log(math.expm1(value))


class OperatorBlock(layers.Layer):
    """An OMBU with operator-type semantics (Keras backend).

    Derivative-op semantics (important precondition)
    ------------------------------------------------
    The derivative ops (``grad``, ``laplacian``, ``derivative``) evaluate the
    closed-form **mean-bias fast-path** ``sigma^(order)(z + mean(biases))`` (see
    :meth:`omnibias.keras.unit.OperatorMultiBiasUnit.analytic_derivative`), *not*
    the literal finite-difference stencil ``sum_k s_k sigma(z + b_k)`` over the
    individual biases. The two coincide only under the **identity-nesting**
    initialisation -- ``init_delta = 0`` (the block's default for these ops),
    which seeds all ``K`` bias columns equal so the mean *is* the shared bias and
    the closed-form ``sigma^(order)`` at that point is the intended derivative.

    With **untied biases** that have drifted apart during training (or a nonzero
    ``init_delta``), ``mean(biases)`` no longer characterises the stencil and the
    op returns the single-point closed-form derivative *at the mean bias* rather
    than the literal multi-bias stencil. Keep ``init_delta = 0`` (and, if exact
    stencil semantics matter, ``learnable_biases=False``) for the derivative ops;
    the activation-level math stays bit-identical to the torch / jax backends
    under this precondition.

    Parameters
    ----------
    op : one of ``identity / grad / laplacian / derivative / band / integral``
    base : str or :class:`ActivationSpec`, default ``"gaussian"``
    channels : int, default 1
    init_delta : float, optional
        Initial bias spread. Defaults to ``0`` for derivative ops (the
        identity-nesting precondition above) and ``1`` for ``band`` / ``integral``.
        A nonzero value breaks the mean-bias == literal-stencil equivalence for
        the derivative ops.
    learnable_biases : bool, default True
    learnable_signs : bool, optional
    derivative_order : int, optional
        Required for ``op="derivative"`` (block uses ``K = order + 1``).
    normalize_integral : bool, default False
    integral_small_width : float, default 1e-4
    """

    def __init__(
        self,
        op: OpName = "identity",
        base: str | ActivationSpec[Any] = "gaussian",
        channels: int = 1,
        init_bias: float = 0.0,
        init_delta: float | None = None,
        learnable_biases: bool = True,
        learnable_signs: bool | None = None,
        derivative_order: int | None = None,
        normalize_integral: bool = False,
        integral_small_width: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if op not in _VALID_OPS:
            raise ValueError(f"op must be one of {list(_VALID_OPS)}, got {op!r}.")
        self.op = op
        self.normalize_integral = normalize_integral
        self.integral_small_width = integral_small_width
        if derivative_order is not None and op in ("grad", "laplacian"):
            raise ValueError(
                f"derivative_order is only valid with op='derivative'; use op={op!r} without it."
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
        self._base_name = base if isinstance(base, str) else base.name

        if op in ("grad", "laplacian", "derivative"):
            required_order = self._derivative_order
            if spec.fastpath is None:
                raise TypeError(
                    f"OperatorBlock(op={op!r}) requires a base activation with a "
                    f"closed-form derivative kernel; activation {spec.name!r} has none."
                )
            try:
                spec.fastpath(ops.zeros((1,)), required_order)
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

        if init_delta is None:
            init_delta = 1.0 if op in ("band", "integral") else 0.0
        if learnable_signs is None:
            learnable_signs = False
        self.init_delta = init_delta
        self.learnable_biases = learnable_biases
        self.learnable_signs = learnable_signs

        dtype = self.variable_dtype
        if op == "band":
            signs_init = _window_signs(channels, K, dtype=dtype)
            offsets = central_bias_offsets(K, init_delta, dtype=dtype)[None, :]
            biases_init = identity_init_biases(channels, K, bias_value=init_bias, dtype=dtype) + offsets
        elif op == "integral":
            signs_init = _window_signs(channels, K, dtype=dtype)
            biases_init = np.empty((channels, K), dtype=dtype)
            biases_init[:, 0] = init_bias
            biases_init[:, 1] = _inverse_softplus(init_delta)
        else:
            signs_init = None
            biases_init = None

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
        self.built = True

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

    def call(self, z: Any) -> Any:
        """Dispatch to the appropriate forward path for this op type."""
        if self.op in ("identity", "band"):
            return self.ombu(z)
        if self.op == "integral":
            return self.ombu.analytic_integral(
                z,
                strict_window=True,
                normalize=self.normalize_integral,
                small_width_threshold=self.integral_small_width,
            )
        return self.ombu.analytic_derivative(z, order=self._derivative_order)

    def literal_forward(self, z: Any) -> Any:
        """Literal activation-space counterpart for diagnostics."""
        if self.op == "integral":
            center = self.ombu.biases[:, 0]
            width = ops.softplus(self.ombu.biases[:, 1])
            half_width = 0.5 * width
            endpoints = ops.stack((center - half_width, center + half_width), axis=-1)
            return multibias_literal_forward(
                z, endpoints, self.ombu.signs, self.ombu.spec.forward
            )
        return self.ombu(z)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "op": self.op,
                "base": self._base_name,
                "channels": self.channels,
                "init_delta": self.init_delta,
                "learnable_biases": self.learnable_biases,
                "learnable_signs": self.learnable_signs,
                "derivative_order": (
                    self._derivative_order if self.op == "derivative" else None
                ),
                "normalize_integral": self.normalize_integral,
                "integral_small_width": self.integral_small_width,
            }
        )
        return config


__all__ = ["OpName", "OperatorBlock"]

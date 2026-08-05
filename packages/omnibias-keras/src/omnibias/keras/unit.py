# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""OperatorMultiBiasUnit (OMBU) for the Keras backend.

A K-bias multi-bias unit applied per channel:

    f_K(z; b, s) = sum_{k=1}^{K} s_k * sigma(z + b_k)

with K learnable biases and K signs per channel on top of a fixed base
activation ``sigma``. Mirrors :class:`omnibias.torch.unit.OperatorMultiBiasUnit`
as a ``keras.layers.Layer`` written against ``keras.ops``.

Two operating regimes:

1. Identity-nested (Lemma 1): tied biases + signs summing to one reduce
   the unit bit-identically to ``sigma(z + b)`` (the default init).
2. Bias-collapse derivative tower: :meth:`analytic_derivative` evaluates
   the closed-form ``sigma^(order)(z + bias_mean)`` via the activation
   fast-path kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omnibias.core.spec import ActivationSpec
from omnibias.keras.activations.registry import get_activation
from omnibias.keras.fastpath.dispatch import (
    multibias_integral_forward,
    multibias_integral_window_forward,
    multibias_literal_forward,
)
from omnibias.keras.identity_init import identity_init_biases, identity_init_signs
from omnibias.keras.stencil import central_bias_offsets

from keras import layers, ops


class OperatorMultiBiasUnit(layers.Layer):
    """The trainable scalar-operator primitive (Keras backend).

    Parameters
    ----------
    num_channels : int
        Number of independent channels (the last input dim).
    K : int, default 2
        Number of bias terms per channel.
    base : str or :class:`ActivationSpec`, default ``"sigmoid"``
        Base activation, by name or spec.
    init_bias : float, default 0.0
        Initial value for every bias entry (Lemma 1 identity nesting).
    init_delta : float, default 0.0
        If non-zero, place biases on the central-difference stencil.
    learnable_biases : bool, default True
    learnable_signs : bool, default False
    biases_init, signs_init : array-like, optional
        Explicit ``(num_channels, K)`` init values.
    """

    def __init__(
        self,
        num_channels: int,
        K: int = 2,
        base: str | ActivationSpec[Any] = "sigmoid",
        init_bias: float = 0.0,
        init_delta: float = 0.0,
        learnable_biases: bool = True,
        learnable_signs: bool = False,
        biases_init: Any = None,
        signs_init: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if init_delta < 0:
            raise ValueError(f"init_delta must be >= 0, got {init_delta}")

        self.num_channels = num_channels
        self.K = K
        self._base_name = base if isinstance(base, str) else base.name
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        self.init_bias = init_bias
        self.init_delta = init_delta
        self.learnable_biases = learnable_biases
        self.learnable_signs = learnable_signs

        dtype = self.variable_dtype

        if biases_init is None:
            biases_np = identity_init_biases(num_channels, K, bias_value=init_bias, dtype=dtype)
            if init_delta > 0 and K > 1:
                offsets = central_bias_offsets(K, init_delta, dtype=dtype)[None, :]
                biases_np = biases_np + offsets
        else:
            biases_np = np.asarray(biases_init, dtype=dtype)
            if biases_np.shape != (num_channels, K):
                raise ValueError(
                    f"biases_init shape {biases_np.shape} does not match ({num_channels}, {K})."
                )

        if signs_init is None:
            signs_np = identity_init_signs(num_channels, K, dtype=dtype)
        else:
            signs_np = np.asarray(signs_init, dtype=dtype)
            if signs_np.shape == (K,):
                signs_np = np.broadcast_to(signs_np, (num_channels, K)).copy()
            elif signs_np.shape != (num_channels, K):
                raise ValueError(
                    f"signs_init shape {signs_np.shape} does not match "
                    f"({num_channels}, {K}) (or ({K},) for broadcast)."
                )

        self.biases = self.add_weight(
            name="biases",
            shape=(num_channels, K),
            initializer="zeros",
            trainable=learnable_biases,
            dtype=dtype,
        )
        self.signs = self.add_weight(
            name="signs",
            shape=(num_channels, K),
            initializer="zeros",
            trainable=learnable_signs,
            dtype=dtype,
        )
        self.biases.assign(biases_np)
        self.signs.assign(signs_np)
        self.built = True

    # ----- forward passes -----

    def call(self, z: Any) -> Any:
        """Literal multi-bias forward ``sum_k s_k * sigma(z + b_k)``."""
        return multibias_literal_forward(z, self.biases, self.signs, self.spec.forward)

    def analytic_derivative(self, z: Any, order: int | None = None) -> Any:
        """Closed-form ``sigma^(order)(z + bias_mean)`` (defaults order ``K-1``).

        This is the **mean-bias fast-path**: it evaluates the closed-form
        ``order``-th derivative once, at the *mean* of the bias columns. It
        equals the literal multi-bias stencil ``sum_k s_k sigma(z + b_k)`` only
        under the identity-nesting precondition (``init_delta = 0`` so all biases
        share a value). For untied biases that have drifted apart it is the
        single-point derivative at ``mean(biases)``, not the literal stencil --
        see :class:`omnibias.keras.blocks.operator.OperatorBlock`.
        """
        if order is None:
            order = self.K - 1
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if self.spec.fastpath is None:
            raise NotImplementedError(
                f"Activation {self.spec.name!r} has no closed-form derivative kernel."
            )
        b_mean = ops.mean(self.biases, axis=-1)
        return self.spec.fastpath(z + b_mean, order)

    def analytic_integral(
        self,
        z: Any,
        *,
        strict_window: bool = False,
        normalize: bool = False,
        small_width_threshold: float = 1e-4,
    ) -> Any:
        """Closed-form sign-weighted antiderivative sum."""
        if strict_window:
            if self.K != 2:
                raise ValueError("strict integral windows require K=2.")
            center = self.biases[:, 0]
            width = ops.softplus(self.biases[:, 1])
            use_taylor = not self.learnable_signs
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

    # ----- introspection -----

    @property
    def bias_spread(self) -> Any:
        return ops.max(self.biases, axis=-1) - ops.min(self.biases, axis=-1)

    @property
    def is_identity_nested(self) -> bool:
        biases = np.asarray(ops.convert_to_numpy(self.biases))
        signs = np.asarray(ops.convert_to_numpy(self.signs))
        tied = bool(np.all(biases.max(axis=-1) - biases.min(axis=-1) == 0))
        sums_to_one = bool(np.all(np.abs(signs.sum(axis=-1) - 1.0) < 1e-6))
        return tied and sums_to_one

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "K": self.K,
                "base": self._base_name,
                "init_bias": self.init_bias,
                "init_delta": self.init_delta,
                "learnable_biases": self.learnable_biases,
                "learnable_signs": self.learnable_signs,
            }
        )
        return config


__all__ = ["OperatorMultiBiasUnit"]

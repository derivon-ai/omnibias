# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""GrowableOperatorMultiBiasUnit (Keras backend).

An OMBU whose bias arity ``K`` can grow during training without changing
its output shape. Mirrors
:class:`omnibias.torch.growable.GrowableOperatorMultiBiasUnit`.

Two growth strategies preserve the unit's output at the moment of growth:

- ``"pair"`` (default; any base activation). Adds two biases at an
  existing location with signs ``(+eta, -eta)`` (sum zero, output
  unchanged).
- ``"saturate"`` (only for activations that vanish as ``z -> -inf``:
  sigmoid, softplus, gaussian, exp, relu, huber). Adds one bias at
  ``-big`` so the new term contributes ~0.

Biases / signs are pre-allocated at ``K_max``; the forward only sums over
the first ``active_K`` columns.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from omnibias.core.spec import ActivationSpec
from omnibias.keras.activations.registry import get_activation
from omnibias.keras.fastpath.dispatch import multibias_literal_forward
from omnibias.keras.identity_init import identity_signs

from keras import layers, ops

GrowStrategy = Literal["pair", "saturate"]

_SATURATE_FRIENDLY: frozenset[str] = frozenset(
    {"sigmoid", "softplus", "gaussian", "exp", "relu", "huber"}
)


class GrowableOperatorMultiBiasUnit(layers.Layer):
    """A multi-bias unit whose arity ``K`` can grow during training.

    Parameters
    ----------
    num_channels : int
    init_K : int, default 1
    K_max : int, default 8
    base : str or :class:`ActivationSpec`, default ``"sigmoid"``
    init_bias : float, default 0.0
    learnable_biases : bool, default True
    learnable_signs : bool, default True
    pair_eta : float, default 0.05
    saturate_big : float, default 25.0
    """

    def __init__(
        self,
        num_channels: int,
        init_K: int = 1,
        K_max: int = 8,
        base: str | ActivationSpec[Any] = "sigmoid",
        init_bias: float = 0.0,
        learnable_biases: bool = True,
        learnable_signs: bool = True,
        pair_eta: float = 0.05,
        saturate_big: float = 25.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if K_max < 1:
            raise ValueError(f"K_max must be >= 1, got {K_max}")
        if not (1 <= init_K <= K_max):
            raise ValueError(f"init_K must be in [1, K_max={K_max}], got {init_K}")
        if pair_eta <= 0:
            raise ValueError(f"pair_eta must be > 0, got {pair_eta}")
        if saturate_big <= 0:
            raise ValueError(f"saturate_big must be > 0, got {saturate_big}")

        self.num_channels = num_channels
        self.K_max = K_max
        self._base_name = base if isinstance(base, str) else base.name
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        self.pair_eta = float(pair_eta)
        self.saturate_big = float(saturate_big)
        self.init_bias = init_bias
        self.learnable_biases = learnable_biases
        self.learnable_signs = learnable_signs

        dtype = self.variable_dtype
        biases_np = np.full((num_channels, K_max), float(init_bias), dtype=dtype)
        signs_np = np.zeros((num_channels, K_max), dtype=dtype)
        signs_np[:, :init_K] = identity_signs(init_K, dtype=dtype)[None, :]

        self.biases = self.add_weight(
            name="biases",
            shape=(num_channels, K_max),
            initializer="zeros",
            trainable=learnable_biases,
            dtype=dtype,
        )
        self.signs = self.add_weight(
            name="signs",
            shape=(num_channels, K_max),
            initializer="zeros",
            trainable=learnable_signs,
            dtype=dtype,
        )
        self.biases.assign(biases_np)
        self.signs.assign(signs_np)
        self._active_K = int(init_K)
        self.built = True

    @property
    def active_K(self) -> int:
        return self._active_K

    @property
    def can_grow(self) -> bool:
        return self._active_K < self.K_max

    def call(self, z: Any) -> Any:
        """Literal multi-bias forward over the active K columns only."""
        K = self._active_K
        biases = self.biases[:, :K]
        signs = self.signs[:, :K]
        return multibias_literal_forward(z, biases, signs, self.spec.forward)

    def grow(
        self,
        strategy: GrowStrategy = "pair",
        *,
        channel: int | None = None,
        anchor_value: float | np.ndarray | None = None,
    ) -> int:
        """Grow ``active_K``. Returns the number of newly-activated columns."""
        K = self._active_K
        biases_np = np.asarray(ops.convert_to_numpy(self.biases))
        signs_np = np.asarray(ops.convert_to_numpy(self.signs))

        if strategy == "pair":
            if K + 2 > self.K_max:
                raise RuntimeError(f"grow(pair) would exceed K_max={self.K_max}: active_K={K}")
            if anchor_value is not None:
                if isinstance(anchor_value, np.ndarray):
                    if anchor_value.shape != (self.num_channels,):
                        raise ValueError(
                            f"anchor_value array must have shape ({self.num_channels},), "
                            f"got {anchor_value.shape}"
                        )
                    anchor = anchor_value.astype(biases_np.dtype)
                else:
                    anchor = np.full(self.num_channels, float(anchor_value), dtype=biases_np.dtype)
            else:
                j = 0 if channel is None else int(channel)
                if not (0 <= j < K):
                    raise ValueError(f"channel must be in [0, {K}), got {j}")
                anchor = biases_np[:, j].copy()
            biases_np[:, K] = anchor
            biases_np[:, K + 1] = anchor
            signs_np[:, K] = +self.pair_eta
            signs_np[:, K + 1] = -self.pair_eta
            self.biases.assign(biases_np)
            self.signs.assign(signs_np)
            self._active_K += 2
            return 2

        if strategy == "saturate":
            if K + 1 > self.K_max:
                raise RuntimeError(f"grow(saturate) would exceed K_max={self.K_max}: active_K={K}")
            if self.spec.name not in _SATURATE_FRIENDLY:
                raise ValueError(
                    f"strategy='saturate' is only safe for activations that vanish "
                    f"at z -> -infinity: one of {sorted(_SATURATE_FRIENDLY)}. "
                    f"Got {self.spec.name!r}; use strategy='pair' instead."
                )
            biases_np[:, K] = -self.saturate_big
            signs_np[:, K] = 0.0
            self.biases.assign(biases_np)
            self.signs.assign(signs_np)
            self._active_K += 1
            return 1

        raise ValueError(f"Unknown strategy {strategy!r}; expected 'pair' or 'saturate'.")

    def active_parameters(self) -> dict[str, Any]:
        K = self._active_K
        return {"biases": self.biases[:, :K], "signs": self.signs[:, :K]}

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "init_K": self._active_K,
                "K_max": self.K_max,
                "base": self._base_name,
                "init_bias": self.init_bias,
                "learnable_biases": self.learnable_biases,
                "learnable_signs": self.learnable_signs,
                "pair_eta": self.pair_eta,
                "saturate_big": self.saturate_big,
            }
        )
        return config


__all__ = ["GrowStrategy", "GrowableOperatorMultiBiasUnit"]

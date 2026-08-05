# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed dense layer: :class:`cmbDense`.

A drop-in for :class:`keras.layers.Dense` followed by an
:class:`OperatorBlock` on the output features. Mirrors
:class:`omnibias.torch.blocks.linear.cmbLinear`.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.spec import ActivationSpec
from omnibias.keras.blocks.operator import OperatorBlock, OpName

from keras import layers


class cmbDense(layers.Layer):
    """``keras.layers.Dense`` + per-feature :class:`OperatorBlock`.

    Parameters
    ----------
    units : int
        Number of output features.
    op : :class:`OpName`, default ``"identity"``
    base : str or :class:`ActivationSpec`, default ``"sigmoid"``
    use_bias : bool, default True
        Dense-layer bias (independent of the OMBU bias terms).
    block_kwargs : dict, optional
        Extra keyword arguments passed to :class:`OperatorBlock`.
    """

    def __init__(
        self,
        units: int,
        op: OpName = "identity",
        base: str | ActivationSpec[Any] = "sigmoid",
        use_bias: bool = True,
        block_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self._op = op
        self._base_name = base if isinstance(base, str) else base.name
        self._use_bias = use_bias
        self._block_kwargs = block_kwargs or {}
        self.dense = layers.Dense(units, use_bias=use_bias)
        self.block = OperatorBlock(
            op=op, base=base, channels=units, **self._block_kwargs
        )

    def call(self, x: Any) -> Any:
        return self.block(self.dense(x))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "op": self._op,
                "base": self._base_name,
                "use_bias": self._use_bias,
                "block_kwargs": self._block_kwargs,
            }
        )
        return config


__all__ = ["cmbDense"]

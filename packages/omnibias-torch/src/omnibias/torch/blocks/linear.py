# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed linear layer: :class:`cmbLinear`.

A drop-in for :class:`torch.nn.Linear` followed by an
:class:`OperatorBlock` on the output channels.
"""

from __future__ import annotations

from typing import Any

from omnibias.torch.activations.registry import ActivationSpec
from omnibias.torch.blocks.operator import OperatorBlock, OpName

from torch import Tensor
from torch import nn as nn


class cmbLinear(nn.Module):
    """``nn.Linear`` + per-channel :class:`OperatorBlock`.

    Parameters
    ----------
    in_features : int
    out_features : int
    op : :class:`OpName`, default ``"identity"``
        Operator role for the output activation.
    base : str or :class:`ActivationSpec`, default ``"sigmoid"``
    bias : bool, default True
        Linear-layer bias (independent of the OMBU bias terms).
    block_kwargs : dict, optional
        Extra keyword arguments passed to :class:`OperatorBlock`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        op: OpName = "identity",
        base: str | ActivationSpec[Tensor] = "sigmoid",
        bias: bool = True,
        block_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.block = OperatorBlock(
            op=op,
            base=base,
            channels=out_features,
            **(block_kwargs or {}),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.block(self.linear(x))
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"op={self.block.op!r}, base={self.block.base_name!r}"
        )


__all__ = ["cmbLinear"]

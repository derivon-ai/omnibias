# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-typed CNN: :class:`CmbNet`.

A small image classifier whose convolution layers each carry an
explicit operator role:

- ``edge``: ``cmbConv2d(op="grad", base="tanh")`` -> Sobel-like gradient
  features.
- ``blob``: ``cmbConv2d(op="laplacian", base="gaussian")`` ->
  Laplacian-of-Gaussian (LoG) blob detectors.
- ``scale``: ``cmbConv2d(op="band", base="gaussian")`` ->
  Difference-of-Gaussian scale-space integrator.

Each layer is end-to-end-trained; the operator role determines the
inductive bias on the *kind* of feature the layer prefers.

The default reference geometry targets MNIST (1 x 28 x 28 -> 10
classes) but the constructor exposes ``in_channels`` / ``num_classes``
for adaptation.
"""

from __future__ import annotations

from omnibias.torch.blocks import cmbConv2d

import torch.nn as nn
from torch import Tensor


class CmbNet(nn.Module):
    """Operator-typed CNN reference.

    Parameters
    ----------
    in_channels : int, default 1
    num_classes : int, default 10
    width : tuple[int, int, int], default ``(16, 32, 64)``
        Output channels of the three operator-typed conv layers.
    base_edge, base_blob, base_scale : str
        Base activations for the three layers; defaults match the
        classical scale-space hierarchy (tanh / gaussian / gaussian).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        width: tuple[int, int, int] = (16, 32, 64),
        base_edge: str = "tanh",
        base_blob: str = "gaussian",
        base_scale: str = "gaussian",
    ) -> None:
        super().__init__()
        c1, c2, c3 = width
        self.edge = cmbConv2d(in_channels, c1, kernel_size=3, padding=1, op="grad", base=base_edge)
        self.pool1 = nn.MaxPool2d(2)
        self.blob = cmbConv2d(c1, c2, kernel_size=3, padding=1, op="laplacian", base=base_blob)
        self.pool2 = nn.MaxPool2d(2)
        self.scale = cmbConv2d(c2, c3, kernel_size=3, padding=1, op="band", base=base_scale)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c3, num_classes)

    def features(self, x: Tensor) -> Tensor:
        x = self.edge(x)
        x = self.pool1(x)
        x = self.blob(x)
        x = self.pool2(x)
        x = self.scale(x)
        x = self.global_pool(x).flatten(1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.head(self.features(x))
        return out


__all__ = ["CmbNet"]

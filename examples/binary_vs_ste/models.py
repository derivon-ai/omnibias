# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Binary-weight / binary-activation networks driven by a swappable quantizer.

Both models read their quantizer from a shared, mutable :class:`QuantCtx` that the
training loop rebinds every step (so the current ``beta`` / arm takes effect).
Weights *and* activations are binarized through the same quantizer, following the
standard BNN convention of keeping the **first and last** layers full precision
(real inputs/logits) and putting ``BatchNorm`` before each activation binarization
-- without it binary activations barely train, which would mask the surrogate's
effect we are trying to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from examples.binary_vs_ste.arms import Quantizer
from examples.binary_vs_ste.data import DatasetSpec


@dataclass
class QuantCtx:
    """Mutable holder for the active quantizer; the training loop rebinds ``fn``.

    ``xnor`` toggles the XNOR-Net per-filter weight scale ``alpha = mean|W|`` (an
    equal-opportunity ablation that lifts every arm's absolute accuracy; it does not
    change the relative surrogate comparison).
    """

    fn: Quantizer
    xnor: bool = False


class BinaryLinear(nn.Module):
    """Linear layer whose weights are binarized by ``ctx.fn`` at every forward."""

    def __init__(self, in_features: int, out_features: int, ctx: QuantCtx, *, bias: bool = True) -> None:
        super().__init__()
        self.ctx = ctx
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        w = self.ctx.fn(self.weight)
        if self.ctx.xnor:
            w = w * self.weight.abs().mean(dim=1, keepdim=True)
        return F.linear(x, w, self.bias)


class BinaryConv2d(nn.Module):
    """2D convolution whose weights are binarized by ``ctx.fn`` at every forward."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ctx: QuantCtx,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        w = self.ctx.fn(self.weight)
        if self.ctx.xnor:
            w = w * self.weight.abs().mean(dim=(1, 2, 3), keepdim=True)
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


class BinaryActivation(nn.Module):
    """Binarize activations through the shared quantizer ``ctx.fn``."""

    def __init__(self, ctx: QuantCtx) -> None:
        super().__init__()
        self.ctx = ctx

    def forward(self, x: Tensor) -> Tensor:
        return self.ctx.fn(x)


class BinaryMLP(nn.Module):
    """BNN multilayer perceptron for the 1-channel image datasets.

    First layer (real weights, real pixel input) -> ``depth`` binary hidden blocks
    (``BinaryLinear -> BatchNorm -> binary activation``) -> real read-out.
    """

    def __init__(self, spec: DatasetSpec, ctx: QuantCtx, *, hidden: int = 1024, depth: int = 2) -> None:
        super().__init__()
        in_dim = spec.channels * spec.size * spec.size
        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.bn_in = nn.BatchNorm1d(hidden)
        self.act_in = BinaryActivation(ctx)
        blocks: list[nn.Module] = []
        for _ in range(max(depth - 1, 0)):
            blocks.append(BinaryLinear(hidden, hidden, ctx, bias=False))
            blocks.append(nn.BatchNorm1d(hidden))
            blocks.append(BinaryActivation(ctx))
        self.blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden, spec.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.act_in(self.bn_in(self.fc_in(x)))
        x = self.blocks(x)
        return self.fc_out(x)


class BinaryConvNet(nn.Module):
    """Small VGG-style BNN for the 3-channel (CIFAR-10) dataset.

    Real first conv and real linear read-out; everything between is binary
    (``BinaryConv2d -> BatchNorm -> binary activation``), with max-pooling to
    32 -> 16 -> 8 -> 4 spatial resolution.
    """

    def __init__(self, spec: DatasetSpec, ctx: QuantCtx, *, width: int = 64) -> None:
        super().__init__()
        c = spec.channels
        w = width

        def block(cin: int, cout: int) -> list[nn.Module]:
            return [
                BinaryConv2d(cin, cout, 3, ctx, padding=1),
                nn.BatchNorm2d(cout),
                BinaryActivation(ctx),
            ]

        layers: list[nn.Module] = [
            nn.Conv2d(c, w, 3, padding=1),
            nn.BatchNorm2d(w),
            BinaryActivation(ctx),
            *block(w, w),
            nn.MaxPool2d(2),  # 32 -> 16
            *block(w, 2 * w),
            *block(2 * w, 2 * w),
            nn.MaxPool2d(2),  # 16 -> 8
            *block(2 * w, 4 * w),
            *block(4 * w, 4 * w),
            nn.MaxPool2d(2),  # 8 -> 4
        ]
        self.features = nn.Sequential(*layers)
        feat = (4 * w) * (spec.size // 8) * (spec.size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(feat, 8 * w, ctx, bias=False),
            nn.BatchNorm1d(8 * w),
            BinaryActivation(ctx),
            nn.Linear(8 * w, spec.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def build_model(spec: DatasetSpec, ctx: QuantCtx) -> nn.Module:
    """Pick the BNN architecture for ``spec``: MLP for 1-channel, ConvNet for 3."""
    if spec.channels == 1:
        return BinaryMLP(spec, ctx)
    return BinaryConvNet(spec, ctx)


__all__ = [
    "BinaryActivation",
    "BinaryConv2d",
    "BinaryConvNet",
    "BinaryLinear",
    "BinaryMLP",
    "QuantCtx",
    "build_model",
]

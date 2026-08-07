# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch readout seam for the frozen-feature linear solver.

Dispatches on the leaf field behind any cage ``.base`` chain. Supports the
linear readouts of :class:`~omnibias.pinn.torch.fields.one_layer.OneLayerVectorField`
(``c`` / ``b`` via ``nn.Linear``),
:class:`~omnibias.pinn.torch.fields.spectral.SpectralVectorField` (``V`` /
``b_t``), and
:class:`~omnibias.pinn.torch.fields.chebyshev.ChebyshevVectorField` (same
``V`` / ``b_t`` contract). Declaring fields without a supported linear readout
(e.g. jet MLPs) raise a named error.
"""

from __future__ import annotations

from typing import Any

import torch
from omnibias.fields._core.field_base import DISPATCH_ATTR
from omnibias.pinn.solver._core.readout import requires_readout_independent
from torch import Tensor


def _leaf(field: Any) -> Any:
    """Unwrap cage wrappers; the outer field must already have passed the gate."""
    while getattr(field, DISPATCH_ATTR, None) == "cage" and hasattr(field, "base"):
        field = field.base
    return field


def _unsupported(leaf: Any) -> TypeError:
    name = type(leaf).__name__
    return TypeError(
        f"{name} declares readout-independent caches but has no supported linear "
        "readout seam. Expected OneLayerVectorField (c.weight / c.bias) or "
        "SpectralVectorField / ChebyshevVectorField (V / b_t); jet and other "
        "nonlinear-in-depth fields must use solve_optimize instead of the "
        "frozen-feature linear path."
    )


def _weight_bias(leaf: Any) -> tuple[Tensor, Tensor]:
    """Return ``(weight, bias)`` Parameter tensors for the live readout."""
    c = getattr(leaf, "c", None)
    if c is not None and hasattr(c, "weight") and hasattr(c, "bias"):
        return c.weight, c.bias
    if hasattr(leaf, "V") and hasattr(leaf, "b_t"):
        return leaf.V, leaf.b_t
    raise _unsupported(leaf)


def readout_size(field: Any) -> tuple[int, int, int]:
    """Return ``(n_out, n_features, n_unknowns)`` for the flat readout vector."""
    requires_readout_independent(field)
    weight, bias = _weight_bias(_leaf(field))
    n_out, n_feat = int(weight.shape[0]), int(weight.shape[1])
    return n_out, n_feat, n_out * n_feat + n_out


def set_readout(field: Any, theta: Tensor) -> None:
    """Write a flat readout vector into the live ``(weight, bias)`` parameters."""
    requires_readout_independent(field)
    weight, bias = _weight_bias(_leaf(field))
    n_out, n_feat = int(weight.shape[0]), int(weight.shape[1])
    with torch.no_grad():
        weight.copy_(theta[: n_out * n_feat].reshape(n_out, n_feat))
        bias.copy_(theta[n_out * n_feat :])


def readout_dtype(field: Any) -> torch.dtype:
    """Dtype of the live readout parameters (and therefore of assembled rows)."""
    requires_readout_independent(field)
    weight, _ = _weight_bias(_leaf(field))
    return weight.dtype


def readout_device(field: Any) -> torch.device:
    """Device of the live readout parameters."""
    requires_readout_independent(field)
    weight, _ = _weight_bias(_leaf(field))
    return weight.device


def empty_rows(field: Any) -> Tensor:
    """A length-zero residual row on the field's dtype/device."""
    requires_readout_independent(field)
    weight, _ = _weight_bias(_leaf(field))
    return weight.new_zeros(0)


def freeze_features(field: Any) -> None:
    """Freeze every non-readout parameter so only the linear readout is free."""
    requires_readout_independent(field)
    leaf = _leaf(field)
    c = getattr(leaf, "c", None)
    if c is not None and hasattr(leaf, "W"):
        leaf.W.weight.requires_grad_(False)
        leaf.W.bias.requires_grad_(False)
        return
    if hasattr(leaf, "V") and hasattr(leaf, "b_t"):
        leaf.W_t.requires_grad_(False)
        leaf.beta_t.requires_grad_(False)
        for layer in getattr(leaf, "_inner_layers", ()):
            for p in layer.parameters():
                p.requires_grad_(False)
        return
    raise _unsupported(leaf)


__all__ = [
    "empty_rows",
    "freeze_features",
    "readout_device",
    "readout_dtype",
    "readout_size",
    "set_readout",
]

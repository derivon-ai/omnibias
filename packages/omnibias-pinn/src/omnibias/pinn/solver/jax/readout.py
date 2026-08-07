# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX readout seam for the frozen-feature linear solver.

Twin of :mod:`omnibias.pinn.solver.torch.readout`. JAX fields are immutable, so
the seam rebuilds a copy via :func:`with_readout` rather than mutating in place.
``freeze_features`` is a gate-only no-op: the hidden weights travel unchanged
into every rebuilt field.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from omnibias.fields._core.field_base import DISPATCH_ATTR
from omnibias.pinn.solver._core.readout import requires_readout_independent


def _leaf(field: Any) -> Any:
    """Unwrap cage wrappers; the outer field must already have passed the gate."""
    while getattr(field, DISPATCH_ATTR, None) == "cage" and hasattr(field, "base"):
        field = field.base
    return field


def _unsupported(leaf: Any) -> TypeError:
    name = type(leaf).__name__
    return TypeError(
        f"{name} declares readout-independent caches but has no supported linear "
        "readout seam. Expected OneLayerVectorField (c / b) or "
        "SpectralVectorField / ChebyshevVectorField (V / b_t); jet and other "
        "nonlinear-in-depth fields must use solve_optimize instead of the "
        "frozen-feature linear path."
    )


def _weight_bias(leaf: Any) -> tuple[Any, Any]:
    """Return ``(weight, bias)`` arrays for the live readout."""
    if hasattr(leaf, "c") and hasattr(leaf, "b") and not hasattr(leaf, "V"):
        return leaf.c, leaf.b
    if hasattr(leaf, "V") and hasattr(leaf, "b_t"):
        return leaf.V, leaf.b_t
    # One-layer always has c/b; prefer that when both somehow exist.
    if hasattr(leaf, "c") and hasattr(leaf, "b"):
        return leaf.c, leaf.b
    raise _unsupported(leaf)


def _is_one_layer(leaf: Any) -> bool:
    return hasattr(leaf, "c") and hasattr(leaf, "b") and not hasattr(leaf, "V")


def readout_size(field: Any) -> tuple[int, int, int]:
    """Return ``(n_out, n_features, n_unknowns)`` for the flat readout vector."""
    requires_readout_independent(field)
    weight, _bias = _weight_bias(_leaf(field))
    n_out, n_feat = int(weight.shape[0]), int(weight.shape[1])
    return n_out, n_feat, n_out * n_feat + n_out


def with_readout(field: Any, weight: Any, bias: Any) -> Any:
    """Return a copy of ``field`` with a new (frozen) readout ``(weight, bias)``.

    A cage is rebuilt around the new base rather than replaced, because an
    affine constrained expression stays affine when only the free function's
    readout changes.
    """
    requires_readout_independent(field)
    if getattr(field, DISPATCH_ATTR, None) == "cage" and hasattr(field, "base"):
        return replace(field, base=with_readout(field.base, weight, bias))
    leaf = field
    if _is_one_layer(leaf) or (hasattr(leaf, "c") and not hasattr(leaf, "V")):
        return replace(leaf, c=weight, b=bias)
    if hasattr(leaf, "V") and hasattr(leaf, "b_t"):
        return replace(leaf, V=weight, b_t=bias)
    raise _unsupported(leaf)


def readout_dtype(field: Any) -> Any:
    """Dtype of the live readout parameters (and therefore of assembled rows)."""
    requires_readout_independent(field)
    weight, _ = _weight_bias(_leaf(field))
    return weight.dtype


def freeze_features(field: Any) -> None:
    """Gate-only no-op: JAX fields are rebuilt with unchanged hidden weights."""
    requires_readout_independent(field)
    _weight_bias(_leaf(field))  # raise for unsupported declaring fields


__all__ = [
    "freeze_features",
    "readout_dtype",
    "readout_size",
    "with_readout",
]

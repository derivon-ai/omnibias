# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Convert a JAX MLP (a list of ``(W, b)`` arrays + activation tags) into a verifier :class:`Network`.

Mirror of :func:`omnibias.verify.torch.ingest.network_from_sequential`; the
two frontends produce bit-identical :class:`Network` objects from equal weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.verify._core.network import (
    GELULayer,
    Layer,
    Network,
    ReLULayer,
    SigmoidLayer,
    TanhLayer,
    affine_layer,
)

_Activation = ReLULayer | TanhLayer | SigmoidLayer | GELULayer

_ACTIVATIONS: dict[str, type[_Activation]] = {
    "relu": ReLULayer,
    "tanh": TanhLayer,
    "sigmoid": SigmoidLayer,
    "gelu": GELULayer,
}


def network_from_params(
    params: Sequence[tuple[Any, Any]], activation: str = "relu", *, final_activation: bool = False
) -> Network:
    """Build a :class:`Network` from ``params = [(W_0, b_0), ...]`` and an activation name.

    ``W`` is ``out x in`` (row-major, as in ``W @ x + b``).  An ``activation``
    layer is inserted after every affine map except the last, unless
    ``final_activation`` is set.  Supported activations: ``relu``, ``tanh``,
    ``sigmoid``.
    """
    act = activation.lower()
    if act not in _ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}; choose from {sorted(_ACTIVATIONS)}")
    make_act = _ACTIVATIONS[act]
    layers: list[Layer] = []
    n = len(params)
    for idx, (w, b) in enumerate(params):
        weight = [[float(x) for x in row] for row in w]
        bias = [float(x) for x in b]
        layers.append(affine_layer(weight, bias))
        if idx < n - 1 or final_activation:
            layers.append(make_act())
    return Network(layers)


__all__ = ["network_from_params"]

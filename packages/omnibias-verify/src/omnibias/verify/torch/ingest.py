# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Convert a trained ``torch.nn`` feed-forward module into a verifier :class:`Network`.

Weights are read off in float64 and frozen into the backend-neutral
:class:`~omnibias.verify._core.network.Network`, so verification never touches
torch again (and stays bit-identical to the jax ingestion).
"""

from __future__ import annotations

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


def network_from_sequential(module: Any) -> Network:
    """Build a :class:`Network` from a ``torch.nn.Sequential`` (or any iterable of layers).

    Recognises ``nn.Linear`` and the ``nn.ReLU`` / ``nn.Tanh`` / ``nn.Sigmoid`` /
    ``nn.GELU`` activations; raises ``TypeError`` on anything else so unsupported
    (and possibly unsound-to-skip) layers are never silently dropped.
    """
    import torch  # noqa: F401  (import guard: torch frontend requires torch)

    layers: list[Layer] = []
    for sub in module:
        name = type(sub).__name__
        if name == "Linear":
            weight = sub.weight.detach().double().tolist()
            bias = (
                sub.bias.detach().double().tolist()
                if sub.bias is not None
                else [0.0] * sub.out_features
            )
            layers.append(affine_layer(weight, bias))
        elif name == "ReLU":
            layers.append(ReLULayer())
        elif name == "Tanh":
            layers.append(TanhLayer())
        elif name == "Sigmoid":
            layers.append(SigmoidLayer())
        elif name == "GELU":
            layers.append(GELULayer())
        else:
            raise TypeError(f"unsupported torch layer for verification: {name}")
    return Network(layers)


__all__ = ["network_from_sequential"]

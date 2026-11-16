# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-neutral description of a dense layer chain.

Every structured derivative program of omnibias reads the same model: an input of dimension ``D`` followed by
dense layers ``z = W a + b``, each followed by an activation or by nothing (an affine layer). Which program is
fastest depends only on the widths, the activation names, which layers carry a bias and the dtype, never on the
parameter values, so those facts are the descriptor a planner keys its decisions on. The descriptor holds no
arrays and imports no tensor framework; JAX and PyTorch adapters build it from their own layer conventions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["ChainDescriptor", "LayerDescriptor"]

_DTYPES = ("float16", "bfloat16", "float32", "float64")


@dataclass(frozen=True)
class LayerDescriptor:
    """One dense layer ``out = activation(W @ in + b)``; ``activation=None`` is affine."""

    in_features: int
    out_features: int
    activation: str | None = None
    bias: bool = True

    def __post_init__(self) -> None:
        for name in ("in_features", "out_features"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.activation is not None and (
            not isinstance(self.activation, str) or not self.activation
        ):
            raise ValueError("activation must be a non-empty name or None")

    def to_config(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation,
            "bias": self.bias,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LayerDescriptor:
        return cls(
            int(config["in_features"]),
            int(config["out_features"]),
            config.get("activation"),
            bool(config.get("bias", True)),
        )


@dataclass(frozen=True)
class ChainDescriptor:
    """A chain of dense layers; consecutive widths must agree."""

    layers: tuple[LayerDescriptor, ...]
    dtype: str = "float64"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise ValueError("a chain needs at least one layer")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
        for i, (a, b) in enumerate(zip(self.layers[:-1], self.layers[1:], strict=True)):
            if a.out_features != b.in_features:
                raise ValueError(
                    f"layer {i + 1} expects {b.in_features} inputs but layer {i} produces {a.out_features}"
                )

    # -- construction -----------------------------------------------------------------------------------------

    @classmethod
    def from_widths(
        cls,
        widths: Sequence[int],
        activations: Sequence[str | None],
        *,
        bias: bool | Sequence[bool] = True,
        dtype: str = "float64",
    ) -> ChainDescriptor:
        """``widths = [D, H_1, ..., H_n, C]`` and one activation (or ``None``) per layer."""
        widths = [int(w) for w in widths]
        if len(widths) < 2:
            raise ValueError("widths needs at least an input and an output width")
        n = len(widths) - 1
        if len(activations) != n:
            raise ValueError(f"expected {n} activations for {n} layers, got {len(activations)}")
        biases = [bool(bias)] * n if isinstance(bias, bool) else [bool(v) for v in bias]
        if len(biases) != n:
            raise ValueError(f"expected {n} bias flags for {n} layers, got {len(biases)}")
        return cls(
            tuple(
                LayerDescriptor(widths[i], widths[i + 1], activations[i], biases[i])
                for i in range(n)
            ),
            dtype,
        )

    @classmethod
    def ridge(
        cls,
        input_dim: int,
        hidden: int,
        activation: str,
        *,
        outputs: int = 1,
        readout_bias: bool = False,
        dtype: str = "float64",
    ) -> ChainDescriptor:
        """``u(x) = sum_h c_h activation(w_h . x + b_h)``: one activated layer and a linear readout."""
        return cls.from_widths(
            [input_dim, hidden, outputs],
            [activation, None],
            bias=[True, readout_bias],
            dtype=dtype,
        )

    # -- structure --------------------------------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def output_dim(self) -> int:
        return self.layers[-1].out_features

    @property
    def widths(self) -> tuple[int, ...]:
        return (self.input_dim, *(layer.out_features for layer in self.layers))

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        return self.widths[1:-1]

    @property
    def activations(self) -> tuple[str | None, ...]:
        return tuple(layer.activation for layer in self.layers)

    @property
    def activation_names(self) -> frozenset[str]:
        """The distinct activation names used (affine layers excluded)."""
        return frozenset(a for a in self.activations if a is not None)

    @property
    def depth(self) -> int:
        """The number of dense layers (a ridge field has two)."""
        return len(self.layers)

    @property
    def parameter_count(self) -> int:
        return sum(
            layer.in_features * layer.out_features + (layer.out_features if layer.bias else 0)
            for layer in self.layers
        )

    @property
    def has_affine_readout(self) -> bool:
        return self.layers[-1].activation is None

    @property
    def is_ridge(self) -> bool:
        """One activated layer followed by an affine readout: the form of ``sum_h c_h sigma(w_h . x + b_h)``."""
        return self.depth == 2 and self.layers[0].activation is not None and self.has_affine_readout

    @property
    def narrowest_hidden(self) -> int | None:
        hidden = self.hidden_widths
        return min(hidden) if hidden else None

    @property
    def first_layer_ratio(self) -> float:
        """``D / H_1``, the quantity first-layer projections gain with (``D`` against the first hidden width)."""
        return self.input_dim / self.layers[0].out_features

    def is_vector_field(self, dim: int | None = None) -> bool:
        """An affine readout whose width equals ``dim`` (default: the input dimension)."""
        return self.has_affine_readout and self.output_dim == (
            self.input_dim if dim is None else dim
        )

    # -- serialization ----------------------------------------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "layers": [layer.to_config() for layer in self.layers]}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ChainDescriptor:
        return cls(
            tuple(LayerDescriptor.from_config(c) for c in config["layers"]),
            str(config.get("dtype", "float64")),
        )

    def fingerprint(self) -> str:
        """A stable hash of the configuration (plan caches key on it)."""
        blob = json.dumps(self.to_config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

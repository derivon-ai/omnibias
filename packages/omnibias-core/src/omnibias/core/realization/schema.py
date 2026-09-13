# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Immutable descriptions of realizations and their observation scope.

The schema stores no tensors or backend imports. A finite observation vector
does not determine a global function unless a separate determining argument
is supplied. Parameter blocks have stable names and contiguous row-major slots.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

OperatorRole = Literal["identity", "grad", "laplacian", "derivative", "band", "integral"]
ObservationKind = Literal["values", "coefficients", "derivatives", "residuals", "integrals"]
RealizationScope = Literal["finite_observations", "polynomial_identity", "declared_domain"]
IntegralKind = Literal["activation_window", "domain_quadrature", "measure"]


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    shape: tuple[int, ...]
    offset: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.name or self.offset < 0 or any(d < 1 for d in self.shape):
            raise ValueError("parameter blocks need a name, nonnegative offset and positive dimensions")

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def stop(self) -> int:
        return self.offset + self.size

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.stop)


@dataclass(frozen=True)
class ParameterLayout:
    blocks: tuple[ParameterBlock, ...]
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if self.version < 1:
            raise ValueError("layout version must be positive")
        offset = 0
        names: set[str] = set()
        for block in self.blocks:
            if block.name in names or block.offset != offset:
                raise ValueError("blocks must have unique names and contiguous offsets")
            names.add(block.name)
            offset = block.stop

    @classmethod
    def from_shapes(
        cls, shapes: Mapping[str, Sequence[int]] | Sequence[tuple[str, Sequence[int]]],
        *, version: int = 1,
    ) -> ParameterLayout:
        items = shapes.items() if isinstance(shapes, Mapping) else shapes
        blocks: list[ParameterBlock] = []
        offset = 0
        for name, shape in items:
            block = ParameterBlock(name, tuple(shape), offset)
            blocks.append(block)
            offset = block.stop
        return cls(tuple(blocks), version)

    @property
    def size(self) -> int:
        return self.blocks[-1].stop if self.blocks else 0

    def block(self, name: str) -> ParameterBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"unknown parameter block {name!r}")

    def block_slice(self, name: str) -> slice:
        return self.block(name).slice

    def fingerprint(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class LayerSpec:
    """Affine layer followed by one of the six activation-level roles.

    ``activation=None`` is a plain affine layer. Window parameters are the
    literal lower/upper shifts, or center/raw-width when ``window_parameterization``
    is ``center_softplus_width``. The latter matches the integral OperatorBlock.
    """

    in_features: int
    out_features: int
    activation: str | None = "tanh"
    op: OperatorRole = "identity"
    derivative_order: int = 0
    weight_name: str = ""
    bias_name: str | None = None
    window_names: tuple[str, str] | None = None
    window_parameterization: Literal["endpoints", "center_softplus_width"] = "endpoints"

    def __post_init__(self) -> None:
        if min(self.in_features, self.out_features) < 1 or self.derivative_order < 0:
            raise ValueError("layer dimensions must be positive and derivative_order nonnegative")
        if self.op not in {"identity", "grad", "laplacian", "derivative", "band", "integral"}:
            raise ValueError(f"unknown operator role {self.op!r}")
        if self.activation is None and self.op != "identity":
            raise ValueError("operator roles require an activation")
        if self.op in {"band", "integral"} and self.window_names is None:
            raise ValueError("band/integral require two live window parameter blocks")
        if self.window_names is not None:
            object.__setattr__(self, "window_names", tuple(self.window_names))
            if len(self.window_names) != 2:
                raise ValueError("window_names must have two entries")
        if self.window_parameterization not in {"endpoints", "center_softplus_width"}:
            raise ValueError("unknown window parameterization")

    @property
    def activation_order(self) -> int:
        return {"grad": 1, "laplacian": 2, "derivative": self.derivative_order}.get(self.op, 0)


@dataclass(frozen=True)
class ObservationSpec:
    """Scoped observations. Legacy integral observations mean domain quadrature.

    Activation windows evaluate the actual integral-role realization at points.
    Domain and measure integrals use the declared weighted numerical rule; its
    provenance identifies the quadrature or sampled measure approximation.
    """
    kind: ObservationKind = "values"
    scope: RealizationScope = "finite_observations"
    derivative_indices: tuple[tuple[int, ...], ...] = ()
    points: tuple[tuple[float, ...], ...] = ()
    weights: tuple[float, ...] = ()
    domain: tuple[tuple[float, float], ...] = ()
    label: str = ""
    provenance: str = ""
    integral_kind: IntegralKind | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"values", "coefficients", "derivatives", "residuals", "integrals"}:
            raise ValueError("unknown observation kind")
        if self.scope not in {"finite_observations", "polynomial_identity", "declared_domain"}:
            raise ValueError("unknown observation scope")
        if self.integral_kind not in {None, "activation_window", "domain_quadrature", "measure"}:
            raise ValueError("unknown integral kind")
        if self.integral_kind is not None and self.kind != "integrals":
            raise ValueError("integral_kind is only valid for integral observations")
        for field in ("derivative_indices", "points", "domain"):
            value = getattr(self, field)
            object.__setattr__(self, field, tuple(tuple(row) for row in value))
        object.__setattr__(self, "weights", tuple(self.weights))
        if any(i < 0 for index in self.derivative_indices for i in index):
            raise ValueError("derivative multi-indices must be nonnegative")
        if self.weights and len(self.weights) != len(self.points):
            raise ValueError("observation weights must match points")
        if any(not math.isfinite(v) for row in self.points for v in row):
            raise ValueError("observation points must be finite")
        if any(not math.isfinite(v) for v in self.weights):
            raise ValueError("observation weights must be finite")
        if any(not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi) for lo, hi in self.domain):
            raise ValueError("domain must contain finite ordered bounds")


@dataclass(frozen=True)
class RealizationSpec:
    layers: tuple[LayerSpec, ...]
    layout: ParameterLayout
    input_dim: int
    output_dim: int
    observations: tuple[ObservationSpec, ...] = ()
    name: str = "realization"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "observations", tuple(self.observations))
        if not self.layers or min(self.input_dim, self.output_dim) < 1:
            raise ValueError("a realization needs layers and positive dimensions")
        width = self.input_dim
        for i, layer in enumerate(self.layers):
            if layer.in_features != width:
                raise ValueError("adjacent realization layer dimensions do not match")
            weight = self.layout.block(layer.weight_name or f"W{i}")
            if weight.shape != (layer.out_features, layer.in_features):
                raise ValueError("weight block shape does not match layer")
            for name in (() if layer.bias_name is None else (layer.bias_name,)) + (layer.window_names or ()):
                if self.layout.block(name).shape != (layer.out_features,):
                    raise ValueError("bias/window block shape does not match layer")
            width = layer.out_features
        if width != self.output_dim:
            raise ValueError("output_dim does not match final layer")

    @classmethod
    def dense(
        cls, widths: Sequence[int], *, activation: str = "tanh",
        output_activation: str | None = None, bias: bool = True, name: str = "dense",
    ) -> RealizationSpec:
        dims = tuple(widths)
        if len(dims) < 2:
            raise ValueError("dense widths must contain input and output dimensions")
        shapes: list[tuple[str, tuple[int, ...]]] = []
        layers: list[LayerSpec] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            shapes.append((f"W{i}", (dout, din)))
            bname = f"b{i}" if bias else None
            if bname is not None:
                shapes.append((bname, (dout,)))
            act = output_activation if i == len(dims) - 2 else activation
            layers.append(LayerSpec(din, dout, act, weight_name=f"W{i}", bias_name=bname))
        return cls(tuple(layers), ParameterLayout.from_shapes(shapes), dims[0], dims[-1], name=name)

    def fingerprint(self) -> str:
        return _digest(asdict(self))


__all__ = ["IntegralKind", "LayerSpec", "ObservationKind", "ObservationSpec", "OperatorRole", "ParameterBlock",
           "ParameterLayout", "RealizationScope", "RealizationSpec"]

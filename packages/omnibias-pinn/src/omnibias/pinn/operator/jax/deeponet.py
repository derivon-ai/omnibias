# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet with closed-form query-coordinate derivatives (JAX).

Parity twin of :mod:`omnibias.pinn.operator.torch.deeponet` (torch↔jax
agreement at ``rtol=1e-11`` in float64). Frozen dataclasses registered as
pytrees so the conditioned field flows through ``jax.grad`` / ``jax.jit``;
``coeffs`` is a pytree leaf (the live per-sample readout).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.state import FieldState
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.architectures.pinn import JetMLP, make_jet_mlp
from omnibias.pinn.jax.fields.base import FieldBase
from omnibias.pinn.jax.fields.jet_mlp import _JetFieldOps
from omnibias.pinn.operator._core.branch import BranchHeadLayout
from omnibias.pinn.operator._core.conditioning import ConditioningSpec
from omnibias.pinn.operator._core.spec import OperatorSpec

#: ``FieldState.extra`` key for the per-evaluation trunk-jet cache.
TRUNK_JET_CACHE_KEY = "_deeponet_trunk_jets"


class PerSampleReadoutError(TypeError):
    """Raised when a shared affine readout is asked of a DeepONet trunk adapter."""


def _layer_norm(x: Array, gamma: Array, beta: Array, eps: float = 1e-5) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


@dataclass(frozen=True)
class _HeadEncoder:
    """Independently normalised head encoder (frozen pytree)."""

    norm_gamma: Array
    norm_beta: Array
    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    spec: JaxActivationSpec
    n_input: int
    encoder_dim: int

    def __call__(self, x: Array) -> Array:
        if x.shape[-1] != self.n_input:
            raise ValueError(
                f"head input trailing dim must be {self.n_input}, got {tuple(x.shape)}"
            )
        h = _layer_norm(x, self.norm_gamma, self.norm_beta)
        n = len(self.weights)
        for i in range(n):
            h = h @ self.weights[i].T + self.biases[i]
            if i < n - 1:
                h = self.spec.forward(h)
        return h


@dataclass(frozen=True)
class _BranchNet:
    """Multi-head encoders + fusion network (frozen pytree)."""

    function_encoder: _HeadEncoder
    parameter_encoder: _HeadEncoder | None
    boundary_encoder: _HeadEncoder | None
    geometry_encoder: _HeadEncoder | None
    fusion_weights: tuple[Array, ...]
    fusion_biases: tuple[Array, ...]
    spec: JaxActivationSpec
    layout: BranchHeadLayout
    n_components: int
    trunk_width: int
    per_sample_bias: bool = True

    @property
    def n_input(self) -> int:
        return int(self.layout.spec.total_dim)

    @property
    def n_sensors(self) -> int:
        return self.n_input

    def __call__(
        self,
        *,
        sensors: Array,
        parameters: Array | None = None,
        boundary: Array | None = None,
        geometry: Array | None = None,
    ) -> tuple[Array, Array | None]:
        parts = [self.function_encoder(sensors)]
        if self.parameter_encoder is not None:
            if parameters is None:
                raise ValueError("parameters required for this branch")
            parts.append(self.parameter_encoder(parameters))
        if self.boundary_encoder is not None:
            if boundary is None:
                raise ValueError("boundary required for this branch")
            parts.append(self.boundary_encoder(boundary))
        if self.geometry_encoder is not None:
            if geometry is None:
                raise ValueError("geometry required for this branch")
            parts.append(self.geometry_encoder(geometry))
        h = jnp.concatenate(parts, axis=-1)
        n = len(self.fusion_weights)
        for i in range(n):
            h = h @ self.fusion_weights[i].T + self.fusion_biases[i]
            if i < n - 1:
                h = self.spec.forward(h)
        leading = h.shape[:-1]
        c_p = self.n_components * self.trunk_width
        coeffs = h[..., :c_p].reshape(*leading, self.n_components, self.trunk_width)
        if self.per_sample_bias:
            return coeffs, h[..., c_p:]
        return coeffs, None


def _head_flatten(enc: _HeadEncoder) -> tuple[tuple[Array, ...], tuple[Any, ...]]:
    leaves = (enc.norm_gamma, enc.norm_beta, *enc.weights, *enc.biases)
    aux = (enc.spec, enc.n_input, enc.encoder_dim, len(enc.weights))
    return leaves, aux


def _head_unflatten(aux: tuple[Any, ...], leaves: tuple[Array, ...]) -> _HeadEncoder:
    spec, n_input, encoder_dim, n = aux
    return _HeadEncoder(
        norm_gamma=leaves[0],
        norm_beta=leaves[1],
        weights=leaves[2 : 2 + n],
        biases=leaves[2 + n :],
        spec=spec,
        n_input=n_input,
        encoder_dim=encoder_dim,
    )


jax.tree_util.register_pytree_node(_HeadEncoder, _head_flatten, _head_unflatten)


def _branch_flatten(
    net: _BranchNet,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    encoders = (
        net.function_encoder,
        net.parameter_encoder,
        net.boundary_encoder,
        net.geometry_encoder,
    )
    leaves: list[Any] = list(encoders)
    leaves.extend(net.fusion_weights)
    leaves.extend(net.fusion_biases)
    aux = (
        net.spec,
        net.layout,
        net.n_components,
        net.trunk_width,
        net.per_sample_bias,
        len(net.fusion_weights),
    )
    return tuple(leaves), aux


def _branch_unflatten(aux: tuple[Any, ...], leaves: tuple[Any, ...]) -> _BranchNet:
    spec, layout, n_components, trunk_width, per_sample_bias, n_fusion = aux
    function_encoder = leaves[0]
    parameter_encoder = leaves[1]
    boundary_encoder = leaves[2]
    geometry_encoder = leaves[3]
    fusion_weights = leaves[4 : 4 + n_fusion]
    fusion_biases = leaves[4 + n_fusion : 4 + 2 * n_fusion]
    return _BranchNet(
        function_encoder=function_encoder,
        parameter_encoder=parameter_encoder,
        boundary_encoder=boundary_encoder,
        geometry_encoder=geometry_encoder,
        fusion_weights=fusion_weights,
        fusion_biases=fusion_biases,
        spec=spec,
        layout=layout,
        n_components=n_components,
        trunk_width=trunk_width,
        per_sample_bias=per_sample_bias,
    )


jax.tree_util.register_pytree_node(_BranchNet, _branch_flatten, _branch_unflatten)


@dataclass(frozen=True)
class DeepONetField(_JetFieldOps, FieldBase):
    """A conditioned DeepONet evaluated as an omnibias PINN field (JAX)."""

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: JetMLP  # type: ignore[assignment]
    coeffs: Array
    bias: Array
    jet_order: int = 2
    n_functions: int = 1
    shared_query_size: int | None = None

    @property
    def trunk(self) -> JetMLP:
        """Alias for the jet trunk (the ``net`` leaf of ``_JetFieldOps``)."""
        return self.net

    def _pre_activations(self, coords: Array) -> Array | None:
        return None

    def _compute_hidden_jet(self, coords: Array, order: int) -> Array:
        """Batched full trunk jet of shape ``(Q, M, p)``."""
        return jax.vmap(lambda xi: self.net._point_jet(xi, order))(coords)

    def _contract(
        self, trunk_jet: Array, coeffs: Array, bias: Array
    ) -> Array:
        B = trunk_jet.shape[0]
        F = coeffs.shape[0]
        if self.shared_query_size is not None:
            Q = self.shared_query_size
            if B != F * Q:
                raise ValueError(
                    f"shared-grid coords leading dim {B} != F*Q = {F}*{Q}"
                )
            coeffs_b = jnp.repeat(coeffs, Q, axis=0)
            if bias.ndim == 1:
                bias_b = jnp.broadcast_to(bias[None, :], (B, bias.shape[0]))
            else:
                bias_b = jnp.repeat(bias, Q, axis=0)
        else:
            if F == 1 and B != 1:
                coeffs_b = jnp.broadcast_to(coeffs, (B, coeffs.shape[1], coeffs.shape[2]))
                if bias.ndim == 1:
                    bias_b = jnp.broadcast_to(bias[None, :], (B, bias.shape[0]))
                else:
                    bias_b = jnp.broadcast_to(bias, (B, bias.shape[1]))
            elif F == B:
                coeffs_b = coeffs
                bias_b = (
                    bias
                    if bias.ndim == 2
                    else jnp.broadcast_to(bias[None, :], (B, bias.shape[0]))
                )
            else:
                raise ValueError(
                    f"cannot align coeffs batch F={F} with coords batch B={B}; "
                    f"use on_grid() for a shared query grid"
                )
        out = jnp.einsum("bmp,bcp->bmc", trunk_jet, coeffs_b)
        out = out.at[:, 0, :].add(bias_b)
        return out

    def _jet_at_least(
        self, state: FieldState[Array], order: int
    ) -> tuple[Array, int]:
        coords = cast(Array, state.coords)
        extra = cast(dict[str, Any], state.extra)
        cache = cast(
            "dict[int, Array]", extra.setdefault(TRUNK_JET_CACHE_KEY, {})
        )
        trunk: Array | None = None
        got_order = -1
        for cached_order in sorted(cache):
            if cached_order >= order:
                trunk = cache[cached_order]
                got_order = cached_order
                break
        if trunk is None:
            want = max(int(order), self.jet_order)
            if self.shared_query_size is not None:
                Q = self.shared_query_size
                trunk = self._compute_hidden_jet(coords[:Q], want)
            else:
                trunk = self._compute_hidden_jet(coords, want)
            cache[want] = trunk
            got_order = want
        if (
            self.shared_query_size is not None
            and trunk.shape[0] == self.shared_query_size
        ):
            trunk = jnp.tile(trunk, (self.n_functions, 1, 1))
        return self._contract(trunk, self.coeffs, self.bias), got_order

    def forward_values(self, coords: Array) -> Array:
        if self.shared_query_size is not None:
            Q = self.shared_query_size
            trunk_val = self.trunk.value(coords[:Q])
            trunk_val = jnp.tile(trunk_val, (self.n_functions, 1))
        else:
            trunk_val = self.trunk.value(coords)
        trunk_jet = trunk_val[:, None, :]
        out = self._contract(trunk_jet, self.coeffs, self.bias)
        return out[:, 0, :]

    def value_component(self, state: FieldState[Array], name: str) -> Array:
        ci = self.components.index(name)
        return self.forward_values(cast(Array, state.coords))[:, ci]

    def on_grid(self, query_coords: Array) -> FieldState[Array]:
        query_coords = jnp.asarray(query_coords)
        if query_coords.ndim != 2:
            raise ValueError(
                f"query_coords must be 2-D (Q, D); got shape {tuple(query_coords.shape)}"
            )
        Q = int(query_coords.shape[0])
        F = self.n_functions
        tiled = jnp.tile(query_coords, (F, 1))
        field = DeepONetField(
            coordinate_spec=self.coordinate_spec,
            components=self.components,
            net=self.net,
            coeffs=self.coeffs,
            bias=self.bias,
            jet_order=self.jet_order,
            n_functions=F,
            shared_query_size=Q,
        )
        return cast(FieldState[Array], field.evaluate(tiled))


@dataclass(frozen=True)
class DeepONetOperator:
    """Trainable DeepONet: ``condition(sensors) -> DeepONetField`` (JAX)."""

    spec: OperatorSpec
    trunk: JetMLP
    branch: _BranchNet
    shared_bias: Array | None
    jet_order: int = 2

    @property
    def coordinate_spec(self) -> CoordinateSpec:
        return self.spec.coordinate_spec

    @property
    def components(self) -> ComponentSpec:
        return self.spec.components

    def _validate_heads(
        self,
        sensors: Array,
        *,
        parameters: Array | None = None,
        boundary: Array | None = None,
        geometry: Array | None = None,
    ) -> tuple[Array, Array | None, Array | None, Array | None]:
        cond = self.spec.conditioning
        assert cond is not None
        sensors = jnp.asarray(sensors)
        if sensors.ndim == 1:
            sensors = sensors[None, :]
        if sensors.ndim != 2:
            raise ValueError(
                f"sensors must be 1-D or 2-D; got shape {tuple(sensors.shape)}"
            )
        if sensors.shape[-1] != cond.n_function_sensors:
            raise ValueError(
                f"sensors trailing dim must be n_function_sensors="
                f"{cond.n_function_sensors}, got {tuple(sensors.shape)}"
            )
        F = int(sensors.shape[0])

        def _check(name: str, t: Array | None, width: int) -> Array | None:
            if width == 0:
                if t is not None:
                    raise ValueError(
                        f"{name} provided but conditioning width is 0"
                    )
                return None
            if t is None:
                raise ValueError(
                    f"{name} required: conditioning width is {width}"
                )
            t = jnp.asarray(t)
            if t.ndim == 1:
                t = t[None, :]
            if t.ndim != 2 or t.shape[-1] != width:
                raise ValueError(
                    f"{name} must have shape (F, {width}) or ({width},); "
                    f"got {tuple(t.shape)}"
                )
            if t.shape[0] == 1 and F > 1:
                t = jnp.broadcast_to(t, (F, width))
            elif int(t.shape[0]) != F:
                raise ValueError(
                    f"{name} batch {t.shape[0]} != sensors batch {F}"
                )
            return t

        return (
            sensors,
            _check("parameters", parameters, cond.n_parameters),
            _check("boundary", boundary, cond.n_boundary_sensors),
            _check("geometry", geometry, cond.n_geometry_probes),
        )

    def condition(
        self,
        sensors: Array,
        *,
        parameters: Array | None = None,
        boundary: Array | None = None,
        geometry: Array | None = None,
    ) -> DeepONetField:
        sensors_v, params_v, boundary_v, geometry_v = self._validate_heads(
            sensors,
            parameters=parameters,
            boundary=boundary,
            geometry=geometry,
        )
        coeffs, bias = self.branch(
            sensors=sensors_v,
            parameters=params_v,
            boundary=boundary_v,
            geometry=geometry_v,
        )
        if bias is None:
            if self.shared_bias is None:
                raise RuntimeError("shared_bias missing for per_sample_bias=False")
            bias = self.shared_bias
        return DeepONetField(
            coordinate_spec=self.spec.coordinate_spec,
            components=self.spec.components,
            net=self.trunk,
            coeffs=coeffs,
            bias=bias,
            jet_order=self.jet_order,
            n_functions=int(sensors_v.shape[0]),
        )


# -- pytree registration --------------------------------------------------------- #

_FieldAux = tuple[CoordinateSpec, ComponentSpec, int, int, int | None]


def _field_flatten(
    field: DeepONetField,
) -> tuple[tuple[Any, ...], _FieldAux]:
    return (field.net, field.coeffs, field.bias), (
        field.coordinate_spec,
        field.components,
        field.jet_order,
        field.n_functions,
        field.shared_query_size,
    )


def _field_unflatten(aux: _FieldAux, leaves: tuple[Any, ...]) -> DeepONetField:
    coordinate_spec, components, jet_order, n_functions, shared_query_size = aux
    obj = DeepONetField.__new__(DeepONetField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "net", leaves[0])
    object.__setattr__(obj, "coeffs", leaves[1])
    object.__setattr__(obj, "bias", leaves[2])
    object.__setattr__(obj, "jet_order", jet_order)
    object.__setattr__(obj, "n_functions", n_functions)
    object.__setattr__(obj, "shared_query_size", shared_query_size)
    return obj


jax.tree_util.register_pytree_node(DeepONetField, _field_flatten, _field_unflatten)


_OpAux = tuple[OperatorSpec, int]


def _op_flatten(
    op: DeepONetOperator,
) -> tuple[tuple[Any, ...], _OpAux]:
    return (op.trunk, op.branch, op.shared_bias), (op.spec, op.jet_order)


def _op_unflatten(aux: _OpAux, leaves: tuple[Any, ...]) -> DeepONetOperator:
    spec, jet_order = aux
    obj = DeepONetOperator.__new__(DeepONetOperator)
    object.__setattr__(obj, "spec", spec)
    object.__setattr__(obj, "trunk", leaves[0])
    object.__setattr__(obj, "branch", leaves[1])
    object.__setattr__(obj, "shared_bias", leaves[2])
    object.__setattr__(obj, "jet_order", jet_order)
    return obj


jax.tree_util.register_pytree_node(DeepONetOperator, _op_flatten, _op_unflatten)


DeepONetField._omnibias_dispatch = "jet_mlp"  # type: ignore[attr-defined]
DeepONetField._omnibias_readout_independent = True  # type: ignore[attr-defined]


def _make_head_encoder(
    n_input: int,
    encoder_dim: int,
    *,
    hidden: int,
    depth: int,
    base: str | JaxActivationSpec,
    key: Array,
    dtype: Any,
) -> tuple[_HeadEncoder, Array]:
    spec = get_activation(base)
    gamma = jnp.ones((n_input,), dtype=dtype)
    beta = jnp.zeros((n_input,), dtype=dtype)
    weights: list[Array] = []
    biases: list[Array] = []
    prev = n_input
    for i in range(depth):
        out = encoder_dim if i == depth - 1 else hidden
        key, wk = jax.random.split(key)
        scale = 1.0 / math.sqrt(prev)
        weights.append(jax.random.normal(wk, (out, prev), dtype=dtype) * scale)
        biases.append(jnp.zeros((out,), dtype=dtype))
        prev = out
    return (
        _HeadEncoder(
            norm_gamma=gamma,
            norm_beta=beta,
            weights=tuple(weights),
            biases=tuple(biases),
            spec=spec,
            n_input=n_input,
            encoder_dim=encoder_dim,
        ),
        key,
    )


def _make_branch(
    conditioning: ConditioningSpec,
    n_components: int,
    trunk_width: int,
    *,
    hidden: int,
    depth: int,
    encoder_dim: int | None,
    encoder_depth: int,
    base: str | JaxActivationSpec,
    per_sample_bias: bool,
    seed: int,
    dtype: Any,
) -> _BranchNet:
    enc_dim = int(encoder_dim if encoder_dim is not None else hidden)
    layout = BranchHeadLayout(conditioning, enc_dim)
    spec = get_activation(base)
    key = jax.random.PRNGKey(seed)
    key, enc_key = jax.random.split(key)
    function_encoder, key = _make_head_encoder(
        conditioning.n_function_sensors,
        enc_dim,
        hidden=hidden,
        depth=encoder_depth,
        base=base,
        key=enc_key,
        dtype=dtype,
    )
    parameter_encoder = None
    boundary_encoder = None
    geometry_encoder = None
    if conditioning.has_parameters:
        key, enc_key = jax.random.split(key)
        parameter_encoder, key = _make_head_encoder(
            conditioning.n_parameters,
            enc_dim,
            hidden=hidden,
            depth=encoder_depth,
            base=base,
            key=enc_key,
            dtype=dtype,
        )
    if conditioning.has_boundary:
        key, enc_key = jax.random.split(key)
        boundary_encoder, key = _make_head_encoder(
            conditioning.n_boundary_sensors,
            enc_dim,
            hidden=hidden,
            depth=encoder_depth,
            base=base,
            key=enc_key,
            dtype=dtype,
        )
    if conditioning.has_geometry:
        key, enc_key = jax.random.split(key)
        geometry_encoder, key = _make_head_encoder(
            conditioning.n_geometry_probes,
            enc_dim,
            hidden=hidden,
            depth=encoder_depth,
            base=base,
            key=enc_key,
            dtype=dtype,
        )
    out = n_components * trunk_width
    if per_sample_bias:
        out += n_components
    fusion_in = layout.fusion_dim
    fusion_weights: list[Array] = []
    fusion_biases: list[Array] = []
    prev = fusion_in
    for _ in range(depth):
        key, wk = jax.random.split(key)
        scale = 1.0 / math.sqrt(prev)
        fusion_weights.append(jax.random.normal(wk, (hidden, prev), dtype=dtype) * scale)
        fusion_biases.append(jnp.zeros((hidden,), dtype=dtype))
        prev = hidden
    key, wk = jax.random.split(key)
    scale = 1.0 / math.sqrt(prev)
    fusion_weights.append(jax.random.normal(wk, (out, prev), dtype=dtype) * scale)
    fusion_biases.append(jnp.zeros((out,), dtype=dtype))
    return _BranchNet(
        function_encoder=function_encoder,
        parameter_encoder=parameter_encoder,
        boundary_encoder=boundary_encoder,
        geometry_encoder=geometry_encoder,
        fusion_weights=tuple(fusion_weights),
        fusion_biases=tuple(fusion_biases),
        spec=spec,
        layout=layout,
        n_components=n_components,
        trunk_width=trunk_width,
        per_sample_bias=per_sample_bias,
    )


def make_deeponet(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    n_sensors: int,
    trunk_width: int = 32,
    trunk_hidden: int = 64,
    trunk_depth: int = 3,
    branch_hidden: int = 64,
    branch_depth: int = 2,
    base: str | JaxActivationSpec = "tanh",
    jet_order: int = 2,
    per_sample_bias: bool = True,
    seed: int = 0,
    dtype: Any = jnp.float64,
    conditioning: Any = None,
) -> DeepONetOperator:
    """Build a randomly-initialised :class:`DeepONetOperator` (JAX)."""
    spec = OperatorSpec(
        coordinate_spec=coordinate_spec,
        components=components,
        n_sensors=n_sensors,
        trunk_width=trunk_width,
        conditioning=conditioning,
    )
    trunk = make_jet_mlp(
        in_dim=spec.ndim,
        hidden=trunk_hidden,
        out_dim=spec.trunk_width,
        depth=trunk_depth,
        activation=base,
        seed=seed,
        dtype=dtype,
    )
    trunk._check_fastpath(jet_order)
    conditioning_resolved = spec.conditioning
    if conditioning_resolved is None:
        from omnibias.pinn.operator._core.conditioning import ConditioningSpec

        conditioning_resolved = ConditioningSpec.function_only(spec.n_sensors)
    branch = _make_branch(
        conditioning_resolved,
        n_components=spec.n_components,
        trunk_width=trunk_width,
        hidden=branch_hidden,
        depth=branch_depth,
        encoder_dim=None,
        encoder_depth=1,
        base=base,
        per_sample_bias=per_sample_bias,
        seed=seed + 1,
        dtype=dtype,
    )
    shared_bias = (
        None
        if per_sample_bias
        else jnp.zeros((spec.n_components,), dtype=dtype)
    )
    return DeepONetOperator(
        spec=spec,
        trunk=trunk,
        branch=branch,
        shared_bias=shared_bias,
        jet_order=jet_order,
    )


__all__ = [
    "DeepONetField",
    "DeepONetOperator",
    "PerSampleReadoutError",
    "TRUNK_JET_CACHE_KEY",
    "make_deeponet",
]

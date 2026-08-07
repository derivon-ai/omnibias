# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``SpectralVectorField`` for the JAX backend.

JAX twin of :class:`omnibias.pinn.torch.fields.spectral.SpectralVectorField`.
Same architecture (D-dim periodic Fourier basis, omnibias temporal head,
multi-component output), same mathematical contract, same closed-form
chain rule. Differences are stylistic: this is a frozen dataclass
holding ``jax.Array`` parameters and registered as a pytree node so the
trainable arrays survive ``jax.grad`` / ``jax.jit``.

For ``time_depth = 1`` the temporal derivative is the single-layer
omnibias fastpath ``sigma^(n)(z) W_t^n``. For deeper temporal MLPs the
time derivatives are the exact omnibias *directional jet*
(:func:`omnibias.jax.jet.mlp_jet`) propagated along the 1-D time axis --
still fully closed-form, with no ``jax.grad`` in the temporal path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.spec import ActivationSpec
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.jet import jet_to_tower, mlp_jet
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.base import FieldBase

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def _axis_basis_dn(
    x: Array, k_vec: Array, K: int, order: int,
) -> Array:
    """Per-axis basis (or its derivative) at points ``x``.

    Returns ``(B, 2K+1)``. Layout is
    ``(1, cos(k_1 x), ..., cos(k_K x), sin(k_1 x), ..., sin(k_K x))``.
    """
    arg = x[..., None] * k_vec
    cos = jnp.cos(arg)
    sin = jnp.sin(arg)
    if order == 0:
        ones = jnp.ones_like(cos[..., :1])
        return jnp.concatenate([ones, cos, sin], axis=-1)
    k_n = k_vec ** order
    rem = order % 4
    if rem == 0:
        cos_slot_fn = cos
        sin_slot_fn = sin
    elif rem == 1:
        cos_slot_fn = -sin
        sin_slot_fn = cos
    elif rem == 2:
        cos_slot_fn = -cos
        sin_slot_fn = -sin
    else:
        cos_slot_fn = sin
        sin_slot_fn = -cos
    cos_slot = k_n * cos_slot_fn
    sin_slot = k_n * sin_slot_fn
    zero = jnp.zeros_like(cos[..., :1])
    return jnp.concatenate([zero, cos_slot, sin_slot], axis=-1)


def _multi_axis_einsum(a: Array, bases: list[Array]) -> Array:
    D = len(bases)
    if D == 0:
        return a
    mode_letters = "jklmnopqr"[:D]
    eq = (
        "bC" + mode_letters + ","
        + ",".join("b" + m for m in mode_letters)
        + "->bC"
    )
    return jnp.einsum(eq, a, *bases)


@dataclass(frozen=True)
class SpectralVectorField(FieldBase):
    """JAX D-dim periodic Fourier vector field with omnibias time head.

    Trainable arrays
    ----------------
    ``W_t``    : ``(M, 1)`` - omnibias time layer weight.
    ``beta_t`` : ``(M,)``   - omnibias time layer bias.
    ``inner_W``: tuple of ``(M, M)`` arrays for inner hidden layers.
    ``inner_b``: tuple of ``(M,)`` arrays for inner hidden layer biases.
    ``V``      : ``(C * (2K+1)^D, M)`` - readout weight.
    ``b_t``    : ``(C * (2K+1)^D,)``   - readout bias.

    Static metadata is held in the dataclass and travels in the pytree
    auxiliary tree-def.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    spec: JaxActivationSpec
    W_t: Array
    beta_t: Array
    inner_W: tuple[Array, ...]
    inner_b: tuple[Array, ...]
    V: Array
    b_t: Array
    K: int = 8
    L: tuple[float, ...] = (2.0 * math.pi,)
    time_hidden: int = 64
    time_depth: int = 1

    @property
    def D_spatial(self) -> int:
        return self.coordinate_spec.n_spatial

    @property
    def C(self) -> int:
        return self.components.n_components

    @property
    def _modes_per_axis(self) -> int:
        return 2 * self.K + 1

    @property
    def _out_per_component(self) -> int:
        return self._modes_per_axis ** self.D_spatial

    @property
    def _coord_axis_indices(self) -> tuple[int, ...]:
        return tuple(
            self.coordinate_spec.axis_index(a)
            for a in self.coordinate_spec.spatial_axes
        )

    @property
    def _time_axis_idx(self) -> int:
        ta = self.coordinate_spec.time_axis
        assert ta is not None
        return self.coordinate_spec.axis_index(ta)

    def k_vec(self, full_axis: int) -> Array:
        """Per-axis ``k`` vector ``(K,)`` for spatial axis ``full_axis``."""
        d = self._coord_axis_indices.index(full_axis)
        L_a = self.L[d]
        ks = jnp.arange(1, self.K + 1, dtype=self.W_t.dtype)
        return (2.0 * math.pi * ks) / L_a

    def _spatial_axis_index(self, full_axis: int) -> int:
        sa = self._coord_axis_indices
        if full_axis not in sa:
            raise ValueError(
                f"axis {full_axis} is not a spatial axis ({sa!r})"
            )
        return sa.index(full_axis)

    def _hidden_t(self, t: Array) -> tuple[Array, Array]:
        """Return ``(z, h)`` where ``z`` is the omnibias pre-activation
        and ``h`` is the final hidden layer output."""
        if t.ndim == 1:
            t_in = t[..., None]
        else:
            t_in = t
        z = t_in @ self.W_t.T + self.beta_t
        h = self.spec.forward(z)
        for W_l, b_l in zip(self.inner_W, self.inner_b, strict=False):
            h = self.spec.forward(h @ W_l.T + b_l)
        return z, h

    def _hidden_t_and_dt(self, t: Array, *, t_order: int = 0) -> Array:
        if t_order == 0:
            _, h = self._hidden_t(t)
            return h
        if self.time_depth > 1:
            return self._hidden_t_and_dt_via_jet(t, t_order=t_order)
        z, _ = self._hidden_t(t)
        fp = self.spec.fastpath
        if fp is None:
            raise ValueError(
                f"activation {self.spec.name!r} has no fastpath kernel"
            )
        sigma_n = fp(z, t_order)
        W_t_pow = self.W_t.squeeze(-1) ** t_order
        return sigma_n * W_t_pow

    def _time_layers(self) -> list[tuple[Array, Array, JaxActivationSpec]]:
        """Temporal MLP as ``mlp_jet`` layers (all activated; the linear readout
        ``V`` is applied after and commutes with ``d/dt``)."""
        layers: list[tuple[Array, Array, JaxActivationSpec]] = [
            (self.W_t, self.beta_t, self.spec),
        ]
        for W_l, b_l in zip(self.inner_W, self.inner_b, strict=False):
            layers.append((W_l, b_l, self.spec))
        return layers

    def _hidden_t_and_dt_via_jet(
        self, t: Array, *, t_order: int,
    ) -> Array:
        """Exact ``d^t_order h(t) / dt^t_order`` of the deep temporal MLP via the
        omnibias directional jet (:func:`omnibias.jax.jet.mlp_jet`) along the 1-D
        time axis -- closed form, no ``jax.grad``, bit-identical to the torch twin.
        """
        if t_order < 1:
            raise ValueError(f"t_order must be >= 1, got {t_order}")
        layers = self._time_layers()
        v = jnp.ones((1,), dtype=self.W_t.dtype)

        def per_b(t_scalar: Array) -> Array:
            x0 = t_scalar.reshape(1)
            tower = jet_to_tower(mlp_jet(x0, v, layers, t_order))
            return tower[t_order]

        out: Array = jax.vmap(per_b)(t.reshape(-1))
        return out

    def _a_from_hidden(self, h: Array, *, t_order: int = 0) -> Array:
        """Apply the live readout ``(V, b_t)`` to a cached temporal feature ``h``."""
        if t_order == 0:
            a = self.b_t + h @ self.V.T
        else:
            a = h @ self.V.T
        return a.reshape(
            -1, self.C, *([self._modes_per_axis] * self.D_spatial),
        )

    def _coeff_blocks_at_t(self, t: Array, *, t_order: int = 0) -> Array:
        h = self._hidden_t_and_dt(t, t_order=t_order)
        return self._a_from_hidden(h, t_order=t_order)

    def _pre_activations(self, coords: Array) -> Array:
        t = coords[..., self._time_axis_idx]
        z, _ = self._hidden_t(t)
        return z

    def _bases_for_state(
        self,
        state: FieldState,
        *,
        order_per_axis: tuple[int, ...] | None = None,
    ) -> list[Array]:
        if order_per_axis is None:
            order_per_axis = tuple(0 for _ in range(self.D_spatial))
        if len(order_per_axis) != self.D_spatial:
            raise ValueError(
                f"order_per_axis has length {len(order_per_axis)} but "
                f"D_spatial = {self.D_spatial}"
            )
        coords = state.coords
        bases: list[Array] = []
        for d, full_axis in enumerate(self._coord_axis_indices):
            x_d = coords[..., full_axis]
            kv = self.k_vec(full_axis)
            bases.append(_axis_basis_dn(x_d, kv, self.K, order_per_axis[d]))
        return bases

    def _coeff_blocks_for_state(
        self, state: FieldState, *, t_order: int = 0,
    ) -> Array:
        """Return ``a(t)`` from a readout-independent cached temporal feature ``h``."""
        key = f"spectral_h_t{t_order}"
        h = state.extra.get(key)
        if h is None:
            t = state.coords[..., self._time_axis_idx]
            h = self._hidden_t_and_dt(t, t_order=t_order)
            state.extra[key] = h
        return self._a_from_hidden(h, t_order=t_order)

    def value_component(self, state: FieldState, name: str) -> Array:
        ci = self.components.index(name)
        a = self._coeff_blocks_for_state(state)
        bases = self._bases_for_state(state)
        return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        ci = self.components.index(name)
        if axis == self._time_axis_idx:
            a = self._coeff_blocks_for_state(state, t_order=order)
            bases = self._bases_for_state(state)
            return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]
        d = self._spatial_axis_index(axis)
        order_per_axis = tuple(
            order if i == d else 0 for i in range(self.D_spatial)
        )
        a = self._coeff_blocks_for_state(state)
        bases = self._bases_for_state(state, order_per_axis=order_per_axis)
        return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        ci = self.components.index(name)
        order_per_axis = [0] * self.D_spatial
        t_order = 0
        for a, o in zip(axes, orders, strict=False):
            if a == self._time_axis_idx:
                t_order += o
            else:
                d = self._spatial_axis_index(a)
                order_per_axis[d] += o
        a_t = self._coeff_blocks_for_state(state, t_order=t_order)
        bases = self._bases_for_state(
            state, order_per_axis=tuple(order_per_axis),
        )
        return _multi_axis_einsum(a_t[:, ci:ci + 1], bases)[:, 0]

    def biharmonic(self, state: FieldState, name: str) -> Array:
        return self.polylaplacian(state, name, k=2)

    def polylaplacian(
        self, state: FieldState, name: str, *, k: int,
    ) -> Array:
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        a = self._coeff_blocks_for_state(state)
        ci = self.components.index(name)
        k2_tables = []
        for full_axis in self._coord_axis_indices:
            kv = self.k_vec(full_axis)
            k2 = kv * kv
            zero = jnp.zeros((1,), dtype=kv.dtype)
            k2_full = jnp.concatenate([zero, k2, k2])
            k2_tables.append(k2_full)
        K2_sum = None
        for d, t in enumerate(k2_tables):
            shape = [1] * self.D_spatial
            shape[d] = -1
            view = t.reshape(*shape)
            K2_sum = view if K2_sum is None else K2_sum + view
        assert K2_sum is not None
        sign = (-1.0) ** k
        mult = sign * (K2_sum ** k)
        a_c = a[:, ci]
        a_w = a_c * mult[None, ...]
        bases = self._bases_for_state(state)
        a_w_4d = a_w[:, None]
        return _multi_axis_einsum(a_w_4d, bases)[:, 0]

    def __repr__(self) -> str:
        return (
            f"SpectralVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, K={self.K}, L={self.L}, "
            f"time_hidden={self.time_hidden}, time_depth={self.time_depth}, "
            f"activation={self.spec.name!r})"
        )


# --- pytree registration --------------------------------------------


def _tree_flatten(field: SpectralVectorField):
    leaves = (
        field.W_t,
        field.beta_t,
        *field.inner_W,
        *field.inner_b,
        field.V,
        field.b_t,
    )
    aux = (
        field.coordinate_spec,
        field.components,
        field.spec,
        field.K,
        field.L,
        field.time_hidden,
        field.time_depth,
        len(field.inner_W),
    )
    return leaves, aux


def _tree_unflatten(aux, leaves):
    (
        coordinate_spec,
        components,
        spec,
        K,
        L,
        time_hidden,
        time_depth,
        n_inner,
    ) = aux
    leaves = tuple(leaves)
    W_t = leaves[0]
    beta_t = leaves[1]
    inner_W = tuple(leaves[2:2 + n_inner])
    inner_b = tuple(leaves[2 + n_inner:2 + 2 * n_inner])
    V = leaves[2 + 2 * n_inner]
    b_t = leaves[3 + 2 * n_inner]
    obj = SpectralVectorField.__new__(SpectralVectorField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "spec", spec)
    object.__setattr__(obj, "W_t", W_t)
    object.__setattr__(obj, "beta_t", beta_t)
    object.__setattr__(obj, "inner_W", inner_W)
    object.__setattr__(obj, "inner_b", inner_b)
    object.__setattr__(obj, "V", V)
    object.__setattr__(obj, "b_t", b_t)
    object.__setattr__(obj, "K", K)
    object.__setattr__(obj, "L", L)
    object.__setattr__(obj, "time_hidden", time_hidden)
    object.__setattr__(obj, "time_depth", time_depth)
    return obj


jax.tree_util.register_pytree_node(
    SpectralVectorField, _tree_flatten, _tree_unflatten,
)


# --- builder ---------------------------------------------------------


def make_spectral_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    K: int = 8,
    L: float | tuple[float, ...] = 2.0 * math.pi,
    time_hidden: int = 64,
    time_depth: int = 1,
    activation: str | JaxActivationSpec = "tanh",
    weight_init_scale: float = 1.0,
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> SpectralVectorField:
    """Build an initialised :class:`SpectralVectorField`."""
    if coordinate_spec.time_axis is None:
        raise ValueError(
            "SpectralVectorField requires a time axis on the coordinate spec"
        )
    spec = (
        activation if isinstance(activation, ActivationSpec)
        else get_activation(activation)
    )
    if spec.fastpath is None:
        raise ValueError(
            f"SpectralVectorField requires a fast-path activation; "
            f"{spec.name!r} has none."
        )
    K = int(K)
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    D_spatial = coordinate_spec.n_spatial
    if isinstance(L, int | float):
        L_t: tuple[float, ...] = tuple(float(L) for _ in range(D_spatial))
    else:
        L_t = tuple(float(x) for x in L)
        if len(L_t) != D_spatial:
            raise ValueError(
                f"L tuple has length {len(L_t)} but coordinate spec has "
                f"{D_spatial} spatial axes"
            )
    if time_depth < 1:
        raise ValueError(f"time_depth must be >= 1, got {time_depth}")

    modes_per_axis = 2 * K + 1
    out_per_component = modes_per_axis ** D_spatial
    out_dim = components.n_components * out_per_component

    key = jax.random.PRNGKey(seed)
    keys = list(jax.random.split(key, 4 + 2 * (time_depth - 1)))
    W_t = jax.random.normal(keys[0], (time_hidden, 1), dtype=dtype) * weight_init_scale
    beta_t = jnp.zeros((time_hidden,), dtype=dtype)
    inner_W = []
    inner_b = []
    for layer_i in range(time_depth - 1):
        kw = keys[2 + 2 * layer_i]
        W_l = jax.random.normal(kw, (time_hidden, time_hidden), dtype=dtype) * (
            weight_init_scale / math.sqrt(time_hidden)
        )
        b_l = jnp.zeros((time_hidden,), dtype=dtype)
        inner_W.append(W_l)
        inner_b.append(b_l)
    V = jnp.zeros((out_dim, time_hidden), dtype=dtype)
    b_t = jnp.zeros((out_dim,), dtype=dtype)

    return SpectralVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        spec=spec,
        W_t=W_t,
        beta_t=beta_t,
        inner_W=tuple(inner_W),
        inner_b=tuple(inner_b),
        V=V,
        b_t=b_t,
        K=K,
        L=L_t,
        time_hidden=time_hidden,
        time_depth=time_depth,
    )


__all__ = ["SpectralVectorField", "make_spectral_vector_field"]

# Marker read by the omnibias-fields backend ops to select the dispatch path.
SpectralVectorField._omnibias_dispatch = "spectral"
SpectralVectorField._omnibias_readout_independent = True

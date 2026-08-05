# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``ChebyshevVectorField`` for the JAX backend.

JAX twin of :class:`omnibias.pinn.torch.fields.chebyshev.ChebyshevVectorField`.
Same architecture (D-dim Chebyshev-T basis, omnibias temporal head,
multi-component output), same closed-form math.
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
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.base import FieldBase

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def _chebyshev_differentiation_matrix(K: int, dtype: Any) -> Array:
    N = K + 1
    rows = []
    for n in range(N):
        cn = 2.0 if n == 0 else 1.0
        row = [0.0] * N
        for m in range(n + 1, N):
            if (m + n) % 2 == 1:
                row[m] = (2.0 * m) / cn
        rows.append(row)
    return jnp.asarray(rows, dtype=dtype)


def _chebyshev_basis(x: Array, K: int) -> Array:
    cols: list[Array] = [jnp.ones_like(x)]
    if K >= 1:
        cols.append(x)
    for n in range(1, K):
        cols.append(2.0 * x * cols[n] - cols[n - 1])
    return jnp.stack(cols, axis=-1)


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
class ChebyshevVectorField(FieldBase):
    """JAX D-dim Chebyshev-T vector field with omnibias time head."""

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    spec: JaxActivationSpec
    W_t: Array
    beta_t: Array
    inner_W: tuple[Array, ...]
    inner_b: tuple[Array, ...]
    V: Array
    b_t: Array
    D_mat: Array
    K: int = 8
    domain: tuple[tuple[float, float], ...] = ((-1.0, 1.0),)
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
        return self.K + 1

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

    def _spatial_axis_index(self, full_axis: int) -> int:
        sa = self._coord_axis_indices
        if full_axis not in sa:
            raise ValueError(
                f"axis {full_axis} is not a spatial axis ({sa!r})"
            )
        return sa.index(full_axis)

    def _rescale_to_unit(self, x: Array, d: int) -> Array:
        lo, hi = self.domain[d]
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def _chain_factor(self, d: int, order: int) -> float:
        lo, hi = self.domain[d]
        return (2.0 / (hi - lo)) ** order

    def _hidden_t(self, t: Array) -> tuple[Array, Array]:
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
            raise NotImplementedError(
                "Closed-form time derivatives only implemented for time_depth=1"
            )
        z, _ = self._hidden_t(t)
        fp = self.spec.fastpath
        if fp is None:
            raise ValueError(
                f"activation {self.spec.name!r} has no fastpath kernel"
            )
        sigma_n = fp(z, t_order)
        W_t_pow = self.W_t.squeeze(-1) ** t_order
        return sigma_n * W_t_pow

    def _coeff_blocks_at_t(self, t: Array, *, t_order: int = 0) -> Array:
        h = self._hidden_t_and_dt(t, t_order=t_order)
        if t_order == 0:
            a = self.b_t + h @ self.V.T
        else:
            a = h @ self.V.T
        return a.reshape(
            -1, self.C, *([self._modes_per_axis] * self.D_spatial),
        )

    def _pre_activations(self, coords: Array) -> Array:
        t = coords[..., self._time_axis_idx]
        z, _ = self._hidden_t(t)
        return z

    def _basis_dn_axis(self, x: Array, d: int, order: int) -> Array:
        xi = self._rescale_to_unit(x, d)
        T = _chebyshev_basis(xi, self.K)
        if order == 0:
            return T
        Dn = self.D_mat
        Dn_pow = jnp.eye(self.K + 1, dtype=Dn.dtype)
        for _ in range(order):
            Dn_pow = Dn_pow @ Dn
        chain = self._chain_factor(d, order)
        return chain * (T @ Dn_pow)

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
            bases.append(self._basis_dn_axis(x_d, d, order_per_axis[d]))
        return bases

    def _coeff_blocks_for_state(
        self, state: FieldState, *, t_order: int = 0,
    ) -> Array:
        key = f"chebyshev_a_t{t_order}"
        cached = state.extra.get(key)
        if cached is not None:
            return cached
        t = state.coords[..., self._time_axis_idx]
        a = self._coeff_blocks_at_t(t, t_order=t_order)
        state.extra[key] = a
        return a

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
        from itertools import product
        ci = self.components.index(name)
        a = self._coeff_blocks_for_state(state)
        result = None
        for ms in product(range(k + 1), repeat=self.D_spatial):
            if sum(ms) != k:
                continue
            coeff = math.factorial(k)
            for m in ms:
                coeff //= math.factorial(m)
            order_per_axis = tuple(2 * m for m in ms)
            bases = self._bases_for_state(
                state, order_per_axis=order_per_axis,
            )
            term = _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]
            term = float(coeff) * term
            result = term if result is None else result + term
        assert result is not None
        return result

    def __repr__(self) -> str:
        return (
            f"ChebyshevVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, K={self.K}, "
            f"domain={self.domain}, "
            f"time_hidden={self.time_hidden}, time_depth={self.time_depth}, "
            f"activation={self.spec.name!r})"
        )


# --- pytree registration -------------------------------------------


def _tree_flatten(field: ChebyshevVectorField):
    leaves = (
        field.W_t,
        field.beta_t,
        *field.inner_W,
        *field.inner_b,
        field.V,
        field.b_t,
        field.D_mat,
    )
    aux = (
        field.coordinate_spec,
        field.components,
        field.spec,
        field.K,
        field.domain,
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
        domain,
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
    D_mat = leaves[4 + 2 * n_inner]
    obj = ChebyshevVectorField.__new__(ChebyshevVectorField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "spec", spec)
    object.__setattr__(obj, "W_t", W_t)
    object.__setattr__(obj, "beta_t", beta_t)
    object.__setattr__(obj, "inner_W", inner_W)
    object.__setattr__(obj, "inner_b", inner_b)
    object.__setattr__(obj, "V", V)
    object.__setattr__(obj, "b_t", b_t)
    object.__setattr__(obj, "D_mat", D_mat)
    object.__setattr__(obj, "K", K)
    object.__setattr__(obj, "domain", domain)
    object.__setattr__(obj, "time_hidden", time_hidden)
    object.__setattr__(obj, "time_depth", time_depth)
    return obj


jax.tree_util.register_pytree_node(
    ChebyshevVectorField, _tree_flatten, _tree_unflatten,
)


# --- builder ---------------------------------------------------------


def make_chebyshev_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    K: int = 8,
    domain: tuple[tuple[float, float], ...] | None = None,
    time_hidden: int = 64,
    time_depth: int = 1,
    activation: str | JaxActivationSpec = "tanh",
    weight_init_scale: float = 1.0,
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> ChebyshevVectorField:
    if coordinate_spec.time_axis is None:
        raise ValueError(
            "ChebyshevVectorField requires a time axis on the coordinate spec"
        )
    spec = (
        activation if isinstance(activation, ActivationSpec)
        else get_activation(activation)
    )
    if spec.fastpath is None:
        raise ValueError(
            f"ChebyshevVectorField requires a fast-path activation; "
            f"{spec.name!r} has none."
        )
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    D_spatial = coordinate_spec.n_spatial
    if domain is None:
        spec_domain = coordinate_spec.domain
        if spec_domain is None:
            resolved = tuple((-1.0, 1.0) for _ in range(D_spatial))
        else:
            resolved = tuple(
                spec_domain[coordinate_spec.axis_index(a)]
                for a in coordinate_spec.spatial_axes
            )
    else:
        resolved = tuple((float(lo), float(hi)) for (lo, hi) in domain)
        if len(resolved) != D_spatial:
            raise ValueError(
                f"domain has length {len(resolved)} but coordinate spec has "
                f"{D_spatial} spatial axes"
            )
    if time_depth < 1:
        raise ValueError(f"time_depth must be >= 1, got {time_depth}")

    modes_per_axis = K + 1
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
    D_mat = _chebyshev_differentiation_matrix(K, dtype=dtype)

    return ChebyshevVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        spec=spec,
        W_t=W_t,
        beta_t=beta_t,
        inner_W=tuple(inner_W),
        inner_b=tuple(inner_b),
        V=V,
        b_t=b_t,
        D_mat=D_mat,
        K=K,
        domain=resolved,
        time_hidden=time_hidden,
        time_depth=time_depth,
    )


__all__ = ["ChebyshevVectorField", "make_chebyshev_vector_field"]

# Marker read by the omnibias-fields backend ops to select the dispatch path.
ChebyshevVectorField._omnibias_dispatch = "chebyshev"

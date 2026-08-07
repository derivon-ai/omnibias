# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``OneLayerVectorField`` -- JAX twin of the torch one-layer field.

Same architecture, same mathematical contract, same closed-form chain
rule. Differences are stylistic: the field is a ``dataclass(frozen=True)``
holding ``jax.Array`` parameters rather than a ``torch.nn.Module``.

For training, register the field as a pytree node so trainable arrays
travel through ``jax.grad`` / ``jax.jit``::

    import jax
    field = OneLayerVectorField(...)
    flat, treedef = jax.tree_util.tree_flatten(field)
    # flat == [W, beta, c, b]; treedef encodes the static metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.spec import ActivationSpec
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.base import FieldBase


@dataclass(frozen=True)
class OneLayerVectorField(FieldBase):
    """JAX one-layer vector PINN field.

    Parameters live as :class:`jax.Array` on the dataclass; metadata is
    plain Python. The class is registered as a pytree so that the leaf
    arrays travel through ``jax.grad``/``jax.jit`` while the metadata
    travels in the auxiliary tree-def.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    spec: JaxActivationSpec
    W: Array          # (H, D)
    beta: Array       # (H,)
    c: Array          # (C, H)
    b: Array          # (C,)
    hidden: int = 64

    def _pre_activations(self, coords: Array) -> Array:
        # z[b, h] = sum_i W[h, i] coords[b, i] + beta[h]
        return coords @ self.W.T + self.beta

    def _sigma(self, z: Array) -> Array:
        return self.spec.forward(z)

    def _sigma_n(self, z: Array, order: int) -> Array:
        if order == 0:
            return self.spec.forward(z)
        fp = self.spec.fastpath
        if fp is None:
            raise ValueError(
                f"activation {self.spec.name!r} has no fastpath kernel"
            )
        return fp(z, order)

    def _spatial_axes(self) -> tuple[int, ...]:
        ta = self.coordinate_spec.time_axis
        return tuple(
            i for i, n in enumerate(self.coordinate_spec.axes) if n != ta
        )

    def _row_norm_sq_spatial(self) -> Array:
        sa = self._spatial_axes()
        if not sa:
            return jnp.zeros(self.hidden, dtype=self.W.dtype)
        W_spatial = self.W[:, list(sa)]
        return (W_spatial * W_spatial).sum(axis=-1)

    # -- closed-form helpers ---------------------------------------

    def value(self, sigma_z: Array, name: str) -> Array:
        ci = self.components.index(name)
        return self.b[ci] + sigma_z @ self.c[ci]

    def value_all(self, sigma_z: Array) -> Array:
        return self.b + sigma_z @ self.c.T

    def forward_values(self, coords: Array) -> Array:
        """Return ``(B, C)`` component values straight from ``coords``.

        The one-line contract a composite field (a partition of unity, a
        multi-patch decomposition) needs from a sub-solution, so composites can
        mix field *types* instead of hard-coding this one.
        """
        return self.value_all(self._sigma(self._pre_activations(coords)))

    def first_partial(self, sigma_p: Array, name: str, axis: int) -> Array:
        ci = self.components.index(name)
        return (sigma_p * self.W[:, axis]) @ self.c[ci]

    def nth_partial(
        self, sigma_n: Array, name: str, axis: int, order: int,
    ) -> Array:
        ci = self.components.index(name)
        return (sigma_n * self.W[:, axis] ** order) @ self.c[ci]

    def mixed_partial(
        self,
        sigma_n: Array,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        ci = self.components.index(name)
        W_factor = jnp.ones(self.hidden, dtype=self.W.dtype)
        for a, o in zip(axes, orders, strict=False):
            W_factor = W_factor * self.W[:, a] ** o
        return (sigma_n * W_factor) @ self.c[ci]

    def gradient_full(self, sigma_p: Array, name: str) -> Array:
        ci = self.components.index(name)
        return (sigma_p * self.c[ci]) @ self.W

    def gradient_spatial(self, sigma_p: Array, name: str) -> Array:
        sa = self._spatial_axes()
        full = self.gradient_full(sigma_p, name)
        if not sa:
            return full[..., :0]
        return full[..., list(sa)]

    def laplacian(self, sigma_pp: Array, name: str) -> Array:
        ci = self.components.index(name)
        row_norm_sq = self._row_norm_sq_spatial()
        return sigma_pp @ (self.c[ci] * row_norm_sq)

    def hessian_full(self, sigma_pp: Array, name: str) -> Array:
        ci = self.components.index(name)
        weights = sigma_pp * self.c[ci]
        return jnp.einsum("bh,hi,hj->bij", weights, self.W, self.W)

    def hessian_spatial(self, sigma_pp: Array, name: str) -> Array:
        sa = self._spatial_axes()
        full = self.hessian_full(sigma_pp, name)
        if not sa:
            return full[..., :0, :0]
        idx = jnp.array(list(sa))
        return full[..., idx, :][..., :, idx]

    def polylaplacian(self, sigma_2k: Array, name: str, k: int) -> Array:
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        ci = self.components.index(name)
        row_norm_sq = self._row_norm_sq_spatial()
        return sigma_2k @ (self.c[ci] * row_norm_sq ** k)

    # -- pytree registration ---------------------------------------

    def tree_flatten(self):
        leaves = (self.W, self.beta, self.c, self.b)
        aux = (self.coordinate_spec, self.components, self.spec, self.hidden)
        return leaves, aux

    # NOTE: pytree registration is performed below via
    # ``jax.tree_util.register_pytree_node`` using the module-level
    # ``_tree_flatten`` / ``_tree_unflatten`` helpers (frozen dataclasses
    # cannot be re-constructed via the standard ``cls(...)`` path because
    # all fields have to be set via ``object.__setattr__``).


# Pytree registration keeps the field through jit/grad transformations.
def _tree_unflatten(aux, leaves):
    coordinate_spec, components, spec, hidden = aux
    W, beta, c, b = leaves
    obj = OneLayerVectorField.__new__(OneLayerVectorField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "spec", spec)
    object.__setattr__(obj, "W", W)
    object.__setattr__(obj, "beta", beta)
    object.__setattr__(obj, "c", c)
    object.__setattr__(obj, "b", b)
    object.__setattr__(obj, "hidden", hidden)
    return obj


def _tree_flatten(field: OneLayerVectorField):
    leaves = (field.W, field.beta, field.c, field.b)
    aux = (field.coordinate_spec, field.components, field.spec, field.hidden)
    return leaves, aux


jax.tree_util.register_pytree_node(
    OneLayerVectorField, _tree_flatten, _tree_unflatten,
)


# ----- builder helper that handles parameter init -------------------


def make_one_layer_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    base: str | JaxActivationSpec = "tanh",
    weight_init_scale: float | None = None,
    bias_init: str = "zeros",
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> OneLayerVectorField:
    """Initialise a fresh :class:`OneLayerVectorField` with random params.

    This mirrors the torch field's constructor (which initialises params
    inside ``__init__``); on JAX we keep params separate so the dataclass
    stays pure / functional.
    """
    spec = base if isinstance(base, ActivationSpec) else get_activation(base)
    if spec.fastpath is None:
        raise ValueError(
            f"OneLayerVectorField requires a fast-path activation; "
            f"{spec.name!r} has none."
        )
    D = coordinate_spec.ndim
    C = components.n_components
    if hidden <= 0:
        raise ValueError(f"hidden must be > 0, got {hidden}")

    scale = (1.0 / math.sqrt(D)) if weight_init_scale is None else float(weight_init_scale)
    key = jax.random.PRNGKey(seed)
    k_W, k_beta, k_c, k_b = jax.random.split(key, 4)
    W = jax.random.normal(k_W, (hidden, D), dtype=dtype) * scale
    c_param = jax.random.normal(k_c, (C, hidden), dtype=dtype) * (
        scale / math.sqrt(hidden)
    )
    if bias_init == "zeros":
        beta = jnp.zeros((hidden,), dtype=dtype)
        b = jnp.zeros((C,), dtype=dtype)
    elif bias_init == "normal":
        beta = jax.random.normal(k_beta, (hidden,), dtype=dtype) * scale
        b = jax.random.normal(k_b, (C,), dtype=dtype) * scale
    else:
        raise ValueError(f"unknown bias_init: {bias_init!r}")
    return OneLayerVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        spec=spec,
        W=W, beta=beta, c=c_param, b=b,
        hidden=hidden,
    )


__all__ = ["OneLayerVectorField", "make_one_layer_vector_field"]

# Marker read by the omnibias-fields backend ops to select the closed-form
# sigma-tower reduction path (avoids a fields -> pinn import cycle).
OneLayerVectorField._omnibias_dispatch = "one_layer"
OneLayerVectorField._omnibias_readout_independent = True

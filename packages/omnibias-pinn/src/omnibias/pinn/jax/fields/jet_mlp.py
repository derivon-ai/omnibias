# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deep and Fourier-feature PINN fields on the exact multivariate jet (JAX twin).

Bit-identical twin of :mod:`omnibias.pinn.torch.fields.jet_mlp`; see that module
for the full exposition. Differences are stylistic: the fields are
``dataclass(frozen=True)`` objects registered as JAX pytrees (so the wrapped
architecture's arrays travel through ``jax.grad`` / ``jax.jit``) rather than
:class:`torch.nn.Module` subclasses.

Both backends read their partials off the same
:func:`omnibias.core.multi_index.multi_indices` row order and drive the same
Faa di Bruno recursion, so the derivatives agree to double-precision round-off.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.multi_index import index_position, multi_index_factorial, multi_indices
from omnibias.jax.activations import JaxActivationSpec
from omnibias.jax.architectures.pinn import (
    FourierFeatureMLP,
    JetMLP,
    make_fourier_feature_mlp,
    make_jet_mlp,
    make_siren,
)
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.base import FieldBase

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState

    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | None]

#: ``FieldState.extra`` key under which the per-evaluation jet cache lives.
JET_CACHE_KEY = "_jet_mlp_jets"


def _polylaplacian_terms(n_spatial: int, k: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    r"""Multinomial expansion of ``Delta^k = (sum_i d_i^2)^k``.

    Returns ``(beta, k! / beta!)`` pairs over the spatial axes with ``|beta| = k``,
    so that ``Delta^k f = sum_beta (k! / beta!) D^{2 beta} f``.
    """
    k_fact = math.factorial(k)
    return tuple(
        (beta, k_fact // multi_index_factorial(beta))
        for beta in multi_indices(n_spatial, k)
        if sum(beta) == k
    )


class JetNet(Protocol):
    """The architecture surface a ``jet_mlp`` field needs.

    Structural rather than nominal so the field module never has to enumerate the
    architectures (the multi-scale ones live in a module that imports *this* one).
    Implemented by every :mod:`omnibias.jax.architectures` jet network.
    """

    in_dim: int
    out_dim: int

    def _point_hidden_jet(self, xi: Array, order: int) -> Array:
        """Single-point readout-independent jet (hidden layers only)."""
        ...

    def _apply_readout_jet(self, hidden_jet: Array) -> Array:
        """Push a (possibly batched) hidden jet through the live affine readout."""
        ...

    def _point_jet(self, xi: Array, order: int) -> Array:
        """Single-point multivariate jet, shape ``(M, out_dim)``."""
        ...

    def _check_fastpath(self, max_order: int) -> None:
        """Raise if any activation lacks a closed-form tower up to ``max_order``."""
        ...

    def value(self, x: Array) -> Array:
        """Plain network value, shape ``(..., out_dim)``."""
        ...


class _JetFieldOps:
    """State-method surface shared by the jet-backed JAX fields.

    Mixin over the frozen dataclasses below; it reads ``self.net``,
    ``self.jet_order``, ``self.coordinate_spec`` and ``self.components`` only, so it
    carries no dataclass fields of its own.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: JetNet
    jet_order: int

    # -- FieldBase plumbing: no single pre-activation tower ---------------------- #

    def _pre_activations(self, coords: Array) -> Array | None:
        """No single ``z``: a deep net has one pre-activation per layer."""
        return None

    # -- jet machinery ---------------------------------------------------------- #

    def _compute_hidden_jet(self, coords: Array, order: int) -> Array:
        """Batched readout-independent jet (hidden layers only).

        Routed through the architecture's ``_point_hidden_jet`` hook so a
        wrapper network -- a band mixture, attention -- overrides one method and
        the field inherits exact derivatives with a live readout.
        """
        out: Array = jax.vmap(lambda xi: self.net._point_hidden_jet(xi, order))(coords)
        return out

    def _jet_at_least(self, state: FieldState, order: int) -> tuple[Array, int]:
        """Return ``(jet, jet_order)`` with ``jet_order >= order``.

        The *hidden* jet is memoised on the state (readout-independent); the
        live affine readout is applied on every call so a frozen-feature column
        sweep against a reused state stays correct. Values take the plain
        forward path (:meth:`value_component`) and never populate this cache.
        """
        cache = cast("dict[int, Array]", state.extra.setdefault(JET_CACHE_KEY, {}))
        hidden: Array | None = None
        got_order = -1
        for cached_order in sorted(cache):
            if cached_order >= order:
                hidden = cache[cached_order]
                got_order = cached_order
                break
        if hidden is None:
            want = max(int(order), self.jet_order)
            hidden = self._compute_hidden_jet(state.coords, want)
            cache[want] = hidden
            got_order = want
        return self.net._apply_readout_jet(hidden), got_order

    def _partial(
        self, state: FieldState, name: str, alpha: tuple[int, ...], order: int
    ) -> Array:
        """``D^alpha f_name`` of shape ``(B,)``, off the live-readout jet."""
        jet, jet_order = self._jet_at_least(state, order)
        pos = index_position(self.coordinate_spec.ndim, jet_order)
        ci = self.components.index(name)
        return jet[:, pos[alpha], ci] * multi_index_factorial(alpha)

    def _spatial_axis_indices(self) -> tuple[int, ...]:
        return tuple(
            self.coordinate_spec.axis_index(a)
            for a in self.coordinate_spec.spatial_axes
        )

    # -- state-method path consumed by the fields ops dispatch ("jet_mlp") ------- #

    def forward_values(self, coords: Array) -> Array:
        """All component values ``(B, C)`` from the plain forward pass."""
        return self.net.value(coords)

    def value_component(self, state: FieldState, name: str) -> Array:
        """Component value from the plain forward pass (no jet).

        A value-only term -- a boundary-condition loss -- never pays for a jet.
        The forward pass and row 0 of the jet are the same function to float64
        round-off (pinned by the attention / jet field tests); derivatives
        still go through :meth:`_jet_at_least` so a residual that also needs
        partials shares one cached hidden jet.
        """
        ci = self.components.index(name)
        return self.forward_values(state.coords)[:, ci]

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1
    ) -> Array:
        r"""``d^order f_name / dx_axis^order`` from the exact jet."""
        if order < 1:
            raise ValueError(f"derivative order must be >= 1, got {order}")
        D = self.coordinate_spec.ndim
        alpha = tuple(order if i == axis else 0 for i in range(D))
        return self._partial(state, name, alpha, order)

    def mixed_partial(
        self, state: FieldState, name: str, axes: tuple[int, ...], orders: tuple[int, ...]
    ) -> Array:
        D = self.coordinate_spec.ndim
        acc = [0] * D
        for a, o in zip(axes, orders, strict=False):
            acc[a] += int(o)
        total = sum(acc)
        if total == 0:
            return self.value_component(state, name)
        return self._partial(state, name, tuple(acc), total)

    # -- fast paths: one jet already holds the whole gradient / Hessian ---------- #

    def gradient_full(self, state: FieldState, name: str) -> Array:
        """``nabla f_name`` over *all* axes, shape ``(B, D)``."""
        D = self.coordinate_spec.ndim
        cols = [
            self._partial(state, name, tuple(1 if j == i else 0 for j in range(D)), 1)
            for i in range(D)
        ]
        return jnp.stack(cols, axis=-1)

    def hessian_full(self, state: FieldState, name: str) -> Array:
        """Full Hessian over all axes, shape ``(B, D, D)``."""
        D = self.coordinate_spec.ndim
        rows = []
        for i in range(D):
            row = []
            for j in range(D):
                alpha = tuple(
                    (1 if k == i else 0) + (1 if k == j else 0) for k in range(D)
                )
                row.append(self._partial(state, name, alpha, 2))
            rows.append(jnp.stack(row, axis=-1))
        return jnp.stack(rows, axis=-2)

    def laplacian(self, state: FieldState, name: str) -> Array:
        """``Delta f_name`` over the spatial axes, shape ``(B,)``."""
        return self.polylaplacian(state, name, k=1)

    def polylaplacian(self, state: FieldState, name: str, *, k: int) -> Array:
        r"""``Delta^k f_name`` via the multinomial expansion of ``(sum_i d_i^2)^k``.

        Every term is a row of the *same* order-``2k`` jet, so the cost is one jet
        regardless of ``k``.
        """
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        sa = self._spatial_axis_indices()
        if not sa:
            raise ValueError("polylaplacian requires at least one spatial axis")
        D = self.coordinate_spec.ndim
        out: Array | None = None
        for beta, coeff in _polylaplacian_terms(len(sa), k):
            acc = [0] * D
            for axis, b in zip(sa, beta, strict=True):
                acc[axis] = 2 * b
            term = self._partial(state, name, tuple(acc), 2 * k)
            out = coeff * term if out is None else out + coeff * term
        assert out is not None
        return out

    def biharmonic(self, state: FieldState, name: str) -> Array:
        """``Delta^2 f_name`` of shape ``(B,)``."""
        return self.polylaplacian(state, name, k=2)


@dataclass(frozen=True)
class JetMLPVectorField(_JetFieldOps, FieldBase):
    r"""Deep multi-output PINN field with exact closed-form input derivatives (JAX).

    JAX twin of :class:`omnibias.pinn.torch.fields.JetMLPVectorField`. Build one with
    :func:`make_jet_mlp_vector_field` (or :func:`make_siren_vector_field`), which
    handles parameter initialisation; the dataclass itself stays pure / functional.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: JetMLP
    jet_order: int = 2

    def __repr__(self) -> str:
        return (
            f"JetMLPVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, depth={self.net.depth}, "
            f"jet_order={self.jet_order})"
        )


@dataclass(frozen=True)
class FourierFeatureVectorField(_JetFieldOps, FieldBase):
    r"""Spectral-bias-mitigating PINN field with a random Fourier front end (JAX).

    JAX twin of :class:`omnibias.pinn.torch.fields.FourierFeatureVectorField`. Build
    one with :func:`make_fourier_feature_vector_field`.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: FourierFeatureMLP
    jet_order: int = 2

    @property
    def scales(self) -> tuple[float, ...]:
        """The frequency bands of the encoding."""
        return self.net.scales

    @property
    def feature_dim(self) -> int:
        """Width of the Fourier encoding ``2 * F * len(scales)``."""
        return self.net.feature_dim

    def __repr__(self) -> str:
        return (
            f"FourierFeatureVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, scales={self.scales}, "
            f"jet_order={self.jet_order})"
        )


# -- pytree registration --------------------------------------------------------- #
#
# The wrapped architecture is itself a registered pytree, so it can be the single
# leaf here and JAX recurses into its weight arrays. Frozen dataclasses cannot be
# rebuilt through ``cls(...)`` during unflattening (JAX passes tracers and object
# placeholders), hence the ``__new__`` + ``object.__setattr__`` construction, the
# same pattern as :mod:`omnibias.pinn.jax.fields.one_layer`.

_Aux = tuple[CoordinateSpec, ComponentSpec, int]


def _flatten(
    field: JetMLPVectorField | FourierFeatureVectorField,
) -> tuple[tuple[Any], _Aux]:
    return (field.net,), (field.coordinate_spec, field.components, field.jet_order)


def _jet_unflatten(aux: _Aux, leaves: tuple[Any]) -> JetMLPVectorField:
    obj = JetMLPVectorField.__new__(JetMLPVectorField)
    _populate(obj, aux, leaves)
    return obj


def _fourier_unflatten(aux: _Aux, leaves: tuple[Any]) -> FourierFeatureVectorField:
    obj = FourierFeatureVectorField.__new__(FourierFeatureVectorField)
    _populate(obj, aux, leaves)
    return obj


def _populate(obj: object, aux: _Aux, leaves: tuple[Any]) -> None:
    coordinate_spec, components, jet_order = aux
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "net", leaves[0])
    object.__setattr__(obj, "jet_order", jet_order)


jax.tree_util.register_pytree_node(JetMLPVectorField, _flatten, _jet_unflatten)
jax.tree_util.register_pytree_node(
    FourierFeatureVectorField, _flatten, _fourier_unflatten
)


# -- builders that handle parameter init ----------------------------------------- #


def validate_jet_field(
    net: JetNet,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    jet_order: int,
) -> None:
    """Shape / order / closed-form checks every ``jet_mlp`` field builder runs.

    Fails loudly at construction rather than deep inside the jet kernel: an
    activation without a closed-form fast path cannot back this field at all.
    """
    if net.in_dim != coordinate_spec.ndim:
        raise ValueError(
            f"net.in_dim {net.in_dim} != coordinate_spec.ndim {coordinate_spec.ndim}"
        )
    if net.out_dim != components.n_components:
        raise ValueError(
            f"net.out_dim {net.out_dim} != components.n_components "
            f"{components.n_components}"
        )
    if jet_order < 1:
        raise ValueError(f"jet_order must be >= 1, got {jet_order}")
    net._check_fastpath(jet_order)


def make_jet_mlp_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    jet_order: int = 2,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> JetMLPVectorField:
    """Initialise a fresh :class:`JetMLPVectorField` with random parameters."""
    net = make_jet_mlp(
        coordinate_spec.ndim,
        hidden,
        out_dim=components.n_components,
        depth=depth,
        activation=base,
        seed=seed,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
    )
    validate_jet_field(net, coordinate_spec, components, jet_order)
    return JetMLPVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        net=net,
        jet_order=jet_order,
    )


def make_fourier_feature_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    num_features: int = 64,
    hidden: int = 64,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    frequency_scale: float | Sequence[float] = 1.0,
    jet_order: int = 2,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> FourierFeatureVectorField:
    """Initialise a fresh :class:`FourierFeatureVectorField` with random parameters.

    A sequence of ``frequency_scale`` values concatenates several bands into one
    encoding -- the multi-scale regime. The frequencies are ordinary pytree leaves,
    so ``jax.grad`` trains them just like the body weights.
    """
    net = make_fourier_feature_mlp(
        coordinate_spec.ndim,
        num_features,
        hidden,
        components.n_components,
        depth,
        base,
        frequency_scale=frequency_scale,
        seed=seed,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
    )
    validate_jet_field(net, coordinate_spec, components, jet_order)
    return FourierFeatureVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        net=net,
        jet_order=jet_order,
    )


def make_siren_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    omega_0: float = 30.0,
    jet_order: int = 2,
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> JetMLPVectorField:
    r"""Build a SIREN (Sitzmann et al. 2020) as an omnibias PINN field (JAX).

    A ``sin``-activation network with the SIREN initialisation; its derivative tower
    is exact at every order (``sin^{(n)}(z) = sin(z + n pi/2)``).
    """
    net = make_siren(
        coordinate_spec.ndim,
        hidden,
        out_dim=components.n_components,
        depth=depth,
        omega_0=omega_0,
        seed=seed,
        dtype=dtype,
    )
    validate_jet_field(net, coordinate_spec, components, jet_order)
    return JetMLPVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        net=net,
        jet_order=jet_order,
    )


__all__ = [
    "FourierFeatureVectorField",
    "JET_CACHE_KEY",
    "JetMLPVectorField",
    "JetNet",
    "make_fourier_feature_vector_field",
    "make_jet_mlp_vector_field",
    "make_siren_vector_field",
    "validate_jet_field",
]

# Marker read by the omnibias-fields backend ops to select the exact multivariate
# jet path (avoids a fields -> pinn import cycle).
JetMLPVectorField._omnibias_dispatch = "jet_mlp"
FourierFeatureVectorField._omnibias_dispatch = "jet_mlp"
JetMLPVectorField._omnibias_readout_independent = True
FourierFeatureVectorField._omnibias_readout_independent = True

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-scale, frequency-aware PINN architectures on the closed-form tower (JAX).

Bit-identical twin of :mod:`omnibias.torch.architectures.multiscale`; see that
module for the exposition. The two constructions are:

* :class:`AdaptiveActivation` -- Jagtap-style ``sigma(n a z)`` with a trainable
  slope ``a``, built as a real :class:`ActivationSpec` through the
  backend-neutral :func:`omnibias.core.spec.tempered` combinator, so the tower
  ``(n a)^k sigma^{(k)}(n a z)`` is exact at every order.
* :class:`MscaleMLP` -- the MscaleDNN band mixture ``u(x) = sum_j f_j(alpha_j x)``,
  whose jet is the sum of the per-band jets.

Everything here is a frozen dataclass registered as a JAX pytree, so the slopes
and band weights are ordinary leaves that ``jax.grad`` / ``jax.jit`` see. Both
backends drive the same Faa di Bruno kernel on the same layer chains, so their
derivatives agree to double-precision round-off.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnibias.core.spec import tempered
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.architectures.pinn import (
    JetMLP,
    _apply_readout_jet_from_layers,
    _check_layer_fastpaths,
    _partials_from_jet,
    _point_hidden_jet_from_layers,
    _value_from_layers,
    make_jet_mlp,
)
from omnibias.jax.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | None]


def _as_band_scales(scales: Sequence[float]) -> tuple[float, ...]:
    """Normalise a band-scale sequence to a tuple of positive floats."""
    out = tuple(float(s) for s in scales)
    if not out:
        raise ValueError("scales must contain at least one band")
    if any(s <= 0.0 for s in out):
        raise ValueError(f"all band scales must be > 0, got {out}")
    return out


# --------------------------------------------------------------------------
# AdaptiveActivation -- trainable-frequency activation as a live spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveActivation:
    r"""Trainable-frequency activation ``sigma(n a z)`` as a live :class:`ActivationSpec`.

    JAX twin of :class:`omnibias.torch.architectures.multiscale.AdaptiveActivation`.
    ``a`` is the single pytree leaf: a scalar for one slope per layer (L-LAAF) or a
    ``(hidden,)`` vector for one per unit (N-LAAF), broadcast over the
    pre-activation. ``slope_scale`` is the fixed amplification ``n``.

    Build one with :func:`make_adaptive_activation` rather than by hand, so the
    initial *effective* slope ``n a`` is the value you asked for.
    """

    a: Array
    base: JaxActivationSpec
    slope_scale: float

    @property
    def scale(self) -> Array:
        """The effective slope ``n a`` at the current parameter value."""
        out: Array = self.slope_scale * self.a
        return out

    @property
    def spec(self) -> JaxActivationSpec:
        """Closed-form :class:`ActivationSpec` of ``sigma(n a z)`` at the current ``a``."""
        return tempered(
            self.base,
            self.scale,
            scale="unit",
            name=f"adaptive_{self.base.name}",
            operator_role=(
                f"Adaptive-frequency {self.base.name}: sigma(n a z) with trainable "
                f"slope a; tower (n a)^k sigma^(k)(n a z)."
            ),
        )


def make_adaptive_activation(
    base: str | JaxActivationSpec = "tanh",
    *,
    slope_scale: float = 1.0,
    width: int | None = None,
    scale_init: float = 1.0,
    dtype: Any = jnp.float64,
) -> AdaptiveActivation:
    """Build an :class:`AdaptiveActivation` whose initial effective slope is ``scale_init``."""
    spec = get_activation(base)
    if spec.fastpath is None:
        raise ValueError(
            f"AdaptiveActivation requires a base activation with a closed-form "
            f"derivative kernel; activation {spec.name!r} has none."
        )
    if slope_scale <= 0.0:
        raise ValueError(f"slope_scale must be > 0, got {slope_scale}")
    if scale_init <= 0.0:
        raise ValueError(f"scale_init must be > 0, got {scale_init}")
    if width is not None and width < 1:
        raise ValueError(f"width must be >= 1 when given, got {width}")
    shape: tuple[int, ...] = () if width is None else (width,)
    a = jnp.full(shape, scale_init / float(slope_scale), dtype=dtype)
    return AdaptiveActivation(a=a, base=spec, slope_scale=float(slope_scale))


def _act_flatten(
    act: AdaptiveActivation,
) -> tuple[tuple[Array], tuple[JaxActivationSpec, float]]:
    return (act.a,), (act.base, act.slope_scale)


def _act_unflatten(
    aux: tuple[JaxActivationSpec, float], leaves: tuple[Array]
) -> AdaptiveActivation:
    base, slope_scale = aux
    return AdaptiveActivation(a=leaves[0], base=base, slope_scale=slope_scale)


jax.tree_util.register_pytree_node(AdaptiveActivation, _act_flatten, _act_unflatten)


# --------------------------------------------------------------------------
# AdaptiveJetMLP -- one trainable slope per hidden layer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveJetMLP:
    r"""Deep MLP with a trainable-frequency activation per hidden layer (JAX).

    JAX twin of :class:`omnibias.torch.architectures.multiscale.AdaptiveJetMLP`.
    The slopes are pytree leaves alongside the weights, so ``jax.grad`` trains
    them together, and every input derivative stays closed form because the slope
    enters through :func:`omnibias.core.spec.tempered`.
    """

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    activations: tuple[AdaptiveActivation, ...]
    in_dim: int
    out_dim: int

    @property
    def depth(self) -> int:
        """Number of hidden (activated) layers."""
        return len(self.weights) - 1

    def _layer_specs(self) -> list[LayerSpec]:
        """``(W, b, spec)`` list with each hidden spec rebuilt at the current slope."""
        n = len(self.weights)
        return [
            (
                self.weights[i],
                self.biases[i],
                None if i == n - 1 else self.activations[i].spec,
            )
            for i in range(n)
        ]

    def _point_hidden_jet(self, xi: Array, order: int) -> Array:
        """Single-point jet through every layer *except* the affine readout.

        Shape ``(M, H)``. Inherits the JetMLP layer-specs path with live slopes.
        """
        return _point_hidden_jet_from_layers(self._layer_specs(), xi, order)

    def _apply_readout_jet(self, hidden_jet: Array) -> Array:
        """Push a (possibly batched) hidden jet through the live affine readout."""
        return _apply_readout_jet_from_layers(self._layer_specs(), hidden_jet)

    def _point_jet(self, xi: Array, order: int) -> Array:
        """Single-point multivariate jet, shape ``(M, out_dim)``."""
        return self._apply_readout_jet(self._point_hidden_jet(xi, order))

    def _check_fastpath(self, max_order: int) -> None:
        """Reject activations without a closed-form tower up to ``max_order``."""
        _check_layer_fastpaths(self._layer_specs(), max_order)

    def slopes(self) -> tuple[Array, ...]:
        """Current effective slope ``n a`` of each hidden layer."""
        return tuple(act.scale for act in self.activations)

    def value(self, x: Array) -> Array:
        """Plain network value ``u(x)``, shape ``(..., out_dim)``."""
        return _value_from_layers(self._layer_specs(), x)

    def __call__(self, x: Array) -> Array:
        return self.value(x)

    def jet(self, x: Array, order: int) -> Array:
        """Batched multivariate jet, shape ``(B, M, out_dim)``."""
        out: Array = jax.vmap(lambda xi: self._point_jet(xi, order))(x)
        return out

    def gradient(self, x: Array) -> Array:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        dim = self.in_dim
        out: Array = jax.vmap(
            lambda xi: jet_gradient(self._point_jet(xi, 1), dim, 1)
        )(x)
        return out

    def hessian(self, x: Array) -> Array:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        dim = self.in_dim
        out: Array = jax.vmap(
            lambda xi: jet_hessian(self._point_jet(xi, 2), dim, 2)
        )(x)
        return out

    def partials(self, x: Array, order: int) -> dict[tuple[int, ...], Array]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order`` (``(B, out_dim)``)."""
        return _partials_from_jet(self.jet(x, order), self.in_dim, order)


_AdaptiveAux = tuple[int, int, int]


def _adaptive_flatten(
    net: AdaptiveJetMLP,
) -> tuple[tuple[Any, ...], _AdaptiveAux]:
    children = (*net.weights, *net.biases, *net.activations)
    return children, (net.in_dim, net.out_dim, len(net.weights))


def _adaptive_unflatten(aux: _AdaptiveAux, leaves: tuple[Any, ...]) -> AdaptiveJetMLP:
    in_dim, out_dim, n = aux
    flat = tuple(leaves)
    return AdaptiveJetMLP(
        weights=flat[:n],
        biases=flat[n : 2 * n],
        activations=flat[2 * n :],
        in_dim=in_dim,
        out_dim=out_dim,
    )


jax.tree_util.register_pytree_node(
    AdaptiveJetMLP, _adaptive_flatten, _adaptive_unflatten
)


def make_adaptive_jet_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int = 1,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    *,
    slope_scale: float = 1.0,
    granularity: str = "layer",
    scale_init: float = 1.0,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> AdaptiveJetMLP:
    """Build a randomly-initialised :class:`AdaptiveJetMLP` (weights as :func:`make_jet_mlp`)."""
    if granularity not in ("layer", "neuron"):
        raise ValueError(f"granularity must be 'layer' or 'neuron', got {granularity!r}")
    plain = make_jet_mlp(
        in_dim,
        hidden,
        out_dim=out_dim,
        depth=depth,
        activation=base,
        seed=seed,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
    )
    activations = tuple(
        make_adaptive_activation(
            base,
            slope_scale=slope_scale,
            width=hidden if granularity == "neuron" else None,
            scale_init=scale_init,
            dtype=dtype,
        )
        for _ in range(depth)
    )
    return AdaptiveJetMLP(
        weights=plain.weights,
        biases=plain.biases,
        activations=activations,
        in_dim=in_dim,
        out_dim=out_dim,
    )


# --------------------------------------------------------------------------
# MscaleMLP -- band mixture u(x) = sum_j f_j(alpha_j x)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MscaleMLP:
    r"""MscaleDNN band mixture ``u(x) = sum_j f_j(alpha_j x)`` with exact jets (JAX).

    JAX twin of :class:`omnibias.torch.architectures.multiscale.MscaleMLP`. Each
    band pre-scales the input by ``alpha_j``, which is the same map as scaling the
    band's first weight matrix, so every band stays an ordinary layer chain and
    the mixture's jet is the sum of the per-band jets -- exact and closed form at
    any order.
    """

    subnets: tuple[JetMLP, ...]
    scales: tuple[float, ...]
    in_dim: int
    out_dim: int

    @property
    def depth(self) -> int:
        """Number of hidden (activated) layers per band."""
        return self.subnets[0].depth

    def _band_layer_specs(self) -> list[list[LayerSpec]]:
        """One ``(W, b, spec)`` chain per band, with ``W_0`` scaled to ``alpha_j W_0``."""
        groups: list[list[LayerSpec]] = []
        for scale, sub in zip(self.scales, self.subnets, strict=True):
            layers = sub._layer_specs()
            w0, b0, act0 = layers[0]
            groups.append([(w0 * scale, b0, act0), *layers[1:]])
        return groups

    def _point_hidden_jet(self, xi: Array, order: int) -> Array:
        """Stack of per-band hidden jets, shape ``(n_bands, M, H)``."""
        bands: list[Array] = []
        for layers in self._band_layer_specs():
            if len(layers) < 2:
                raise ValueError(
                    "each Mscale band needs a hidden layer plus a readout"
                )
            bands.append(mlp_jet_mv(xi, layers[:-1], order))
        return jnp.stack(bands, axis=0)

    def _apply_readout_jet(self, hidden_jet: Array) -> Array:
        """Apply each band's live readout and sum.

        ``hidden_jet`` is ``(n_bands, M, H)`` or ``(B, n_bands, M, H)``.
        """
        band_layers = self._band_layer_specs()
        total: Array | None = None
        batched = hidden_jet.ndim == 4
        for i, layers in enumerate(band_layers):
            w, b, _spec = layers[-1]
            h_i = hidden_jet[:, i] if batched else hidden_jet[i]
            out_i = jnp.tensordot(h_i, w, axes=([-1], [-1]))
            if b is not None:
                if batched:
                    out_i = out_i.at[:, 0, :].add(b)
                else:
                    out_i = out_i.at[0].add(b)
            total = out_i if total is None else total + out_i
        assert total is not None
        return total

    def _point_jet(self, xi: Array, order: int) -> Array:
        """Single-point jet of the mixture: the sum of the per-band jets."""
        return self._apply_readout_jet(self._point_hidden_jet(xi, order))

    def _check_fastpath(self, max_order: int) -> None:
        """Reject activations without a closed-form tower up to ``max_order``."""
        for sub in self.subnets:
            sub._check_fastpath(max_order)

    def value(self, x: Array) -> Array:
        """Plain mixture value ``sum_j f_j(alpha_j x)``, shape ``(..., out_dim)``."""
        total: Array | None = None
        for layers in self._band_layer_specs():
            h = _value_from_layers(layers, x)
            total = h if total is None else total + h
        assert total is not None
        return total

    def __call__(self, x: Array) -> Array:
        return self.value(x)

    def jet(self, x: Array, order: int) -> Array:
        """Batched multivariate jet, shape ``(B, M, out_dim)``."""
        out: Array = jax.vmap(lambda xi: self._point_jet(xi, order))(x)
        return out

    def gradient(self, x: Array) -> Array:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        dim = self.in_dim
        out: Array = jax.vmap(
            lambda xi: jet_gradient(self._point_jet(xi, 1), dim, 1)
        )(x)
        return out

    def hessian(self, x: Array) -> Array:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        dim = self.in_dim
        out: Array = jax.vmap(
            lambda xi: jet_hessian(self._point_jet(xi, 2), dim, 2)
        )(x)
        return out

    def partials(self, x: Array, order: int) -> dict[tuple[int, ...], Array]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order`` (``(B, out_dim)``)."""
        return _partials_from_jet(self.jet(x, order), self.in_dim, order)


_MscaleAux = tuple[tuple[float, ...], int, int]


def _mscale_flatten(net: MscaleMLP) -> tuple[tuple[Any, ...], _MscaleAux]:
    return net.subnets, (net.scales, net.in_dim, net.out_dim)


def _mscale_unflatten(aux: _MscaleAux, leaves: tuple[Any, ...]) -> MscaleMLP:
    scales, in_dim, out_dim = aux
    return MscaleMLP(
        subnets=tuple(leaves), scales=scales, in_dim=in_dim, out_dim=out_dim
    )


jax.tree_util.register_pytree_node(MscaleMLP, _mscale_flatten, _mscale_unflatten)


def make_mscale_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int = 1,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    *,
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> MscaleMLP:
    """Build a randomly-initialised :class:`MscaleMLP`.

    ``hidden`` is the *total* width, split evenly across the bands (the MscaleDNN
    convention), so the mixture costs about as much as one MLP of that width. Each
    band is seeded independently, mirroring the torch builder's fresh draws.
    """
    band_scales = _as_band_scales(scales)
    if hidden < 1:
        raise ValueError(f"hidden must be >= 1, got {hidden}")
    band_hidden = max(1, hidden // len(band_scales))
    subnets = tuple(
        make_jet_mlp(
            in_dim,
            band_hidden,
            out_dim=out_dim,
            depth=depth,
            activation=base,
            seed=seed + j,
            weight_init_scale=weight_init_scale,
            dtype=dtype,
        )
        for j in range(len(band_scales))
    )
    return MscaleMLP(
        subnets=subnets, scales=band_scales, in_dim=in_dim, out_dim=out_dim
    )


__all__ = [
    "AdaptiveActivation",
    "AdaptiveJetMLP",
    "MscaleMLP",
    "make_adaptive_activation",
    "make_adaptive_jet_mlp",
    "make_mscale_mlp",
]

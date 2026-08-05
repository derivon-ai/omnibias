# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deep, arbitrary-order omnibias PINNs for JAX -- bit-identical twins of torch.

:class:`JetMLP` is the JAX counterpart of
:class:`omnibias.torch.architectures.pinn.JetMLP`: a deep fully-connected network
whose *every* mixed input partial ``D^alpha u(x)`` up to total order ``N`` comes
**exactly** from one forward pass of :func:`omnibias.jax.jet_mv.mlp_jet_mv` -- no
``jax.grad`` stacking, no finite differences. Because both backends call the same
Faa di Bruno multivariate-jet kernel, the derivatives are *bit-identical* across
torch and JAX in double precision (enable ``jax_enable_x64``).

Spectral-bias mitigation mirrors the torch surface:

* :class:`FourierFeatureMLP` lifts the input through a random Fourier-feature
  encoding ``gamma(x) = [cos(B x), sin(B x)]`` (Tancik et al. 2020). Because
  ``cos(z) = sin(z + pi/2)`` the encoding is a single omnibias ``sin`` layer, so the
  composite still yields every closed-form derivative through ``mlp_jet_mv``. Build
  one with :func:`make_fourier_feature_mlp`.
* :func:`make_siren` builds a SIREN (Sitzmann et al. 2020): a :class:`JetMLP` with
  ``sin`` activations and the SIREN initialisation; its derivative tower is exact for
  every order (``sin^{(n)}(z) = sin(z + n pi/2)``).

Each network is a frozen dataclass registered as a JAX pytree (its weight/bias arrays
are the leaves), so it flows through ``jax.grad`` / ``jax.jit`` for training while
omnibias supplies the spatial differential operator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnibias.core.multi_index import multi_index_factorial, multi_indices
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | None]


# --------------------------------------------------------------------------
# Shared closed-form-derivative readout (operates on a ``(W, b, spec)`` list)
# --------------------------------------------------------------------------


def _value_from_layers(layers: list[LayerSpec], x: Array) -> Array:
    """Plain forward value ``u(x)`` from a layer list (``spec=None`` = affine readout)."""
    h = x
    for w, b, spec in layers:
        h = h @ w.T
        if b is not None:
            h = h + b
        if spec is not None:
            h = spec.forward(h)
    out: Array = h
    return out


def _jet_batched(layers: list[LayerSpec], x: Array, order: int) -> Array:
    out: Array = jax.vmap(lambda xi: mlp_jet_mv(xi, layers, order))(x)
    return out


def _gradient_batched(layers: list[LayerSpec], in_dim: int, x: Array) -> Array:
    out: Array = jax.vmap(lambda xi: jet_gradient(mlp_jet_mv(xi, layers, 1), in_dim, 1))(x)
    return out


def _hessian_batched(layers: list[LayerSpec], in_dim: int, x: Array) -> Array:
    out: Array = jax.vmap(lambda xi: jet_hessian(mlp_jet_mv(xi, layers, 2), in_dim, 2))(x)
    return out


def _value_grad_hessian_batched(
    layers: list[LayerSpec], in_dim: int, x: Array
) -> tuple[Array, Array, Array]:
    def f(xi: Array) -> tuple[Array, Array, Array]:
        j = mlp_jet_mv(xi, layers, 2)
        return j[0], jet_gradient(j, in_dim, 2), jet_hessian(j, in_dim, 2)

    res = jax.vmap(f)(x)
    value_b: Array = res[0]
    grad_b: Array = res[1]
    hess_b: Array = res[2]
    return value_b, grad_b, hess_b


def _partials_from_jet(jet_b: Array, in_dim: int, order: int) -> dict[tuple[int, ...], Array]:
    idx = multi_indices(in_dim, order)
    return {
        alpha: jet_b[:, i] * multi_index_factorial(alpha) for i, alpha in enumerate(idx)
    }


def _as_scales(frequency_scale: float | Sequence[float]) -> tuple[float, ...]:
    """Normalise the ``frequency_scale`` argument to a tuple of positive floats."""
    scales: tuple[float, ...]
    if isinstance(frequency_scale, int | float):
        scales = (float(frequency_scale),)
    else:
        scales = tuple(float(s) for s in frequency_scale)
    if not scales:
        raise ValueError("frequency_scale must contain at least one scale")
    if any(s <= 0.0 for s in scales):
        raise ValueError(f"all frequency scales must be > 0, got {scales}")
    return scales


# --------------------------------------------------------------------------
# JetMLP -- uniform deep MLP
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JetMLP:
    r"""Deep MLP with exact omnibias multivariate-jet input derivatives (JAX).

    Parameters
    ----------
    weights, biases
        Per-linear parameters; ``len == depth + 1`` (the last is the affine
        readout). ``weights[i]`` has shape ``(out_i, in_i)`` and ``biases[i]``
        shape ``(out_i,)``.
    spec
        Base activation with a closed-form derivative fast path.
    in_dim, out_dim
        Input coordinate count and output component count.
    """

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    spec: JaxActivationSpec
    in_dim: int
    out_dim: int

    @property
    def depth(self) -> int:
        """Number of hidden (activated) layers."""
        return len(self.weights) - 1

    def _layer_specs(self) -> list[LayerSpec]:
        n = len(self.weights)
        return [
            (self.weights[i], self.biases[i], None if i == n - 1 else self.spec)
            for i in range(n)
        ]

    def value(self, x: Array) -> Array:
        """Plain network value ``u(x)``, shape ``(..., out_dim)``."""
        return _value_from_layers(self._layer_specs(), x)

    def __call__(self, x: Array) -> Array:
        return self.value(x)

    def jet(self, x: Array, order: int) -> Array:
        """Batched multivariate jet, shape ``(B, M, out_dim)``."""
        return _jet_batched(self._layer_specs(), x, order)

    def gradient(self, x: Array) -> Array:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        return _gradient_batched(self._layer_specs(), self.in_dim, x)

    def hessian(self, x: Array) -> Array:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        return _hessian_batched(self._layer_specs(), self.in_dim, x)

    def value_grad_hessian(self, x: Array) -> tuple[Array, Array, Array]:
        """One jet -> ``(value, gradient, Hessian)`` for 2nd-order PDE residuals."""
        return _value_grad_hessian_batched(self._layer_specs(), self.in_dim, x)

    def partials(self, x: Array, order: int) -> dict[tuple[int, ...], Array]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order`` (``(B, out_dim)``)."""
        return _partials_from_jet(self.jet(x, order), self.in_dim, order)


def _tree_flatten(
    net: JetMLP,
) -> tuple[tuple[Array, ...], tuple[JaxActivationSpec, int, int, int]]:
    leaves = (*net.weights, *net.biases)
    aux = (net.spec, net.in_dim, net.out_dim, len(net.weights))
    return leaves, aux


def _tree_unflatten(
    aux: tuple[JaxActivationSpec, int, int, int], leaves: tuple[Array, ...]
) -> JetMLP:
    spec, in_dim, out_dim, n = aux
    flat = tuple(leaves)
    weights = flat[:n]
    biases = flat[n : 2 * n]
    return JetMLP(
        weights=weights, biases=biases, spec=spec, in_dim=in_dim, out_dim=out_dim
    )


jax.tree_util.register_pytree_node(JetMLP, _tree_flatten, _tree_unflatten)


# --------------------------------------------------------------------------
# FourierFeatureMLP -- sin-encoded random Fourier features (spectral-bias cure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FourierFeatureMLP:
    r"""Spectral-bias-mitigating MLP with a sin-encoded random Fourier front end (JAX).

    The input is lifted by ``gamma(x) = [cos(B x), sin(B x)]`` (Tancik et al. 2020) and
    a standard MLP body acts on it. Because ``cos(z) = sin(z + pi/2)`` the encoding is a
    single omnibias ``sin`` layer
    (``gamma(x) = sin([B; B] x + [pi/2; 0])``), so ``D^alpha u(x)`` stays exactly closed
    form to arbitrary order through :func:`omnibias.jax.jet_mv.mlp_jet_mv`. Bit-identical
    twin of :class:`omnibias.torch.architectures.pinn.FourierFeatureMLP`.

    Parameters
    ----------
    w_ff, b_ff
        The sin-encoding layer: ``w_ff`` has shape ``(2 F_total, in_dim)`` and equals
        ``[B; B]``; ``b_ff`` is ``[pi/2 * 1_{F_total}; 0_{F_total}]``.
    weights, biases
        Body parameters (hidden layers + affine readout).
    base_spec
        Body activation with a closed-form derivative fast path.
    in_dim, out_dim
        Input coordinate count and output component count.
    num_features
        Fourier features per band (``F``); the encoding width is ``2 * F * len(scales)``.
    scales
        Frequency bands used to build ``B`` (kept for introspection).
    """

    w_ff: Array
    b_ff: Array
    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    base_spec: JaxActivationSpec
    in_dim: int
    out_dim: int
    num_features: int
    scales: tuple[float, ...]

    @property
    def feature_dim(self) -> int:
        """Width of the Fourier encoding ``2 * F * len(scales)``."""
        return int(self.w_ff.shape[0])

    @property
    def depth(self) -> int:
        """Number of hidden (activated) body layers."""
        return len(self.weights) - 1

    def _layer_specs(self) -> list[LayerSpec]:
        sin_spec = get_activation("sin")
        specs: list[LayerSpec] = [(self.w_ff, self.b_ff, sin_spec)]
        n = len(self.weights)
        for i in range(n):
            specs.append(
                (self.weights[i], self.biases[i], None if i == n - 1 else self.base_spec)
            )
        return specs

    def value(self, x: Array) -> Array:
        """Plain network value ``u(x)``, shape ``(..., out_dim)``."""
        return _value_from_layers(self._layer_specs(), x)

    def __call__(self, x: Array) -> Array:
        return self.value(x)

    def jet(self, x: Array, order: int) -> Array:
        """Batched multivariate jet, shape ``(B, M, out_dim)``."""
        return _jet_batched(self._layer_specs(), x, order)

    def gradient(self, x: Array) -> Array:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        return _gradient_batched(self._layer_specs(), self.in_dim, x)

    def hessian(self, x: Array) -> Array:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        return _hessian_batched(self._layer_specs(), self.in_dim, x)

    def value_grad_hessian(self, x: Array) -> tuple[Array, Array, Array]:
        """One jet -> ``(value, gradient, Hessian)`` for 2nd-order PDE residuals."""
        return _value_grad_hessian_batched(self._layer_specs(), self.in_dim, x)

    def partials(self, x: Array, order: int) -> dict[tuple[int, ...], Array]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order`` (``(B, out_dim)``)."""
        return _partials_from_jet(self.jet(x, order), self.in_dim, order)


def _ff_tree_flatten(
    net: FourierFeatureMLP,
) -> tuple[
    tuple[Array, ...],
    tuple[JaxActivationSpec, int, int, int, int, tuple[float, ...]],
]:
    leaves = (net.w_ff, net.b_ff, *net.weights, *net.biases)
    aux = (
        net.base_spec,
        net.in_dim,
        net.out_dim,
        len(net.weights),
        net.num_features,
        net.scales,
    )
    return leaves, aux


def _ff_tree_unflatten(
    aux: tuple[JaxActivationSpec, int, int, int, int, tuple[float, ...]],
    leaves: tuple[Array, ...],
) -> FourierFeatureMLP:
    base_spec, in_dim, out_dim, n, num_features, scales = aux
    flat = tuple(leaves)
    w_ff = flat[0]
    b_ff = flat[1]
    rest = flat[2:]
    weights = rest[:n]
    biases = rest[n : 2 * n]
    return FourierFeatureMLP(
        w_ff=w_ff,
        b_ff=b_ff,
        weights=weights,
        biases=biases,
        base_spec=base_spec,
        in_dim=in_dim,
        out_dim=out_dim,
        num_features=num_features,
        scales=scales,
    )


jax.tree_util.register_pytree_node(
    FourierFeatureMLP, _ff_tree_flatten, _ff_tree_unflatten
)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def make_jet_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int = 1,
    depth: int = 3,
    activation: str | JaxActivationSpec = "tanh",
    *,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> JetMLP:
    """Build a randomly-initialised :class:`JetMLP` (Glorot-ish scaled normals)."""
    if in_dim < 1 or hidden < 1 or out_dim < 1:
        raise ValueError("in_dim, hidden and out_dim must all be >= 1")
    if depth < 1:
        raise ValueError(f"depth (number of hidden layers) must be >= 1, got {depth}")
    spec = get_activation(activation)
    key = jax.random.PRNGKey(seed)
    dims = [hidden] * depth + [out_dim]
    weights: list[Array] = []
    biases: list[Array] = []
    prev = in_dim
    for d in dims:
        key, wk = jax.random.split(key)
        scale = weight_init_scale / math.sqrt(prev)
        weights.append(jax.random.normal(wk, (d, prev), dtype=dtype) * scale)
        biases.append(jnp.zeros((d,), dtype=dtype))
        prev = d
    return JetMLP(
        weights=tuple(weights), biases=tuple(biases), spec=spec, in_dim=in_dim, out_dim=out_dim
    )


def make_fourier_feature_mlp(
    in_dim: int,
    num_features: int = 64,
    hidden: int = 64,
    out_dim: int = 1,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    *,
    frequency_scale: float | Sequence[float] = 1.0,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> FourierFeatureMLP:
    """Build a randomly-initialised :class:`FourierFeatureMLP`.

    The frequency matrix is ``B ~ N(0, (2 pi * scale)^2)`` per band, concatenated across
    ``frequency_scale`` for a multi-scale encoding; the body is Glorot-ish like
    :func:`make_jet_mlp`. ``depth=0`` gives a pure random-Fourier-feature model.
    """
    if in_dim < 1:
        raise ValueError(f"in_dim must be >= 1, got {in_dim}")
    if num_features < 1:
        raise ValueError(f"num_features must be >= 1, got {num_features}")
    if hidden < 1:
        raise ValueError(f"hidden must be >= 1, got {hidden}")
    if out_dim < 1:
        raise ValueError(f"out_dim must be >= 1, got {out_dim}")
    if depth < 0:
        raise ValueError(f"depth (hidden layers after encoding) must be >= 0, got {depth}")
    scales = _as_scales(frequency_scale)
    spec = get_activation(base)
    key = jax.random.PRNGKey(seed)

    bands: list[Array] = []
    for s in scales:
        key, bk = jax.random.split(key)
        bands.append(jax.random.normal(bk, (num_features, in_dim), dtype=dtype) * (2.0 * math.pi * s))
    b_mat = jnp.concatenate(bands, axis=0)  # (F_total, in_dim)
    f_total = int(b_mat.shape[0])
    w_ff = jnp.concatenate([b_mat, b_mat], axis=0)  # (2 F_total, in_dim)
    b_ff = jnp.concatenate(
        [jnp.full((f_total,), 0.5 * math.pi, dtype=dtype), jnp.zeros((f_total,), dtype=dtype)]
    )

    weights: list[Array] = []
    biases: list[Array] = []
    prev = 2 * f_total
    dims = [hidden] * depth + [out_dim]
    for d in dims:
        key, wk = jax.random.split(key)
        scale = weight_init_scale / math.sqrt(prev)
        weights.append(jax.random.normal(wk, (d, prev), dtype=dtype) * scale)
        biases.append(jnp.zeros((d,), dtype=dtype))
        prev = d
    return FourierFeatureMLP(
        w_ff=w_ff,
        b_ff=b_ff,
        weights=tuple(weights),
        biases=tuple(biases),
        base_spec=spec,
        in_dim=in_dim,
        out_dim=out_dim,
        num_features=num_features,
        scales=scales,
    )


def make_siren(
    in_dim: int,
    hidden: int,
    out_dim: int = 1,
    depth: int = 3,
    *,
    omega_0: float = 30.0,
    seed: int = 0,
    dtype: Any = jnp.float64,
) -> JetMLP:
    r"""Build a SIREN (Sitzmann et al. 2020) as an omnibias :class:`JetMLP`.

    Bit-identical twin of :func:`omnibias.torch.architectures.pinn.make_siren`: a
    ``sin``-activation MLP whose first-layer frequency is scaled by ``omega_0`` (folded
    into its weights), with weights ``U(-1/fan_in, 1/fan_in)`` on the first layer and
    ``U(-sqrt(6/fan_in)/omega_0, +...)`` thereafter and zero biases. Every input
    derivative is exact for all orders via ``sin^{(n)}(z) = sin(z + n pi/2)``.
    """
    if in_dim < 1 or hidden < 1 or out_dim < 1:
        raise ValueError("in_dim, hidden and out_dim must all be >= 1")
    if depth < 1:
        raise ValueError(f"depth (number of hidden layers) must be >= 1, got {depth}")
    if omega_0 <= 0.0:
        raise ValueError(f"omega_0 must be > 0, got {omega_0}")
    spec = get_activation("sin")
    key = jax.random.PRNGKey(seed)
    dims = [hidden] * depth + [out_dim]
    weights: list[Array] = []
    biases: list[Array] = []
    prev = in_dim
    for i, d in enumerate(dims):
        key, wk = jax.random.split(key)
        if i == 0:
            bound = 1.0 / prev
            w = jax.random.uniform(wk, (d, prev), dtype=dtype, minval=-bound, maxval=bound)
            w = w * omega_0  # fold omega_0 into the first-layer frequency
        else:
            bound = math.sqrt(6.0 / prev) / omega_0
            w = jax.random.uniform(wk, (d, prev), dtype=dtype, minval=-bound, maxval=bound)
        weights.append(w)
        biases.append(jnp.zeros((d,), dtype=dtype))
        prev = d
    return JetMLP(
        weights=tuple(weights), biases=tuple(biases), spec=spec, in_dim=in_dim, out_dim=out_dim
    )


__all__ = [
    "FourierFeatureMLP",
    "JetMLP",
    "make_fourier_feature_mlp",
    "make_jet_mlp",
    "make_siren",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``SpectralVectorField`` -- D-dimensional periodic Fourier basis with
deep-omnibias temporal head, generalised to multiple output channels.

Architecture
------------

For ``D`` spatial axes (each with ``2K+1`` Fourier modes) plus a time
axis, and ``C`` output channels:

.. math::
    f_c(x_1, ..., x_D, t) = \\sum_{j_1, ..., j_D} a_c^{j_1...j_D}(t)\\,
        \\phi_{j_1}(x_1) \\cdots \\phi_{j_D}(x_D)

where the per-axis basis ``\\phi`` is the standard ``2K+1``-mode layout
``(1, cos(k_1 x), ..., cos(k_K x), sin(k_1 x), ..., sin(k_K x))``
introduced in :mod:`research.experiments.cahn_hilliard.spectral_field_3d`.

The temporal coefficient ``a_c^{j_1...j_D}(t)`` is the output of a deep
MLP whose first hidden layer is the omnibias fast-path layer; the MLP
shares its hidden activations across all components ``c`` and all
spectral indices, with a final linear readout to
``C * (2K+1)^D`` outputs. This gives perfectly closed-form spatial
derivatives of arbitrary order via diagonal/shifted multipliers, while
the temporal derivative ``d/dt`` is closed-form through the single
omnibias hidden layer.

Important: the spatial derivatives along the *spatial* axes are exact
(diagonal in coefficient space); the time derivative is also exact
through the omnibias closed-form chain rule on the temporal MLP. There
is *no* autograd in the inner loop.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from omnibias.core.spec import ActivationSpec
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet import jet_to_tower, mlp_jet
from torch import Tensor
from torch.func import vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def _axis_basis_dn(
    x: Tensor, k_vec: Tensor, K: int, order: int,
) -> Tensor:
    """Return ``d^order/dx^order`` of the per-axis basis at points ``x``.

    Shape ``(B, 2K+1)``. The layout is
    ``(1, cos(k_1 x), ..., cos(k_K x), sin(k_1 x), ..., sin(k_K x))``.
    """
    arg = x.unsqueeze(-1) * k_vec                  # (B, K)
    cos = torch.cos(arg)
    sin = torch.sin(arg)
    if order == 0:
        ones = torch.ones_like(cos[..., :1])
        return torch.cat([ones, cos, sin], dim=-1)
    k_n = k_vec.pow(order)                          # (K,)
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
    else:  # 3
        cos_slot_fn = sin
        sin_slot_fn = -cos
    cos_slot = k_n * cos_slot_fn                    # (B, K)
    sin_slot = k_n * sin_slot_fn                    # (B, K)
    zero = torch.zeros_like(cos[..., :1])
    return torch.cat([zero, cos_slot, sin_slot], dim=-1)


def _multi_axis_einsum(
    a: Tensor, bases: list[Tensor],
) -> Tensor:
    """Compute ``einsum`` over coefficient tensor and per-axis bases.

    ``a`` has shape ``(B, C, n_1, ..., n_D)``; each ``bases[d]`` has
    shape ``(B, n_d)``. Returns ``(B, C)``.
    """
    D = len(bases)
    if D == 0:
        return a
    mode_letters = "jklmnopqr"[:D]
    eq = (
        "bC" + mode_letters + ","
        + ",".join("b" + m for m in mode_letters)
        + "->bC"
    )
    return torch.einsum(eq, a, *bases)


class SpectralVectorField(FieldBase):
    """D-dimensional periodic Fourier vector field with omnibias time head.

    Parameters
    ----------
    coordinate_spec
        Must declare a time axis. The non-time axes are the spatial
        Fourier axes.
    components
        Output component spec; ``C = components.n_components`` is the
        number of channels.
    K
        Number of Fourier mode pairs per spatial axis. Output dim per
        component is ``(2K+1)^D``. Memory: ``B * C * (2K+1)^D``.
    L
        Spatial period. Either a scalar (same for all axes) or a tuple of
        D floats. Default ``2 * pi`` (matches the existing 2D NS / KS /
        CH solvers).
    time_hidden
        Width of the temporal MLP hidden layer ``M``.
    time_depth
        Number of hidden activations in the temporal MLP. ``time_depth=1``
        is the original shallow head; ``>= 3`` recommended for stiff
        equations like Cahn-Hilliard.
    activation
        Base activation for the temporal MLP. Must be Riccati-class.
    weight_init_scale
        Scale of the MLP weight init. Default 1.0.
    dtype
        Torch dtype. Default ``torch.float64``.

    Notes
    -----
    The spatial axes are determined by ``coordinate_spec.spatial_axes``;
    1D, 2D, and 3D are explicitly supported and tested. Higher
    dimensions in principle work but the ``(2K+1)^D`` blow-up makes
    them impractical above D=4.
    """

    spec: ActivationSpec[Tensor]

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        K: int = 8,
        L: float | tuple[float, ...] = 2.0 * math.pi,
        time_hidden: int = 64,
        time_depth: int = 1,
        activation: str | ActivationSpec[Tensor] = "tanh",
        weight_init_scale: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            coordinate_spec=coordinate_spec, components=components,
        )
        if coordinate_spec.time_axis is None:
            raise ValueError(
                "SpectralVectorField requires a time axis on the coordinate spec"
            )
        self.K = int(K)
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {self.K}")
        self.D_spatial = coordinate_spec.n_spatial
        self.C = components.n_components
        if isinstance(L, int | float):
            self.L: tuple[float, ...] = tuple(float(L) for _ in range(self.D_spatial))
        else:
            self.L = tuple(float(x) for x in L)
            if len(self.L) != self.D_spatial:
                raise ValueError(
                    f"L tuple has length {len(self.L)} but coordinate spec has "
                    f"{self.D_spatial} spatial axes"
                )
        self.time_hidden = int(time_hidden)
        self.time_depth = int(time_depth)
        if self.time_depth < 1:
            raise ValueError(f"time_depth must be >= 1, got {self.time_depth}")
        self.spec = (
            activation if isinstance(activation, ActivationSpec)
            else get_activation(activation)
        )
        if self.spec.fastpath is None:
            raise ValueError(
                f"SpectralVectorField requires a fast-path activation; "
                f"{self.spec.name!r} has none."
            )
        self._coord_axis_indices = tuple(
            coordinate_spec.axis_index(a)
            for a in coordinate_spec.spatial_axes
        )
        self._time_axis_idx = coordinate_spec.axis_index(
            coordinate_spec.time_axis
        )

        # Time MLP. Same structure as SpectralOmnibiasField{2,3}D.
        self.W_t = nn.Parameter(
            torch.randn(self.time_hidden, 1, dtype=dtype) * weight_init_scale,
        )
        self.beta_t = nn.Parameter(torch.zeros(self.time_hidden, dtype=dtype))
        self._inner_layers = nn.ModuleList()
        for _ in range(self.time_depth - 1):
            layer = nn.Linear(self.time_hidden, self.time_hidden, dtype=dtype)
            with torch.no_grad():
                layer.weight.normal_(std=weight_init_scale / math.sqrt(self.time_hidden))
                layer.bias.zero_()
            self._inner_layers.append(layer)

        self._modes_per_axis = 2 * self.K + 1
        self._out_per_component = self._modes_per_axis ** self.D_spatial
        self._out_dim = self.C * self._out_per_component
        self.V = nn.Parameter(
            torch.zeros(self._out_dim, self.time_hidden, dtype=dtype),
        )
        self.b_t = nn.Parameter(torch.zeros(self._out_dim, dtype=dtype))

        # Per-axis k vectors stored as buffers; constructed once.
        for ax_i, L_a in enumerate(self.L):
            ks = torch.arange(1, self.K + 1, dtype=dtype)
            kv = (2.0 * math.pi * ks) / L_a
            self.register_buffer(f"_k_vec_{ax_i}", kv, persistent=False)

    # ------------- internals ------------------------------------------

    def k_vec(self, full_axis: int) -> Tensor:
        """Per-axis ``k`` vector ``(K,)`` for the *full* coord-spec axis index."""
        d = self._spatial_axis_index(full_axis)
        return getattr(self, f"_k_vec_{d}")

    def _spatial_axis_index(self, full_axis: int) -> int:
        """Map a coordinate-spec axis index to a *spatial* axis index."""
        sa = self._coord_axis_indices
        if full_axis not in sa:
            raise ValueError(
                f"axis {full_axis} is not a spatial axis "
                f"({sa!r}); spatial derivatives only supported there"
            )
        return sa.index(full_axis)

    def _hidden_t(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Pre-activation ``z`` (one omnibias layer) and final hidden ``h``.

        Returns
        -------
        z : Tensor of shape ``(B, M)``
        h : Tensor of shape ``(B, M)``
        """
        if t.dim() == 1:
            t_in = t.unsqueeze(-1)               # (B, 1)
        else:
            t_in = t
        z = t_in @ self.W_t.T + self.beta_t       # (B, M)
        h = self.spec.forward(z)                  # (B, M)
        for layer in self._inner_layers:
            h = self.spec.forward(layer(h))
        return z, h

    def _hidden_t_and_dt(
        self, t: Tensor, *, t_order: int = 0,
    ) -> Tensor:
        """Return the temporal MLP hidden layer or its ``t_order``-th
        derivative wrt ``t``.

        For ``t_order == 0`` returns ``h(t)``; for ``t_order >= 1`` returns
        the closed-form ``d^order h / dt^order``. The shallow head
        (``time_depth == 1``) uses the single-layer omnibias chain rule
        ``sigma^(order)(z) * W_t^order``; the deep head (``time_depth > 1``)
        uses the exact omnibias *directional jet*
        (:func:`omnibias.torch.jet.mlp_jet`) propagated along the 1-D time
        axis, so the whole derivative tower is closed-form -- there is no
        autograd anywhere in the temporal path.
        """
        if t_order == 0:
            _, h = self._hidden_t(t)
            return h
        if self.time_depth > 1:
            return self._hidden_t_and_dt_via_jet(t, t_order=t_order)
        # Closed-form for shallow time MLP:  h(t) = sigma(W_t t + beta_t).
        z, _ = self._hidden_t(t)
        sigma_n = (
            self.spec.fastpath(z, t_order)
            if self.spec.fastpath is not None
            else None
        )
        if sigma_n is None:
            raise ValueError(
                f"activation {self.spec.name!r} has no fastpath kernel"
            )
        # d^n/dt^n sigma(W_t * t + beta_t) = W_t^n * sigma^(n)(W_t t + beta_t)
        W_t_pow = self.W_t.squeeze(-1).pow(t_order)            # (M,)
        return sigma_n * W_t_pow                               # (B, M)

    def _time_layers(
        self,
    ) -> list[tuple[Tensor, Tensor, ActivationSpec[Tensor]]]:
        """The temporal MLP as ``mlp_jet`` layers ``h(t) = (sigma . affine)^L(t)``.

        Every layer is activated; the linear readout ``V`` is applied *after*
        the derivative tower and commutes with ``d/dt`` because it is linear.
        """
        layers: list[tuple[Tensor, Tensor, ActivationSpec[Tensor]]] = [
            (self.W_t, self.beta_t, self.spec),
        ]
        for layer in self._inner_layers:
            assert isinstance(layer, nn.Linear)
            layers.append((layer.weight, layer.bias, self.spec))
        return layers

    def _hidden_t_and_dt_via_jet(
        self, t: Tensor, *, t_order: int,
    ) -> Tensor:
        """Exact ``d^t_order h(t) / dt^t_order`` of the deep temporal MLP.

        The time axis is one-dimensional, so ordinary time derivatives are the
        directional derivatives of the temporal MLP along ``v = 1``. The omnibias
        directional jet (:func:`omnibias.torch.jet.mlp_jet`) returns the whole
        Taylor tower in a single forward pass through the closed-form
        ``sigma^(k)`` kernels; :func:`jet_to_tower` rescales coefficient
        ``t_order`` to the raw derivative. No autograd; exact to float epsilon
        and bit-identical to the JAX twin.
        """
        if t_order < 1:
            raise ValueError(f"t_order must be >= 1, got {t_order}")
        layers = self._time_layers()
        v = torch.ones(1, dtype=self.W_t.dtype, device=self.W_t.device)

        def per_b(t_scalar: Tensor) -> Tensor:
            x0 = t_scalar.reshape(1)
            tower: Tensor = jet_to_tower(mlp_jet(x0, v, layers, t_order))
            return tower[t_order]

        out: Tensor = vmap(per_b)(t.reshape(-1))
        return out

    def _a_from_hidden(self, h: Tensor, *, t_order: int = 0) -> Tensor:
        """Apply the live readout ``(V, b_t)`` to a cached temporal feature ``h``.

        ``h`` depends only on ``t`` and the frozen temporal weights; ``a`` is
        rebuilt on every call so a frozen-feature column sweep against a reused
        :class:`~omnibias.fields._core.state.FieldState` stays correct.
        """
        if t_order == 0:
            a = self.b_t + h @ self.V.T
        else:
            a = h @ self.V.T
        return a.view(-1, self.C, *([self._modes_per_axis] * self.D_spatial))

    def _coeff_blocks_at_t(
        self, t: Tensor, *, t_order: int = 0,
    ) -> Tensor:
        """Return ``a(t)`` reshaped as ``(B, C, n_1, ..., n_D)``.

        ``t_order`` selects ``d^t_order a / dt^t_order``.
        """
        h = self._hidden_t_and_dt(t, t_order=t_order)
        return self._a_from_hidden(h, t_order=t_order)

    def _make_sigma_cache(self, coords: Tensor):  # type: ignore[override]
        # Use the time-axis pre-activation as the sigma cache key.
        t = coords[..., self._time_axis_idx]
        z, _ = self._hidden_t(t)
        from omnibias.pinn._core.sigma_cache import SigmaCache
        return SigmaCache(z=z)

    def _pre_activations(self, coords: Tensor) -> Tensor:  # not used
        t = coords[..., self._time_axis_idx]
        z, _ = self._hidden_t(t)
        return z

    # ------------- accessors consumed by the ops -----------------------

    def _bases_for_state(
        self,
        state: FieldState,
        *,
        order_per_axis: tuple[int, ...] | None = None,
    ) -> list[Tensor]:
        """Per-axis basis tables ``(B, 2K+1)`` with given derivative
        orders.

        ``order_per_axis`` has length ``self.D_spatial`` (in spatial-axis
        order, i.e. matching ``coordinate_spec.spatial_axes``). Default
        ``None`` is "all zeros" -> the value basis.
        """
        if order_per_axis is None:
            order_per_axis = tuple(0 for _ in range(self.D_spatial))
        if len(order_per_axis) != self.D_spatial:
            raise ValueError(
                f"order_per_axis has length {len(order_per_axis)} but "
                f"D_spatial = {self.D_spatial}"
            )
        coords = state.coords
        bases: list[Tensor] = []
        for d, full_axis in enumerate(self._coord_axis_indices):
            x_d = coords[..., full_axis]
            kv = self.k_vec(full_axis)
            bases.append(_axis_basis_dn(x_d, kv, self.K, order_per_axis[d]))
        return bases

    def _coeff_blocks_for_state(
        self, state: FieldState, *, t_order: int = 0,
    ) -> Tensor:
        """Return ``a(t)`` from a readout-independent cached temporal feature ``h``.

        The expensive ``h = _hidden_t_and_dt(t)`` is memoised on the state; the
        live readout ``(V, b_t)`` is applied on every call.
        """
        key = f"spectral_h_t{t_order}"
        h = state.extra.get(key)
        if h is None:
            t = state.coords[..., self._time_axis_idx]
            h = self._hidden_t_and_dt(t, t_order=t_order)
            state.extra[key] = h
        return self._a_from_hidden(h, t_order=t_order)

    # The ops module calls these:

    def value_component(self, state: FieldState, name: str) -> Tensor:
        ci = self.components.index(name)
        a = self._coeff_blocks_for_state(state)             # (B, C, ...)
        bases = self._bases_for_state(state)
        return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
        """``d^order f_name / d(coord_axis)^order``.

        For a *spatial* axis the derivative is closed-form (one
        Fourier-shift per order). For the *time* axis, it uses the
        closed-form chain rule on the omnibias hidden layer (when
        ``time_depth=1``).
        """
        if order == 0:
            return self.value_component(state, name)
        ci = self.components.index(name)
        if axis == self._time_axis_idx:
            a = self._coeff_blocks_for_state(state, t_order=order)
            bases = self._bases_for_state(state)
            return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]
        # Spatial axis -> shift the basis on that axis.
        d = self._spatial_axis_index(axis)
        order_per_axis = tuple(order if i == d else 0 for i in range(self.D_spatial))
        a = self._coeff_blocks_for_state(state)
        bases = self._bases_for_state(state, order_per_axis=order_per_axis)
        return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        """Mixed partial. Splits into "spatial part" + "time part"."""
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
        bases = self._bases_for_state(state, order_per_axis=tuple(order_per_axis))
        return _multi_axis_einsum(a_t[:, ci:ci + 1], bases)[:, 0]

    def biharmonic(self, state: FieldState, name: str) -> Tensor:
        """Spatial biharmonic ``Delta^2 f``."""
        return self.polylaplacian(state, name, k=2)

    def polylaplacian(
        self, state: FieldState, name: str, *, k: int,
    ) -> Tensor:
        """Spatial ``Delta^k f`` via diagonal Fourier multipliers.

        ``Delta^k phi_{j_1,...,j_D} = (-1)^k * (k_x^2 + k_y^2 + ...)^k *
        phi_{j_1,...,j_D}``.
        """
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        a = self._coeff_blocks_for_state(state)              # (B, C, ...)
        ci = self.components.index(name)
        # Build the per-axis k^2 tables aligned with the (2K+1) layout:
        #   index 0: 0, indices 1..K: k_m^2, indices K+1..2K: k_m^2.
        k2_tables = []
        for full_axis in self._coord_axis_indices:
            kv = self.k_vec(full_axis)
            k2 = kv * kv
            zero = torch.zeros(1, dtype=kv.dtype, device=kv.device)
            k2_full = torch.cat([zero, k2, k2])              # (2K+1,)
            k2_tables.append(k2_full)
        # Sum over spatial axes: K2_total[j_1,...,j_D] = sum_d k2_tables[d][j_d]
        # Build via broadcasting.
        K2_sum = None
        for d, t in enumerate(k2_tables):
            shape = [1] * self.D_spatial
            shape[d] = -1
            view = t.view(*shape)
            K2_sum = view if K2_sum is None else K2_sum + view
        assert K2_sum is not None
        # Multiply by (-1)^k * K2^k
        sign = (-1.0) ** k
        mult = sign * K2_sum.pow(k)                          # (n_1, ..., n_D)
        a_c = a[:, ci]                                        # (B, n_1, ..., n_D)
        a_w = a_c * mult.unsqueeze(0)                         # (B, n_1, ..., n_D)
        bases = self._bases_for_state(state)                  # value bases
        # Reshape a_w to (B, 1, ...) for the einsum helper.
        a_w_4d = a_w.unsqueeze(1)
        return _multi_axis_einsum(a_w_4d, bases)[:, 0]

    def __repr__(self) -> str:
        return (
            f"SpectralVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, K={self.K}, L={self.L}, "
            f"time_hidden={self.time_hidden}, time_depth={self.time_depth}, "
            f"activation={self.spec.name!r})"
        )


__all__ = ["SpectralVectorField"]

# Marker read by the omnibias-fields backend ops to select the dispatch path.
SpectralVectorField._omnibias_dispatch = "spectral"
SpectralVectorField._omnibias_readout_independent = True

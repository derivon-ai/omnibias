# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``OneLayerVectorField`` -- one-hidden-layer multi-output omnibias field.

This is the multi-component generalisation of
:class:`omnibias.torch.architectures.PINNOMBU`. Every output channel
shares a *single* hidden layer of width ``H`` so that all per-channel
derivatives reuse the same fast-path :math:`\\sigma^{(n)}(z)` evaluations.

Architecture
------------

For ``D`` input axes (spatial + time) and ``C`` output channels:

.. math::
    z_h(x) &= \\sum_i W_{hi} x_i + \\beta_h, \\qquad h = 1, ..., H \\\\
    f_c(x) &= b_c + \\sum_h c_{ch}\\, \\sigma(z_h),
        \\qquad c = 1, ..., C

All spatial / temporal derivatives reduce to chain-rule formulas in
``\\sigma^{(n)}(z_h)``:

- :math:`\\partial_a f_c = \\sum_h c_{ch}\\, W_{ha}\\, \\sigma'(z_h)`,
- :math:`\\Delta f_c = \\sum_h c_{ch}\\, \\|W_{h,\\text{spatial}}\\|^2\\, \\sigma''(z_h)`,
- ... and so on up to any order the activation's fast-path supports.

The fused-state contract: every derivative call goes through the
:class:`FieldState`'s :class:`SigmaCache`, so each :math:`\\sigma^{(n)}`
is computed at most once per residual.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.base import FieldBase
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from torch import Tensor


class OneLayerVectorField(FieldBase):
    """One-layer multi-output omnibias field.

    Parameters
    ----------
    coordinate_spec
        :class:`CoordinateSpec` describing the input axes.
    components
        :class:`ComponentSpec` listing the output channels (with named
        groups, e.g. ``"velocity": ("u", "v", "w")``).
    hidden
        Number of hidden units :math:`H`. Default 64.
    base
        Base activation. Either the registry name (``"tanh"``,
        ``"sigmoid"``, ``"softplus"``, ``"gaussian"``, ``"exp"``) or an
        :class:`ActivationSpec`. Must have a fast-path so derivative
        towers are closed-form.
    weight_init_scale
        Scale of the initial Gaussian weights for ``W`` and ``c``.
        Default ``1.0 / sqrt(D)`` (standard fan-in scaling).
    bias_init
        Either ``"zeros"`` (default) or ``"normal"`` (small random).
    dtype
        Default tensor dtype. Default :class:`torch.float64` because all
        v0.1 parity tests are run in double precision.
    """

    spec: ActivationSpec[Tensor]
    W: nn.Linear
    c: nn.Linear

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        hidden: int = 64,
        base: str | ActivationSpec[Tensor] = "tanh",
        weight_init_scale: float | None = None,
        bias_init: str = "zeros",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            coordinate_spec=coordinate_spec, components=components,
        )
        self.hidden = int(hidden)
        if self.hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {self.hidden}")
        D = coordinate_spec.ndim
        C = components.n_components
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        if self.spec.fastpath is None:
            raise ValueError(
                f"OneLayerVectorField requires a fast-path activation; "
                f"{self.spec.name!r} has none. Try 'tanh', 'sigmoid', 'softplus', "
                "'gaussian', or 'exp'."
            )

        # Hidden weights and biases.
        self.W = nn.Linear(D, self.hidden, bias=True, dtype=dtype)
        # Output / readout weights and biases.
        self.c = nn.Linear(self.hidden, C, bias=True, dtype=dtype)

        scale = (1.0 / math.sqrt(D)) if weight_init_scale is None else float(weight_init_scale)
        with torch.no_grad():
            self.W.weight.normal_(mean=0.0, std=scale)
            self.c.weight.normal_(mean=0.0, std=scale / math.sqrt(self.hidden))
            if bias_init == "zeros":
                self.W.bias.zero_()
                self.c.bias.zero_()
            elif bias_init == "normal":
                self.W.bias.normal_(mean=0.0, std=scale)
                self.c.bias.normal_(mean=0.0, std=scale)
            else:
                raise ValueError(f"unknown bias_init: {bias_init!r}")

    # ------------------------------------------------------------------
    # Closed-form helpers consumed by the torch ops dispatch.
    # ------------------------------------------------------------------

    def _pre_activations(self, coords: Tensor) -> Tensor:
        """Return ``z = W coords + beta`` of shape ``(B, H)``."""
        return self.W(coords)

    def _sigma(self, z: Tensor) -> Tensor:
        return self.spec.forward(z)

    def _sigma_n(self, z: Tensor, order: int) -> Tensor:
        if order == 0:
            return self.spec.forward(z)
        fp = self.spec.fastpath
        assert fp is not None, "fastpath checked at __init__"
        return fp(z, order)

    def _spatial_axes(self) -> tuple[int, ...]:
        """Integer indices of the *spatial* axes (everything but time)."""
        ta = self.coordinate_spec.time_axis
        return tuple(
            i for i, name in enumerate(self.coordinate_spec.axes)
            if name != ta
        )

    def _row_norm_sq_spatial(self) -> Tensor:
        """``sum_{a in spatial} W[h, a]^2`` per hidden unit, shape ``(H,)``."""
        sa = self._spatial_axes()
        if not sa:
            return torch.zeros(
                self.hidden, dtype=self.W.weight.dtype, device=self.W.weight.device,
            )
        W_spatial = self.W.weight[:, list(sa)]
        return (W_spatial * W_spatial).sum(dim=-1)

    # Closed-form value: ``f_c(x) = b_c + sum_h c[c,h] * sigma(z_h)``.
    def value(self, sigma_z: Tensor, name: str) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]  # (H,)
        b_c = self.c.bias[ci]
        return b_c + sigma_z @ c_row

    def value_all(self, sigma_z: Tensor) -> Tensor:
        """Return ``(B, C)`` stacked component values."""
        return self.c(sigma_z)

    def forward_values(self, coords: Tensor) -> Tensor:
        """Return ``(B, C)`` component values straight from ``coords``.

        The one-line contract a composite field (a partition of unity, a
        multi-patch decomposition) needs from a sub-solution, so composites can
        mix field *types* instead of hard-coding this one.
        """
        return self.value_all(self._sigma(self._pre_activations(coords)))

    # First-order partial: ``df_c/dx_axis = sum_h c[c,h] W[h,axis] sigma'(z_h)``.
    def first_partial(self, sigma_p: Tensor, name: str, axis: int) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]                 # (H,)
        W_axis = self.W.weight[:, axis]           # (H,)
        return (sigma_p * W_axis) @ c_row

    # ``n``-th pure partial: ``d^n f_c/dx_axis^n = sum_h c[c,h] W[h,axis]^n sigma^(n)(z_h)``.
    def nth_partial(
        self, sigma_n: Tensor, name: str, axis: int, order: int,
    ) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]
        W_axis = self.W.weight[:, axis]
        return (sigma_n * W_axis.pow(order)) @ c_row

    # Mixed partial: ``d^|orders| f_c / prod_a dx_a^{orders_a}``.
    def mixed_partial(
        self,
        sigma_n: Tensor,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]
        W_factor = torch.ones_like(self.W.weight[:, 0])  # (H,)
        for a, o in zip(axes, orders, strict=False):
            W_factor = W_factor * self.W.weight[:, a].pow(o)
        return (sigma_n * W_factor) @ c_row

    # Gradient of f_c wrt all spatial+time axes:
    #     (B, D) = sigma'(z) @ (c_row[None, :] * W[h, :]).
    def gradient_full(self, sigma_p: Tensor, name: str) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]                                   # (H,)
        W = self.W.weight                                          # (H, D)
        return (sigma_p * c_row) @ W                                # (B, D)

    # Spatial-only gradient.
    def gradient_spatial(self, sigma_p: Tensor, name: str) -> Tensor:
        sa = self._spatial_axes()
        full = self.gradient_full(sigma_p, name)
        if not sa:
            return full[..., :0]
        return full[..., list(sa)]

    # Closed-form Laplacian (spatial axes only).
    def laplacian(self, sigma_pp: Tensor, name: str) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]                                   # (H,)
        row_norm_sq = self._row_norm_sq_spatial()                   # (H,)
        return (sigma_pp * (c_row * row_norm_sq)) @ torch.ones_like(c_row)

    # Closed-form Hessian: H_x f_c = W^T diag(sigma''(z) c[c,:]) W
    # restricted to spatial axes when needed.
    def hessian_full(self, sigma_pp: Tensor, name: str) -> Tensor:
        ci = self.components.index(name)
        c_row = self.c.weight[ci]                                   # (H,)
        weights = sigma_pp * c_row                                  # (B, H)
        W = self.W.weight                                          # (H, D)
        return torch.einsum("bh,hi,hj->bij", weights, W, W)

    def hessian_spatial(self, sigma_pp: Tensor, name: str) -> Tensor:
        sa = self._spatial_axes()
        full = self.hessian_full(sigma_pp, name)
        if not sa:
            return full[..., :0, :0]
        idx = list(sa)
        return full[..., idx, :][..., :, idx]

    # k-th polylaplacian: ``Delta^k f = sum_h c[c,h] sigma^{(2k)}(z_h) ||W_spatial[h]||^{2k}``.
    def polylaplacian(self, sigma_2k: Tensor, name: str, k: int) -> Tensor:
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        ci = self.components.index(name)
        c_row = self.c.weight[ci]
        row_norm_sq = self._row_norm_sq_spatial()
        return (sigma_2k * (c_row * row_norm_sq.pow(k))) @ torch.ones_like(c_row)

    def __repr__(self) -> str:
        return (
            f"OneLayerVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, hidden={self.hidden}, "
            f"base={self.spec.name!r})"
        )


__all__ = ["OneLayerVectorField"]

# Marker read by the omnibias-fields backend ops to select the closed-form
# sigma-tower reduction path (avoids a fields -> pinn import cycle).
OneLayerVectorField._omnibias_dispatch = "one_layer"

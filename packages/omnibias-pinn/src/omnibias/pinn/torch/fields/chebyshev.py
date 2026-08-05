# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``ChebyshevVectorField`` -- D-dimensional Chebyshev-T vector field
with omnibias temporal head.

Architecture
------------

For ``D`` *non-periodic* spatial axes (each with ``K + 1`` Chebyshev
modes ``T_0, ..., T_K``) plus a time axis, and ``C`` output channels:

.. math::
    f_c(x_1, ..., x_D, t) = \\sum_{n_1, ..., n_D} a_c^{n_1...n_D}(t)\\,
        T_{n_1}(\\xi_1) \\cdots T_{n_D}(\\xi_D),

where ``\\xi_d = 2 (x_d - L_d^{lo})/(L_d^{hi} - L_d^{lo}) - 1`` rescales
each spatial input to the Chebyshev domain ``[-1, 1]``.

The temporal coefficient ``a_c^{n_1...n_D}(t)`` is the output of a deep
MLP whose first hidden layer is the omnibias fast-path layer; the
final readout is to ``C * (K + 1)^D`` outputs which is reshaped per
component.

Spatial derivatives use the Chebyshev *differentiation matrix*
``D[n, m] = 2m / c_n`` if ``m > n`` and ``m + n`` is odd, else 0
(``c_0 = 2``, ``c_n = 1`` for ``n >= 1``). This converts coefficients
of ``f`` to coefficients of ``f'`` in the same basis. Higher
derivatives use ``D^k``. The per-axis cost is one ``(K+1) x (K+1)``
matmul per derivative order; this is essentially free for small
``K`` (the polynomial regime; <= 24 modes is typical).

Time derivatives are closed-form through the omnibias fast-path on
the temporal MLP for ``time_depth = 1`` (the default).
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
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def _chebyshev_differentiation_matrix(K: int, dtype: torch.dtype) -> Tensor:
    """Return the (K+1, K+1) differentiation matrix for the basis ``{T_n}``.

    ``D[n, m] = 2m / c_n`` if ``m > n`` and ``(m + n)`` is odd else 0.
    With ``c_0 = 2`` and ``c_n = 1`` for ``n >= 1``.
    """
    N = K + 1
    D = torch.zeros((N, N), dtype=dtype)
    for n in range(N):
        cn = 2.0 if n == 0 else 1.0
        for m in range(n + 1, N):
            if (m + n) % 2 == 1:
                D[n, m] = (2.0 * m) / cn
    return D


def _chebyshev_basis(x: Tensor, K: int) -> Tensor:
    """Evaluate Chebyshev basis ``T_0, ..., T_K`` at points ``x in [-1, 1]``.

    Returns ``(B, K + 1)`` via the recurrence
    ``T_0 = 1``, ``T_1 = x``, ``T_{n+1} = 2 x T_n - T_{n-1}``.

    Implemented with a Python list so the autograd graph stays clean
    (no in-place writes to a pre-allocated tensor).
    """
    cols: list[Tensor] = [torch.ones_like(x)]
    if K >= 1:
        cols.append(x)
    for n in range(1, K):
        cols.append(2.0 * x * cols[n] - cols[n - 1])
    return torch.stack(cols, dim=-1)


def _multi_axis_einsum(a: Tensor, bases: list[Tensor]) -> Tensor:
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


class ChebyshevVectorField(FieldBase):
    """D-dim Chebyshev-T vector field with omnibias time head.

    Parameters
    ----------
    coordinate_spec
        Must declare a time axis. Spatial axes should be *non-periodic*
        (Chebyshev's natural setting) and have a finite ``domain`` if
        you want results outside the default ``[-1, 1]`` range.
    components
        Output component spec; ``C = components.n_components``.
    K
        Polynomial order. Output dim per component is ``(K + 1)^D``.
        Memory: ``B * C * (K + 1)^D``. ``K <= 24`` is typical.
    domain
        Optional ``(lo, hi)`` per spatial axis. If ``None`` the spec's
        ``domain`` (if any) is used; otherwise default ``[-1, 1]``
        per axis.
    time_hidden, time_depth, activation, weight_init_scale, dtype
        Same semantics as :class:`SpectralVectorField`.
    """

    spec: ActivationSpec[Tensor]

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
        K: int = 8,
        domain: tuple[tuple[float, float], ...] | None = None,
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
                "ChebyshevVectorField requires a time axis on the coordinate spec"
            )
        self.K = int(K)
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        self.D_spatial = coordinate_spec.n_spatial
        self.C = components.n_components
        if domain is None:
            spec_domain = coordinate_spec.domain
            if spec_domain is None:
                resolved = tuple((-1.0, 1.0) for _ in range(self.D_spatial))
            else:
                resolved = tuple(
                    spec_domain[coordinate_spec.axis_index(a)]
                    for a in coordinate_spec.spatial_axes
                )
        else:
            resolved = tuple(
                (float(lo), float(hi)) for (lo, hi) in domain
            )
            if len(resolved) != self.D_spatial:
                raise ValueError(
                    f"domain has length {len(resolved)} but coordinate spec "
                    f"has {self.D_spatial} spatial axes"
                )
        self.domain: tuple[tuple[float, float], ...] = resolved
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
                f"ChebyshevVectorField requires a fast-path activation; "
                f"{self.spec.name!r} has none."
            )
        self._coord_axis_indices = tuple(
            coordinate_spec.axis_index(a)
            for a in coordinate_spec.spatial_axes
        )
        self._time_axis_idx = coordinate_spec.axis_index(
            coordinate_spec.time_axis
        )

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

        self._modes_per_axis = self.K + 1
        self._out_per_component = self._modes_per_axis ** self.D_spatial
        self._out_dim = self.C * self._out_per_component
        self.V = nn.Parameter(
            torch.zeros(self._out_dim, self.time_hidden, dtype=dtype),
        )
        self.b_t = nn.Parameter(torch.zeros(self._out_dim, dtype=dtype))

        D_mat = _chebyshev_differentiation_matrix(self.K, dtype=dtype)
        self.register_buffer("_D_mat", D_mat, persistent=False)

    # ----- bookkeeping ----------------------------------------------

    def _spatial_axis_index(self, full_axis: int) -> int:
        sa = self._coord_axis_indices
        if full_axis not in sa:
            raise ValueError(
                f"axis {full_axis} is not a spatial axis ({sa!r})"
            )
        return sa.index(full_axis)

    def _rescale_to_unit(self, x: Tensor, d: int) -> Tensor:
        """Rescale axis-d input from its domain to ``[-1, 1]``."""
        lo, hi = self.domain[d]
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def _chain_factor(self, d: int, order: int) -> float:
        """Multiplicative factor from chain rule when rescaling."""
        lo, hi = self.domain[d]
        return (2.0 / (hi - lo)) ** order

    # ----- temporal MLP ---------------------------------------------

    def _hidden_t(self, t: Tensor) -> tuple[Tensor, Tensor]:
        if t.dim() == 1:
            t_in = t.unsqueeze(-1)
        else:
            t_in = t
        z = t_in @ self.W_t.T + self.beta_t
        h = self.spec.forward(z)
        for layer in self._inner_layers:
            h = self.spec.forward(layer(h))
        return z, h

    def _hidden_t_and_dt(
        self, t: Tensor, *, t_order: int = 0,
    ) -> Tensor:
        if t_order == 0:
            _, h = self._hidden_t(t)
            return h
        if self.time_depth > 1:
            raise NotImplementedError(
                "Closed-form time derivatives only implemented for time_depth=1"
            )
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
        W_t_pow = self.W_t.squeeze(-1).pow(t_order)
        return sigma_n * W_t_pow

    def _coeff_blocks_at_t(
        self, t: Tensor, *, t_order: int = 0,
    ) -> Tensor:
        h = self._hidden_t_and_dt(t, t_order=t_order)
        if t_order == 0:
            a = self.b_t + h @ self.V.T
        else:
            a = h @ self.V.T
        return a.view(
            -1, self.C, *([self._modes_per_axis] * self.D_spatial),
        )

    def _make_sigma_cache(self, coords: Tensor):  # type: ignore[override]
        t = coords[..., self._time_axis_idx]
        z, _ = self._hidden_t(t)
        from omnibias.pinn._core.sigma_cache import SigmaCache
        return SigmaCache(z=z)

    def _pre_activations(self, coords: Tensor) -> Tensor:
        t = coords[..., self._time_axis_idx]
        z, _ = self._hidden_t(t)
        return z

    # ----- coefficient-side derivatives -----------------------------

    def _basis_dn_axis(
        self,
        x: Tensor,
        d: int,
        order: int,
    ) -> Tensor:
        """Compute the ``order``-th derivative basis along axis ``d``.

        Returns ``(B, K + 1)``. Built by composing the Chebyshev
        differentiation matrix with the value basis evaluated at the
        rescaled inputs, then multiplying by the chain-rule factor.
        """
        xi = self._rescale_to_unit(x, d)
        T = _chebyshev_basis(xi, self.K)                          # (B, K + 1)
        if order == 0:
            return T
        # The transformation `b = D @ a` converts coefficients
        # `a = (a_0, ..., a_K)` of `f = sum_n a_n T_n` to coefficients
        # `b = (b_0, ..., b_K)` of `f' = sum_n b_n T_n`. To get a
        # *derivative basis* (per-mode derivative tensor) we apply
        # `D^T` to the value basis vector: `(D^T)^order @ T(x)`.
        Dn = self._D_mat
        Dn_pow = torch.eye(self.K + 1, dtype=Dn.dtype, device=Dn.device)
        for _ in range(order):
            Dn_pow = Dn_pow @ Dn
        chain = self._chain_factor(d, order)
        return chain * (T @ Dn_pow)

    def _bases_for_state(
        self,
        state: FieldState,
        *,
        order_per_axis: tuple[int, ...] | None = None,
    ) -> list[Tensor]:
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
            bases.append(self._basis_dn_axis(x_d, d, order_per_axis[d]))
        return bases

    def _coeff_blocks_for_state(
        self, state: FieldState, *, t_order: int = 0,
    ) -> Tensor:
        key = f"chebyshev_a_t{t_order}"
        cached = state.extra.get(key)
        if cached is not None:
            return cached
        t = state.coords[..., self._time_axis_idx]
        a = self._coeff_blocks_at_t(t, t_order=t_order)
        state.extra[key] = a
        return a

    # ----- ops surface --------------------------------------------

    def value_component(self, state: FieldState, name: str) -> Tensor:
        ci = self.components.index(name)
        a = self._coeff_blocks_for_state(state)
        bases = self._bases_for_state(state)
        return _multi_axis_einsum(a[:, ci:ci + 1], bases)[:, 0]

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
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
    ) -> Tensor:
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

    def biharmonic(self, state: FieldState, name: str) -> Tensor:
        return self.polylaplacian(state, name, k=2)

    def polylaplacian(
        self, state: FieldState, name: str, *, k: int,
    ) -> Tensor:
        """``Delta^k f`` via composition of Laplacians.

        For Chebyshev the Laplacian is *not* diagonal in coefficient
        space (unlike the Fourier case), so we don't get the same O(1)
        eigenvalue trick. Instead we compose ``2k``-th pure derivatives
        plus mixed cross-terms via the standard expansion of
        ``Delta^k`` -- which is still exact (no autograd) and much
        cheaper than autograd-built ``Delta^k``.
        """
        if k < 1:
            raise ValueError(f"polylaplacian k must be >= 1, got {k}")
        # Use the multinomial expansion: Delta^k = sum_{m_1 + ... + m_D = k}
        #   (k choose m_1, ..., m_D) * prod_d d^{2 m_d}/dx_d^{2 m_d}.
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


__all__ = ["ChebyshevVectorField"]

# Marker read by the omnibias-fields backend ops to select the dispatch path.
ChebyshevVectorField._omnibias_dispatch = "chebyshev"

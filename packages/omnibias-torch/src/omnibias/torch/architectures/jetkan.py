# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-KAN: univariate multi-pack edges with exact jets (theory 02-03).

Each edge is ``phi(u) = sum_g c_g sigma^(n_g)(alpha_g u + mu_g)``. Scales are
``exp(log_scale)`` so they stay positive. Exactness is of the **model jet**,
not the target; the Kolmogorov-Arnold theorem does **not** justify this
architecture (its inner functions may be non-smooth).

Directional jets use :func:`~omnibias.torch.jet.compose_jet`; mixed partials
use :func:`~omnibias.torch.jet_mv.compose_jet_mv` (the same Faà di Bruno
kernel as :func:`~omnibias.torch.jet_mv.mlp_jet_mv`). Do not reimplement
Bell polynomials here.

Refinement in this wave is the honest subset: zero-weight pack birth, and
optional :class:`~omnibias.torch.growable.GrowableOperatorMultiBiasUnit`
order growth on torch. Full residual-driven pack birth/death (03-13) stays
designed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from omnibias.core.multipack import MultiPackSpec, PackSpec
from omnibias.core.spectral_design import BandPlan
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.growable import GrowableOperatorMultiBiasUnit
from omnibias.torch.jet import compose_jet
from omnibias.torch.jet_mv import compose_jet_mv, identity_jet

import torch
import torch.nn as nn
from torch import Tensor


def _edge_tower(
    u0: Tensor,
    weights: Tensor,
    means: Tensor,
    log_scales: Tensor,
    orders: tuple[int, ...],
    spec: ActivationSpec[Tensor],
    order: int,
) -> Tensor:
    """``phi^(k)(u0)`` for ``k = 0..order``. ``weights``/``means``/``log_scales`` are ``(G,)``."""
    if spec.fastpath is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form derivative kernel"
        )
    fp = spec.fastpath
    rows: list[Tensor] = []
    for k in range(order + 1):
        acc: Tensor | None = None
        for g, n in enumerate(orders):
            alpha = torch.exp(log_scales[g])
            term = weights[g] * (alpha ** k) * fp(alpha * u0 + means[g], n + k)
            acc = term if acc is None else acc + term
        assert acc is not None
        rows.append(acc)
    return torch.stack(rows, dim=0)


class _JetKANLayer(nn.Module):
    """One KAN layer: ``y_q = sum_p phi_{q,p}(x_p)``."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        orders: tuple[int, ...],
        n_packs: int,
        extra_packs: int,
        base: ActivationSpec[Tensor],
        growable: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.orders = tuple(int(n) for n in orders)
        self.n_packs = int(n_packs)
        self.capacity = int(n_packs) + int(extra_packs)
        self.act_spec = base
        g = self.capacity
        # First ``n_packs`` live; extra slots start at weight 0 (birth invariant).
        w0 = torch.zeros((n_out, n_in, g), dtype=dtype)
        w0[..., : self.n_packs] = 1.0 / float(max(self.n_packs, 1))
        self.weights = nn.Parameter(w0)
        self.means = nn.Parameter(torch.zeros((n_out, n_in, g), dtype=dtype))
        self.log_scales = nn.Parameter(torch.zeros((n_out, n_in, g), dtype=dtype))
        self.register_buffer("_active_g", torch.tensor(self.n_packs, dtype=torch.long))
        self.growable_unit: GrowableOperatorMultiBiasUnit | None
        if growable:
            self.growable_unit = GrowableOperatorMultiBiasUnit(
                num_channels=n_out * n_in,
                init_K=1,
                K_max=8,
                base=base,
            )
        else:
            self.growable_unit = None

    @property
    def active_g(self) -> int:
        return int(cast(Tensor, self._active_g).item())

    def _edge_sum(self, x: Tensor, g_live: int) -> Tensor:
        if self.act_spec.fastpath is None:
            raise NotImplementedError(
                f"Activation {self.act_spec.name!r} has no closed-form derivative kernel"
            )
        fp = self.act_spec.fastpath
        # x: (..., n_in) -> y: (..., n_out)
        acc: Tensor | None = None
        orders = self.orders
        n_edge_packs = min(g_live, len(orders))
        for g in range(g_live):
            n = orders[g] if g < n_edge_packs else orders[-1]
            alpha = torch.exp(self.log_scales[..., g])
            u = x.unsqueeze(-2)  # (..., 1, n_in)
            z = alpha * u + self.means[..., g]
            term = self.weights[..., g] * fp(z, n)
            acc = term if acc is None else acc + term
        assert acc is not None
        return acc.sum(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        y = self._edge_sum(x, self.active_g)
        unit = self.growable_unit
        if unit is not None:
            batch = x.shape[:-1]
            z = x.unsqueeze(-2).expand(*batch, self.n_out, self.n_in)
            flat = z.reshape(*batch, self.n_out * self.n_in)
            y = y + unit(flat).reshape(*batch, self.n_out, self.n_in).sum(dim=-1)
        return y

    def directional_jet(self, in_jet: Tensor, order: int) -> Tensor:
        """``in_jet`` is ``(N+1, n_in)``; returns ``(N+1, n_out)``."""
        g_live = self.active_g
        outs: list[Tensor] = []
        for q in range(self.n_out):
            acc: Tensor | None = None
            for p in range(self.n_in):
                u_jet = in_jet[:, p]
                orders = tuple(
                    self.orders[g] if g < len(self.orders) else self.orders[-1]
                    for g in range(g_live)
                )
                tower = _edge_tower(
                    u_jet[0],
                    self.weights[q, p, :g_live],
                    self.means[q, p, :g_live],
                    self.log_scales[q, p, :g_live],
                    orders,
                    self.act_spec,
                    order,
                )
                part = compose_jet(u_jet, tower)
                acc = part if acc is None else acc + part
            assert acc is not None
            outs.append(acc)
        return torch.stack(outs, dim=-1)

    def mixed_jet(self, in_jets: Tensor, dim: int, order: int) -> Tensor:
        """``in_jets`` is ``(M, n_in)`` multivariate; returns ``(M, n_out)``."""
        g_live = self.active_g
        outs: list[Tensor] = []
        for q in range(self.n_out):
            acc: Tensor | None = None
            for p in range(self.n_in):
                u_jet = in_jets[:, p]
                orders = tuple(
                    self.orders[g] if g < len(self.orders) else self.orders[-1]
                    for g in range(g_live)
                )
                tower = _edge_tower(
                    u_jet[0],
                    self.weights[q, p, :g_live],
                    self.means[q, p, :g_live],
                    self.log_scales[q, p, :g_live],
                    orders,
                    self.act_spec,
                    order,
                )
                part = compose_jet_mv(u_jet, tower, dim, order)
                acc = part if acc is None else acc + part
            assert acc is not None
            outs.append(acc)
        return torch.stack(outs, dim=-1)


@dataclass(frozen=True)
class JetKANConfig:
    """Widths include input and output, e.g. ``(2, 2, 1)``."""

    widths: tuple[int, ...]
    packs_per_edge: int = 3
    extra_packs: int = 2
    orders: tuple[int, ...] = (0, 1, 2)
    base: str = "tanh"
    growable: bool = False

    def __post_init__(self) -> None:
        if len(self.widths) < 2:
            raise ValueError("widths must include at least input and output")
        if any(int(w) < 1 for w in self.widths):
            raise ValueError("each width must be >= 1")
        if int(self.packs_per_edge) < 1:
            raise ValueError("packs_per_edge must be >= 1")
        if int(self.extra_packs) < 0:
            raise ValueError("extra_packs must be >= 0")
        if not self.orders:
            raise ValueError("orders must be non-empty")
        if any(int(n) < 0 for n in self.orders):
            raise ValueError("orders must be >= 0")


def jetkan_from_band_plan(
    plan: BandPlan,
    widths: tuple[int, ...],
    *,
    base: str = "tanh",
    growable: bool = False,
) -> JetKANConfig:
    """Optional 01-07 helper: one live pack per band-plan channel.

    01-06 wavelet frames stay concept; this only copies the channel count.
    """
    return JetKANConfig(
        widths=tuple(int(w) for w in widths),
        packs_per_edge=int(plan.n_channels),
        extra_packs=2,
        orders=tuple(int(n) for n in plan.orders),
        base=base,
        growable=growable,
    )


class JetKAN(nn.Module):
    """Stacked univariate multi-pack edges. Kolmogorov-Arnold is not a proof."""

    def __init__(self, config: JetKANConfig, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.config = config
        dt = torch.get_default_dtype() if dtype is None else dtype
        act = get_activation(config.base)
        layers: list[_JetKANLayer] = []
        for n_in, n_out in zip(config.widths[:-1], config.widths[1:], strict=True):
            layers.append(
                _JetKANLayer(
                    n_in,
                    n_out,
                    orders=config.orders,
                    n_packs=config.packs_per_edge,
                    extra_packs=config.extra_packs,
                    base=act,
                    growable=config.growable,
                    dtype=dt,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.layers:
            assert isinstance(layer, _JetKANLayer)
            h = layer(h)
        return h

    def jet(self, x: Tensor, order: int, direction: Tensor) -> Tensor:
        """Directional Taylor jet of the model along ``x + t direction``, via ``compose_jet``.

    Coefficients follow :func:`~omnibias.torch.jet.mlp_jet`: ``jet_to_tower``
    recovers raw directional derivatives.
    """
        x0 = torch.as_tensor(x)
        v = torch.as_tensor(direction)
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        rows = [x0]
        if order >= 1:
            rows.append(v)
        rows.extend(torch.zeros_like(x0) for _ in range(max(order - 1, 0)))
        jet = torch.stack(rows[: order + 1], dim=0)
        for layer in self.layers:
            assert isinstance(layer, _JetKANLayer)
            jet = layer.directional_jet(jet, order)
        return jet

    def jet_mv(self, x: Tensor, total_order: int) -> Tensor:
        """All mixed partials of the model to ``total_order``, via ``compose_jet_mv``."""
        x0 = torch.as_tensor(x)
        if x0.ndim != 1:
            raise ValueError(f"jet_mv expects a 1-D point, got shape {tuple(x0.shape)}")
        dim = int(x0.shape[0])
        jet = identity_jet(x0, total_order)
        for layer in self.layers:
            assert isinstance(layer, _JetKANLayer)
            jet = layer.mixed_jet(jet, dim, total_order)
        return jet

    def refine(self, kind: str = "pack") -> None:
        """Zero-weight pack birth, or GrowableOMBU order growth (torch only)."""
        name = str(kind).lower()
        if name == "pack":
            for layer in self.layers:
                assert isinstance(layer, _JetKANLayer)
                if layer.active_g >= layer.capacity:
                    raise RuntimeError("no unused pack slots remain")
                buf = cast(Tensor, layer._active_g)
                buf.copy_(buf + 1)
            return
        if name == "order":
            grew = False
            for layer in self.layers:
                assert isinstance(layer, _JetKANLayer)
                if layer.growable_unit is not None and layer.growable_unit.can_grow:
                    layer.growable_unit.grow(strategy="pair")
                    grew = True
            if not grew:
                raise RuntimeError("order growth needs growable=True and free K slots")
            return
        raise ValueError(f"unknown refine kind {kind!r}; expected 'pack' or 'order'")


def edge_functions(model: JetKAN) -> tuple[MultiPackSpec, ...]:
    """Inspect live edge packs (interpretability; means/weights as floats)."""
    specs: list[MultiPackSpec] = []
    for layer in model.layers:
        assert isinstance(layer, _JetKANLayer)
        g_live = layer.active_g
        w = layer.weights.detach()
        m = layer.means.detach()
        for q in range(layer.n_out):
            for p in range(layer.n_in):
                packs = []
                for g in range(g_live):
                    n = layer.orders[g] if g < len(layer.orders) else layer.orders[-1]
                    packs.append(
                        PackSpec(
                            order=int(n),
                            mean=float(m[q, p, g]),
                            weight=float(w[q, p, g]),
                        )
                    )
                specs.append(MultiPackSpec(tuple(packs)))
    return tuple(specs)


__all__ = [
    "JetKAN",
    "JetKANConfig",
    "edge_functions",
    "jetkan_from_band_plan",
]

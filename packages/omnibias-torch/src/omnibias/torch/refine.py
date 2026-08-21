# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Adaptive pack refinement (theory 03-13), PyTorch twin.

:class:`AdaptivePackBank` is a pre-allocated bank of tempered packs

    c * alpha^n sigma^(n)(alpha (z - center))

Birth and *p*-type growth write a new slot with ``c = 0`` (bit-identical
output). Death deactivates a slot and reports
:attr:`~omnibias.core.refine.RefineReport.death_perturbation`.

**Optimizer-state policy** (same as :class:`GrowableOperatorMultiBiasUnit`):
slots are allocated up to ``max_packs``. Inactive slots do not enter the
forward sum, so they receive no gradient and Adam-style moments stay at
zero until first activation. Death leaves whatever moment state the slot
had; a later reuse of that slot inherits it. Callers who hold an
optimizer may zero that slot's moments on death. Do not reconstruct the
module mid-step.

Literal-``K`` growth of a single OMBU still goes through
:meth:`GrowableOperatorMultiBiasUnit.grow` via :func:`grow_ombu`. That
path is torch-only (there is no jax ``GrowableOMBU``). Pack-bank
*p*-type growth is the collapsed sibling and is the jax-parity path.

The refine driver is eager. Do not wrap it in ``torch.compile``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from omnibias.core.refine import (
    Indicator,
    RefinedPack,
    RefinePolicy,
    RefineReport,
    assert_zero_perturbation,
    certified_local_error,
    pack_contributions,
    refine_bank,
)
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.growable import GrowableOperatorMultiBiasUnit, GrowStrategy

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import grad

ResidualFn = Callable[[Tensor], Tensor]
ScalarFn = Callable[[Tensor], Tensor]


def _as_1d(z: Tensor) -> Tensor:
    if z.ndim == 0:
        return z.reshape(1)
    if z.ndim == 1:
        return z
    if z.shape[-1] == 1:
        return z.reshape(-1)
    raise ValueError(f"AdaptivePackBank expects a 1-D field, got shape {tuple(z.shape)}")


def scalar_jet_1d(fn: ScalarFn, x0: float, order: int) -> list[float]:
    """``phi^(k)(x0)`` for ``k = 0 .. order`` via nested reverse-mode."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    x = torch.tensor(float(x0), dtype=torch.get_default_dtype())

    def restriction(s: Tensor) -> Tensor:
        out = fn(s.reshape(()))
        if out.ndim != 0:
            out = out.reshape(())
        return out

    derivs = [float(restriction(x).detach())]
    current: Callable[[Tensor], Tensor] = restriction
    for _ in range(order):
        current = grad(current)
        derivs.append(float(current(x)))
    return derivs


def pack_response(
    z: Tensor,
    center: Tensor,
    weight: Tensor,
    scale: Tensor,
    order: int,
    spec: ActivationSpec[Tensor],
) -> Tensor:
    """``c * alpha^n sigma^(n)(alpha (z - center))``."""
    if spec.fastpath is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form derivative kernel"
        )
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    u = scale * (z - center)
    return weight * (scale ** order) * spec.fastpath(u, order)


class AdaptivePackBank(nn.Module):
    """Pre-allocated tempered pack bank with birth / growth / death.

    Parameters
    ----------
    packs:
        Initial live packs. Length must be ``<= max_packs``.
    max_packs:
        Slot budget. Optimizer moments for unused slots stay at zero.
    max_order:
        Highest derivative order a slot may hold (static loop bound).
    base:
        Activation with a closed-form fastpath.
    """

    act_spec: ActivationSpec[Tensor]
    centers: Tensor
    weights: Tensor
    scales: Tensor
    orders: Tensor
    ages: Tensor
    active: Tensor

    def __init__(
        self,
        packs: Sequence[RefinedPack],
        *,
        max_packs: int = 16,
        max_order: int = 8,
        base: str | ActivationSpec[Tensor] = "exp",
        learnable_centers: bool = True,
        learnable_weights: bool = True,
        learnable_scales: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if int(max_packs) < 1:
            raise ValueError(f"max_packs must be >= 1, got {max_packs}")
        if int(max_order) < 0:
            raise ValueError(f"max_order must be >= 0, got {max_order}")
        live = tuple(packs)
        if len(live) > int(max_packs):
            raise ValueError(f"{len(live)} initial packs exceed max_packs={max_packs}")
        for pack in live:
            if pack.order > int(max_order):
                raise ValueError(
                    f"pack order {pack.order} exceeds max_order={max_order}"
                )
        self.max_packs = int(max_packs)
        self.max_order = int(max_order)
        self.act_spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        if self.act_spec.fastpath is None:
            raise NotImplementedError(
                f"Activation {self.act_spec.name!r} has no closed-form derivative kernel"
            )
        dt = torch.get_default_dtype() if dtype is None else dtype
        centers0 = torch.zeros(self.max_packs, dtype=dt)
        weights0 = torch.zeros(self.max_packs, dtype=dt)
        scales0 = torch.ones(self.max_packs, dtype=dt)
        orders0 = torch.zeros(self.max_packs, dtype=torch.long)
        ages0 = torch.zeros(self.max_packs, dtype=torch.long)
        active0 = torch.zeros(self.max_packs, dtype=torch.bool)
        for i, pack in enumerate(live):
            centers0[i] = pack.center
            weights0[i] = pack.weight
            scales0[i] = pack.scale
            orders0[i] = pack.order
            ages0[i] = pack.age
            active0[i] = True
        if learnable_centers:
            self.centers = nn.Parameter(centers0)
        else:
            self.register_buffer("centers", centers0)
        if learnable_weights:
            self.weights = nn.Parameter(weights0)
        else:
            self.register_buffer("weights", weights0)
        if learnable_scales:
            self.scales = nn.Parameter(scales0)
        else:
            self.register_buffer("scales", scales0)
        self.register_buffer("orders", orders0)
        self.register_buffer("ages", ages0)
        self.register_buffer("active", active0)

    @property
    def n_active(self) -> int:
        return int(self.active.sum().item())

    def active_packs(self) -> list[RefinedPack]:
        """Live packs in slot order (stable, used as the refine algebra list)."""
        out: list[RefinedPack] = []
        for i in range(self.max_packs):
            if bool(self.active[i].item()):
                out.append(self._pack_at(i))
        return out

    def _pack_at(self, slot: int) -> RefinedPack:
        return RefinedPack(
            order=int(self.orders[slot].item()),
            center=float(self.centers[slot].detach()),
            weight=float(self.weights[slot].detach()),
            scale=float(self.scales[slot].detach()),
            age=int(self.ages[slot].item()),
        )

    def _key(self, pack: RefinedPack) -> tuple[int, float, float, float]:
        return (pack.order, pack.center, pack.scale, pack.weight)

    def forward(self, z: Tensor) -> Tensor:
        """1-D field ``z`` → same leading shape."""
        flat = _as_1d(z)
        if self.act_spec.fastpath is None:
            raise NotImplementedError(
                f"Activation {self.act_spec.name!r} has no closed-form derivative kernel"
            )
        out = torch.zeros_like(flat)
        for i in range(self.max_packs):
            if not bool(self.active[i].item()):
                continue
            n = int(self.orders[i].item())
            out = out + pack_response(
                flat,
                self.centers[i],
                self.weights[i],
                self.scales[i],
                n,
                self.act_spec,
            )
        return out.reshape(z.shape) if z.ndim != 1 and z.shape[-1] == 1 else out

    def pack_term_norms(self, z: Tensor) -> list[float]:
        """L2 norms of each live pack's term on ``z``, in slot order."""
        flat = _as_1d(z)
        norms: list[float] = []
        for i in range(self.max_packs):
            if not bool(self.active[i].item()):
                continue
            n = int(self.orders[i].item())
            term = pack_response(
                flat,
                self.centers[i].detach(),
                self.weights[i].detach(),
                self.scales[i].detach(),
                n,
                self.act_spec,
            )
            norms.append(float(term.square().sum().sqrt().item()))
        return norms

    def _write_slot(self, slot: int, pack: RefinedPack) -> None:
        with torch.no_grad():
            self.centers[slot] = pack.center
            self.weights[slot] = pack.weight
            self.scales[slot] = pack.scale
            self.orders[slot] = pack.order
            self.ages[slot] = pack.age
            self.active[slot] = True

    def _deactivate(self, slot: int) -> None:
        with torch.no_grad():
            self.active[slot] = False
            self.weights[slot] = 0.0
            self.ages[slot] = 0

    def sync_from(self, packs: Sequence[RefinedPack]) -> None:
        """Write a refined list back onto slots, keeping live packs in place."""
        occupied: dict[tuple[int, float, float, float], int] = {}
        for i in range(self.max_packs):
            if bool(self.active[i].item()):
                occupied[self._key(self._pack_at(i))] = i
        assigned: set[int] = set()
        for pack in packs:
            key = self._key(pack)
            slot = occupied.get(key)
            if slot is not None and slot not in assigned:
                with torch.no_grad():
                    self.ages[slot] = pack.age
                assigned.add(slot)
                continue
            free = next((i for i in range(self.max_packs) if i not in assigned), None)
            if free is None:
                raise RuntimeError(f"max_packs={self.max_packs} exhausted")
            self._write_slot(free, pack)
            assigned.add(free)
        for i in range(self.max_packs):
            if i not in assigned:
                self._deactivate(i)

    def extra_repr(self) -> str:
        return (
            f"n_active={self.n_active}/{self.max_packs}, "
            f"base={self.act_spec.name!r}, max_order={self.max_order}"
        )


def grow_ombu(
    unit: GrowableOperatorMultiBiasUnit,
    strategy: GrowStrategy = "pair",
    **kwargs: object,
) -> int:
    """Delegate *p*-type growth to the existing growable OMBU. Do not reimplement it."""
    return unit.grow(strategy, **kwargs)  # type: ignore[arg-type]


def _jet_source(
    residual_fn: ResidualFn,
    target_fn: ResidualFn | None,
    x0: float,
    order: int,
) -> list[float]:
    source = target_fn if target_fn is not None else residual_fn

    def scalar(s: Tensor) -> Tensor:
        val = source(s.reshape(1))
        return val.reshape(())

    return scalar_jet_1d(scalar, x0, order)


def refine(
    bank: AdaptivePackBank,
    residual_fn: ResidualFn,
    probes: Tensor,
    policy: RefinePolicy,
    *,
    step: int,
    target_fn: ResidualFn | None = None,
    probe_jet: tuple[float, Sequence[float]] | None = None,
) -> RefineReport:
    """One refine tick. ``step`` is recorded for callers; ages increment here.

    Singularity / scale-flow read ``target_fn`` when given (the field being
    resolved), otherwise the residual jet. Residual / certified-error read
    ``residual_fn``. ``probe_jet`` overrides the jet (used by G3).
    """
    del step
    probes_1d = _as_1d(probes)
    residual = _as_1d(residual_fn(probes_1d))
    if residual.shape != probes_1d.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} != probes {tuple(probes_1d.shape)}"
        )
    before = None
    if policy.debug_zero_perturbation:
        before = [float(v) for v in bank(probes_1d).detach().reshape(-1)]

    packs = bank.active_packs()
    field = _as_1d(bank(probes_1d)).detach()
    field_norm = float(field.square().sum().sqrt().item())
    contrib = pack_contributions(bank.pack_term_norms(probes_1d), field_norm)
    residuals = [float(v) for v in residual.detach()]
    probe_list = [float(v) for v in probes_1d.detach()]

    certified: list[float] | None = None
    if policy.indicator is Indicator.CERTIFIED_ERROR:
        if len(probe_list) < 2:
            spacing = 1.0
        else:
            spacing = abs(probe_list[1] - probe_list[0])
        certified = []
        for x in probe_list:
            jet = _jet_source(residual_fn, None, x, policy.jet_order)
            certified.append(certified_local_error(jet, spacing))

    peak_location: float | None = None
    derivatives: list[float] | None = None
    if probe_jet is not None:
        peak_location, derivatives = float(probe_jet[0]), [float(v) for v in probe_jet[1]]
    elif policy.indicator in (Indicator.SINGULARITY, Indicator.SCALE_FLOW):
        values = residuals if certified is None else certified
        peak_i = max(range(len(values)), key=lambda i: abs(values[i]))
        peak_location = probe_list[peak_i]
        derivatives = _jet_source(residual_fn, target_fn, peak_location, policy.jet_order)

    working, report = refine_bank(
        packs,
        policy,
        probes=probe_list,
        residuals=residuals,
        contributions=contrib,
        derivatives=derivatives,
        certified_values=certified,
        peak_location=peak_location,
    )
    bank.sync_from(working)
    if before is not None and (report.born or report.grown) and not report.died:
        after = [float(v) for v in bank(probes_1d).detach().reshape(-1)]
        assert_zero_perturbation(before, after)
    return report


__all__ = [
    "AdaptivePackBank",
    "GrowStrategy",
    "Indicator",
    "RefinePolicy",
    "RefineReport",
    "RefinedPack",
    "grow_ombu",
    "pack_response",
    "refine",
    "scalar_jet_1d",
]

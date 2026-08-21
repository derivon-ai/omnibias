# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Adaptive pack refinement (theory 03-13), JAX twin.

Bit-identical partner of :mod:`omnibias.torch.refine` on the same pack
list and the same residual / jet numbers. Structural decisions run
through :func:`omnibias.core.refine.refine_bank` (host-side). Do not
``jit`` the driver.

There is no jax :class:`GrowableOperatorMultiBiasUnit`. Literal-``K``
growth stays on the torch growable unit; this module's *p*-type move is
the collapsed sibling (new ``sigma^(n+1)`` slot with ``c = 0``).

Enable 64-bit JAX before the first array if you need torch parity
(:mod:`omnibias.jax.precision`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

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
from omnibias.jax.activations import JaxActivationSpec, get_activation

import jax
import jax.numpy as jnp
from jax import Array

ResidualFn = Callable[[Array], Array]
ScalarFn = Callable[[Array], Array]


def _as_1d(z: Array) -> Array:
    z_a = jnp.asarray(z)
    if z_a.ndim == 0:
        return z_a.reshape((1,))
    if z_a.ndim == 1:
        return z_a
    if z_a.shape[-1] == 1:
        return z_a.reshape((-1,))
    raise ValueError(f"AdaptivePackBank expects a 1-D field, got shape {tuple(z_a.shape)}")


def scalar_jet_1d(fn: ScalarFn, x0: float, order: int) -> list[float]:
    """``phi^(k)(x0)`` for ``k = 0 .. order`` via nested ``jax.grad``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    def restriction(s: Array) -> Array:
        out = fn(jnp.reshape(s, ()))
        return jnp.reshape(out, ())

    x = jnp.asarray(float(x0))
    derivs = [float(restriction(x))]
    current: Callable[[Array], Array] = restriction
    for _ in range(order):
        current = jax.grad(current)
        derivs.append(float(current(x)))
    return derivs


def pack_response(
    z: Array,
    center: Array,
    weight: Array,
    scale: Array,
    order: int,
    spec: JaxActivationSpec,
) -> Array:
    """``c * alpha^n sigma^(n)(alpha (z - center))``."""
    if spec.fastpath is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form derivative kernel"
        )
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    u = scale * (z - center)
    return weight * (scale ** order) * spec.fastpath(u, order)


@dataclass
class AdaptivePackBank:
    """Functional tempered pack bank (pytree of slot arrays)."""

    centers: Array
    weights: Array
    scales: Array
    orders: Array
    ages: Array
    active: Array
    max_order: int
    base: str

    @property
    def max_packs(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_active(self) -> int:
        return int(jnp.asarray(self.active).sum())


def init_pack_bank(
    packs: Sequence[RefinedPack],
    *,
    max_packs: int = 16,
    max_order: int = 8,
    base: str | JaxActivationSpec = "exp",
) -> AdaptivePackBank:
    """Allocate slots and write the initial live packs into the prefix."""
    if int(max_packs) < 1:
        raise ValueError(f"max_packs must be >= 1, got {max_packs}")
    if int(max_order) < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    live = tuple(packs)
    if len(live) > int(max_packs):
        raise ValueError(f"{len(live)} initial packs exceed max_packs={max_packs}")
    act = get_activation(base) if isinstance(base, str) else base
    if act.fastpath is None:
        raise NotImplementedError(
            f"Activation {act.name!r} has no closed-form derivative kernel"
        )
    for pack in live:
        if pack.order > int(max_order):
            raise ValueError(f"pack order {pack.order} exceeds max_order={max_order}")
        act.fastpath(jnp.zeros(()), pack.order)
    centers = [0.0] * int(max_packs)
    weights = [0.0] * int(max_packs)
    scales = [1.0] * int(max_packs)
    orders = [0] * int(max_packs)
    ages = [0] * int(max_packs)
    active = [False] * int(max_packs)
    for i, pack in enumerate(live):
        centers[i] = pack.center
        weights[i] = pack.weight
        scales[i] = pack.scale
        orders[i] = pack.order
        ages[i] = pack.age
        active[i] = True
    return AdaptivePackBank(
        centers=jnp.asarray(centers),
        weights=jnp.asarray(weights),
        scales=jnp.asarray(scales),
        orders=jnp.asarray(orders, dtype=jnp.int32),
        ages=jnp.asarray(ages, dtype=jnp.int32),
        active=jnp.asarray(active, dtype=bool),
        max_order=int(max_order),
        base=act.name,
    )


def bank_forward(bank: AdaptivePackBank, z: Array) -> Array:
    """1-D field ``z`` → same leading shape."""
    spec = get_activation(bank.base)
    flat = _as_1d(z)
    out = jnp.zeros_like(flat)
    for i in range(bank.max_packs):
        if not bool(bank.active[i]):
            continue
        n = int(bank.orders[i])
        out = out + pack_response(
            flat,
            bank.centers[i],
            bank.weights[i],
            bank.scales[i],
            n,
            spec,
        )
    if z.ndim != 1 and tuple(z.shape[-1:]) == (1,):
        return out.reshape(z.shape)
    return out


def bank_active_packs(bank: AdaptivePackBank) -> list[RefinedPack]:
    out: list[RefinedPack] = []
    for i in range(bank.max_packs):
        if bool(bank.active[i]):
            out.append(
                RefinedPack(
                    order=int(bank.orders[i]),
                    center=float(bank.centers[i]),
                    weight=float(bank.weights[i]),
                    scale=float(bank.scales[i]),
                    age=int(bank.ages[i]),
                )
            )
    return out


def _key(pack: RefinedPack) -> tuple[int, float, float, float]:
    return (pack.order, pack.center, pack.scale, pack.weight)


def _pack_at(bank: AdaptivePackBank, slot: int) -> RefinedPack:
    return RefinedPack(
        order=int(bank.orders[slot]),
        center=float(bank.centers[slot]),
        weight=float(bank.weights[slot]),
        scale=float(bank.scales[slot]),
        age=int(bank.ages[slot]),
    )


def bank_sync(bank: AdaptivePackBank, packs: Sequence[RefinedPack]) -> AdaptivePackBank:
    """Write a refined list back onto slots, keeping live packs in place."""
    occupied: dict[tuple[int, float, float, float], int] = {}
    for i in range(bank.max_packs):
        if bool(bank.active[i]):
            occupied[_key(_pack_at(bank, i))] = i
    centers = [float(v) for v in bank.centers]
    weights = [float(v) for v in bank.weights]
    scales = [float(v) for v in bank.scales]
    orders = [int(v) for v in bank.orders]
    ages = [int(v) for v in bank.ages]
    active = [bool(v) for v in bank.active]
    assigned: set[int] = set()
    for pack in packs:
        key = _key(pack)
        slot = occupied.get(key)
        if slot is not None and slot not in assigned:
            ages[slot] = pack.age
            assigned.add(slot)
            continue
        free = next((i for i in range(bank.max_packs) if i not in assigned), None)
        if free is None:
            raise RuntimeError(f"max_packs={bank.max_packs} exhausted")
        centers[free] = pack.center
        weights[free] = pack.weight
        scales[free] = pack.scale
        orders[free] = pack.order
        ages[free] = pack.age
        active[free] = True
        assigned.add(free)
    for i in range(bank.max_packs):
        if i not in assigned:
            active[i] = False
            weights[i] = 0.0
            ages[i] = 0
    return replace(
        bank,
        centers=jnp.asarray(centers, dtype=bank.centers.dtype),
        weights=jnp.asarray(weights, dtype=bank.weights.dtype),
        scales=jnp.asarray(scales, dtype=bank.scales.dtype),
        orders=jnp.asarray(orders, dtype=bank.orders.dtype),
        ages=jnp.asarray(ages, dtype=bank.ages.dtype),
        active=jnp.asarray(active, dtype=bool),
    )


def pack_term_norms(bank: AdaptivePackBank, z: Array) -> list[float]:
    spec = get_activation(bank.base)
    flat = _as_1d(z)
    norms: list[float] = []
    for i in range(bank.max_packs):
        if not bool(bank.active[i]):
            continue
        n = int(bank.orders[i])
        term = pack_response(
            flat,
            bank.centers[i],
            bank.weights[i],
            bank.scales[i],
            n,
            spec,
        )
        norms.append(float(jnp.sqrt(jnp.sum(term * term))))
    return norms


def _jet_source(
    residual_fn: ResidualFn,
    target_fn: ResidualFn | None,
    x0: float,
    order: int,
) -> list[float]:
    source = target_fn if target_fn is not None else residual_fn

    def scalar(s: Array) -> Array:
        val = source(jnp.reshape(s, (1,)))
        return jnp.reshape(val, ())

    return scalar_jet_1d(scalar, x0, order)


def refine(
    bank: AdaptivePackBank,
    residual_fn: ResidualFn,
    probes: Array,
    policy: RefinePolicy,
    *,
    step: int,
    target_fn: ResidualFn | None = None,
    probe_jet: tuple[float, Sequence[float]] | None = None,
) -> tuple[AdaptivePackBank, RefineReport]:
    """One refine tick. Returns the updated bank (functional)."""
    del step
    probes_1d = _as_1d(probes)
    residual = _as_1d(residual_fn(probes_1d))
    if residual.shape != probes_1d.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} != probes {tuple(probes_1d.shape)}"
        )
    before = None
    if policy.debug_zero_perturbation:
        before = [float(v) for v in bank_forward(bank, probes_1d).reshape(-1)]

    packs = bank_active_packs(bank)
    field = bank_forward(bank, probes_1d)
    field_norm = float(jnp.sqrt(jnp.sum(field * field)))
    contrib = pack_contributions(pack_term_norms(bank, probes_1d), field_norm)
    residuals = [float(v) for v in residual]
    probe_list = [float(v) for v in probes_1d]

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
    updated = bank_sync(bank, working)
    if before is not None and (report.born or report.grown) and not report.died:
        after = [float(v) for v in bank_forward(updated, probes_1d).reshape(-1)]
        assert_zero_perturbation(before, after)
    return updated, report


__all__ = [
    "AdaptivePackBank",
    "Indicator",
    "RefinePolicy",
    "RefineReport",
    "RefinedPack",
    "bank_active_packs",
    "bank_forward",
    "bank_sync",
    "init_pack_bank",
    "pack_response",
    "pack_term_norms",
    "refine",
    "scalar_jet_1d",
]

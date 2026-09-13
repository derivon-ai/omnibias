# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Functional, preallocated collision bank with versioned host transitions.

Forward, retraction and parameter replacement are JIT-compatible. Proposals,
checkpoint ingestion and acceptance run between optimizer steps on the host.
The explicit parameter dictionary excludes integer/bool representation state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Protocol, TypeVar, cast

from omnibias.core.confluence import initialize_moments
from omnibias.core.realization.transition import SlotUpdate, TransitionProposal, snapshot_digest
from omnibias.core.refine import RefinedPack
from omnibias.jax.activations import get_activation
from omnibias.jax.confluence import collision_atom, moment_atom, retract_spread
from omnibias.jax.jet import _sigma_tower
from omnibias.jax.refine import AdaptivePackBank

import jax
import jax.numpy as jnp
from jax import Array

PARAMETER_NAMES = ("centers", "weights", "scales", "rho", "odd_moments", "directions",
                   "direction_bias", "moment_coefficients")
STATE_NAMES = PARAMETER_NAMES + ("orders", "ages", "active", "modes", "slot_ids", "transition_version")
StateT = TypeVar("StateT")


class OptimizerStateAdapter(Protocol[StateT]):
    def __call__(self, state: StateT, bank: ConfluentPackBank,
                 slots: tuple[int, ...]) -> StateT: ...


class ExactTransportAdapter(Protocol[StateT]):
    def __call__(self, state: StateT, bank: ConfluentPackBank,
                 permutation: tuple[int, ...] | None, sign_slots: tuple[int, ...]) -> StateT: ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NamedMomentState:
    """Explicit elementwise optimizer moments and a preserved shared step count.

    Each moment dictionary has exactly the bank's parameter names/shapes. This
    represents Adam's first/second moments or an SGD momentum without guessing
    the structure of an arbitrary optimizer's state. Adapt other states through
    ``OptimizerStateAdapter``.
    """

    step: Array
    moments: dict[str, dict[str, Array]]

    def tree_flatten(self) -> tuple[tuple[Array, dict[str, dict[str, Array]]], None]:
        return (self.step, self.moments), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Array, dict[str, dict[str, Array]]]) -> NamedMomentState:
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConfluentPackBank:
    centers: Array
    weights: Array
    scales: Array
    rho: Array
    odd_moments: Array
    directions: Array
    direction_bias: Array
    moment_coefficients: Array
    orders: Array
    ages: Array
    active: Array
    modes: Array
    slot_ids: Array
    transition_version: Array
    max_order: int
    base: str

    def tree_flatten(self) -> tuple[tuple[Array, ...], tuple[int, str]]:
        return tuple(getattr(self, name) for name in STATE_NAMES), (self.max_order, self.base)

    @classmethod
    def tree_unflatten(cls, aux: tuple[int, str], children: tuple[Array, ...]) -> ConfluentPackBank:
        values: dict[str, Any] = dict(zip(STATE_NAMES, children, strict=True))
        return cls(**values, max_order=aux[0], base=aux[1])

    @classmethod
    def create(cls, packs: Sequence[RefinedPack], *, max_packs: int = 16, max_order: int = 8,
               base: str = "sigmoid", dtype: Any = None) -> ConfluentPackBank:
        return init_confluent_bank(packs, max_packs=max_packs, max_order=max_order, base=base, dtype=dtype)

    @property
    def max_packs(self) -> int:
        return int(self.centers.shape[0])

    @property
    def parameters(self) -> dict[str, Array]:
        return {name: getattr(self, name) for name in PARAMETER_NAMES}

    def with_parameters(self, parameters: Mapping[str, Array]) -> ConfluentPackBank:
        if set(parameters) != set(PARAMETER_NAMES):
            raise ValueError("parameters must contain exactly the trainable bank blocks")
        for name, value in parameters.items():
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"parameter shape changed for {name}")
        return replace(self, **dict(cast(Mapping[str, Any], parameters)))

    def forward(self, z: Array) -> Array:
        return bank_forward(self, z)

    def slot_terms(self, z: Array) -> Array:
        return bank_slot_terms(self, z)

    def pack_term_norms(self, z: Array) -> list[float]:
        terms = self.slot_terms(z)
        return [float(jnp.sqrt(jnp.sum(terms[i] ** 2))) for i in range(self.max_packs) if bool(self.active[i])]

    def retract(self) -> ConfluentPackBank:
        return replace(self, rho=retract_spread(self.rho))

    def portable_snapshot(self) -> dict[str, Any]:
        snapshot = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "values": value.reshape(-1).tolist()}
                for name in STATE_NAMES for value in (getattr(self, name),)}
        snapshot["__configuration__"] = {"activation": self.base, "max_order": self.max_order,
                                           "max_packs": self.max_packs, "schema_version": 1,
                                           "series_terms": 6, "series_radius": 0.01}
        return snapshot

    def checkpoint(self) -> dict[str, Any]:
        return {"schema_version": 1, "base": self.base, "max_order": self.max_order,
                "state": self.portable_snapshot()}

    def propose_cluster(self, slots: tuple[int, ...], *, order: int,
                        error_budget: float) -> TransitionProposal:
        """Common-weight ordinary cluster -> bounded truncated moment expansion."""
        if (not slots or len(set(slots)) != len(slots) or order < 0 or order > self.max_order
                or any(i < 0 or i >= self.max_packs for i in slots)):
            raise ValueError('valid distinct slots and an order within the bank budget required')
        first = slots[0]
        scale = Fraction(float(self.scales[first]))
        if scale == 0:
            raise ValueError('nonzero common weight required')
        for i in slots:
            if (not bool(self.active[i]) or int(self.modes[i]) != 0 or int(self.orders[i]) != 0
                    or Fraction(float(self.scales[i])) != scale):
                raise ValueError('cluster requires ordinary order-zero atoms with a common weight')
        centers = tuple(Fraction(float(self.centers[i])) for i in slots)
        center = sum(centers, Fraction())/len(slots)
        moments = initialize_moments(tuple(-scale*c for c in centers),
                                     tuple(float(self.weights[i]) for i in slots),
                                     center=-scale*center, order=order)
        row = moments.moments+(0.0,)*(self.max_order-order)
        updates = [SlotUpdate('centers', first, float(center)), SlotUpdate('modes', first, 2),
                   SlotUpdate('orders', first, order), SlotUpdate('moment_coefficients', first, row)]
        for i in slots[1:]:
            updates.extend((SlotUpdate('active', i, 0), SlotUpdate('weights', i, 0)))
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  tuple(int(self.slot_ids[i]) for i in slots), tuple(updates),
                                  'cluster_to_moments', 'finite-spread truncation with explicit remainder',
                                  error_budget)

    def propose_pair(self, first: int, second: int, *, error_budget: float = 0.0) -> TransitionProposal:
        if first == second or any(i < 0 or i >= self.max_packs for i in (first, second)):
            raise ValueError("two distinct live slots required")
        for i in (first, second):
            if not bool(self.active[i]) or int(self.modes[i]) != 0 or int(self.orders[i]) != 0:
                raise ValueError("pair proposal requires ordinary order-zero live slots")
        scale = Fraction(float(self.scales[first]))
        if scale != Fraction(float(self.scales[second])) or scale == 0:
            raise ValueError("bias confluence requires identical nonzero weights")
        b = [-scale * Fraction(float(self.centers[i])) for i in (first, second)]
        a = [Fraction(float(self.weights[i])) for i in (first, second)]
        half = abs(b[0] - b[1]) / 2
        mean = (b[0] + b[1]) / 2
        odd = (b[0] - b[1]) * (a[0] - a[1]) / 2
        values: list[tuple[str, int, Fraction | int]] = [("centers", first, -mean / scale), ("weights", first, a[0] + a[1]),
                  ("rho", first, half * half), ("odd_moments", first, odd),
                  ("modes", first, 1), ("directions", first, 0), ("direction_bias", first, 1),
                  ("active", second, 0), ("weights", second, 0), ("odd_moments", second, 0)]
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[first]), int(self.slot_ids[second])),
                                  tuple(SlotUpdate(name, i, float(v)) for name, i, v in values),
                                  "duplicate_merge" if half == 0 else "centered_pair",
                                  "exact_real_coordinate_identity_with_recorded_float_operands", error_budget)

    def propose_derivative(self, slot: int, *, error_budget: float) -> TransitionProposal:
        self._check_slot(slot)
        if not bool(self.active[slot]) or int(self.modes[slot]) != 1:
            raise ValueError("a live centered-pair slot is required")
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[slot]),), (SlotUpdate("rho", slot, 0.0),),
                                  "pair_to_derivative", "limit_atom; finite-spread error requires acceptance",
                                  error_budget)

    def propose_remove_zero(self, slot: int) -> TransitionProposal:
        self._check_slot(slot)
        if not bool(self.active[slot]) or int(self.modes[slot]) != 0 or float(self.weights[slot]) != 0:
            raise ValueError("an exactly zero ordinary output coefficient is required")
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[slot]),), (SlotUpdate("active", slot, 0),),
                                  "remove_zero", "exact_zero_output")

    def _check_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.max_packs:
            raise ValueError("slot is outside the bank")

    def propose_birth(self, slot: int, pack: RefinedPack) -> TransitionProposal:
        """Birth/reuse an inactive slot with exactly zero output and fresh moments."""
        self._check_slot(slot)
        if bool(self.active[slot]):
            raise ValueError("birth requires an allocated inactive slot")
        if pack.weight != 0 or pack.order > self.max_order:
            raise ValueError("birth requires zero output weight and an order within budget")
        updates = (SlotUpdate("centers", slot, pack.center), SlotUpdate("weights", slot, 0),
                   SlotUpdate("scales", slot, pack.scale), SlotUpdate("orders", slot, pack.order),
                   SlotUpdate("ages", slot, pack.age), SlotUpdate("active", slot, 1),
                   SlotUpdate("modes", slot, 0), SlotUpdate("rho", slot, 0),
                   SlotUpdate("odd_moments", slot, 0), SlotUpdate("directions", slot, 0),
                   SlotUpdate("direction_bias", slot, 1),
                   SlotUpdate("moment_coefficients", slot, (0.0,) * (self.max_order + 1)))
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[slot]),), updates, "zero_output_birth",
                                  "exact_zero_output_with_full_slot_state_initialization")


def init_confluent_bank(packs: Sequence[RefinedPack], *, max_packs: int = 16, max_order: int = 8,
                        base: str = "sigmoid", dtype: Any = None) -> ConfluentPackBank:
    if base not in ("sigmoid", "tanh") or max_packs < 1 or max_order < 0:
        raise ValueError("positive capacity, nonnegative order, and sigmoid/tanh are required")
    live = tuple(packs)
    if len(live) > max_packs or any(pack.order > max_order for pack in live):
        raise ValueError("initial packs exceed capacity or supported order")
    centers, weights, scales = [0.0] * max_packs, [0.0] * max_packs, [1.0] * max_packs
    orders, ages, active = [0] * max_packs, [0] * max_packs, [False] * max_packs
    for i, pack in enumerate(live):
        centers[i], weights[i], scales[i] = pack.center, pack.weight, pack.scale
        orders[i], ages[i], active[i] = pack.order, pack.age, True
    center_array = jnp.asarray(centers, dtype=dtype)
    dt = center_array.dtype
    # Match the workspace's 64-bit integer snapshot schema when x64 is enabled.
    it = jnp.asarray(0).dtype
    return ConfluentPackBank(center_array, jnp.asarray(weights, dtype=dt), jnp.asarray(scales, dtype=dt),
                             jnp.zeros(max_packs, dtype=dt), jnp.zeros(max_packs, dtype=dt),
                             jnp.zeros(max_packs, dtype=dt), jnp.ones(max_packs, dtype=dt),
                             jnp.zeros((max_packs, max_order + 1), dtype=dt),
                             jnp.asarray(orders, dtype=it), jnp.asarray(ages, dtype=it),
                             jnp.asarray(active), jnp.zeros(max_packs, dtype=it),
                             jnp.arange(max_packs, dtype=it), jnp.asarray(0, dtype=it), max_order, base)


def migrate_legacy(bank: AdaptivePackBank) -> ConfluentPackBank:
    """Add ordinary-mode state to an existing functional bank without moving slots."""
    fresh = init_confluent_bank((), max_packs=bank.max_packs, max_order=bank.max_order,
                               base=bank.base, dtype=bank.centers.dtype)
    return replace(fresh, centers=bank.centers, weights=bank.weights, scales=bank.scales,
                   orders=bank.orders, ages=bank.ages, active=bank.active)


def bank_forward(bank: ConfluentPackBank, z: Array) -> Array:
    terms = bank_slot_terms(bank, z)
    result = jnp.zeros_like(z)
    for i in range(bank.max_packs):
        result = result + terms[i]
    return result


def bank_slot_terms(bank: ConfluentPackBank, z: Array) -> Array:
    """One shared registered tower per slot, reused by all three mode branches."""
    spec = get_activation(bank.base)
    terms = []
    for i in range(bank.max_packs):
        argument = bank.scales[i] * (z - bank.centers[i])
        tower = _sigma_tower(spec, argument, max(bank.max_order, 11))
        ordinary = jnp.zeros_like(z)
        for order in range(bank.max_order + 1):
            value = bank.weights[i] * (bank.scales[i] ** order) * tower[order]
            ordinary = ordinary + jnp.where(bank.orders[i] == order, value, jnp.zeros_like(z))
        pair = collision_atom(argument, bank.rho[i], bank.directions[i] * z + bank.direction_bias[i],
                              bank.weights[i], bank.odd_moments[i], activation=bank.base, tower=tower)
        moments = moment_atom(argument, bank.moment_coefficients[i], activation=bank.base, tower=tower)
        value = jnp.where(bank.modes[i] == 0, ordinary, jnp.where(bank.modes[i] == 1, pair, moments))
        terms.append(jnp.where(bank.active[i], value, jnp.zeros_like(z)))
    return jnp.stack(terms)


def bank_from_checkpoint(checkpoint: Mapping[str, Any]) -> ConfluentPackBank:
    """Validate a portable checkpoint. This host API never silently truncates dtype."""
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    state = checkpoint["state"]
    if set(state) != set(STATE_NAMES) | {"__configuration__"}:
        raise ValueError("checkpoint contains missing or unknown state blocks")
    capacity = state["centers"]["shape"][0]
    fresh = init_confluent_bank((), max_packs=capacity, max_order=checkpoint["max_order"],
                               base=checkpoint["base"], dtype=state["centers"]["dtype"])
    if state["__configuration__"] != fresh.portable_snapshot()["__configuration__"]:
        raise ValueError("checkpoint configuration disagrees with its bound state")
    values: dict[str, Any] = {}
    for name in STATE_NAMES:
        item = state[name]
        value = jnp.asarray(item["values"], dtype=item["dtype"]).reshape(tuple(item["shape"]))
        if value.shape != getattr(fresh, name).shape or str(value.dtype) != item["dtype"]:
            raise ValueError(f"checkpoint shape/dtype not supported for {name}")
        if name in PARAMETER_NAMES and (not jnp.issubdtype(value.dtype, jnp.floating)
                                         or value.dtype != fresh.centers.dtype):
            raise ValueError("checkpoint trainables must share a real floating dtype")
        if name == "active" and value.dtype != jnp.bool_:
            raise ValueError("checkpoint activity flags must be boolean")
        if name not in PARAMETER_NAMES + ("active",) and not jnp.issubdtype(value.dtype, jnp.integer):
            raise ValueError("checkpoint representation metadata must be integer")
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError("checkpoint state must be finite")
        values[name] = value
    candidate = replace(fresh, **values)
    if (not bool(jnp.all((candidate.modes >= 0) & (candidate.modes <= 2)))
            or not bool(jnp.all((candidate.orders >= 0) & (candidate.orders <= candidate.max_order)))
            or not bool(jnp.all(candidate.rho >= 0))
            or not bool(jnp.all(candidate.slot_ids >= 0))
            or not bool(jnp.all(candidate.ages >= 0))
            or int(candidate.transition_version) < 0
            or len(set(candidate.slot_ids.tolist())) != capacity):
        raise ValueError("checkpoint representation state is invalid")
    return candidate


def reset_slot_state(state: NamedMomentState, bank: ConfluentPackBank,
                     slots: tuple[int, ...]) -> NamedMomentState:
    """Reset affected moment rows, retaining all other rows and the shared clock."""
    if not isinstance(state, NamedMomentState):
        raise TypeError("unknown optimizer state requires an explicit OptimizerStateAdapter")
    moments: dict[str, dict[str, Array]] = {}
    for kind, entries in state.moments.items():
        if set(entries) != set(PARAMETER_NAMES):
            raise ValueError("each named moment must match the complete bank parameter dictionary")
        moments[kind] = {}
        for name, value in entries.items():
            if value.shape != getattr(bank, name).shape:
                raise ValueError("moment shape does not match parameter")
            moments[kind][name] = value.at[jnp.asarray(slots)].set(0)
    return NamedMomentState(state.step, moments)


def commit_transition(
    bank: ConfluentPackBank, proposal: TransitionProposal, optimizer_state: StateT,
    *, between_steps: bool, acceptance: Callable[[ConfluentPackBank, TransitionProposal], bool],
    postcondition: Callable[[ConfluentPackBank], bool] | None = None,
    state_adapter: OptimizerStateAdapter[StateT] | None = None,
) -> tuple[ConfluentPackBank, StateT, bool]:
    """Functional host transaction. Rejection returns both original objects.

    The acceptance callback checks the actual requested error budget before
    any state is constructed. A failed postcondition discards both candidate
    model and adapted optimizer state. No callback here is itself a proof.
    """
    if not between_steps:
        raise RuntimeError("commit only between optimizer steps")
    if (proposal.source_version != int(bank.transition_version)
            or proposal.source_digest != snapshot_digest(bank.portable_snapshot())):
        raise ValueError("stale transition proposal")
    ids = bank.slot_ids.tolist()
    if any(i not in ids for i in proposal.slot_ids):
        raise ValueError("proposal contains unknown stable slot IDs")
    slots = tuple(ids.index(i) for i in proposal.slot_ids)
    if not acceptance(bank, proposal):
        return bank, optimizer_state, False
    changed: dict[str, Any] = {}
    for update in proposal.updates:
        if update.name not in STATE_NAMES or update.name in ("slot_ids", "transition_version"):
            raise ValueError("proposal may only update representation tensors")
        tensor = changed.get(update.name, getattr(bank, update.name))
        if tensor.ndim == 0 or update.index not in slots:
            raise ValueError("update must refer to an affected slot")
        value = jnp.asarray(update.value, dtype=tensor.dtype)
        if value.shape != tensor[update.index].shape:
            raise ValueError("slot update shape does not match the preallocated row")
        changed[update.name] = tensor.at[update.index].set(value)
    candidate = replace(bank, **changed, transition_version=bank.transition_version + 1)
    if state_adapter is None:
        if not isinstance(optimizer_state, NamedMomentState):
            raise TypeError("unknown optimizer state requires an explicit OptimizerStateAdapter")
        candidate_state = cast(StateT, reset_slot_state(optimizer_state, candidate, slots))
    else:
        candidate_state = state_adapter(optimizer_state, candidate, slots)
    if postcondition is not None and not postcondition(candidate):
        return bank, optimizer_state, False
    return candidate, candidate_state, True


def detect_transitions(bank: ConfluentPackBank, conditioning_score: Callable[[ConfluentPackBank], float],
                       *, error_budget: float = 0.0) -> tuple[TransitionProposal, ...]:
    baseline = conditioning_score(bank)
    ranked: list[tuple[float, tuple[int, ...], TransitionProposal]] = []
    for first in range(bank.max_packs):
        for second in range(first + 1, bank.max_packs):
            try:
                proposal = bank.propose_pair(first, second, error_budget=error_budget)
            except ValueError:
                continue
            state = NamedMomentState(jnp.asarray(0), {})
            candidate, _, accepted = commit_transition(bank, proposal, state, between_steps=True,
                                                       acceptance=lambda _bank, _proposal: True)
            if accepted:
                improvement = conditioning_score(candidate) - baseline
                if improvement > 0:
                    ranked.append((-improvement, proposal.slot_ids, proposal))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked)


def transport_optimizer_state(
    state: NamedMomentState, bank: ConfluentPackBank,
    permutation: tuple[int, ...] | None, sign_slots: tuple[int, ...],
) -> NamedMomentState:
    """Permute all moment rows; sign-transform named first moments only."""
    if not isinstance(state, NamedMomentState):
        raise TypeError("unknown optimizer state requires an explicit ExactTransportAdapter")
    odd = {"first", "momentum", "gradient_average"}
    even = {"second", "variance", "max_second", "sum_squares", "absolute"}
    moments: dict[str, dict[str, Array]] = {}
    for kind, entries in state.moments.items():
        if set(entries) != set(PARAMETER_NAMES):
            raise ValueError("each named moment must match the complete bank parameter dictionary")
        if sign_slots and kind not in odd | even:
            raise TypeError("unknown moment parity requires an explicit ExactTransportAdapter")
        moments[kind] = {}
        for name, value in entries.items():
            if value.shape != getattr(bank, name).shape:
                raise ValueError("moment shape does not match parameter")
            result = value if permutation is None else value[jnp.asarray(permutation)]
            if sign_slots and kind in odd and name in ("weights", "scales"):
                result = result.at[jnp.asarray(sign_slots)].multiply(-1)
            moments[kind][name] = result
    return NamedMomentState(state.step, moments)


def _exact_transport(
    bank: ConfluentPackBank, optimizer_state: StateT, *, between_steps: bool,
    permutation: tuple[int, ...] | None = None, sign_slots: tuple[int, ...] = (),
    state_adapter: ExactTransportAdapter[StateT] | None = None,
) -> tuple[ConfluentPackBank, StateT]:
    if not between_steps:
        raise RuntimeError("transport only between optimizer steps")
    if permutation is not None and (len(permutation) != bank.max_packs
                                    or sorted(permutation) != list(range(bank.max_packs))):
        raise ValueError("permutation must contain every allocated slot exactly once")
    if sign_slots:
        if len(set(sign_slots)) != len(sign_slots) or any(i < 0 or i >= bank.max_packs for i in sign_slots):
            raise ValueError("sign slots must be distinct allocated slots")
        if bank.base != "tanh" or any(int(bank.modes[i]) != 0 for i in sign_slots):
            raise ValueError("sign relabeling supports ordinary tanh slots only")
    changed: dict[str, Any] = {}
    if permutation is not None:
        for name in STATE_NAMES:
            if name != "transition_version":
                changed[name] = getattr(bank, name)[jnp.asarray(permutation)]
    if sign_slots:
        for name in ("weights", "scales"):
            changed[name] = getattr(bank, name).at[jnp.asarray(sign_slots)].multiply(-1)
    candidate = replace(bank, **changed, transition_version=bank.transition_version + 1)
    if state_adapter is None:
        if not isinstance(optimizer_state, NamedMomentState):
            raise TypeError("unknown optimizer state requires an explicit ExactTransportAdapter")
        next_state = cast(StateT, transport_optimizer_state(optimizer_state, candidate, permutation, sign_slots))
    else:
        next_state = state_adapter(optimizer_state, candidate, permutation, sign_slots)
    return candidate, next_state


def permute_bank(
    bank: ConfluentPackBank, permutation: Sequence[int], optimizer_state: StateT,
    *, between_steps: bool, state_adapter: ExactTransportAdapter[StateT] | None = None,
) -> tuple[ConfluentPackBank, StateT]:
    """Move stable IDs, every state row and its moments through one exact permutation."""
    return _exact_transport(bank, optimizer_state, between_steps=between_steps,
                            permutation=tuple(permutation), state_adapter=state_adapter)


def relabel_tanh_sign(
    bank: ConfluentPackBank, slots: Sequence[int], optimizer_state: StateT,
    *, between_steps: bool, state_adapter: ExactTransportAdapter[StateT] | None = None,
) -> tuple[ConfluentPackBank, StateT]:
    """Ordinary tanh symmetry, including first-/second-moment transport."""
    return _exact_transport(bank, optimizer_state, between_steps=between_steps,
                            sign_slots=tuple(slots), state_adapter=state_adapter)


__all__ = ["ConfluentPackBank", "ExactTransportAdapter", "NamedMomentState", "OptimizerStateAdapter",
           "PARAMETER_NAMES", "STATE_NAMES", "bank_forward", "bank_from_checkpoint", "bank_slot_terms", "commit_transition",
           "detect_transitions", "init_confluent_bank", "migrate_legacy", "permute_bank",
           "relabel_tanh_sign", "reset_slot_state", "transport_optimizer_state"]

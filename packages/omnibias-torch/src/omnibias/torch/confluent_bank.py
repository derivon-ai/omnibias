# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Opt-in preallocated confluent refinement bank and atomic structural commits.

Legacy AdaptivePackBank behavior is preserved. This subclass adds stable IDs,
representation modes, and optimizer-aware transactions without replacing any
Parameter object. Call commit only between optimizer steps, after zero_grad.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from fractions import Fraction
from typing import Any, Protocol

from omnibias.core.confluence import initialize_moments
from omnibias.core.realization.transition import SlotUpdate, TransitionProposal, snapshot_digest
from omnibias.core.refine import RefinedPack
from omnibias.torch.confluence import collision_atom, moment_atom
from omnibias.torch.jet import _sigma_tower
from omnibias.torch.refine import AdaptivePackBank

import torch
from torch import Tensor, nn


class OptimizerStateAdapter(Protocol):
    def __call__(self, optimizer: torch.optim.Optimizer, bank: ConfluentPackBank,
                 slots: tuple[int, ...]) -> None: ...


class ExactTransportAdapter(Protocol):
    def __call__(self, optimizer: torch.optim.Optimizer, bank: ConfluentPackBank,
                 permutation: tuple[int, ...] | None, sign_slots: tuple[int, ...]) -> None: ...


class ConfluentPackBank(AdaptivePackBank):
    """Ordinary (0), centered-pair (1), and moment/derivative (2) slots.

    A pair uses weights=m0, odd_moments=m1, rho=h**2 and eta=v*x+c.
    A moment slot uses Taylor-normalized moment_coefficients. Legacy state
    dictionaries migrate additively to ordinary mode. Slot IDs never recycle;
    activating a formerly inactive slot resets its optimizer moments.
    """
    rho: Tensor
    odd_moments: Tensor
    directions: Tensor
    direction_bias: Tensor
    moment_coefficients: Tensor
    modes: Tensor
    slot_ids: Tensor
    transition_version: Tensor

    def __init__(self, packs: Sequence[RefinedPack], *, max_packs: int = 16,
                 max_order: int = 8, base: str = 'sigmoid',
                 dtype: torch.dtype | None = None) -> None:
        if base not in ('sigmoid', 'tanh'):
            raise ValueError('the confluent bank supports sigmoid and tanh')
        super().__init__(packs, max_packs=max_packs, max_order=max_order, base=base, dtype=dtype)
        dt = self.centers.dtype
        self.rho = nn.Parameter(torch.zeros(max_packs, dtype=dt))
        self.odd_moments = nn.Parameter(torch.zeros(max_packs, dtype=dt))
        self.directions = nn.Parameter(torch.zeros(max_packs, dtype=dt))
        self.direction_bias = nn.Parameter(torch.ones(max_packs, dtype=dt))
        self.moment_coefficients = nn.Parameter(torch.zeros((max_packs, max_order+1), dtype=dt))
        self.register_buffer('modes', torch.zeros(max_packs, dtype=torch.long))
        self.register_buffer('slot_ids', torch.arange(max_packs, dtype=torch.long))
        self.register_buffer('transition_version', torch.tensor(0, dtype=torch.long))

    def forward(self, z: Tensor) -> Tensor:
        terms = self.slot_terms(z)
        result = torch.zeros_like(z)
        for i in range(self.max_packs):
            result = result + terms[i]
        return result

    def slot_terms(self, z: Tensor) -> Tensor:
        """Actual ordinary/pair/moment contributions, including zero inactive rows.

        One registered tower per allocated slot serves all three branches.
        The pair's stable finite formula separately evaluates three sigmoids.
        """
        terms = []
        for i in range(self.max_packs):
            argument = self.scales[i] * (z-self.centers[i])
            tower = _sigma_tower(self.act_spec, argument, max(self.max_order, 11))
            ordinary = torch.zeros_like(z)
            for order in range(self.max_order+1):
                value = self.weights[i] * (self.scales[i] ** order) * tower[order]
                ordinary = ordinary + torch.where(self.orders[i] == order, value, torch.zeros_like(z))
            pair = collision_atom(argument, self.rho[i], self.directions[i]*z+self.direction_bias[i],
                                  self.weights[i], self.odd_moments[i], activation=self.act_spec.name,
                                  tower=tower)
            moments = moment_atom(argument, self.moment_coefficients[i], activation=self.act_spec.name,
                                  tower=tower)
            value = torch.where(self.modes[i] == 0, ordinary,
                                torch.where(self.modes[i] == 1, pair, moments))
            terms.append(torch.where(self.active[i], value, torch.zeros_like(z)))
        return torch.stack(terms)

    def pack_term_norms(self, z: Tensor) -> list[float]:
        """Host L2 diagnostics of actual live contributions in slot order."""
        terms = self.slot_terms(z).detach()
        return [float(terms[i].square().sum().sqrt()) for i in range(self.max_packs) if bool(self.active[i])]

    def active_packs(self) -> list[RefinedPack]:
        if bool(((self.modes != 0) & self.active).any()):
            raise RuntimeError('confluent modes cannot be represented as RefinedPack; use slot_terms or portable_snapshot')
        return super().active_packs()

    def sync_from(self, packs: Sequence[RefinedPack]) -> None:
        raise RuntimeError('confluent banks require transactional proposals and commit_transition')

    def propose_birth(self, slot: int, pack: RefinedPack) -> TransitionProposal:
        """Birth/reuse an inactive slot with exactly zero output and fresh moments."""
        if slot < 0 or slot >= self.max_packs or bool(self.active[slot]):
            raise ValueError('birth requires an allocated inactive slot')
        if pack.weight != 0 or pack.order > self.max_order:
            raise ValueError('birth requires zero output weight and an order within budget')
        updates = (SlotUpdate('centers', slot, pack.center), SlotUpdate('weights', slot, 0),
                   SlotUpdate('scales', slot, pack.scale), SlotUpdate('orders', slot, pack.order),
                   SlotUpdate('ages', slot, pack.age), SlotUpdate('active', slot, 1),
                   SlotUpdate('modes', slot, 0), SlotUpdate('rho', slot, 0),
                   SlotUpdate('odd_moments', slot, 0), SlotUpdate('directions', slot, 0),
                   SlotUpdate('direction_bias', slot, 1),
                   SlotUpdate('moment_coefficients', slot, (0.0,)*(self.max_order+1)))
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[slot]),), updates, 'zero_output_birth',
                                  'exact_zero_output_with_full_slot_state_initialization')

    def retract(self) -> None:
        """Boundary projection after an optimizer step; attained rho=0 is retained."""
        with torch.no_grad():
            self.rho.clamp_(min=0)

    def portable_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {name: {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.'),
                       'values': value.detach().cpu().reshape(-1).tolist()}
                for name, value in self.state_dict().items()}
        snapshot['__configuration__'] = {'activation': self.act_spec.name, 'max_order': self.max_order,
                                         'max_packs': self.max_packs, 'schema_version': 1,
                                         'series_terms': 6, 'series_radius': 0.01}
        return snapshot

    def propose_cluster(self, slots: tuple[int, ...], *, order: int,
                        error_budget: float) -> TransitionProposal:
        """Common-weight ordinary cluster -> bounded truncated moment expansion."""
        if (not slots or len(set(slots)) != len(slots) or order < 0 or order > self.max_order
                or any(i < 0 or i >= self.max_packs for i in slots)):
            raise ValueError('valid distinct slots and an order within the bank budget required')
        first = slots[0]
        scale = Fraction(float(self.scales[first].detach()))
        if scale == 0:
            raise ValueError('nonzero common weight required')
        for i in slots:
            if (not bool(self.active[i]) or int(self.modes[i]) != 0 or int(self.orders[i]) != 0
                    or Fraction(float(self.scales[i].detach())) != scale):
                raise ValueError('cluster requires ordinary order-zero atoms with a common weight')
        centers = tuple(Fraction(float(self.centers[i].detach())) for i in slots)
        center = sum(centers, Fraction())/len(slots)
        moments = initialize_moments(tuple(-scale*c for c in centers),
                                     tuple(float(self.weights[i].detach()) for i in slots),
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
        """Ordinary common-weight bias pair -> stable moments (host planning)."""
        if first == second or any(i < 0 or i >= self.max_packs for i in (first, second)):
            raise ValueError('two distinct live slots required')
        for i in (first, second):
            if not bool(self.active[i]) or int(self.modes[i]) != 0 or int(self.orders[i]) != 0:
                raise ValueError('pair proposal requires ordinary order-zero live slots')
        scale = Fraction(float(self.scales[first].detach()))
        if scale != Fraction(float(self.scales[second].detach())) or scale == 0:
            raise ValueError('bias confluence requires identical nonzero weights')
        b = [-scale * Fraction(float(self.centers[i].detach())) for i in (first, second)]
        a = [Fraction(float(self.weights[i].detach())) for i in (first, second)]
        h = abs(b[0]-b[1])/2
        mean = (b[0]+b[1])/2
        odd = (b[0]-b[1])*(a[0]-a[1])/2
        values: list[tuple[str, int, Fraction | int]] = [('centers', first, -mean/scale), ('weights', first, a[0]+a[1]),
                  ('rho', first, h*h), ('odd_moments', first, odd),
                  ('modes', first, 1), ('directions', first, 0), ('direction_bias', first, 1),
                  ('active', second, 0), ('weights', second, 0), ('odd_moments', second, 0)]
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[first]), int(self.slot_ids[second])),
                                  tuple(SlotUpdate(n, i, float(v)) for n, i, v in values),
                                  'duplicate_merge' if h == 0 else 'centered_pair',
                                  'exact_real_coordinate_identity_with_recorded_float_operands', error_budget)

    def propose_derivative(self, slot: int, *, error_budget: float) -> TransitionProposal:
        if not bool(self.active[slot]) or int(self.modes[slot]) != 1:
            raise ValueError('a live centered-pair slot is required')
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[slot]),), (SlotUpdate('rho', slot, 0.0),),
                                  'pair_to_derivative', 'limit_atom; finite-spread error requires acceptance',
                                  error_budget)

    def propose_remove_zero(self, slot: int) -> TransitionProposal:
        if (not bool(self.active[slot]) or int(self.modes[slot]) != 0
                or float(self.weights[slot].detach()) != 0.0):
            raise ValueError('an exactly zero ordinary output coefficient is required')
        return TransitionProposal(snapshot_digest(self.portable_snapshot()), int(self.transition_version),
                                  (int(self.slot_ids[slot]),), (SlotUpdate('active', slot, 0),),
                                  'remove_zero', 'exact_zero_output')

    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str,
                             local_metadata: dict[str, Any], strict: bool,
                             missing_keys: list[str], unexpected_keys: list[str],
                             error_msgs: list[str]) -> None:
        # Additive legacy migration. All new slots start in ordinary mode.
        for name in ('rho', 'odd_moments', 'directions', 'direction_bias', 'moment_coefficients',
                     'modes', 'slot_ids', 'transition_version'):
            key = prefix+name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def reset_slot_state(optimizer: torch.optim.Optimizer, bank: ConfluentPackBank,
                     slots: tuple[int, ...]) -> None:
    """Reset affected elementwise moments, preserving other slots and scalar steps.

    LBFGS history couples all coordinates and is cleared. Unknown optimizers
    must provide an explicit adapter. Shared Adam step counters are preserved;
    this is a moment reset, not a fresh independent per-slot optimizer clock.
    """
    allowed = (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.RMSprop,
               torch.optim.Adagrad, torch.optim.Adadelta, torch.optim.Adamax)
    if isinstance(optimizer, torch.optim.LBFGS):
        optimizer.state.clear()
        return
    if type(optimizer) not in allowed:
        raise TypeError('unknown optimizer requires an explicit optimizer state adapter')
    for parameter in bank.parameters():
        for value in optimizer.state.get(parameter, {}).values():
            if isinstance(value, Tensor) and value.shape == parameter.shape:
                value[list(slots)] = 0


def commit_transition(bank: ConfluentPackBank, proposal: TransitionProposal,
                      optimizer: torch.optim.Optimizer, *, between_steps: bool,
                      acceptance: Callable[[ConfluentPackBank, TransitionProposal], bool],
                      postcondition: Callable[[ConfluentPackBank], bool] | None = None,
                      state_adapter: OptimizerStateAdapter | None = None) -> bool:
    """Atomic acceptance/update/reset/postcheck transaction; failure restores all state.

    Acceptance runs against the source model, before any mutation. It must check
    the requested budget (a certified consumer supplies this callback). A false
    postcondition rolls back parameters, buffers, and optimizer history.
    """
    if not between_steps or any(p.grad is not None for p in bank.parameters()):
        raise RuntimeError('commit between optimizer steps after zero_grad(set_to_none=True)')
    if (proposal.source_version != int(bank.transition_version)
            or proposal.source_digest != snapshot_digest(bank.portable_snapshot())):
        raise ValueError('stale transition proposal')
    ids = [int(i) for i in bank.slot_ids]
    if any(i not in ids for i in proposal.slot_ids):
        raise ValueError('proposal contains unknown stable slot IDs')
    slots = tuple(ids.index(i) for i in proposal.slot_ids)
    if not acceptance(bank, proposal):
        return False
    source = {k: v.clone() for k, v in bank.state_dict().items()}
    optimizer_source = deepcopy(optimizer.state_dict())
    try:
        with torch.no_grad():
            state = bank.state_dict(keep_vars=True)
            for update in proposal.updates:
                if update.name not in state or update.name in ('slot_ids', 'transition_version'):
                    raise ValueError('proposal may only update representation tensors')
                tensor = state[update.name]
                if tensor.ndim == 0 or update.index not in slots:
                    raise ValueError('update must refer to an affected slot')
                value = torch.as_tensor(update.value, dtype=tensor.dtype, device=tensor.device)
                if value.shape != tensor[update.index].shape:
                    raise ValueError('slot update shape does not match the preallocated row')
                tensor[update.index] = value
            (state_adapter or reset_slot_state)(optimizer, bank, slots)
            bank.transition_version.add_(1)
        if postcondition is not None and not postcondition(bank):
            bank.load_state_dict(source)
            optimizer.load_state_dict(optimizer_source)
            return False
    except BaseException:
        bank.load_state_dict(source)
        optimizer.load_state_dict(optimizer_source)
        raise
    return True


def detect_transitions(bank: ConfluentPackBank,
                       conditioning_score: Callable[[ConfluentPackBank], float], *,
                       error_budget: float = 0.0) -> tuple[TransitionProposal, ...]:
    """Rank compatible pair proposals by measured conditioning improvement.

    Scores are numerical diagnostics on the caller's declared observations.
    Stable slot IDs break ties. Certified acceptance remains a separate step.
    """
    baseline = conditioning_score(bank)
    ranked: list[tuple[float, tuple[int, ...], TransitionProposal]] = []
    for first in range(bank.max_packs):
        for second in range(first+1, bank.max_packs):
            try:
                proposal = bank.propose_pair(first, second, error_budget=error_budget)
            except ValueError:
                continue
            candidate = deepcopy(bank)
            with torch.no_grad():
                for update in proposal.updates:
                    target = getattr(candidate, update.name)
                    target[update.index] = torch.as_tensor(update.value, dtype=target.dtype, device=target.device)
            improvement = conditioning_score(candidate) - baseline
            if improvement > 0:
                ranked.append((-improvement, proposal.slot_ids, proposal))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked)


def transport_optimizer_state(
    optimizer: torch.optim.Optimizer, bank: ConfluentPackBank,
    permutation: tuple[int, ...] | None, sign_slots: tuple[int, ...],
) -> None:
    """Exact elementwise state transport for the named supported optimizers.

    Permutations move every moment row. Tanh sign relabeling negates first
    moments of weight/scale while keeping squared/absolute moments unchanged.
    Coupled optimizer histories require an explicit transport adapter.
    """
    allowed = (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.RMSprop,
               torch.optim.Adagrad, torch.optim.Adadelta, torch.optim.Adamax)
    if type(optimizer) not in allowed:
        raise TypeError('unknown or coupled optimizer requires an explicit ExactTransportAdapter')
    odd = {'exp_avg', 'momentum_buffer', 'grad_avg'}
    even = {'exp_avg_sq', 'max_exp_avg_sq', 'square_avg', 'sum', 'acc_delta', 'exp_inf'}
    for name, parameter in bank.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                continue
            if permutation is not None:
                value.copy_(value[list(permutation)].clone())
            if sign_slots and name in ('weights', 'scales'):
                if key in odd:
                    value[list(sign_slots)] *= -1
                elif key not in even:
                    raise TypeError(f'unknown moment {key!r} requires an explicit ExactTransportAdapter')


def _exact_transport(
    bank: ConfluentPackBank, optimizer: torch.optim.Optimizer, *, between_steps: bool,
    permutation: tuple[int, ...] | None = None, sign_slots: tuple[int, ...] = (),
    state_adapter: ExactTransportAdapter | None = None,
) -> None:
    if not between_steps or any(parameter.grad is not None for parameter in bank.parameters()):
        raise RuntimeError('transport between steps after zero_grad(set_to_none=True)')
    if permutation is not None and (len(permutation) != bank.max_packs
                                    or sorted(permutation) != list(range(bank.max_packs))):
        raise ValueError('permutation must contain every allocated slot exactly once')
    if sign_slots:
        if len(set(sign_slots)) != len(sign_slots) or any(i < 0 or i >= bank.max_packs for i in sign_slots):
            raise ValueError('sign slots must be distinct allocated slots')
        if bank.act_spec.name != 'tanh' or any(int(bank.modes[i]) != 0 for i in sign_slots):
            raise ValueError('sign relabeling supports ordinary tanh slots only')
    source = {name: value.clone() for name, value in bank.state_dict().items()}
    optimizer_source = deepcopy(optimizer.state_dict())
    try:
        with torch.no_grad():
            if permutation is not None:
                for name, value in bank.state_dict(keep_vars=True).items():
                    if name != 'transition_version':
                        value.copy_(value[list(permutation)].clone())
            if sign_slots:
                bank.weights[list(sign_slots)] *= -1
                bank.scales[list(sign_slots)] *= -1
            (state_adapter or transport_optimizer_state)(optimizer, bank, permutation, sign_slots)
            bank.transition_version.add_(1)
    except BaseException:
        bank.load_state_dict(source)
        optimizer.load_state_dict(optimizer_source)
        raise


def permute_bank(
    bank: ConfluentPackBank, permutation: Sequence[int], optimizer: torch.optim.Optimizer,
    *, between_steps: bool, state_adapter: ExactTransportAdapter | None = None,
) -> None:
    """Relabel every slot and its optimizer moments, preserving Parameter objects.

    Stable IDs move with their atoms. Floating sum order may change rounding;
    the underlying realization and all state coordinates are permuted exactly.
    """
    _exact_transport(bank, optimizer, between_steps=between_steps, permutation=tuple(permutation),
                     state_adapter=state_adapter)


def relabel_tanh_sign(
    bank: ConfluentPackBank, slots: Sequence[int], optimizer: torch.optim.Optimizer,
    *, between_steps: bool, state_adapter: ExactTransportAdapter | None = None,
) -> None:
    """Ordinary tanh symmetry (weight, scale)->(-weight,-scale), for every order."""
    _exact_transport(bank, optimizer, between_steps=between_steps, sign_slots=tuple(slots),
                     state_adapter=state_adapter)


__all__ = ['ConfluentPackBank', 'ExactTransportAdapter', 'OptimizerStateAdapter', 'commit_transition',
           'detect_transitions', 'permute_bank', 'relabel_tanh_sign', 'reset_slot_state',
           'transport_optimizer_state']

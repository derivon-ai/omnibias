# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private live-ledger sign authority for known emitted Adam block programs.

No numerical module is imported at top level. These handles are process-local
authority, not JSON certificates or a security boundary against private bypass.
"""

import hashlib
import json
import weakref
from dataclasses import dataclass

CONTRACT = "owned-emitted-moment-nonnegative/carry/v1"
MAX_LIVE = 16
MAX_METADATA_BYTES = 8 * 1024**2
ROLES = ("U", "V", "bias", "readout", "offset")


def _require(value, reason):
    if not value:
        raise ArithmeticError(reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class _Input:
    token: int
    inputs: tuple
    indices: tuple
    outputs: tuple
    closed: object
    graph: str
    profile: tuple
    origin: str


@dataclass(frozen=True)
class _Output:
    token: int
    groups: tuple
    graph: str
    profile: tuple
    origin: str


@dataclass(frozen=True)
class _Permit:
    token: int
    state: object
    state_bounds: object
    signature: tuple
    moments: tuple
    origin: str


class MomentAuthority:
    """Owned by one known RuntimeBlock; retains no predecessor trace/packet.

    Input proofs are box-intersect-NN theorems, with all five moment predicates
    and optimizer/source identity in their profile. Current active authority is
    independently checked on every use, including an existing broad proof.
    """

    def __init__(self, engine):
        self._engine = weakref.ref(engine)
        self._requests = weakref.WeakValueDictionary()
        self._outputs = weakref.WeakValueDictionary()
        self._permits = weakref.WeakValueDictionary()
        self._next = 0

    def _runtime(self):
        from . import bounds, runtime_block

        engine = self._engine()
        _require(type(engine) is runtime_block.RuntimeBlock, "exact known block issuer required")
        return engine, runtime_block.jax, runtime_block.np, bounds

    def _issue(self, registry, factory, *values):
        _require(len(registry) < MAX_LIVE, "live moment-authority hardcap16 exhausted")
        live = (*self._requests.values(), *self._outputs.values(), *self._permits.values())
        _require(sum(len(v.origin.encode()) + 1024 for v in live)
                 + len(values[-1].encode()) + 1024 <= MAX_METADATA_BYTES,
                 "moment-authority metadata hardcap exhausted")
        self._next += 1
        value = factory(self._next, *values)
        registry[value.token] = value
        return value

    def profile(self):
        engine, _, _, _ = self._runtime()
        return (CONTRACT, engine.engine_id, engine.optimizer_profile(), ROLES)

    def _moments(self, tree):
        engine, _, _, bounds = self._runtime()
        _require(type(tree) is engine.state_type, "actual TrainingState declaration required")
        values = tree.second_moment
        _require(type(values) is tuple and len(values) == 5, "exact tuple5 moment roles required")
        for value in values:
            _require(bounds.is_owned_bound(value), "owned originating moment Bound required")
        return values

    def validate_permit(self, permit, state, declared):
        engine, _, _, _ = self._runtime()
        _require(type(permit) is _Permit and self._permits.get(permit.token) is permit,
                 "foreign, expired or fabricated moment permit")
        _require(permit.state is state and permit.state_bounds is declared,
                 "moment permit changed actual state or originating Bound tree")
        _require(engine.array_signature(state) == permit.signature,
                 "moment permit actual device state changed")
        _require(all(a is b for a, b in zip(self._moments(declared), permit.moments, strict=True)),
                 "moment permit Bound role identity changed")
        return permit

    def active(self, state, declared, carry):
        """Return only after ideal AND actual NN has an established origin."""
        _, _, np, _ = self._runtime()
        moments = self._moments(declared)
        if carry is not None:
            self.validate_permit(carry._nn_permit, state, declared)
            return {"kind": "issued_same_call_carry", "token": carry._nn_permit.token,
                    "carry": carry.token, "profile": self.profile()}
        _require(all(np.all(b.lo >= b.error) for b in moments),
                 "external moments require complete ideal+actual NN enclosure")
        return {"kind": "complete_external_enclosure", "profile": self.profile()}

    def prepare(self, inputs, shaped, *, state, active_bounds, carry, graph, closed):
        engine, jax, _, _ = self._runtime()
        active = self.active(state, active_bounds, carry)
        values = tuple(jax.tree.leaves(inputs))
        indexed = jax.tree.unflatten(jax.tree.structure(inputs), list(range(len(values))))
        _require(type(indexed[0]) is engine.state_type, "actual input state PyTree required")
        indices = tuple(indexed[0].second_moment)
        _require(len(indices) == 5 and len(set(indices)) == 5, "complete five input moment roles")
        leaves = jax.tree.leaves(shaped)
        output = jax.tree.unflatten(jax.tree.structure(shaped), list(range(len(leaves))))
        _require(type(output) is engine.trace_type, "actual BlockTrace PyTree required")
        groups = (tuple(output.final_state.second_moment),
                  tuple(output.history.accepted.second_moment),
                  tuple(output.history.proposed.second_moment))
        _require(all(len(group) == 5 for group in groups), "complete emitted moment output roles")
        # Current active enclosure must fit each declared proof role. This is
        # separate from the NN predicate: neither enclosure/error is replaced.
        _, _, np, _ = self._runtime()
        for index, active_bound in zip(indices, self._moments(active_bounds), strict=True):
            proof_bound = values[index]
            _require(proof_bound.lo.shape == active_bound.lo.shape
                     and np.all(proof_bound.lo <= active_bound.lo)
                     and np.all(proof_bound.hi >= active_bound.hi)
                     and np.all(proof_bound.error >= active_bound.error),
                     "current moment ideal/error bounds outside declared NN proof box")
        origin = _json({"active": active, "graph": graph, "profile": self.profile()})
        return self._issue(self._requests, _Input, values, indices, groups,
                           closed, graph, self.profile(), origin)

    def _validate_request(self, request, closed, inputs):
        _require(type(request) is _Input and self._requests.get(request.token) is request
                 and request.closed is closed and request.profile == self.profile(),
                 "foreign graph, profile or expired live interpreter request")
        _require(len(inputs) == len(request.inputs)
                 and all(a is b for a, b in zip(inputs, request.inputs, strict=True)),
                 "interpreter request changed exact original input Bound objects")

    def _input_facts(self, request, ledger, original, compressed):
        """Called by the new interpreter around its known compression operation."""
        _, _, np, _ = self._runtime()
        _require(type(request) is _Input and self._requests.get(request.token) is request,
                 "issued live interpreter request required")
        _require(len(original) == len(compressed) == len(request.inputs)
                 and all(a is b for a, b in zip(original, request.inputs, strict=True)),
                 "precompression Bound origin/arity changed")
        facts = []
        selected = set(request.indices)
        for index, (source, target) in enumerate(zip(original, compressed, strict=True)):
            arrays = ledger._joint_representatives(
                (source.lo, source.hi, source.error, target.lo, target.hi, target.error))
            ledger._visit(sum(int(a.size) for a in arrays))
            _require(source.lo.shape == target.lo.shape and all(
                np.array_equal(a, b) for a, b in zip(arrays[:3], arrays[3:], strict=True)),
                "known compression did not preserve coordinates and all three bounds")
            facts.append(ledger._issue(target, "issued_box_intersect_NN_input_after_known_compression",
                                       nonnegative=True, location=("owned_carry", index))
                         if index in selected else ledger.seed(target))
        return facts

    def _own_outputs(self, request, ledger, values):
        """LIVE ledger check and owned copy happen before interpreter fact loss."""
        _, _, _, bounds = self._runtime()
        _require(self._requests.get(request.token) is request, "foreign output request")
        _require(len(values) == len(values.facts), "actual output witness arity changed")
        selected = {index for group in request.outputs for index in group}
        for index in selected:
            _require(ledger._valid(values.facts[index], values[index]).nonnegative,
                     "emitted moment lacks BOTH ideal+actual nonnegative proof")
        owned = [bounds.immutable_bound(value) for value in values]
        groups = tuple(tuple(owned[index] for index in group) for group in request.outputs)
        packet = self._issue(self._outputs, _Output, groups, request.graph,
                             request.profile, request.origin)
        self._requests.pop(request.token)
        return owned, packet

    def validate_output(self, output, trace_bounds):
        _, _, _, _ = self._runtime()
        _require(type(output) is _Output and self._outputs.get(output.token) is output
                 and output.profile == self.profile(), "foreign or stale emitted NN output")
        actual = (self._moments(trace_bounds.final_state),
                  self._moments(trace_bounds.history.accepted),
                  self._moments(trace_bounds.history.proposed))
        _require(all(a is b for row, other in zip(actual, output.groups, strict=True)
                     for a, b in zip(row, other, strict=True)),
                 "emitted NN output Bound identity/roles changed")
        return output

    def carry(self, outcome, state, declared, *, group, index=None):
        """Known copy/slice link from the ACTUAL issued same-call trace only."""
        engine, jax, np, _ = self._runtime()
        outcome.validate()
        output = self.validate_output(outcome.certificate._nn_output,
                                      outcome.certificate.trace_bounds)
        _require(group in (0, 2), "only final accepted or failed proposed state may carry")
        source_bounds = outcome.certificate.trace_bounds.final_state if group == 0 else None
        if group == 0:
            _require(declared is source_bounds and index is None,
                     "OK carry must preserve SAME complete Bound tree")
            source_state = outcome.trace.final_state
        else:
            _require(type(index) is int and 0 <= index < outcome.certificate.packet.steps,
                     "exact failed proposal time slice required")
            _require(int(jax.device_get(outcome.trace.failed_index)) == index,
                     "proposal permit must refer to actual failed slot")
            source_state = jax.tree.map(lambda a: a[index], outcome.trace.history.proposed)
            for target, source in zip(self._moments(declared), output.groups[2], strict=True):
                _require(all(np.array_equal(a, b[index]) for a, b in zip(
                    (target.lo, target.hi, target.error),
                    (source.lo, source.hi, source.error), strict=True)),
                    "known proposed-bound time slice changed error or ideal enclosure")
        # The caller owns the known copy operation; enforce its complete state
        # result by device equality in the issued operation, not host hashing.
        _require(jax.tree.structure(source_state) == jax.tree.structure(state)
                 and all(a.shape == b.shape and a.dtype == b.dtype
                         for a, b in zip(jax.tree.leaves(source_state),
                                         jax.tree.leaves(state), strict=True)),
                 "known copy changed complete state shape/dtype/tree")
        _require(bool(jax.device_get(jax.tree.reduce(
            lambda a, b: a & b, jax.tree.map(lambda a, b: engine.array_equal(a, b),
                                           source_state, state)))),
                 "known state-copy link differs from same-call source")
        origin = _json({"contract": CONTRACT, "certificate": outcome.certificate.token,
                        "graph": output.graph, "group": group, "index": index,
                        "profile": self.profile(),
                        "source_offset": outcome.certificate.packet.source_offset,
                        "steps": outcome.certificate.packet.steps,
                        "output_token": output.token})
        return self._issue(self._permits, _Permit, state, declared,
                           engine.array_signature(state), self._moments(declared), origin)

    def rebind(self, permit, state, declared):
        self.validate_permit(permit, state, declared)
        return self._issue(self._permits, _Permit, state, declared, permit.signature,
                           permit.moments, permit.origin)

    def _abandon_unpublished(self, permit):
        """Known failed terminal issuance only; do not retain a failed handle.

        Registration removal is idempotent and never revokes a different
        equal-valued/token object. Monotonic IDs are deliberately not rolled back.
        """
        if type(permit) is _Permit and self._permits.get(permit.token) is permit:
            self._permits.pop(permit.token)
            return True
        return False

    def storage_report(self):
        """Referenced Bound payload is shared with existing charged owners."""
        handles = (*self._outputs.values(), *self._permits.values(), *self._requests.values())
        metadata = sum(len(value.origin.encode()) + 1024 for value in handles)
        return {"live_requests": len(self._requests), "live_outputs": len(self._outputs),
                "live_permits": len(self._permits), "metadata_reserve_bytes": metadata,
                "shared_bound_payload_counted_by_block": True,
                "python_object_overhead_and_transient_hashing_outside_payload": True}

    def profile_id(self):
        return hashlib.sha256(_json(self.profile()).encode()).hexdigest()

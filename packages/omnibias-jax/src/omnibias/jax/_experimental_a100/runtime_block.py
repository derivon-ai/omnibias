# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private bounded issued-packet scan; SOURCE ONLY until final-source validation.

A packet receipt is not a block proof. The actual emitted guarded scan must
qualify, then this component checks initial input containment on device before
issuing runnable authority. Known context builders own objective/VG semantics.
"""

import json
import math
import time
import uuid
import weakref
from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import (
    bounded_optimizer,
    bounds,
    device_training,
    planner,
    runtime_prepared,
    scheduled_optimizer,
)
from ._bound_carry_authority import MomentAuthority

MAX_STEPS = 16
MAX_BYTES = 3 * 1024**3
MAX_EQUATIONS = 100000
RADIUS_CONTRACT = "outward-real-halfwidth/nextafter-two-stage/v1"


def outward_radius(lo, hi, delta):
    """Upper bound max((hi-lo)/2,16|delta|) in EXACT REAL binary64 inputs.

    Two outward operations are necessary: outward final endpoints alone cannot
    repair an RN halfwidth underestimate. Finite overflow refusal is deliberate.
    """
    lo, hi, delta = (np.asarray(a, dtype=np.float64) for a in (lo, hi, delta))
    _require(lo.shape == hi.shape == delta.shape
             and all(np.all(np.isfinite(a)) for a in (lo, hi, delta))
             and np.all(lo <= hi), "finite matching recenter radius arrays required")
    with np.errstate(over="ignore", invalid="ignore"):
        diff_up = np.nextafter(hi - lo, np.inf)
        half_up = np.nextafter(diff_up / 2, np.inf)
        old_radius = np.where(lo == hi, 0.0, half_up)
        radius = np.maximum(old_radius, 16 * np.abs(delta))
    _require(np.all(np.isfinite(radius)), "outward recenter radius overflow")
    return radius


def positive(value, maximum, label):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(label + " outside positive hard cap")
    return value


def payload_estimate(steps, parameter_entries, input_entries, projection_entries, counter_bytes):
    """Conservative logical payloads; executable/owners are separately added.

    TraceStep stores TWO complete states, delta, actual gradient and 3 scalars.
    Stack contains X/Z0 and all six per-slot box fields. No alias credit.
    """
    positive(steps, MAX_STEPS, "block length")
    for value in (parameter_entries, input_entries, projection_entries):
        positive(value, MAX_BYTES, "entry count")
    if counter_bytes not in (4, 8):
        raise ValueError("32/64-bit counter required")
    p, x, z = (8 * n for n in (parameter_entries, input_entries, projection_entries))
    state = 3 * p + counter_bytes
    history = steps * (2 * state + 2 * p + 8 + 4 + 1)
    stack = steps * (2 * p + 3 * x + 3 * z)
    return {
        "state_bytes": state,
        "full_history_bytes": history,
        "stack_device_bytes": stack,
        "trace_output_bytes": state + history + 4 + 4 + 1,
        "scope": "logical payload estimate before compilation; no alias credit; retained proof/owner/executable bytes separate",
    }


def resume_start(failed_index, steps):
    positive(steps, MAX_STEPS, "block length")
    if type(failed_index) is not int or not 0 <= failed_index < steps:
        raise ValueError("actual failed slot required")
    return failed_index + 1


CONTRACT = "owned-runtime-block/moment-carry-outward-radius/v1"
SKIPPED = 4


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(tree):
    return tuple(runtime_prepared._device_signature(a) for a in jax.tree.leaves(tree))


def _bytes(tree):
    return sum(a.nbytes for a in jax.tree.leaves(tree))


def _owned(tree):
    return jax.tree.map(bounds.immutable_bound, tree)


def _bound_bytes(tree):
    owners = {}
    for b in jax.tree.leaves(tree):
        owners.update(bounds.bound_storage_owners(b))
    return sum(size for _, size in owners.values())


def _validate_bounds(tree):
    for b in jax.tree.leaves(tree):
        _require(bounds.is_owned_bound(b), "owned originating bound required")


def _point(tree):
    return jax.tree.map(bounds.Bound.point, tree)


def _bound_identity(tree):
    _validate_bounds(tree)
    return planner.digest_record(
        {
            "tree": str(jax.tree.structure(tree)),
            "triplets": [
                [bounds.constant_digest(a) for a in (b.lo, b.hi, b.error)]
                for b in jax.tree.leaves(tree)
            ],
        }
    )


def _all_inside(value, lo, hi):
    return device_training.inside(value, lo, hi)


class BlockTrace(NamedTuple):
    final_state: device_training.TrainingState
    history: bounded_optimizer.TraceStep
    failed_index: object
    status: object
    entry_valid: object


class EntryGuard(NamedTuple):
    state_lower: object
    state_upper: object
    points_lower: object
    points_upper: object
    projection_lower: object
    projection_upper: object
    boxes_lower: object
    boxes_upper: object


@dataclass(frozen=True)
class Packet:
    entries: tuple
    points: object
    projection: object
    boxes: object
    point_bound: object
    projection_bound: object
    box_bounds: object
    token: str
    source_offset: int
    _engine: object
    _signature: tuple
    _report_json: str

    @property
    def steps(self):
        return len(self.entries)

    @property
    def report(self):
        return json.loads(self._report_json)

    def validate(self):
        engine = self._engine
        engine.validate()
        _require(engine._packets.get(id(self)) is self, "packet was not issued by this engine")
        _require(1 <= self.steps <= MAX_STEPS, "bounded nonempty packet required")
        for batch, receipt in self.entries:
            engine.context.check_certificate(batch, receipt)
        _require(
            _signature((self.points, self.projection, self.boxes)) == self._signature,
            "owned stack device identity changed",
        )
        _validate_bounds((self.point_bound, self.projection_bound, self.box_bounds))
        return self


@dataclass(frozen=True)
class BlockCertificate:
    packet: Packet
    state_bounds: object
    guard: EntryGuard
    trace_bounds: BlockTrace
    executable: object
    token: str
    carry: object
    carried_state_signature: tuple
    _engine: object
    _signature: tuple
    _report_json: str
    _nn_output: object = None

    @property
    def report(self):
        return json.loads(self._report_json)

    def validate(self):
        self.packet.validate()
        _require(
            self._engine._certificates.get(id(self)) is self,
            "block certificate was not issued by actual scan qualification",
        )
        _validate_bounds((self.state_bounds, self.trace_bounds))
        self._engine._moment_authority.validate_output(self._nn_output, self.trace_bounds)
        _require(_signature(self.guard) == self._signature, "entry guard arrays changed")
        if self.carry is not None:
            _require(
                isinstance(self.carry, (Resume, Carry)), "issued carry provenance type required"
            )
            self.carry.validate()
            _require(
                self.carry._engine is self._engine
                and self.carry.packet is self.packet
                and self.state_bounds is self.carry.state_bounds,
                "certificate changed its originating carried state/bounds/packet",
            )
            _require(
                self.carried_state_signature
                == self.carry._signature
                == _signature(self.carry.state),
                "carried state identity changed",
            )
        else:
            _require(
                self.carried_state_signature == (),
                "external state-domain scope has no carry signature",
            )
        report = self.report
        _require(
            report["state_origin"].get("carry_token")
            == (self.carry.token if self.carry is not None else None),
            "saved state origin differs from issued carry",
        )
        _require(report["state_origin"].get("moment_NN_permit")
                 == (self.carry._nn_permit.token if self.carry is not None else None)
                 and report.get("moment_NN_output_token") == self._nn_output.token,
                 "saved NN provenance differs from live issued handles")
        _require(
            report["admissible"]
            and report["certificate_issued"]
            and report["runtime_input_containment_checked"]
            and report["packet_token"] == self.packet.token,
            "block report is not issued authority for this packet",
        )
        return self


@dataclass(frozen=True)
class Outcome:
    trace: BlockTrace
    certificate: BlockCertificate
    _engine: object
    _signature: tuple
    _report_json: str

    @property
    def report(self):
        return json.loads(self._report_json)

    def validate(self):
        self.certificate.validate()
        _require(
            self._engine._outcomes.get(id(self)) is self,
            "outcome was not returned by the certified actual executable",
        )
        _require(_signature(self.trace) == self._signature, "outcome device history changed")
        return self


@dataclass(frozen=True)
class Resume:
    state: object
    state_bounds: object
    packet: object
    failed_receipt: object
    next_index: int
    _source_json: str
    _engine: object
    _signature: tuple
    _report_json: str
    _nn_permit: object = None

    @property
    def token(self):
        return self.report["carry_token"]

    @property
    def report(self):
        return json.loads(self._report_json)

    @property
    def source_record(self):
        return json.loads(self._source_json)

    def validate(self):
        self._engine.validate()
        _require(
            self._engine._resumes.get(id(self)) is self,
            "resume token was not issued after proposal requalification",
        )
        self.failed_receipt.validate()
        _require(self.failed_receipt.admissible, "failed-slot proposal receipt refused")
        origin = self.source_record
        _require(
            origin["engine_id"] == self._engine.engine_id
            and origin["kind"] == "DOMAIN_EXIT"
            and self.next_index == origin["source_offset"] + origin["failed_index"] + 1,
            "detached resume origin/offset changed",
        )
        if self.packet is not None:
            self.packet.validate()
            _require(
                self.packet._engine is self._engine
                and self.packet.source_offset == self.next_index,
                "resume packet source offset/authority differs",
            )
        _require(_signature(self.state) == self._signature, "adopted proposed state changed")
        _validate_bounds(self.state_bounds)
        self._engine._moment_authority.validate_permit(self._nn_permit, self.state, self.state_bounds)
        return self


@dataclass(frozen=True)
class Carry:
    state: object
    state_bounds: object
    packet: Packet
    next_index: int
    _source_json: str
    _engine: object
    _signature: tuple
    _report_json: str
    _nn_permit: object = None

    @property
    def token(self):
        return self.report["carry_token"]

    @property
    def report(self):
        return json.loads(self._report_json)

    @property
    def source_record(self):
        return json.loads(self._source_json)

    def validate(self):
        self._engine.validate()
        _require(
            self._engine._carries.get(id(self)) is self,
            "successful carry was not issued by this exact engine",
        )
        self.packet.validate()
        origin = self.source_record
        _require(
            origin["engine_id"] == self._engine.engine_id
            and origin["kind"] == "OK"
            and self.next_index == origin["source_offset"] + origin["steps"]
            and self.packet._engine is self._engine
            and self.packet.source_offset == self.next_index,
            "successful carry packet/source offset/authority differs",
        )
        _require(
            _signature(self.state) == self._signature, "successful carried state identity changed"
        )
        _validate_bounds(self.state_bounds)
        self._engine._moment_authority.validate_permit(self._nn_permit, self.state, self.state_bounds)
        return self


class RuntimeBlock:
    """Bounded fixed/fresh stack, same-call trace, actual graph proof and resume.

    This candidate does not implement streamed-input overlap or donation.
    Retained context and receipts are charged conservatively, without aliases.
    """

    def __init__(self, context, *, limits=None, tolerances=None, optimizer_kind="bounded", **optimizer_options):
        _require(
            type(context) is runtime_prepared.RuntimePreparedContext,
            "sealed combined context required",
        )
        context._validate_context()
        limits = planner.Limits(cells=1) if limits is None else limits
        tolerances = planner.Tolerances() if tolerances is None else tolerances
        _require(
            isinstance(limits, planner.Limits) and isinstance(tolerances, planner.Tolerances),
            "typed planner limits/tolerances required",
        )
        positive(limits.equations, min(MAX_EQUATIONS, context.max_equations), "equations")
        positive(limits.state_bytes, min(MAX_BYTES, context.max_state_bytes), "state bytes")
        _require(asdict(tolerances) == asdict(planner.Tolerances()), "unchanged accuracy contract")
        self.context, self.limits, self.tolerances = context, limits, tolerances
        _require(optimizer_kind in ("bounded", "schedule"), "explicit known optimizer selection required")
        optimizer_type = (bounded_optimizer.BoundedTraceAdam if optimizer_kind == "bounded"
                          else scheduled_optimizer.ScheduleTraceAdam)
        self.optimizer = optimizer_type(context.value_gradient, **optimizer_options)
        self.optimizer_kind = optimizer_kind
        self._moment_authority = MomentAuthority(self)
        self._packets, self._certificates = (
            weakref.WeakValueDictionary(),
            weakref.WeakValueDictionary(),
        )
        self._outcomes, self._resumes = weakref.WeakValueDictionary(), weakref.WeakValueDictionary()
        self._carries = weakref.WeakValueDictionary()
        self._continued = {}
        self._engine_id = planner.digest_record(
            {"contract": CONTRACT, "context": context.structure_key, "nonce": uuid.uuid4().hex}
        )
        self._issuance = {kind: 0 for kind in ("packet", "certificate", "carry")}
        self._committed = {}  # weak outcome references; one successful proposal adoption
        self._terminal_bindings = {}  # weak source Resume references; binding is single-use
        self._identity = (
            id(context),
            id(self.optimizer),
            id(self._moment_authority),
            self.optimizer_kind,
            limits,
            tolerances,
            context.structure_key,
            self._engine_id,
        )

    state_type = device_training.TrainingState
    trace_type = BlockTrace
    array_signature = staticmethod(_signature)
    array_equal = staticmethod(lambda a, b: jnp.all(a == b))

    def optimizer_profile(self):
        selected = (bounded_optimizer.BoundedTraceAdam if self.optimizer_kind == "bounded"
                    else scheduled_optimizer.ScheduleTraceAdam)
        _require(type(self.optimizer) is selected, "exact selected Adam implementation required")
        o = self.optimizer
        profile = {"kind": self.optimizer_kind, "beta1": o.beta1.hex(), "beta2": o.beta2.hex(),
                   "learning_rate": o.learning_rate.hex(), "epsilon": o.epsilon.hex(),
                   "max_accepted_steps": o.max_accepted_steps, "bits": o.bits}
        if self.optimizer_kind == "schedule":
            o.validate_schedule()
            profile["schedule"] = o.storage_report()
        return json.dumps(profile, sort_keys=True, allow_nan=False)

    def constant_bounds(self):
        binding = bounds.ConstantBindings().extend(self.context.value_gradient.constant_bounds)
        if self.optimizer_kind == "schedule":
            binding.extend(self.optimizer.constant_bounds)
        return binding

    def moment_storage_bytes(self):
        # Reserve the entire issuer metadata cap BEFORE any permit/export;
        # shared Bound payload is already charged through packet/proof/state.
        result = 8 * 1024**2
        if self.optimizer_kind == "schedule":
            report = self.optimizer.storage_report()
            result += report["retained_host_array_payload_bytes"] + report["retained_device_array_payload_bytes"]
            result += report["retained_Python_record_bytes_conservative_estimate"]
        return result

    def validate(self):
        self.context._validate_context()
        self.optimizer_profile()
        _require(
            (
                id(self.context),
                id(self.optimizer),
                id(self._moment_authority),
                self.optimizer_kind,
                self.limits,
                self.tolerances,
                self.context.structure_key,
                self._engine_id,
            )
            == self._identity,
            "block structure/limits changed",
        )

    @property
    def engine_id(self):
        return self._engine_id

    def _next_token(self, kind, payload):
        _require(kind in self._issuance, "known issuance kind required")
        value = self._issuance[kind]
        _require(
            type(value) is int and 0 <= value < (1 << 63) - 1,
            "monotonic issuance counter exhausted",
        )
        self._issuance[kind] = value + 1
        return planner.digest_record(
            {
                "contract": CONTRACT,
                "engine": self.engine_id,
                "kind": kind,
                "issuance": value,
                "origin": payload,
            }
        )

    @staticmethod
    def _executable_bytes(executable):
        info = executable.memory_analysis()
        _require(info is not None, "existing executable memory estimate unavailable")
        count = (
            info.argument_size_in_bytes
            + info.output_size_in_bytes
            + info.temp_size_in_bytes
            - info.alias_size_in_bytes
        )
        _require(count >= 0, "negative executable accounting")
        return count

    def _additional_storage(self, *, packet=None, certificate=None):
        # Charge all OTHER live issued storage, including caller-retained objects.
        # Deliberately no alias credit between arrays/views or proof categories.
        total = 0
        for p in list(self._packets.values()):
            if p is not packet:
                total += _bytes((p.points, p.projection, p.boxes))
                total += _bound_bytes((p.point_bound, p.projection_bound, p.box_bounds))
                total += sum(self.context._certificate_payload_bytes(c) for _, c in p.entries)
        for c in list(self._certificates.values()):
            if c is not certificate:
                total += _bytes(c.guard) + _bound_bytes((c.state_bounds, c.trace_bounds))
                total += c.report["executable_bytes"] + len(c._report_json.encode())
        total += sum(_bytes(o.trace) for o in list(self._outcomes.values()))
        total += sum(
            _bytes(r.state)
            + _bound_bytes(r.state_bounds)
            + self.context._certificate_payload_bytes(r.failed_receipt)
            + len(r._source_json.encode())
            + len(r._report_json.encode())
            for r in list(self._resumes.values())
        )
        total += sum(
            _bytes(c.state)
            + _bound_bytes(c.state_bounds)
            + len(c._source_json.encode())
            + len(c._report_json.encode())
            for c in list(self._carries.values())
        )
        return total

    def _base_storage(self, entries):
        context = self.context.storage_report()
        unique = {id(receipt): receipt for _, receipt in entries}
        existing = self.context._compiled or {}
        return (
            self.moment_storage_bytes()
            + context["fixed_context_host_payload_bytes"]
            + context["owned_context_device_payload_bytes"]
            + context["certificate_retained_payload_bytes"]
            + sum(self._executable_bytes(e) for e in existing.values())
            + sum(self.context._certificate_payload_bytes(c) for c in unique.values())
        )

    def _budget(self, value, message):
        _require(value <= min(self.limits.state_bytes, self.context.max_host_bytes), message)

    def prepare(self, entries, *, source_offset=0):
        """Own an explicit tuple of issued (RuntimeBatch, admissible receipt)."""
        self.validate()
        _require(
            type(source_offset) is int and 0 <= source_offset <= self.optimizer.max_accepted_steps,
            "nonnegative bounded source offset required",
        )
        _require(type(entries) is tuple, "explicit issued entry tuple required")
        positive(len(entries), MAX_STEPS, "block length")
        positive(len(self._packets) + 1, 16, "live owned packets")
        for item in entries:
            _require(type(item) is tuple and len(item) == 2, "exact batch/receipt pair required")
            self.context.check_certificate(*item)
        batch = entries[0][0]
        p = sum(math.prod(shape) for shape in self.context._parameter_shapes)
        estimate = payload_estimate(
            len(entries), p, math.prod(batch.points.shape), math.prod(batch.projection.shape), 8
        )
        # Analysis bounds (3 binary64 arrays/element) plus actual device stack,
        # full trace and retained receipts are included before stack allocation.
        projected = (
            self._base_storage(entries)
            + self._additional_storage()
            + 12 * estimate["stack_device_bytes"]
            + 4 * estimate["trace_output_bytes"]
            + 12 * estimate["state_bytes"]
        )
        self._budget(projected, "packet/full-history preflight exceeds resource budget")
        started = time.perf_counter()
        xbound = bounds.immutable_bound(
            bounds.Bound.point(np.stack([b.host_points for b, _ in entries]))
        )
        zitems = [b.projection_bound for b, _ in entries]
        zbound = bounds.immutable_bound(
            bounds.Bound(
                *(np.stack([getattr(b, attr) for b in zitems]) for attr in ("lo", "hi", "error"))
            )
        )
        # The SAME exact host threshold construction used by issued R4 receipts.
        host_boxes = []
        for b, receipt in entries:
            construction = receipt.active_construction
            construction = b.projection_bound if construction is None else construction
            lo, hi = construction.enlarged()
            host_boxes.append(
                bounded_optimizer.TraceBoxes(
                    tuple(v.lo for v in receipt.domain.parameters),
                    tuple(v.hi for v in receipt.domain.parameters),
                    receipt.domain.inputs.lo,
                    receipt.domain.inputs.hi,
                    lo,
                    hi,
                )
            )
        stacked_host = jax.tree.map(lambda *v: np.stack(v), *host_boxes)
        box_bounds = _owned(_point(stacked_host))
        points = jnp.stack([b.points for b, _ in entries])
        projection = jnp.stack([b.projection for b, _ in entries])
        boxes = jax.tree.map(
            lambda *v: jnp.stack(v), *(bounded_optimizer.TraceBoxes(*c.boxes) for _, c in entries)
        )
        token = self._next_token(
            "packet",
            {
                "context": self.context.structure_key,
                "entries": [[b.token, c._key] for b, c in entries],
                "source_offset": source_offset,
            },
        )
        report = {
            "contract": CONTRACT,
            "engine_id": self.engine_id,
            "token": token,
            "steps": len(entries),
            "logical_payload_estimate": estimate,
            "conservative_preflight_bytes": projected,
            "setup_dispatch_s": time.perf_counter() - started,
            "source_offset": source_offset,
            "receipt_ids": [
                {"batch": b.token, "receipt": c._key, "proof": c.report.get("proof_origin", c._key)}
                for b, c in entries
            ],
            "no_input_or_parameter_device_readback": True,
            "fixed_or_fresh": "fixed" if len({id(b) for b, _ in entries}) == 1 else "fresh/mixed",
            "public_integration": False,
            "performance_evidence": False,
        }
        packet = Packet(
            entries,
            points,
            projection,
            boxes,
            xbound,
            zbound,
            box_bounds,
            token,
            source_offset,
            self,
            _signature((points, projection, boxes)),
            json.dumps(report, sort_keys=True),
        )
        self._packets[id(packet)] = packet
        packet.validate()
        # Synchronization/transfer completion is charged by complete setup callers.
        return packet

    def _entry_valid(self, state, points, projection, boxes, guard, steps):
        return (
            _all_inside(state, guard.state_lower, guard.state_upper)
            & jnp.all(jnp.stack([jnp.all(v >= 0) for v in state.second_moment]))
            & self.optimizer.counter_valid(state.accepted_steps, headroom=steps)
            & _all_inside(points, guard.points_lower, guard.points_upper)
            & _all_inside(projection, guard.projection_lower, guard.projection_upper)
            & _all_inside(boxes, guard.boxes_lower, guard.boxes_upper)
        )

    def _scan(self, state, points, projection, boxes, guard):
        steps = points.shape[0]
        positive(steps, MAX_STEPS, "static block length")
        entry = self._entry_valid(state, points, projection, boxes, guard, steps)
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def body(carry, row):
            accepted, active, failed, status = carry
            index, x, z, slot = row

            def skip(_):
                code = jnp.where((index == 0) & ~entry, device_training.INVALID_INPUT, SKIPPED)
                return bounded_optimizer.TraceStep(
                    accepted,
                    accepted,
                    zero,
                    jnp.asarray(0.0, jnp.float64),
                    zero,
                    code.astype(jnp.int32),
                    jnp.asarray(False),
                )

            trace = jax.lax.cond(
                active, lambda _: self.optimizer.step(accepted, x, z, slot, headroom=1), skip, None
            )
            stopped = active & (trace.status != device_training.OK)
            failed = jnp.where(stopped, index, failed)
            status = jnp.where(stopped, trace.status, status)
            return (
                (trace.accepted, active & (trace.status == device_training.OK), failed, status),
                trace,
            )

        initial = (
            state,
            entry,
            jnp.where(entry, -1, 0).astype(jnp.int32),
            jnp.where(entry, device_training.OK, device_training.INVALID_INPUT).astype(jnp.int32),
        )
        final, history = jax.lax.scan(
            body,
            initial,
            (jnp.arange(steps, dtype=jnp.int32), points, projection, boxes),
            length=steps,
        )
        return BlockTrace(final[0], history, final[2], final[3], entry)

    def _own_initial(self, state, declared, sample, *, carried=False, carry=None):
        _require(
            isinstance(state, device_training.TrainingState)
            and isinstance(declared, device_training.TrainingState),
            "complete TrainingState bounds required",
        )
        self.context._check_args(state.params, sample.points, sample.projection)
        actual, tree = jax.tree.flatten(state)
        values, other = jax.tree.flatten(declared)
        _require(
            tree == other and len(actual) == len(values),
            "complete initial state bound tree required",
        )
        self._budget(
            sum(24 * math.prod(a.shape) for a in actual), "initial-state bound copy budget"
        )
        for a, b in zip(actual, values, strict=True):
            _require(
                isinstance(b, bounds.Bound)
                and b.lo.shape == b.hi.shape == b.error.shape == a.shape,
                "exact initial-state bound shapes required",
            )
            _require(
                carried or not np.any(b.error),
                "external initial states require zero error; use issued resume provenance",
            )
        counter = declared.accepted_steps
        _require(
            counter.lo.shape == ()
            and not np.any(counter.error)
            and np.array_equal(counter.lo, counter.hi)
            and np.all(counter.lo == np.floor(counter.lo))
            and 0 <= float(counter.lo) <= self.optimizer.max_accepted_steps,
            "exact nonnegative point-counter analysis contract required",
        )
        if carried:
            _require(carry is not None, "issued current carry needed for nonzero-error moments")
            self._moment_authority.validate_permit(carry._nn_permit, state, declared)
        else:
            _require(all(np.all(b.lo >= b.error) for b in declared.second_moment),
                     "root complete ideal+actual nonnegative second-moment enclosure required")
        if carried:
            _validate_bounds(declared)
            return declared
        return _owned(declared)

    def _guard(self, state, state_bounds, packet):
        lower = jax.tree.map(lambda b: b.enlarged()[0], state_bounds)
        upper = jax.tree.map(lambda b: b.enlarged()[1], state_bounds)
        lower = lower._replace(
            accepted_steps=np.asarray(lower.accepted_steps, dtype=state.accepted_steps.dtype)
        )
        upper = upper._replace(
            accepted_steps=np.asarray(upper.accepted_steps, dtype=state.accepted_steps.dtype)
        )
        xlo, xhi = packet.point_bound.enlarged()
        zlo, zhi = packet.projection_bound.enlarged()
        blo = jax.tree.map(lambda b: b.lo, packet.box_bounds)
        bhi = jax.tree.map(lambda b: b.hi, packet.box_bounds)
        host = EntryGuard(lower, upper, xlo, xhi, zlo, zhi, blo, bhi)
        target = self.context.owner.base.device
        device = jax.tree.map(lambda a: jax.device_put(a, target), host)
        return device, _owned(_point(host))

    def analyze(self, state, packet, state_bounds, *, _carry=None):
        """Conditional ACTUAL-scan analysis; report alone never grants execution."""
        packet.validate()
        _require(packet._engine is self, "same-engine packet required")
        _require(
            _carry is not None or packet.source_offset == 0,
            "nonroot packet requires issued carry; zero-error trajectory reset refused",
        )
        if _carry is not None:
            _require(isinstance(_carry, (Resume, Carry)), "issued carry type required")
            _carry.validate()
            _require(
                _carry._engine is self
                and _carry.state is state
                and _carry.packet is packet
                and _carry.state_bounds is state_bounds,
                "exact issued carry provenance required",
            )
        sample = packet.entries[0][0]
        self.optimizer._metadata(
            state,
            sample.points,
            sample.projection,
            bounded_optimizer.TraceBoxes(*packet.entries[0][1].boxes),
        )
        _require(
            state.accepted_steps.shape == ()
            and np.dtype(state.accepted_steps.dtype).kind in "iu"
            and np.dtype(state.accepted_steps.dtype).itemsize in (4, 8),
            "scalar32/64-bit counter metadata required before proof copies",
        )
        declared = self._own_initial(state, state_bounds, sample, carried=_carry is not None, carry=_carry)
        guard, guard_bounds = self._guard(state, declared, packet)
        args = (state, packet.points, packet.projection, packet.boxes, guard)
        inputs = (
            declared,
            packet.point_bound,
            packet.projection_bound,
            packet.box_bounds,
            guard_bounds,
        )
        report = {
            "contract": CONTRACT,
            "packet_token": packet.token,
            "engine_id": self.engine_id,
            "admissible": False,
            "certificate_issued": False,
            "runtime_input_containment_checked": False,
            "nonvacuous_evaluated_case": False,
            "steps": packet.steps,
            "source_offset": packet.source_offset,
            "source_receipts": packet.report["receipt_ids"],
            "construction_error_preserved": True,
            "construction_error_max": float(packet.projection_bound.error.max()),
            "counter_scope": "exact ideal nonnegative point; T-headroom once then inner1",
            "scope": "actual emitted full trace; conditional on originating state/X/Z/threshold bounds",
            "native_pow": False,
            "public_integration": False,
            "performance_evidence": False,
            "tolerances": asdict(self.tolerances),
            "limits": asdict(self.limits),
            "assumptions": "unchanged RN binary64/gradual underflow/finite compatible compiler and conditional native transcendental contracts; trusted generated VG; owned transfers exact",
        }
        output, frozen, nn_output = None, None, None
        started = time.perf_counter()
        try:
            self._budget(
                self._base_storage(packet.entries)
                + self._additional_storage(packet=packet)
                + _bound_bytes(inputs)
                + _bytes(args),
                "actual analysis retained input/owner budget exceeded",
            )

            # Graph-specific slots must be rebuilt; never reuse forward slot numbers.
            def function(*args):
                return self._scan(*args)

            function.constant_bounds = self.constant_bounds()
            frozen = planner.freeze_callable(function, args)
            graph = frozen.closed_jaxpr
            report["traced_graph"] = str(graph)
            identity = planner._emitted_graph_identity(graph)
            report["emitted_graph_identity"] = identity
            report["emitted_graph_id"] = planner.digest_record(identity)
            shaped = jax.eval_shape(frozen, *args)
            # Reserve original owners/runtime arrays and the second owned output
            # snapshot BEFORE interpreting/allocating its unowned proof output.
            output_copy_reserve = sum(24 * math.prod(a.shape) for a in jax.tree.leaves(shaped))
            external = (
                self._base_storage(packet.entries)
                + self._additional_storage(packet=packet)
                + _bytes(args)
                + output_copy_reserve
            )
            interpreter_budget = (
                min(self.limits.state_bytes, self.context.max_host_bytes) - external
            )
            _require(
                interpreter_budget >= _bound_bytes(inputs),
                "external payload plus actual scan-proof output exceeds budget",
            )
            report["interpreter_bound_payload_budget_bytes"] = interpreter_budget
            report["output_copy_reserve_bytes"] = output_copy_reserve
            request = self._moment_authority.prepare(
                inputs, shaped, state=state, active_bounds=declared, carry=_carry,
                graph=report["emitted_graph_id"], closed=graph)
            values, meta, nn_output = bounds._interpret_owned(
                graph, jax.tree.leaves(inputs), self.limits.equations,
                bounded_optimizer._strict_slots(graph, frozen.constant_bounds),
                interpreter_budget, authority=self._moment_authority, request=request)
            report["moment_NN_profile"] = self._moment_authority.profile()
            report["moment_NN_profile_id"] = self._moment_authority.profile_id()
            report["moment_NN_output_token"] = nn_output.token
            _require(
                len(values) == len(jax.tree.leaves(shaped)), "complete actual block trace required"
            )
            output = _owned(jax.tree.unflatten(jax.tree.structure(shaped), values))
            out = []

            def summarize(label, physical, expected, *, gradient=False):
                for index, (b, a) in enumerate(
                    zip(jax.tree.leaves(physical), jax.tree.leaves(expected), strict=True)
                ):
                    if np.dtype(a.dtype).kind in "biu":
                        summary = {
                            "admissible": bool(
                                np.array_equal(b.lo, b.hi)
                                and not np.any(b.error)
                                and np.all(b.lo == np.floor(b.lo))
                            ),
                            "discrete": True,
                        }
                    else:
                        atol, rtol = (
                            (self.tolerances.gradient_atol, self.tolerances.gradient_rtol)
                            if gradient
                            else (self.tolerances.forward_atol, self.tolerances.forward_rtol)
                        )
                        summary = b.summary(atol, rtol)
                    out.append({"field": label, "leaf": index, **summary})

            summarize("final_state", output.final_state, shaped.final_state)
            for field in bounded_optimizer.TraceStep._fields:
                summarize(
                    "history/" + field,
                    getattr(output.history, field),
                    getattr(shaped.history, field),
                    gradient=field == "gradient",
                )
            for field in ("failed_index", "status", "entry_valid"):
                summarize(field, getattr(output, field), getattr(shaped, field))
            report.update(
                admissible=all(v["admissible"] for v in out), output_bounds=out, graph=meta
            )
            report["nonvacuous_evaluated_case"] = bool(
                report["admissible"] and np.any(output.history.evaluated.lo == 1)
            )
            if not report["admissible"]:
                report["reason"] = "actual full-scan error contract inconclusive"
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            report.update(reason=str(error), exception=type(error).__name__)
        report["analysis_s"] = time.perf_counter() - started
        return report, output, declared, guard, guard_bounds, frozen, nn_output

    def qualify(self, state, packet, state_bounds, *, _carry=None):
        """Issue authority only after graph proof, device binding and memory checks.

        Returns (BlockCertificate|None, preserved inspectable report). Unsupported
        or inconclusive analysis never issues a boolean-only authority object.
        """
        report, output, declared, guard, guard_bounds, frozen, nn_output = self.analyze(
            state, packet, state_bounds, _carry=_carry
        )
        if not report["admissible"] or not report["nonvacuous_evaluated_case"]:
            report.setdefault("reason", "no nonvacuous qualified evaluated slot")
            return None, report
        started = time.perf_counter()
        try:
            _require(
                bool(
                    self._entry_valid(
                        state, packet.points, packet.projection, packet.boxes, guard, packet.steps
                    )
                ),
                "actual state/input/threshold containment failed",
            )
            report["runtime_input_containment_checked"] = True
            executable = (
                jax.jit(frozen)
                .lower(state, packet.points, packet.projection, packet.boxes, guard)
                .compile()
            )
            info = executable.memory_analysis()
            _require(info is not None, "block executable memory estimate unavailable")
            executable_bytes = (
                info.argument_size_in_bytes
                + info.output_size_in_bytes
                + info.temp_size_in_bytes
                - info.alias_size_in_bytes
            )
            _require(executable_bytes >= 0, "negative block executable accounting")
            total = (
                self._base_storage(packet.entries)
                + self._additional_storage(packet=packet)
                + _bound_bytes(
                    (
                        declared,
                        output,
                        guard_bounds,
                        packet.point_bound,
                        packet.projection_bound,
                        packet.box_bounds,
                    )
                )
                + _bytes((packet.points, packet.projection, packet.boxes, guard))
                + executable_bytes
            )
            report.update(
                executable_bytes=executable_bytes,
                retained_payload_and_executable_bytes=total,
                resource_scope="conservative sum of current and other live issued storage, no alias credit; Python/compiler allocator, unrelated engine state and process peaks separate",
            )
            origin = {
                "scope": "issued_carried_trajectory"
                if _carry is not None
                else "external_zero_error_state_domain",
                "carry_token": _carry.token if _carry is not None else None,
                "moment_NN_permit": _carry._nn_permit.token if _carry is not None else None,
                "bound_identity": _bound_identity(declared),
                "source_offset": packet.source_offset,
            }
            token = self._next_token(
                "certificate",
                {
                    "packet": packet.token,
                    "graph": report["emitted_graph_id"],
                    "state_origin": origin,
                },
            )
            report["state_origin"] = origin
            report["certificate_token"] = token
            report["certificate_issued"] = True
            report["compile_and_binding_s"] = time.perf_counter() - started
            self._budget(
                total + len(json.dumps(report, sort_keys=True, allow_nan=False).encode()),
                "block certificate+executable+proof budget exceeded",
            )
            positive(len(self._certificates) + 1, 16, "live block certificates")
            receipt = BlockCertificate(
                packet,
                declared,
                guard,
                output,
                executable,
                token,
                _carry,
                _signature(_carry.state) if _carry is not None else (),
                self,
                _signature(guard),
                json.dumps(report, sort_keys=True, allow_nan=False),
                nn_output,
            )
            self._certificates[id(receipt)] = receipt
            receipt.validate()
            return receipt, report
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            report.update(
                certificate_issued=False, reason=str(error), exception=type(error).__name__
            )
            return None, report

    def run(self, state, certificate):
        certificate.validate()
        _require(certificate._engine is self, "same-engine block certificate required")
        packet = certificate.packet
        if certificate.carry is not None:
            _require(
                state is certificate.carry.state
                and _signature(state) == certificate.carried_state_signature,
                "carried certificate requires the exact issued trajectory state",
            )
        self.context._check_args(
            state.params, packet.entries[0][0].points, packet.entries[0][0].projection
        )
        positive(len(self._outcomes) + 1, 16, "live owned outcomes")
        estimate = packet.report["logical_payload_estimate"]
        self._budget(
            certificate.report["retained_payload_and_executable_bytes"]
            + self._additional_storage(packet=packet, certificate=certificate)
            + estimate["trace_output_bytes"],
            "owned full-history execution payload budget exceeded",
        )
        # The exact guard is part of the compiled scan, so each call rechecks all
        # actual initial state/moments/counter/input/threshold values on DEVICE.
        trace = jax.block_until_ready(
            certificate.executable(
                state, packet.points, packet.projection, packet.boxes, certificate.guard
            )
        )
        report = {
            "contract": CONTRACT,
            "certificate_token": certificate.token,
            "actual_single_call_gradients": True,
            "history_bytes": _bytes(trace),
            "runtime_containment_guard_executed": True,
            "arithmetic_certificate_applies": bool(trace.entry_valid),
            "source_offset": packet.source_offset,
            "timing_scope": "full audit-returning graph; all output traffic must be charged",
            "public_integration": False,
            "performance_evidence": False,
        }
        result = Outcome(
            trace, certificate, self, _signature(trace), json.dumps(report, sort_keys=True)
        )
        self._outcomes[id(result)] = result
        return result.validate()

    def _source_record(self, outcome, kind):
        outcome.validate()
        packet = outcome.certificate.packet
        return {
            "kind": kind,
            "engine_id": self.engine_id,
            "certificate_token": outcome.certificate.token,
            "packet_token": packet.token,
            "graph_id": outcome.certificate.report["emitted_graph_id"],
            "state_origin": outcome.certificate.report["state_origin"],
            "source_offset": packet.source_offset,
            "steps": packet.steps,
            "failed_index": int(outcome.trace.failed_index),
            "receipt_ids": packet.report["receipt_ids"],
        }

    def continue_ok(self, outcome, next_packet):
        """Detach a successful final-state proof; do not adopt or replay an update."""
        outcome.validate()
        next_packet.validate()
        report = {
            "contract": CONTRACT,
            "carry_issued": False,
            "public_integration": False,
            "performance_evidence": False,
            "additional_VG": 0,
            "update_adopted": False,
        }
        started = time.perf_counter()
        try:
            _require(
                outcome._engine is self and next_packet._engine is self,
                "same-engine outcome and next packet required",
            )
            previous = self._continued.get(id(outcome))
            _require(
                previous is None or previous() is not outcome,
                "continuation from this successful outcome was already issued",
            )
            _require(
                bool(outcome.trace.entry_valid)
                and int(outcome.trace.status) == device_training.OK
                and int(outcome.trace.failed_index) == -1,
                "actual successful complete block required",
            )
            origin = self._source_record(outcome, "OK")
            cursor = origin["source_offset"] + origin["steps"]
            _require(next_packet.source_offset == cursor, "next packet has wrong source offset")
            declared = outcome.certificate.trace_bounds.final_state
            _validate_bounds(declared)
            self._require_carried_domain(
                declared.params, next_packet.entries[0][1].domain.parameters
            )
            self._budget(
                self._base_storage(next_packet.entries)
                + self._additional_storage()
                + _bytes(outcome.trace.final_state)
                + _bound_bytes(declared),
                "compact successful carry ownership budget exceeded",
            )
            # Existing exact device-copy ownership helper, without enabling donation.
            state, copy_report = device_training.copy_state_for_donation(
                outcome.trace.final_state, synchronize=True
            )
            token_id = self._next_token(
                "carry",
                {
                    "kind": "OK",
                    "source": origin,
                    "next_packet": next_packet.token,
                    "bounds": _bound_identity(declared),
                },
            )
            report.update(
                carry_issued=True,
                carry_token=token_id,
                next_index=cursor,
                source=origin,
                new_receipts=next_packet.report["receipt_ids"],
                carried_error_preserved=True,
                compact_parent_detached=True,
                independent_copy=copy_report,
                setup_s=time.perf_counter() - started,
            )
            positive(len(self._carries) + 1, 16, "live successful carries")
            self._budget(
                self._base_storage(next_packet.entries)
                + self._additional_storage()
                + _bytes(state)
                + _bound_bytes(declared)
                + len(json.dumps(origin, sort_keys=True).encode())
                + len(json.dumps(report, sort_keys=True).encode()),
                "compact carry report/owner budget exceeded",
            )
            link_started = time.perf_counter()
            nn_permit = self._moment_authority.carry(outcome, state, declared, group=0)
            report["moment_authority_copy_link_s"] = time.perf_counter() - link_started
            report["setup_s"] = time.perf_counter() - started
            token = Carry(
                state,
                declared,
                next_packet,
                cursor,
                json.dumps(origin, sort_keys=True),
                self,
                _signature(state),
                json.dumps(report, sort_keys=True),
                nn_permit,
            )
            self._carries[id(token)] = token
            token.validate()
            key = id(outcome)
            self._continued[key] = weakref.ref(outcome, lambda _ref: self._continued.pop(key, None))
            return token, report
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            report.update(
                carry_issued=False,
                reason=str(error),
                exception=type(error).__name__,
                setup_s=time.perf_counter() - started,
            )
            return None, report

    @staticmethod
    def _require_carried_domain(carried_params, replacements):
        for carried_box, replacement in zip(carried_params, replacements, strict=True):
            _require(
                np.all(carried_box.lo >= replacement.lo)
                and np.all(carried_box.hi <= replacement.hi),
                "carried ideal proposal exceeds replacement domain",
            )
            enlarged_lo, enlarged_hi = carried_box.enlarged()
            _require(
                np.all(enlarged_lo >= replacement.lo) and np.all(enlarged_hi <= replacement.hi),
                "carried proposal error exceeds replacement domain",
            )

    def recenter(self, outcome, *, reusable_domains=False):
        """Qualify finite proposal, adopt it exactly once, resume after failed slot.

        This explicit DOMAIN_EXIT boundary reads proposed state/delta to CPU;
        normal execution never hashes or reads full live parameters/inputs.
        Returns (Resume|None, report). Refusal never changes last accepted state.
        """
        outcome.validate()
        _require(
            outcome._engine is self and type(reusable_domains) is bool,
            "same-engine outcome and explicit policy required",
        )
        trace, old = outcome.trace, outcome.certificate.packet
        failed = int(trace.failed_index)
        report = {
            "contract": CONTRACT,
            "resume_issued": False,
            "accepted_state_preserved": True,
            "public_integration": False,
            "performance_evidence": False,
        }
        started = time.perf_counter()
        try:
            previous_commit = self._committed.get(id(outcome))
            _require(
                previous_commit is None or previous_commit() is not outcome,
                "proposal from this outcome was already adopted",
            )
            _require(
                int(trace.status) == device_training.DOMAIN_EXIT and bool(trace.entry_valid),
                "certified actual DOMAIN_EXIT required",
            )
            begin = resume_start(failed, old.steps)
            proposed = jax.tree.map(lambda a: a[failed], trace.history.proposed)
            delta = jax.tree.map(lambda a: a[failed], trace.history.delta)
            _require(
                bool(device_training.finite((proposed, delta))),
                "finite proposed optimizer state required",
            )
            carried = _owned(
                jax.tree.map(
                    lambda b: bounds.Bound(b.lo[failed], b.hi[failed], b.error[failed]),
                    outcome.certificate.trace_bounds.history.proposed,
                )
            )
            # Explicitly charged full proposal/delta readback ONLY for recentering.
            host_params = tuple(np.asarray(a) for a in jax.device_get(proposed.params))
            host_delta = tuple(np.asarray(a) for a in jax.device_get(delta))
            previous = old.entries[failed][1].domain.parameters
            boxes = []
            for p, d, before in zip(host_params, host_delta, previous, strict=True):
                with np.errstate(over="raise", invalid="raise"):
                    radius = outward_radius(before.lo, before.hi, d)
                    lo, hi = np.nextafter(p - radius, -np.inf), np.nextafter(p + radius, np.inf)
                _require(np.isfinite(lo).all() and np.isfinite(hi).all(), "recenter overflow")
                boxes.append(bounds.Bound(lo, hi, 0.0))
            # Guard the originating ideal ranges AND their carried arithmetic
            # uncertainty, separately from actual proposed-point containment.
            self._require_carried_domain(carried.params, boxes)
            new = []
            for batch, prior in old.entries[failed:]:
                domain = planner.Domain(tuple(boxes), prior.domain.inputs)
                receipt = (
                    self.context.qualify_reusable_domain(proposed.params, batch, domain)
                    if reusable_domains
                    else self.context.qualify(
                        proposed.params,
                        batch,
                        planner.Domain(tuple(boxes), bounds.Bound.point(batch.host_points)),
                    )
                )
                self.context.check_certificate(batch, receipt)
                new.append((batch, receipt))
            packet = (
                self.prepare(tuple(new[1:]), source_offset=old.source_offset + begin)
                if begin < old.steps
                else None
            )
            report.update(
                resume_issued=True,
                next_index=old.source_offset + begin,
                failed_slot=failed,
                original_failed_index=old.source_offset + failed,
                original_source_offset=old.source_offset,
                terminal=packet is None,
                committed_proposal_once=True,
                additional_VG_at_recenter=0,
                old_receipts=old.report["receipt_ids"],
                new_receipts=[
                    {
                        "batch": b.token,
                        "receipt": c._key,
                        "proof": c.report.get("proof_origin", c._key),
                    }
                    for b, c in new
                ],
                requalification_receipts=len(new),
                carried_error_preserved=True,
                proposed_readback_bytes=_bytes(proposed.params) + _bytes(delta),
                recenter_and_qualification_s=time.perf_counter() - started,
                scope="adopt certified finite proposal after failed-slot and suffix requalification; suffix starts failed+1, no repeated update",
            )
            positive(len(self._resumes) + 1, 16, "live resume receipts")
            origin = self._source_record(outcome, "DOMAIN_EXIT")
            self._budget(
                self._base_storage(tuple(new))
                + self._additional_storage()
                + _bytes(proposed)
                + _bound_bytes(carried),
                "compact resume copy budget exceeded",
            )
            proposed, copy_report = device_training.copy_state_for_donation(
                proposed, synchronize=True
            )
            token_id = self._next_token(
                "carry",
                {
                    "kind": "DOMAIN_EXIT",
                    "source": origin,
                    "next_packet": packet.token if packet is not None else None,
                    "failed_receipt": new[0][1]._key,
                    "bounds": _bound_identity(carried),
                },
            )
            report.update(
                carry_token=token_id,
                source=origin,
                compact_parent_detached=True,
                independent_copy=copy_report,
                recenter_and_qualification_s=time.perf_counter() - started,
            )
            self._budget(
                self._base_storage(tuple(new))
                + self._additional_storage()
                + _bytes(proposed)
                + _bound_bytes(carried)
                + len(json.dumps(origin, sort_keys=True).encode())
                + len(json.dumps(report, sort_keys=True).encode()),
                "compact resume report/owner budget exceeded",
            )
            link_started = time.perf_counter()
            nn_permit = self._moment_authority.carry(
                outcome, proposed, carried, group=2, index=failed)
            report["moment_authority_copy_link_s"] = time.perf_counter() - link_started
            report["recenter_and_qualification_s"] = time.perf_counter() - started
            report["radius_contract"] = RADIUS_CONTRACT
            token = Resume(
                proposed,
                carried,
                packet,
                new[0][1],
                old.source_offset + begin,
                json.dumps(origin, sort_keys=True),
                self,
                _signature(proposed),
                json.dumps(report, sort_keys=True),
                nn_permit,
            )
            self._resumes[id(token)] = token
            token.validate()
            key = id(outcome)
            self._committed[key] = weakref.ref(outcome, lambda _ref: self._committed.pop(key, None))
            return token, report
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            report.update(
                resume_issued=False,
                reason=str(error),
                exception=type(error).__name__,
                recenter_and_qualification_s=time.perf_counter() - started,
            )
            return None, report

    def bind_terminal_resume(self, token, next_packet):
        """Bind an already-adopted last-slot proposal to its next owned packet.

        No update, evaluation, qualification, state copy or Bound conversion is
        performed. The old terminal Resume remains inspectable. This operation
        supplies missing packet authority, never mathematical proof for a scan.
        The resulting Resume must pass the ordinary qualify/run entry checks.
        """
        report = {
            "contract": CONTRACT,
            "resume_bound": False,
            "update_adopted": False,
            "additional_VG": 0,
            "additional_optimizer_updates": 0,
            "original_adopted_state_preserved": True,
            "public_integration": False,
            "performance_evidence": False,
        }
        issued = None
        nn_permit = None
        started = time.perf_counter()
        try:
            self.validate()
            _require(type(token) is Resume, "exact issued terminal Resume required")
            token.validate()
            _require(token._engine is self, "same-engine terminal Resume required")
            _require(token.packet is None, "only a terminal Resume can bind a new packet")
            prior = self._terminal_bindings.get(id(token))
            _require(
                prior is None or prior() is not token,
                "terminal Resume was already bound to a next packet",
            )
            _require(type(next_packet) is Packet, "exact issued next Packet required")
            next_packet.validate()
            _require(
                next_packet._engine is self
                and next_packet.source_offset == token.next_index,
                "next packet has different engine or global source offset",
            )
            origin = token.source_record
            _require(
                origin["kind"] == "DOMAIN_EXIT"
                and origin["failed_index"] == origin["steps"] - 1
                and token.next_index == origin["source_offset"] + origin["steps"],
                "terminal Resume must originate at the actual packet's last slot",
            )
            old_report = token.report
            _require(
                old_report["terminal"] is True
                and old_report["committed_proposal_once"] is True
                and old_report["additional_VG_at_recenter"] == 0
                and old_report["carried_error_preserved"] is True,
                "issued recenter/adoption report required",
            )
            # These are already-issued next-batch certificates. Checking their
            # domains does not qualify a new domain or compute a new gradient.
            for _, receipt in next_packet.entries:
                self._require_carried_domain(token.state_bounds.params, receipt.domain.parameters)
            positive(len(self._resumes) + 1, 16, "live resume receipts")
            positive(len(self._terminal_bindings) + 1, 16, "live terminal binding origins")
            report.update(
                resume_bound=True,
                source_resume_token=token.token,
                source=origin,
                next_index=token.next_index,
                next_packet_token=next_packet.token,
                new_receipts=next_packet.report["receipt_ids"],
                failed_receipt=token.failed_receipt._key,
                committed_proposal_once=True,
                carried_error_preserved=True,
                same_state_object=True,
                same_bound_objects=True,
                terminal=False,
            )
            report["carry_token"] = self._next_token(
                "carry",
                {
                    "kind": "TERMINAL_RESUME_BINDING",
                    "source_resume": token.token,
                    "source": origin,
                    "next_packet": next_packet.token,
                    "failed_receipt": token.failed_receipt._key,
                    "bounds": _bound_identity(token.state_bounds),
                },
            )
            report["binding_pre_issuance_s"] = time.perf_counter() - started
            report["cost_scope"] = (
                "nested observation before final budget/registration; whole-call "
                "binding cost must be measured by the owned outer phase"
            )
            source_json = token._source_json
            report_json = json.dumps(report, sort_keys=True)
            self._budget(
                self._base_storage(next_packet.entries)
                + self._additional_storage()
                + _bytes(token.state)
                + _bound_bytes(token.state_bounds)
                + self.context._certificate_payload_bytes(token.failed_receipt)
                + len(source_json.encode())
                + len(report_json.encode()),
                "terminal binding ownership/report budget exceeded",
            )
            nn_permit = self._moment_authority.rebind(
                token._nn_permit, token.state, token.state_bounds)
            issued = Resume(
                token.state,
                token.state_bounds,
                next_packet,
                token.failed_receipt,
                token.next_index,
                source_json,
                self,
                token._signature,
                report_json,
                nn_permit,
            )
            self._resumes[id(issued)] = issued
            issued.validate()
            key = id(token)
            self._terminal_bindings[key] = weakref.ref(
                token, lambda _ref: self._terminal_bindings.pop(key, None)
            )
            return issued, report
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            if issued is not None:
                self._resumes.pop(id(issued), None)
            if nn_permit is not None:
                self._moment_authority._abandon_unpublished(nn_permit)
            report.update(
                resume_bound=False,
                reason=str(error),
                exception=type(error).__name__,
                failed_binding_observation_s=time.perf_counter() - started,
            )
            return None, report

    def qualify_carry(self, token):
        _require(isinstance(token, (Carry, Resume)), "issued continuation token type required")
        token.validate()
        _require(token._engine is self, "same-engine continuation token required")
        if token.packet is None:
            return None, {
                "completed_after_adopting_last_proposal": True,
                "next_index": token.next_index,
                "certificate_issued": False,
                "public_integration": False,
                "terminal": True,
                "additional_VG": 0,
                "committed_state": "issued Resume.state",
                "carry_token": token.token,
            }
        return self.qualify(token.state, token.packet, token.state_bounds, _carry=token)

    def qualify_resume(self, token):
        _require(isinstance(token, Resume), "issued DOMAIN_EXIT resume token required")
        return self.qualify_carry(token)

    def storage_report(self):
        """Logical retained payload only, including externally retained issued objects.

        Shared owners are deliberately overcounted across packet/certificate
        categories; interpreter arithmetic temporaries/compiler/Python/RSS remain
        separate. A per-call payload cap is not a whole-process allocator cap.
        """
        self.validate()
        packets = list(self._packets.values())
        certificates = list(self._certificates.values())
        outcomes = list(self._outcomes.values())
        resumes = list(self._resumes.values())
        carries = list(self._carries.values())
        return {
            "live_packets": len(packets),
            "live_certificates": len(certificates),
            "live_outcomes": len(outcomes),
            "live_resumes": len(resumes),
            "live_successful_carries": len(carries),
            "engine_id": self.engine_id,
            "monotonic_next_issuance": dict(self._issuance),
            "packet_stack_and_bound_bytes": sum(
                _bytes((p.points, p.projection, p.boxes))
                + _bound_bytes((p.point_bound, p.projection_bound, p.box_bounds))
                for p in packets
            ),
            "certificate_conservative_total_bytes": sum(
                c.report["retained_payload_and_executable_bytes"] for c in certificates
            ),
            "actual_outcome_history_bytes": sum(_bytes(o.trace) for o in outcomes),
            "resume_state_and_bound_bytes": sum(
                _bytes(r.state)
                + _bound_bytes(r.state_bounds)
                + self.context._certificate_payload_bytes(r.failed_receipt)
                + len(r._source_json.encode())
                + len(r._report_json.encode())
                for r in resumes
            ),
            "compact_successful_carry_bytes": sum(
                _bytes(c.state)
                + _bound_bytes(c.state_bounds)
                + len(c._source_json.encode())
                + len(c._report_json.encode())
                for c in carries
            ),
            "moment_authority": self._moment_authority.storage_report(),
            "moment_authority_reserved_metadata_capacity_bytes": 8 * 1024**2,
            "moment_and_schedule_reserved_bytes": self.moment_storage_bytes(),
            "optimizer_kind": self.optimizer_kind,
            "optimizer_schedule_storage": self.optimizer.storage_report()
                if self.optimizer_kind == "schedule" else None,
            "context": self.context.storage_report(),
            "scope": "conservative retained sums; shared storage may repeat; per-call cap not RSS guarantee",
            "donation_or_overlap_claim": False,
            "public_integration": False,
        }

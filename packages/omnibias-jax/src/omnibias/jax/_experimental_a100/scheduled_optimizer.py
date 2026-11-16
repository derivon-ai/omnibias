# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private source-only fixed-beta schedule candidate; no promotion/default change.

The INTERNAL generated VG has the same trusted mathematical boundary as R2.
The known builder constructs every power/error constant; arbitrary replacement
tables, live beta/hypergradients and arbitrary callback equivalence are excluded.
Final source composition and actual emitted-graph/backend tests remain required.
"""

import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from . import _adam_bias_schedule_math as schedule_math
from . import bounds, device_training
from .bounded_optimizer import BoundedTraceAdam, TraceStep, _require

CONTRACT = "owned-adam-bias-schedule/actual-trace/v1"
MATH_SOURCE_SHA256 = "d89edb6a4c6e84c749b0a4beb43bc09032fff5debda18f27b0ffd6639e58f9d3"
R2_SOURCE_SHA256 = "3524a6436fb93a18bd6e4e2466708dc1a857f4a337482995cb8cbcf9444972e1"
MAX_PAYLOAD_BYTES = 3 * 1024**3


def _outward_fraction(value):
    """Exact rational in[0,1] -> outward binary64 endpoints, without native pow."""
    _require(type(value) is Fraction and 0 <= value <= 1, "fraction in[0,1] required")
    rounded = schedule_math.round_nonnegative_binary64(value)
    represented = float(rounded)
    _require(Fraction.from_float(represented) == rounded, "exact dyadic serialization required")
    lower = math.nextafter(represented, -math.inf) if rounded > value else represented
    upper = math.nextafter(represented, math.inf) if rounded < value else represented
    _require(math.isfinite(lower) and math.isfinite(upper), "outward fraction overflow")
    _require(
        Fraction.from_float(lower) <= value <= Fraction.from_float(upper),
        "outward fraction failure",
    )
    return lower, upper


def _prepared_upper(start, entries, headroom):
    _require(type(start) is int and start >= 0, "nonnegative prepared start required")
    _require(
        type(entries) is int and 1 <= entries <= schedule_math.MAX_ENTRIES,
        "bounded prepared entries required",
    )
    _require(type(headroom) is int and headroom > 0, "positive integer headroom required")
    return None if headroom > entries else start + entries - headroom


def _immutable_vector(values):
    """Internal originating values; no external array is silently snapshotted."""
    raw = np.asarray(tuple(values), dtype=np.float64).tobytes(order="C")
    return np.frombuffer(raw, dtype=np.float64)


def _fraction_records(schedule):
    return tuple(
        {
            "exponent": entry.exponent,
            "beta_hex": entry.beta_hex,
            "stored_hex": entry.stored_hex,
            "ideal_numerator_hex": hex(entry.ideal.numerator),
            "ideal_denominator_hex": hex(entry.ideal.denominator),
            "absolute_error_numerator_hex": hex(entry.absolute_error.numerator),
            "absolute_error_denominator_hex": hex(entry.absolute_error.denominator),
            "origin": entry.origin,
            "slot": entry.slot,
        }
        for row in schedule.entries
        for entry in row
    )


def _bind_origin_transfer(binding, value, snapshot, bound, prior_tables):
    """Different logical constants cannot reuse one device-object authority.

    Immutable storage may be physically shared by distinct JAX objects. This
    checks logical object provenance, not physical-buffer uniqueness.
    """
    _require(
        not any(value is table.values for table in prior_tables),
        "different power-table origins cannot alias one device object",
    )
    return binding.bind_owned_transfer(value, snapshot, bound)


def _exact_record_size(schedule):
    """Conservative retained Python-record estimate, not whole-process memory."""
    total = (
        sys.getsizeof(schedule) + sys.getsizeof(schedule.__dict__) + sys.getsizeof(schedule.entries)
    )
    for value in (
        schedule.start_accepted_steps,
        schedule.maximum_accepted_steps,
        schedule.bits,
        schedule.beta1_hex,
        schedule.beta2_hex,
        schedule.origin,
        schedule.ideal_storage_bit_upper_bound,
    ):
        total += sys.getsizeof(value)
    for row in schedule.entries:
        total += sys.getsizeof(row)
        for entry in row:
            total += sys.getsizeof(entry) + sys.getsizeof(entry.__dict__)
            for value in (
                entry.exponent,
                entry.bits,
                entry.beta_hex,
                entry.stored_hex,
                entry.origin,
                entry.slot,
            ):
                total += sys.getsizeof(value)
            for value in (entry.ideal, entry.absolute_error):
                total += (
                    sys.getsizeof(value)
                    + sys.getsizeof(value.numerator)
                    + sys.getsizeof(value.denominator)
                )
    return total


@dataclass(frozen=True, eq=False)
class _BiasTable:
    values: object
    snapshot: object
    bound: object
    channel: int
    slots: tuple


@dataclass(frozen=True, kw_only=True)
class ScheduleTraceAdam(BoundedTraceAdam):
    """Explicit opt-in candidate. The existing R2 class/default is unchanged.

    `schedule_entries` and its exact-arithmetic budget are frozen setup choices.
    Exhaustion refuses before VG; it never extends/rebuilds a table implicitly.
    Current sealed contexts/factories do not accept this new exact type yet.
    """

    schedule_entries: int
    schedule_start_accepted_steps: int = 0
    ideal_storage_bit_limit: int = 1_000_000
    max_schedule_payload_bytes: int = MAX_PAYLOAD_BYTES
    _schedule: object = field(init=False, repr=False, compare=False)
    _tables: tuple = field(init=False, repr=False, compare=False)
    _schedule_bindings: object = field(init=False, repr=False, compare=False)
    _table_identity: tuple = field(init=False, repr=False, compare=False)
    _retained_python_record_bytes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        _require(jax.config.x64_enabled, "binary64 configuration required before schedule setup")
        _require(
            type(self.max_schedule_payload_bytes) is int
            and 1 <= self.max_schedule_payload_bytes <= MAX_PAYLOAD_BYTES,
            "schedule payload budget exceeds hard cap3GiB",
        )
        _require(
            type(self.schedule_entries) is int
            and 1 <= self.schedule_entries <= schedule_math.MAX_ENTRIES,
            "bounded declared schedule entries required",
        )
        # Two CPU value vectors + six Bound vectors + two device value vectors.
        payload = 80 * self.schedule_entries
        _require(
            payload <= self.max_schedule_payload_bytes, "schedule retained payload budget exhausted"
        )
        schedule = schedule_math.prepare_schedule(
            beta1=self.beta1,
            beta2=self.beta2,
            start_accepted_steps=self.schedule_start_accepted_steps,
            count=self.schedule_entries,
            maximum_accepted_steps=self.max_accepted_steps,
            ideal_storage_bit_limit=self.ideal_storage_bit_limit,
        )
        tables, binding = [], bounds.ConstantBindings()
        for channel in range(2):
            entries = tuple(row[channel] for row in schedule.entries)
            snapshot = _immutable_vector(float.fromhex(entry.stored_hex) for entry in entries)
            lower, upper, errors = [], [], []
            for entry in entries:
                lo, hi = _outward_fraction(entry.ideal)
                lower.append(lo)
                upper.append(hi)
                errors.append(_outward_fraction(entry.absolute_error)[1])
            bound = bounds.immutable_bound(
                bounds.Bound(
                    np.asarray(lower, dtype=np.float64),
                    np.asarray(upper, dtype=np.float64),
                    np.asarray(errors, dtype=np.float64),
                )
            )
            value = jax.device_put(snapshot)  # The originating immutable bytes, no arithmetic.
            _bind_origin_transfer(binding, value, snapshot, bound, tables)
            tables.append(
                _BiasTable(
                    value, snapshot, bound, channel + 1, tuple(entry.slot for entry in entries)
                )
            )
        object.__setattr__(self, "_schedule", schedule)
        object.__setattr__(self, "_tables", tuple(tables))
        object.__setattr__(self, "_schedule_bindings", binding)
        object.__setattr__(
            self,
            "_table_identity",
            tuple(
                (
                    id(table),
                    id(table.values),
                    id(table.snapshot),
                    id(table.bound),
                    table.channel,
                    table.slots,
                )
                for table in tables
            ),
        )
        object.__setattr__(self, "_retained_python_record_bytes", _exact_record_size(schedule))
        self.validate_schedule()

    def validate_schedule(self):
        """Setup/reuse metadata checks; no live parameter hashing/GPU readback.

        A compiled device call cannot execute this host method. The final issued
        context must call it before cached-certificate reuse/dispatch. Deliberate
        private-field bypass is outside this ownership contract.
        """
        _require(
            type(self._tables) is tuple and len(self._tables) == 2,
            "complete owned two-table schedule required",
        )
        identity = tuple(
            (
                id(table),
                id(table.values),
                id(table.snapshot),
                id(table.bound),
                table.channel,
                table.slots,
            )
            for table in self._tables
        )
        _require(identity == self._table_identity, "originating schedule owners changed")
        for table in self._tables:
            _require(
                type(table) is _BiasTable and len(table.slots) == self.schedule_entries,
                "originating table metadata differs",
            )
            _require(
                bounds.immutable_bound(table.bound) is table.bound,
                "owned immutable table bounds required",
            )
            _require(
                self._schedule_bindings.owned_host_snapshot(table.values) is table.snapshot,
                "owned originating table transfer required",
            )
            _require(
                table.snapshot.shape == (self.schedule_entries,)
                and table.snapshot.dtype == np.dtype("float64"),
                "complete binary64 schedule vector required",
            )
        return self

    @property
    def constant_bounds(self):
        self.validate_schedule()
        return bounds.ConstantBindings().extend(self._schedule_bindings)

    @property
    def analysis_contract(self):
        return {
            **super().analysis_contract,
            "bias_power_contract": CONTRACT,
            "live_beta_or_hypergradient_support": False,
            "table_error_reset": False,
        }

    def counter_valid(self, counter, *, headroom=1):
        self.validate_schedule()
        ordinary = super().counter_valid(counter, headroom=headroom)
        upper = _prepared_upper(self.schedule_start_accepted_steps, self.schedule_entries, headroom)
        if upper is None:
            return ordinary & jnp.asarray(False)
        return ordinary & (counter >= self.schedule_start_accepted_steps) & (counter <= upper)

    def _metadata(self, state, points, projection, boxes):
        self.validate_schedule()
        return super()._metadata(state, points, projection, boxes)

    def _lookup(self, channel, next_count):
        """INTERNAL: counter_valid already established current range/headroom."""
        index = next_count - jnp.asarray(
            self.schedule_start_accepted_steps + 1, dtype=next_count.dtype
        )
        index = jax.lax.convert_element_type(index, np.int32)
        return jax.lax.dynamic_slice_in_dim(self._tables[channel].values, index, 1, axis=0).reshape(
            ()
        )

    def storage_report(self, *, include_exact_records=False):
        self.validate_schedule()
        return {
            "contract": CONTRACT,
            "math_source_sha256": MATH_SOURCE_SHA256,
            "parent_optimizer_source_sha256": R2_SOURCE_SHA256,
            "origin": self._schedule.origin,
            "beta_hex": [self.beta1.hex(), self.beta2.hex()],
            "start_accepted_steps": self.schedule_start_accepted_steps,
            "prepared_entries": self.schedule_entries,
            "maximum_accepted_steps": self.max_accepted_steps,
            "bits_of_reproduced_R2_circuit": self.bits,
            "ideal_storage_bit_limit": self.ideal_storage_bit_limit,
            "ideal_storage_bit_upper_bound": self._schedule.ideal_storage_bit_upper_bound,
            "retained_host_array_payload_bytes": 64 * self.schedule_entries,
            "retained_device_array_payload_bytes": 16 * self.schedule_entries,
            "retained_Python_record_bytes_conservative_estimate": self._retained_python_record_bytes,
            "payload_preflight_bytes": 80 * self.schedule_entries,
            "payload_limit_bytes": self.max_schedule_payload_bytes,
            "resource_scope": "retained schedule array payload only; exact-record estimate is separate; VG constants, interpreter/executable/temporary/transfer/setup/whole-process peaks are not bounded by this preflight",
            "additional_unmeasured_metadata": "binding dictionaries, array object headers, owner objects, generated report serialization and transient construction storage are outside the retained-array preflight; final composition must account for these separately",
            "entry_slot_origins": [list(table.slots) for table in self._tables],
            "exact_records": list(_fraction_records(self._schedule))
            if include_exact_records
            else None,
            "ownership_scope": "owned immutable bytes and device transfers; metadata checked before setup/reuse by the final context; no per-call live parameter hashing/readback",
            "counter_policy": "accepted count; rejected attempts consume no entry; no implicit extension; T-headroom once at block entry and inner default1",
            "runtime_input_containment_checked": False,
            "certificate_issued": False,
            "public_integration": False,
            "performance_evidence": False,
        }

    def analyze_trace(self, *args, extra_bindings=None, **kwargs):
        """Analyze actual emitted lookup+VG+Adam trace, retaining conditional scope."""
        binding = self.constant_bounds
        binding.extend(extra_bindings)
        result = super().analyze_trace(*args, extra_bindings=binding, **kwargs)
        result["bias_schedule"] = self.storage_report(include_exact_records=True)
        result["assumptions"] += (
            "; immutable device transfer preserves binary64 table bytes; table constants reproduce the exact RN-even R2 fixed-bit multiplication circuit under its stated arithmetic assumptions"
        )
        return result

    def step(self, state, points, projection, boxes, *, headroom=1):
        """One actual VG evaluation, only on a valid guarded state/input/count."""
        self._metadata(state, points, projection, boxes)
        cvalid = self.counter_valid(state.accepted_steps, headroom=headroom)
        pvalid = device_training.inside(state.params, boxes.parameter_lower, boxes.parameter_upper)
        xvalid = jnp.all(
            jnp.isfinite(points) & (points >= boxes.input_lower) & (points <= boxes.input_upper)
        )
        mvalid = device_training.finite((state.first_moment, state.second_moment))
        zvalid = jnp.all(
            jnp.isfinite(projection)
            & (projection >= boxes.projection_lower)
            & (projection <= boxes.projection_upper)
        )
        valid = cvalid & pvalid & xvalid & zvalid & mvalid
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def evaluate(_):
            value, gradient = self.value_gradient(state.params, points, projection)
            _require(
                value.shape == ()
                and np.dtype(value.dtype) == np.dtype("float64")
                and (jax.tree.structure(gradient) == jax.tree.structure(state.params)),
                "generated scalar/complete gradient required",
            )
            _require(
                all(
                    (
                        g.shape == p.shape and np.dtype(g.dtype) == np.dtype("float64")
                        for g, p in zip(
                            jax.tree.leaves(gradient), jax.tree.leaves(state.params), strict=True
                        )
                    )
                ),
                "complete binary64 gradient leaves required",
            )
            count = state.accepted_steps + jnp.asarray(1, dtype=state.accepted_steps.dtype)
            beta1, beta2 = (
                jnp.asarray(self.beta1, jnp.float64),
                jnp.asarray(self.beta2, jnp.float64),
            )
            decay1, decay2 = (
                jnp.asarray(1.0, jnp.float64) - beta1,
                jnp.asarray(1.0, jnp.float64) - beta2,
            )
            first = jax.tree.map(lambda m, g: beta1 * m + decay1 * g, state.first_moment, gradient)
            second = jax.tree.map(
                lambda v, g: beta2 * v + decay2 * g * g, state.second_moment, gradient
            )
            b1 = self._lookup(0, count)
            b2 = self._lookup(1, count)
            delta = jax.tree.map(
                lambda m, v: (
                    -self.learning_rate * (m / (1 - b1)) / (jnp.sqrt(v / (1 - b2)) + self.epsilon)
                ),
                first,
                second,
            )
            proposed = device_training.TrainingState(
                jax.tree.map(jnp.add, state.params, delta), first, second, count
            )
            finite = device_training.finite((value, gradient, proposed, delta))
            in_domain = device_training.inside(
                proposed.params, boxes.parameter_lower, boxes.parameter_upper
            )
            status = jnp.where(
                finite,
                jnp.where(in_domain, device_training.OK, device_training.DOMAIN_EXIT),
                device_training.NONFINITE,
            ).astype(jnp.int32)
            accepted = jax.lax.cond(
                status == device_training.OK, lambda _: proposed, lambda _: state, None
            )
            return TraceStep(accepted, proposed, delta, value, gradient, status, jnp.asarray(True))

        def reject(_):
            status = jnp.where(
                cvalid & pvalid & xvalid & zvalid,
                device_training.NONFINITE,
                device_training.INVALID_INPUT,
            ).astype(jnp.int32)
            return TraceStep(
                state, state, zero, jnp.asarray(0.0, jnp.float64), zero, status, jnp.asarray(False)
            )

        return jax.lax.cond(valid, evaluate, reject, None)

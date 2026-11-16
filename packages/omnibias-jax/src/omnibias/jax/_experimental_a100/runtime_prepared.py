# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Proposed ridge-only runtime diagonal geometry/domain reuse; not staged or validated.

Copy this module only into an explicitly manifested private candidate package.
It requires the separate FrozenDependencies.validate_ownership hardening. Fresh
X and saved Z0 are runtime operands; only the immutable base/requested norms are closed.
The initial training interface performs one guarded Adam step per prepared batch.
It does not provide a multi-batch device block, donation, or an overlap claim.
"""

import json
import math
import time
import traceback
import weakref
from collections import OrderedDict
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import (
    bounds,
    cache_manager,
    device_training,
    planner,
    prefetch,
    providers,
    ridge_diagonal_geometry,
    streaming_objective,
)
from .certificates import matrix_product
from .ridge_diagonal_geometry import RidgeProgram, prepare_ridge_base

PROVIDERS = frozenset(("parity", "polynomial", "sech", "recurrence", "jax_native", "jax_analytic"))
CONTRACT = "runtime-prepared-ridge-streaming/first-gradient/v2"


def _runtime_identity_start(device):
    snapshot = cache_manager.environment_snapshot()
    return snapshot, cache_manager._environment_device_for(snapshot, device)


def _runtime_identity_source(snapshot, name, source_file):
    return cache_manager._environment_source(snapshot, name, source_file)


def _runtime_identity_check(snapshot, digest, generation):
    cache_manager._require_environment_current(snapshot)
    _require(
        snapshot["digest"] == digest and snapshot["generation"] == generation,
        "runtime environment identity metadata changed",
    )


def _binding_bound_identity(value):
    return (id(value), id(value.lo), id(value.hi), id(value.error))


def _binding_table_identity(table):
    """Current metadata identity, never a device read or numerical data hash.

    A table object's id alone cannot detect replacing a slot with a different
    Bound. Track the actual entries and origin dictionaries too. Existing saved
    _Origin records independently validate every immutable Bound array layout.
    """
    return (
        id(table),
        id(table.slots),
        tuple((key, _binding_bound_identity(value)) for key, value in table.slots.items()),
        id(table.objects),
        tuple(
            (key, id(value), _binding_bound_identity(bound), digest)
            for key, (value, bound, digest) in table.objects.items()
        ),
        id(table.legacy),
        tuple((key, _binding_bound_identity(value)) for key, value in table.legacy.items()),
        id(table.owned_transfers),
        tuple((key, id(value)) for key, value in table.owned_transfers.items()),
        id(table.owned_transfer_origins),
        tuple((key, id(value)) for key, value in table.owned_transfer_origins.items()),
    )


def _freeze_binding_table(table):
    """Seal owned mapping copies after graph construction, before issuance."""
    _require(
        isinstance(table, bounds.ConstantBindings),
        "typed originating binding table required",
    )
    for name in (
        "slots",
        "objects",
        "legacy",
        "owned_transfers",
        "owned_transfer_origins",
    ):
        setattr(table, name, MappingProxyType(dict(getattr(table, name))))
    return table


def _validate_streaming_selection(
    scalar_objective, specification, chunk_rows, batch_rows, descriptor_type
):
    """Static scope decision before geometry, graph tracing or device copies."""
    if specification is None:
        _require(
            chunk_rows is None,
            "chunk rows require an explicit streaming objective descriptor",
        )
        return False
    _require(
        scalar_objective is None,
        "arbitrary scalar callable cannot be silently replaced by streaming",
    )
    _require(
        type(specification) is descriptor_type,
        "exact known streaming LeastSquares descriptor required",
    )
    _positive_integer(chunk_rows, "chunk rows", batch_rows)
    return True


class RuntimeBoxes(NamedTuple):
    """Runtime thresholds; Z0 encloses its ideal interval enlarged by roundoff."""

    parameter_lower: object
    parameter_upper: object
    input_lower: object
    input_upper: object
    projection_lower: object
    projection_upper: object

    def ordinary(self):
        return device_training.Boxes(*self[:4])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_integer(value, name, maximum=None):
    _require(type(value) is int and value > 0, name + " must be a positive integer")
    _require(maximum is None or value <= maximum, name + " exceeds supported maximum")
    return value


def _signature(array):
    """Cheap retained-layout check, deliberately not a numerical data hash."""
    _require(isinstance(array, np.ndarray), "immutable CPU ndarray required")
    owner = array
    seen = set()
    while isinstance(owner, np.ndarray) and owner.base is not None and id(owner) not in seen:
        seen.add(id(owner))
        owner = owner.base
    _require(
        isinstance(owner, bytes) and not array.flags.writeable,
        "retained CPU storage must be immutable bytes",
    )
    return (
        id(array),
        array.shape,
        array.dtype.str,
        array.strides,
        array.__array_interface__["data"][0],
        id(owner),
        len(owner),
    )


@dataclass(frozen=True)
class _Origin:
    array: object
    signature: tuple

    @classmethod
    def capture(cls, array):
        return cls(array, _signature(array))

    def validate(self):
        _require(
            _signature(self.array) == self.signature,
            "retained CPU array metadata/storage changed",
        )


def _device_signature(array):
    _require(
        isinstance(array, jax.Array) and not array.is_deleted(),
        "existing nondeleted JAX array required",
    )
    _require(
        array.is_fully_addressable and len(array.devices()) == 1,
        "one fully addressable device required",
    )
    return (id(array), array.shape, array.dtype.str, tuple(map(str, array.devices())))


def _domain_origins(domain):
    return tuple(
        _Origin.capture(array)
        for box in (*domain.parameters, domain.inputs)
        for array in (box.lo, box.hi, box.error)
    )


def _slot_bindings(closed, source):
    """Bind immutable graph-local slots once, avoiding constant GPU hashes in proofs.

    These path labels deliberately match bounds.interpret. Unknown primitive
    families remain interpreter refusals; they never acquire a fabricated rule.
    """
    result, constants = bounds.ConstantBindings(), []

    def visit(value, path=()):
        graph = getattr(value, "jaxpr", value)
        for index, constant in enumerate(getattr(value, "consts", ())):
            bound = source.resolve(constant, path, index)
            if bound is None:
                raise NotImplementedError(
                    "unowned emitted constant: explicit originating binding required"
                )
            result.bind_slot(path, index, bound)
            constants.append((constant, _device_signature(constant)))
        for index, equation in enumerate(graph.eqns):
            name, params = equation.primitive.name, equation.params
            if name in (
                "jit",
                "pjit",
                "custom_jvp_call",
                "custom_vjp_call",
                "closed_call",
                "remat2",
            ):
                sub = params.get("jaxpr", params.get("call_jaxpr"))
                visit(sub, path + (f"{index}:{name}",))
            elif name == "scan":
                visit(params["jaxpr"], path + (f"{index}:scan",))
            elif name == "cond":
                for branch, sub in enumerate(params["branches"]):
                    visit(sub, path + (f"{index}:cond:{branch}",))

    visit(closed)
    return result, tuple(constants)


def _literal_objective_bindings(closed):
    """Exact supplied objective data, owned by the known setup-time freezer.

    These snapshots declare binary64 data or exact boolean/integer metadata
    in the scalar objective; they are NOT cached field computations. Device reads/hashes occur at
    setup only. An equal-valued cached n0 keeps its different originating
    construction bound. Legacy value joins can widen, never erase, errors.
    """
    result = bounds.ConstantBindings()
    seen = set()

    def visit(value):
        if hasattr(value, "jaxpr"):
            for constant in getattr(value, "consts", ()):
                if id(constant) in seen:
                    continue
                seen.add(id(constant))
                supplied = np.asarray(constant)
                if supplied.dtype == np.dtype("float64"):
                    _require(
                        np.isfinite(supplied).all(),
                        "captured objective binary64 literal data must be finite",
                    )
                elif supplied.dtype.kind == "b":
                    pass  # The interpreter represents exact False/True as 0/1.
                elif supplied.dtype.kind in "iu":
                    # Compare in the originating integer dtype, before any
                    # float conversion; never abs(int64_min) or rounded equality.
                    if np.any(supplied > 2**53) or (
                        supplied.dtype.kind == "i" and np.any(supplied < -(2**53))
                    ):
                        raise NotImplementedError(
                            "objective integer metadata outside exact binary64 range"
                        )
                else:
                    raise NotImplementedError(
                        "objective literals require float64 data or exact boolean/integer metadata"
                    )
                result.bind_object(constant, bounds.Bound.point(supplied))
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            for equation in value.eqns:
                visit(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(closed)
    return result


def _owned_domain(domain):
    _require(isinstance(domain, planner.Domain), "explicit parameter/real-X Domain required")
    _require(type(domain.parameters) is tuple, "explicit parameter-box tuple required")
    return planner.Domain(
        tuple(bounds.immutable_bound(box) for box in domain.parameters),
        bounds.immutable_bound(domain.inputs),
    )


def _default_objective(params, points, outputs):
    del params
    return sum(jnp.mean((value - 0.1 * points[:, 0]) ** 2) for value in outputs)


def _reject_objective_derivative_overrides(graph):
    """A full scalar objective is supported; unproved derivative overrides are not."""

    def walk(value):
        if hasattr(value, "jaxpr"):
            walk(value.jaxpr)
        elif hasattr(value, "eqns"):
            for equation in value.eqns:
                if equation.primitive.name in (
                    "stop_gradient",
                    "custom_jvp_call",
                    "custom_vjp_call",
                ):
                    raise NotImplementedError(
                        "objective derivative override outside declared contract"
                    )
                walk(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(graph)


def _validate_emitted_contract(forward, value_gradient, args, expected_output_count):
    """Metadata validation only; public context always generates its own gradient."""
    outputs = jax.eval_shape(forward, *args)
    _require(
        type(outputs) is tuple and len(outputs) == expected_output_count,
        "complete operator output tuple required",
    )
    for value in outputs:
        _require(
            value.shape == (args[1].shape[0],) and np.dtype(value.dtype) == np.dtype("float64"),
            "every operator requires a full binary64 batch vector",
        )
    packet = jax.eval_shape(value_gradient, *args)
    _require(
        type(packet) is tuple and len(packet) == 2,
        "scalar objective and complete parameter gradient required",
    )
    value, gradient = packet
    _require(
        value.shape == () and np.dtype(value.dtype) == np.dtype("float64"),
        "objective must be a binary64 scalar",
    )
    _require(
        jax.tree.structure(gradient) == jax.tree.structure(args[0]),
        "complete parameter-gradient pytree required",
    )
    actual, expected = jax.tree.leaves(gradient), jax.tree.leaves(args[0])
    _require(len(actual) == len(expected) == 5, "all five live parameter gradients required")
    for entry, parameter in zip(actual, expected, strict=True):
        _require(
            entry.shape == parameter.shape and np.dtype(entry.dtype) == np.dtype("float64"),
            "complete gradient shape/binary64 contract required",
        )
    return outputs


@dataclass(frozen=True)
class _Project:
    shape: tuple
    base_signature: tuple

    def __call__(self, points, frozen):
        _require(
            tuple(frozen) == ("base",),
            "one explicit immutable base dependency required",
        )
        _require(
            _signature(frozen["base"]) == self.base_signature,
            "frozen base metadata changed before production",
        )
        _require(
            isinstance(points, np.ndarray)
            and points.shape == self.shape
            and points.dtype == np.float64
            and np.isfinite(points).all(),
            "finite fixed-shape binary64 incoming points required",
        )
        with np.errstate(over="raise", invalid="raise"):
            projection = points @ frozen["base"].T
        _require(np.isfinite(projection).all(), "saved projection overflow")
        enclosure = matrix_product(points, frozen["base"].T)
        lower, upper = enclosure.enlarged()
        _require(
            np.all((projection >= lower) & (projection <= upper)),
            "saved projection lies outside its construction enclosure",
        )
        # Both payload and metadata are snapshotted by the existing buffer.
        return prefetch.HostPrepared(
            (points, projection),
            {
                "points": points,
                "projection": projection,
                "lo": enclosure.lo,
                "hi": enclosure.hi,
                "error": enclosure.error,
            },
        )


@dataclass(frozen=True)
class RuntimeBatch:
    points: object
    projection: object
    base_token: str
    input_token: str
    token: str
    preparation: object
    _context: object
    _origins: tuple
    _identity: tuple

    @property
    def host_points(self):
        self.validate()
        return self.preparation.host_metadata.values["points"]

    @property
    def host_projection(self):
        self.validate()
        return self.preparation.host_metadata.values["projection"]

    @property
    def projection_bound(self):
        self.validate()
        metadata = self.preparation.host_metadata.values
        return bounds.Bound(metadata["lo"], metadata["hi"], metadata["error"])

    @property
    def index(self):
        return self.preparation.index

    def validate(self):
        self._context._validate_context()
        _require(
            self._context._issued_batches.get(id(self)) is self,
            "prepared batch was not issued by this owned producer",
        )
        self.preparation.host_metadata.validate_ownership()
        for origin in self._origins:
            origin.validate()
        current = (
            self.base_token,
            self.input_token,
            self.token,
            id(self.preparation),
            _device_signature(self.points),
            _device_signature(self.projection),
        )
        _require(current == self._identity, "prepared batch token/device identity changed")
        _require(
            self.base_token == self._context.owner.frozen.token
            and self.input_token == self.preparation.input_token
            and self.base_token == self.preparation.dependency_token,
            "prepared batch belongs to different base or input",
        )
        _require(
            self.preparation.payload[0] is self.points
            and self.preparation.payload[1] is self.projection,
            "prepared device payload was replaced",
        )
        metadata = self.preparation.host_metadata.values
        _require(
            metadata["points"].shape == self.points.shape
            and metadata["projection"].shape == self.projection.shape
            and all(
                metadata[name].shape == self.projection.shape for name in ("lo", "hi", "error")
            ),
            "retained real-X/Z0 construction metadata shape differs",
        )
        self._context._check_args(None, self.points, self.projection)
        return self

    def storage_report(self):
        self.validate()
        return {
            "host_metadata_bytes": self.preparation.host_metadata.nbytes,
            "device_payload_bytes": self.points.nbytes + self.projection.nbytes,
            "host_points_retained": True,
            "input_token": self.input_token,
            "base_token": self.base_token,
            "batch_token": self.token,
            "host_preparation_s": self.preparation.host_preparation_s,
            "transfer_dispatch_s": self.preparation.transfer_dispatch_s,
            "scope": "payload/metadata only; shared base, compiler caches and process peaks separate",
        }


class _Stream:
    def __init__(self, context, batches, prefetched):
        context._validate_context()
        _require(
            context._active_stream is None,
            "only one bounded input stream per context is supported",
        )
        self._context = context
        cls = prefetch.DoubleBuffer if prefetched else prefetch.SynchronousBuffer
        producer = _Project(context.batch_shape, context._base_origin.signature)
        # Reserve the context's fixed payload and entire bounded proof cache.
        remaining = context.owner.frozen.nbytes + context._stream_host_reserve
        self._buffer = cls(
            batches,
            producer,
            context.owner.frozen,
            device=context.owner.base.device,
            max_host_bytes=remaining,
        )
        self.report = {
            "maximum_future_batches": int(prefetched),
            "host_X_and_Z0_retained": True,
            "fixed_owner_reused": True,
            "same_synchronous_and_prefetched_math": True,
            "cuda_overlap": "not established by host timelines",
        }
        self._closed = False
        context._active_stream = self

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._context._validate_context()
            preparation = next(self._buffer)
            preparation.host_metadata.validate_ownership()
            metadata = preparation.host_metadata.values
            _require(
                set(metadata) == {"points", "projection", "lo", "hi", "error"},
                "complete prepared construction metadata required",
            )
            points, projection = preparation.require_dependencies(self._context.owner.frozen)
            token = cache_manager.digest_record(
                {
                    "contract": CONTRACT,
                    "base": preparation.dependency_token,
                    "input": preparation.input_token,
                    "metadata": preparation.host_metadata.token,
                    "issuance": self._context._batch_issuance,
                }
            )
            self._context._batch_issuance += 1
            identity = (
                preparation.dependency_token,
                preparation.input_token,
                token,
                id(preparation),
                _device_signature(points),
                _device_signature(projection),
            )
            batch = RuntimeBatch(
                points,
                projection,
                preparation.dependency_token,
                preparation.input_token,
                token,
                preparation,
                self._context,
                tuple(_Origin.capture(value) for value in metadata.values()),
                identity,
            )
            self._context._issued_batches[id(batch)] = batch
            batch.validate()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._closed:
            try:
                self._buffer.close()
            finally:
                self._closed = True
                self._context._active_stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class _ReusableProof:
    domain: object
    construction: object
    boxes: object
    identity: str
    _context: object
    _report_json: str
    _origins: tuple
    _box_signatures: tuple

    @property
    def report(self):
        return json.loads(self._report_json)

    def validate(self, context):
        context._validate_context()
        _require(
            self._context is context and context._issued_reusable_proofs.get(id(self)) is self,
            "reusable proof was not issued by this exact context",
        )
        for origin in self._origins:
            origin.validate()
        arrays = [
            a
            for b in (*self.domain.parameters, self.domain.inputs, self.construction)
            for a in (b.lo, b.hi, b.error)
        ]
        _require(
            len(arrays) == len(self._origins)
            and all(a is origin.array for a, origin in zip(arrays, self._origins, strict=True)),
            "reusable proof changed its originating domain/construction arrays",
        )
        _require(
            tuple(_device_signature(a) for a in jax.tree.leaves(self.boxes))
            == self._box_signatures,
            "reusable runtime thresholds changed or were deleted",
        )
        _require(
            self.report["admissible"]
            and self.report["structure_key"] == context.structure_key
            and self.report["proof_origin"] == self.identity,
            "reusable proof is not admissible for this structure",
        )
        return self

    def retained_extra_bytes(self):
        owners = {origin.signature[-2]: origin.signature[-1] for origin in self._origins}
        return (
            sum(owners.values()) + len(self._report_json.encode()) + self.device_threshold_bytes()
        )

    def device_threshold_bytes(self):
        """Strongly retained proof thresholds, even for a narrower receipt."""
        return sum(array.nbytes for array in jax.tree.leaves(self.boxes))


@dataclass(frozen=True)
class Certificate:
    admissible: bool
    batch_token: str
    structure_key: str
    domain: object
    boxes: object
    _report_json: str
    _context: object
    _batch: RuntimeBatch
    _domain_origins: tuple
    _box_signatures: tuple
    _key: str
    _reusable_proof: object = None
    _active_construction: object = None

    @property
    def report(self):
        return json.loads(self._report_json)

    @property
    def active_construction(self):
        self.validate()
        return self._active_construction

    def validate(self):
        self._batch.validate()
        _require(
            self._context._issued_certificates.get(id(self)) is self,
            "certificate was not issued by this qualifier",
        )
        _require(
            self._context.structure_key == self.structure_key
            and self._batch.token == self.batch_token,
            "certificate belongs to a different structure/batch",
        )
        for origin in self._domain_origins:
            origin.validate()
        _require(
            tuple(_device_signature(a) for a in jax.tree.leaves(self.boxes))
            == self._box_signatures,
            "certificate runtime thresholds changed or were deleted",
        )
        _require(self.admissible is self.report["admissible"], "certificate status changed")
        if self._reusable_proof is not None:
            proof = self._reusable_proof.validate(self._context)
            _require(
                isinstance(self._active_construction, bounds.Bound)
                and self.report["proof_origin"] == proof.identity,
                "fresh receipt changed its immutable proof origin",
            )
            arrays = [
                a
                for b in (
                    *self.domain.parameters,
                    self.domain.inputs,
                    self._active_construction,
                )
                for a in (b.lo, b.hi, b.error)
            ]
            _require(
                len(arrays) == len(self._domain_origins)
                and all(
                    a is origin.array
                    for a, origin in zip(arrays, self._domain_origins, strict=True)
                ),
                "receipt changed active requested domain/construction ownership",
            )
            _require(
                all(
                    self._context._contains(outer, inner)
                    for outer, inner in zip(
                        (
                            *proof.domain.parameters,
                            proof.domain.inputs,
                            proof.construction,
                        ),
                        (
                            *self.domain.parameters,
                            self.domain.inputs,
                            self._active_construction,
                        ),
                        strict=True,
                    )
                ),
                "active requested domain exceeds containing proof",
            )
        return self

    def storage_report(self):
        self.validate()
        owners = {}
        domain_origin_count = 3 * (len(self.domain.parameters) + 1)
        for origin in self._domain_origins[:domain_origin_count]:
            owners[origin.signature[-2]] = origin.signature[-1]
        active_owners = {
            origin.signature[-2]: origin.signature[-1]
            for origin in self._domain_origins[domain_origin_count:]
        }
        return {
            "owned_domain_host_bytes": sum(owners.values()),
            "active_construction_host_bytes": sum(active_owners.values()),
            "runtime_threshold_device_bytes": sum(a.nbytes for a in jax.tree.leaves(self.boxes)),
            "report_bytes": len(self._report_json.encode()),
            "reusable_proof_payload_bytes": (
                self._reusable_proof.retained_extra_bytes()
                if self._reusable_proof is not None
                else 0
            ),
            "reusable_proof_device_threshold_bytes": (
                self._reusable_proof.device_threshold_bytes()
                if self._reusable_proof is not None
                else 0
            ),
            "threshold_accounting_scope": "active and proof thresholds charged separately; shared buffers may be counted twice",
            "batch": self._batch.storage_report(),
        }


class RuntimePreparedContext:
    """One owned executable structure, with separate per-batch certificates."""

    def __init__(
        self,
        base,
        operators,
        params,
        batch_shape,
        *,
        provider,
        scalar_objective,
        device,
        max_host_bytes,
        max_state_bytes,
        max_equations,
        max_certificate_entries,
        streaming_objective_spec=None,
        chunk_rows=None,
    ):
        if not jax.config.x64_enabled:
            raise ValueError("enable binary64 before preparing runtime inputs")
        _require(
            callable(getattr(prefetch.FrozenDependencies, "validate_ownership", None)),
            "requires separate FrozenDependencies.validate_ownership hardening",
        )
        self.max_host_bytes = _positive_integer(max_host_bytes, "host bytes", 3 * 1024**3)
        self.max_state_bytes = _positive_integer(max_state_bytes, "state bytes", 3 * 1024**3)
        self.max_equations = _positive_integer(max_equations, "equations", 100000)
        self.max_certificate_entries = _positive_integer(
            max_certificate_entries, "certificate entries", 16
        )
        _require(provider in PROVIDERS, "unsupported existing activation provider")
        self.provider = provider
        self.batch_shape = tuple(batch_shape)
        _require(
            len(self.batch_shape) == 2 and all(type(n) is int and n > 0 for n in self.batch_shape),
            "positive fixed batch shape required",
        )
        use_streaming = _validate_streaming_selection(
            scalar_objective,
            streaming_objective_spec,
            chunk_rows,
            self.batch_shape[0],
            streaming_objective.LeastSquares,
        )
        _require(
            not isinstance(base, jax.Array),
            "explicit CPU base snapshot required at preparation",
        )
        base = np.asarray(base)
        _require(
            base.ndim == 2
            and min(base.shape) > 0
            and base.dtype == np.float64
            and np.isfinite(base).all()
            and base.shape[1] == self.batch_shape[1],
            "finite binary64 base(H,D) matching points required",
        )
        # Ridge invariants are A0 and optional H row norms, never an H-by-H metric.
        # The typed factory separately checks its norm-construction scratch allowance.
        h, d = base.shape
        _require(
            base.nbytes * 2 < self.max_host_bytes,
            "owned base setup exceeds host budget",
        )
        self._operators = tuple(operators)
        _require(self._operators, "nonempty operator tuple required")
        for operator in self._operators:
            operator.check_dimension(d)
            _require(operator.order >= 0, "negative derivative order")
            if operator.order > 16 or operator.mode == "directional":
                raise NotImplementedError("existing constant-coordinate operators through16 only")
        self._parameter_shapes = self._check_parameter_shapes(params, h, d)
        self._parameter_tree = jax.tree.structure(params)
        estimated = sum(math.prod(shape) for shape in self._parameter_shapes) * 8
        estimated += 8 * self.batch_shape[0] * (d + h)
        _require(
            estimated <= self.max_state_bytes,
            "runtime argument storage exceeds state budget",
        )
        started = time.perf_counter()
        self._environment_snapshot, device = _runtime_identity_start(device)
        self._environment = self._environment_snapshot["digest"]
        self._environment_generation = self._environment_snapshot["generation"]
        frozen = prefetch.FrozenDependencies({"base": base})
        self._owner = prepare_ridge_base(
            frozen, self._operators, device=device, max_host_bytes=self.max_host_bytes
        )
        self._owner.validate()
        self._base_origin = _Origin.capture(self._owner.frozen.values["base"])
        self._fixed_origins = tuple(
            _Origin.capture(a) for a in self._owner.metadata.values.values()
        )
        geometry_arrays = (
            (self._owner.base, self._owner.norms) if self._owner.need_norms else (self._owner.base,)
        )
        self._owner_device_signatures = tuple(_device_signature(value) for value in geometry_arrays)
        context_bindings = self._owner.constant_bounds()
        operators = self._operators
        self._ridge_program = RidgeProgram(self._owner, provider=provider)
        # Only forward is used. The context below independently constructs the
        # supplied full scalar objective and its COMPLETE parameter gradient.
        raw_forward = self._ridge_program.forward
        raw_forward.constant_bounds = context_bindings
        template_x = jnp.zeros(self.batch_shape, dtype=jnp.float64)
        template_z = jnp.zeros((self.batch_shape[0], h), dtype=jnp.float64)
        args = (params, template_x, template_z)
        self._forward = planner.freeze_callable(raw_forward, args)
        self._streaming_program = None
        self._objective_only = None
        self._objective_only_graph = None
        self._objective_only_bindings = None
        self._objective_only_identity_json = None
        if not use_streaming:
            objective = _default_objective if scalar_objective is None else scalar_objective
            _require(callable(objective), "explicit full scalar objective must be callable")
            objective_args = (
                params,
                template_x,
                tuple(jnp.zeros((self.batch_shape[0],), dtype=jnp.float64) for _ in operators),
            )
            objective_graph = jax.make_jaxpr(objective)(*objective_args)
            _reject_objective_derivative_overrides(objective_graph)
            owned_objective = planner.freeze_callable(objective, objective_args)
            objective_bindings = _literal_objective_bindings(owned_objective.closed_jaxpr)
            shaped = jax.eval_shape(owned_objective, *objective_args)
            _require(
                shaped.shape == () and np.dtype(shaped.dtype) == np.dtype("float64"),
                "full scalar objective must return a binary64 scalar",
            )

            def loss(p, points, projection):
                return owned_objective(p, points, self._forward(p, points, projection))

            generated = jax.value_and_grad(loss, argnums=0)
            generated.constant_bounds = bounds.ConstantBindings().extend(
                self._forward.constant_bounds
            )
            generated.constant_bounds.extend(objective_bindings)
            objective_identity = planner._emitted_graph_identity(owned_objective.closed_jaxpr)
            self._streaming_config_json = json.dumps(
                {"mode": "full-output-objective"}, sort_keys=True
            )
        else:
            # A descriptor is the declared objective, never an inferred rewrite
            # of an arbitrary callable or the different RidgeProgram objective.
            runtime = {
                "jax": jax,
                "jnp": jnp,
                "np": np,
                "modules": {
                    "ridge_diagonal_geometry": ridge_diagonal_geometry,
                    "providers": providers,
                    "bounds": bounds,
                    "planner": planner,
                },
            }
            self._streaming_program = streaming_objective.StreamingProgram(
                runtime,
                self._owner,
                batch=self.batch_shape[0],
                chunk=chunk_rows,
                provider=provider,
                objective=streaming_objective_spec,
                max_state_bytes=self.max_state_bytes,
            )
            # Preserve n0/A0 originating errors. Generated geometry constants
            # must NOT pass through the supplied-literal-objective freezer.
            generated = self._streaming_program.value_gradient
            self._objective_only = planner.freeze_callable(self._streaming_program.objective, args)
            shaped = jax.eval_shape(self._objective_only, *args)
            _require(
                shaped.shape == () and np.dtype(shaped.dtype) == np.dtype("float64"),
                "streaming scalar objective must be binary64",
            )
            self._objective_only_graph = self._objective_only.closed_jaxpr
            objective_identity = planner._emitted_graph_identity(self._objective_only_graph)
            self._objective_only_identity_json = json.dumps(
                objective_identity, sort_keys=True, allow_nan=False
            )
            self._streaming_config_json = json.dumps(
                {
                    "mode": "explicit-lowrank-objective-streaming",
                    "objective": asdict(streaming_objective_spec),
                    "layout": dict(self._streaming_program.layout),
                    "implementation_source": _runtime_identity_source(
                        self._environment_snapshot,
                        "streaming_objective.py",
                        streaming_objective.__file__,
                    ),
                    "scalar_graph_digest": cache_manager.digest_record(objective_identity),
                },
                sort_keys=True,
                allow_nan=False,
            )
        self._value_gradient = planner.freeze_callable(generated, args)
        _validate_emitted_contract(self._forward, self._value_gradient, args, len(operators))
        self._forward_graph = self._forward.closed_jaxpr
        self._gradient_graph = self._value_gradient.closed_jaxpr
        self._bindings = bounds.ConstantBindings().extend(self._forward.constant_bounds)
        self._bindings.extend(self._value_gradient.constant_bounds)
        if self._objective_only is not None:
            self._bindings.extend(self._objective_only.constant_bounds)
        # Each actual emitted graph gets its own immutable slot table: slots
        # from different graphs must never be conflated by numerical index.
        self._forward_bindings, fconstants = _slot_bindings(self._forward_graph, self._bindings)
        self._gradient_bindings, gconstants = _slot_bindings(self._gradient_graph, self._bindings)
        oconstants = ()
        if self._objective_only_graph is not None:
            self._objective_only_bindings, oconstants = _slot_bindings(
                self._objective_only_graph, self._bindings
            )
        self._constant_signatures = tuple(
            {
                id(value): (value, signature)
                for value, signature in (*fconstants, *gconstants, *oconstants)
            }.values()
        )
        owned_device_arrays = {id(value): value for value, _ in self._constant_signatures}
        owned_device_arrays.update({id(value): value for value in geometry_arrays})
        self._owned_device_bytes = sum(value.nbytes for value in owned_device_arrays.values())
        _require(
            self._owned_device_bytes + estimated <= self.max_state_bytes,
            "owned constants and runtime arguments exceed state budget",
        )
        tables = (
            self._forward.constant_bounds,
            self._value_gradient.constant_bounds,
            self._bindings,
            self._forward_bindings,
            self._gradient_bindings,
        )
        if self._objective_only is not None:
            tables += (
                self._objective_only.constant_bounds,
                self._objective_only_bindings,
            )
        bound_values = []
        for table in tables:
            bound_values.extend(table.slots.values())
            bound_values.extend(item[1] for item in table.objects.values())
            bound_values.extend(table.legacy.values())
        self._binding_origins = tuple(
            _Origin.capture(array)
            for bound in bound_values
            for array in (bound.lo, bound.hi, bound.error)
        )
        bound_owners = {}
        for bound in bound_values:
            bound_owners.update(bounds.bound_storage_owners(bound))
        # A table ID alone would not notice ordinary slot/Bound replacement.
        # Own read-only map copies, then retain the CURRENT mapping identities.
        for table in tables:
            _freeze_binding_table(table)
        self._binding_tables_identity = self._current_binding_tables_identity()
        self._fixed_host_bytes = (
            self._owner.frozen.nbytes
            + self._owner.metadata.nbytes
            + sum(size for _, size in bound_owners.values())
        )
        self._streaming_metadata_bytes = len(self._streaming_config_json.encode()) + (
            len(self._objective_only_identity_json.encode())
            if self._objective_only_identity_json is not None
            else 0
        )
        _require(
            self._streaming_metadata_bytes <= min(self.max_host_bytes // 8, 16 * 1024**2),
            "streaming scalar graph/configuration metadata exceeds setup budget",
        )
        self._fixed_host_bytes += self._streaming_metadata_bytes
        self._stream_host_reserve = 2 * 8 * self.batch_shape[0] * (3 * d + 5 * h)
        self._certificate_budget = (
            self.max_host_bytes - self._fixed_host_bytes - self._stream_host_reserve
        )
        _require(
            self._certificate_budget > 0,
            "fixed metadata and retained-X double buffer exhaust host budget",
        )
        graphs = ((self._forward_graph, len(operators)), (self._gradient_graph, 6))
        if self._objective_only_graph is not None:
            graphs += ((self._objective_only_graph, 1),)
        for graph, count in graphs:
            _require(
                len(graph.jaxpr.invars) == 7 and len(graph.jaxpr.outvars) == count,
                "actual three-argument graph has an incomplete input/output inventory",
            )
            _require(
                len(graph.jaxpr.eqns) <= self.max_equations,
                "emitted graph budget exceeded",
            )
        self._graph_builds = 3 if self._objective_only is None else 4
        self._structure_key = cache_manager.digest_record(
            {
                "contract": CONTRACT,
                "module_source": _runtime_identity_source(
                    self._environment_snapshot, "runtime_prepared.py", __file__
                ),
                "base": self._owner.frozen.token,
                "requested_norms": self._owner.metadata.token,
                "ridge_module_source": _runtime_identity_source(
                    self._environment_snapshot,
                    "ridge_diagonal_geometry.py",
                    ridge_diagonal_geometry.__file__,
                ),
                "provider": provider,
                "operators": [operator.to_config() for operator in operators],
                "parameter_shapes": self._parameter_shapes,
                "batch_shape": self.batch_shape,
                "state_bytes": self.max_state_bytes,
                "equations": self.max_equations,
                "environment": self._environment,
                "forward": planner._emitted_graph_identity(self._forward_graph),
                "value_gradient": planner._emitted_graph_identity(self._gradient_graph),
                "objective": objective_identity,
                "streaming_configuration": json.loads(self._streaming_config_json),
            }
        )
        self._training_step = self._make_training_step()
        self._compiled = None
        self._compiled_memory = MappingProxyType({})
        self._compiled_identity = ()
        self._compile_calls = 0
        self._certificates = OrderedDict()
        self._certificate_hits = 0
        self._issued_batches = weakref.WeakValueDictionary()
        self._issued_certificates = weakref.WeakValueDictionary()
        self._issued_reusable_proofs = weakref.WeakValueDictionary()
        self._batch_issuance = 0
        self._active_stream = None
        _runtime_identity_check(
            self._environment_snapshot, self._environment, self._environment_generation
        )
        self._setup_s = time.perf_counter() - started
        self._context_metadata = self._metadata_signature()

    @staticmethod
    def _check_parameter_shapes(params, h, d):
        _require(
            type(params) is tuple and len(params) == 5,
            "live (U,V,bias,readout,offset) tuple required",
        )
        shapes = tuple(tuple(p.shape) for p in params)
        _require(
            all(np.dtype(p.dtype) == np.dtype("float64") for p in params),
            "binary64 live parameters required",
        )
        rank = shapes[0][1] if len(shapes[0]) == 2 else 0
        _require(
            rank > 0 and shapes == ((h, rank), (d, rank), (h,), (h,), ()),
            "live lowrank parameter shapes differ",
        )
        return shapes

    @property
    def owner(self):
        return self._owner

    @property
    def structure_key(self):
        return self._structure_key

    @property
    def forward(self):
        return self._forward

    @property
    def value_gradient(self):
        return self._value_gradient

    @property
    def objective_only(self):
        _require(
            self._objective_only is not None,
            "standalone scalar path requires an explicit streaming objective",
        )
        return self._objective_only

    @property
    def training_step(self):
        return self._training_step

    @property
    def graph_builds(self):
        return self._graph_builds

    @property
    def compile_calls(self):
        return self._compile_calls

    def _metadata_signature(self):
        return (
            self.provider,
            self.batch_shape,
            self.max_host_bytes,
            self.max_state_bytes,
            self.max_equations,
            self.max_certificate_entries,
            self._parameter_shapes,
            self._parameter_tree,
            self._structure_key,
            self._certificate_budget,
            self._fixed_host_bytes,
            self._stream_host_reserve,
            self._owned_device_bytes,
            id(self._owner),
            id(self._ridge_program),
            id(self._forward),
            id(self._value_gradient),
            id(self._training_step),
            id(self._forward_graph),
            id(self._gradient_graph),
            id(self._forward_bindings),
            id(self._gradient_bindings),
            id(self._streaming_program),
            self._streaming_config_json,
            id(self._environment_snapshot),
            self._environment,
            self._environment_generation,
            self._objective_only_identity_json,
            self._streaming_metadata_bytes,
            id(self._objective_only),
            id(self._objective_only_graph),
            id(self._objective_only_bindings),
        )

    def _current_binding_tables_identity(self):
        tables = (
            self._forward.constant_bounds,
            self._value_gradient.constant_bounds,
            self._bindings,
            self._forward_bindings,
            self._gradient_bindings,
        )
        if self._objective_only is not None:
            tables += (
                self._objective_only.constant_bounds,
                self._objective_only_bindings,
            )
        return tuple(_binding_table_identity(table) for table in tables)

    def _validate_context(self):
        _require(
            self._metadata_signature() == self._context_metadata,
            "runtime context structural/resource metadata changed",
        )
        _require(
            self._current_binding_tables_identity() == self._binding_tables_identity,
            "originating constant table or slot/Bound identity changed",
        )
        actual_compiled = (
            ()
            if self._compiled is None
            else tuple(
                (name, id(value), self._compiled_memory[name])
                for name, value in self._compiled.items()
            )
        )
        _require(
            len(self._compiled_memory) == len(actual_compiled) <= 4
            and actual_compiled == self._compiled_identity,
            "compiled path/cache identity or memory record changed",
        )
        self._owner.validate()
        self._base_origin.validate()
        for origin in self._fixed_origins:
            origin.validate()
        for origin in self._binding_origins:
            origin.validate()
        for constant, signature in self._constant_signatures:
            _require(
                _device_signature(constant) == signature,
                "owned emitted-graph constant was deleted or replaced",
            )
        geometry_arrays = (
            (self._owner.base, self._owner.norms) if self._owner.need_norms else (self._owner.base,)
        )
        _require(
            tuple(_device_signature(value) for value in geometry_arrays)
            == self._owner_device_signatures,
            "fixed ridge geometry device ownership changed",
        )
        _runtime_identity_check(
            self._environment_snapshot, self._environment, self._environment_generation
        )

    def _check_args(self, params, points, projection):
        _require(
            points.shape == self.batch_shape and np.dtype(points.dtype) == np.dtype("float64"),
            "runtime real-X shape/binary64 mismatch",
        )
        _require(
            projection.shape == (self.batch_shape[0], self._parameter_shapes[0][0])
            and np.dtype(projection.dtype) == np.dtype("float64"),
            "runtime Z0 shape/binary64 mismatch",
        )
        if params is not None:
            _require(
                jax.tree.structure(params) == self._parameter_tree
                and tuple(tuple(p.shape) for p in jax.tree.leaves(params)) == self._parameter_shapes
                and all(np.dtype(p.dtype) == np.dtype("float64") for p in jax.tree.leaves(params)),
                "runtime complete parameter metadata differs",
            )

    def stream(self, batches, *, prefetched=False):
        _require(type(prefetched) is bool, "explicit prefetch mode required")
        return _Stream(self, batches, prefetched)

    def make_domain(self, host_params, batch, radius=1e-5):
        """Explicit CPU preparation helper; never implicitly reads a device parameter."""
        batch.validate()
        _require(batch._context is self, "batch belongs to another context")
        _require(
            np.isfinite(radius) and radius > 0,
            "finite positive parameter radius required",
        )
        _require(
            type(host_params) is tuple
            and len(host_params) == 5
            and not any(isinstance(p, jax.Array) for p in host_params),
            "explicit CPU parameter snapshots required; no implicit device readback",
        )
        arrays = tuple(np.asarray(p) for p in host_params)
        _require(
            tuple(p.shape for p in arrays) == self._parameter_shapes
            and all(p.dtype == np.float64 and np.isfinite(p).all() for p in arrays),
            "finite binary64 CPU parameter snapshots required",
        )
        boxes = tuple(
            bounds.Bound(np.nextafter(p - radius, -np.inf), np.nextafter(p + radius, np.inf), 0.0)
            for p in arrays
        )
        return _owned_domain(planner.Domain(boxes, bounds.Bound.point(batch.host_points)))

    def _validate_domain(self, domain, batch):
        _require(len(domain.parameters) == 5, "complete parameter domain required")
        for box, shape in zip(domain.parameters, self._parameter_shapes, strict=True):
            _require(
                box.lo.shape == shape and box.hi.shape == shape and box.error.shape == shape,
                "exact parameter-domain shapes required",
            )
        _require(
            all(not np.any(box.error) for box in (*domain.parameters, domain.inputs)),
            "parameter and real-X Domain require zero-initial-error boxes",
        )
        points = batch.host_points
        _require(
            domain.inputs.lo.shape == points.shape
            and domain.inputs.hi.shape == points.shape
            and np.array_equal(domain.inputs.lo, points)
            and np.array_equal(domain.inputs.hi, points),
            "initial scope requires the exact owned real-X point domain",
        )

    def _make_training_step(self):
        def step(state, points, projection, boxes):
            self._check_args(state.params, points, projection)
            _require(
                isinstance(state, device_training.TrainingState),
                "TrainingState required",
            )
            _require(
                jax.tree.structure(state.first_moment) == self._parameter_tree
                and jax.tree.structure(state.second_moment) == self._parameter_tree,
                "complete optimizer moment pytrees required",
            )
            for moments in (state.first_moment, state.second_moment):
                _require(
                    all(
                        g.shape == shape and np.dtype(g.dtype) == np.dtype("float64")
                        for g, shape in zip(
                            jax.tree.leaves(moments),
                            self._parameter_shapes,
                            strict=True,
                        )
                    ),
                    "optimizer moment metadata differs",
                )
            _require(
                isinstance(boxes, RuntimeBoxes)
                and type(boxes.parameter_lower) is tuple
                and type(boxes.parameter_upper) is tuple
                and len(boxes.parameter_lower) == len(boxes.parameter_upper) == 5,
                "complete runtime parameter bounds required",
            )
            for lo, hi, shape in zip(
                boxes.parameter_lower,
                boxes.parameter_upper,
                self._parameter_shapes,
                strict=True,
            ):
                _require(
                    lo.shape == hi.shape == shape
                    and np.dtype(lo.dtype) == np.dtype(hi.dtype) == np.dtype("float64"),
                    "runtime parameter-bound metadata differs",
                )
            _require(
                np.broadcast_shapes(boxes.input_lower.shape, points.shape) == points.shape
                and np.broadcast_shapes(boxes.input_upper.shape, points.shape) == points.shape
                and np.dtype(boxes.input_lower.dtype)
                == np.dtype(boxes.input_upper.dtype)
                == np.dtype("float64"),
                "runtime real-X threshold metadata differs",
            )
            _require(
                boxes.projection_lower.shape == boxes.projection_upper.shape == projection.shape
                and np.dtype(boxes.projection_lower.dtype)
                == np.dtype(boxes.projection_upper.dtype)
                == np.dtype("float64"),
                "runtime constructed-Z0 threshold metadata differs",
            )
            _require(
                state.accepted_steps.shape == ()
                and np.dtype(state.accepted_steps.dtype).kind in "iu",
                "scalar integer optimizer step counter required",
            )
            ordinary = device_training.make_adam_step(
                lambda p, x: self._value_gradient(p, x, projection)
            )

            def reject(_):
                return device_training.StepResult(
                    state,
                    state,
                    jax.tree.map(jnp.zeros_like, state.params),
                    jnp.asarray(0.0, jnp.float64),
                    jnp.asarray(device_training.INVALID_INPUT, jnp.int32),
                    jnp.asarray(False),
                )

            zvalid = jnp.all(
                jnp.isfinite(projection)
                & (projection >= boxes.projection_lower)
                & (projection <= boxes.projection_upper)
            )
            return jax.lax.cond(
                zvalid,
                lambda _: ordinary(state, points, boxes.ordinary()),
                reject,
                None,
            )

        return step

    def compile(self, params, batch, *, requests=None):
        """Compile only requested known paths, keeping a bounded four-path cache.

        The default retains the prior F/VG/training bundle. The scalar-only
        call requests just its own executable; no unrelated F/VG/Adam compile.
        """
        self._validate_context()
        batch.validate()
        _require(batch._context is self, "prepared batch belongs to another context")
        self._check_args(params, batch.points, batch.projection)
        requests = ("forward", "value_gradient", "training_step") if requests is None else requests
        _require(
            type(requests) is tuple
            and 1 <= len(requests) <= 4
            and all(type(name) is str for name in requests)
            and len(set(requests)) == len(requests),
            "bounded distinct compile requests required",
        )
        available = {"forward", "value_gradient", "training_step"}
        if self._objective_only is not None:
            available.add("objective_only")
        _require(set(requests) <= available, "unknown or unavailable compile path")
        compiled = {} if self._compiled is None else dict(self._compiled)
        memory = dict(self._compiled_memory)
        missing = tuple(name for name in requests if name not in compiled)
        if not missing:
            return {
                "cache_hit": True,
                "compile_s": 0.0,
                "compile_calls": self._compile_calls,
                "requested_paths": list(requests),
                "compiled_paths": sorted(compiled),
                "structure_key": self.structure_key,
            }
        # Runtime thresholds remain arguments. Construct the full Adam state
        # only when that path is actually requested.
        args = (params, batch.points, batch.projection)
        definitions = {
            "forward": (self._forward, args),
            "value_gradient": (self._value_gradient, args),
        }
        if "objective_only" in missing:
            definitions["objective_only"] = (self._objective_only, args)
        if "training_step" in missing:
            boxes = RuntimeBoxes(
                tuple(jnp.zeros_like(p) for p in params),
                tuple(jnp.zeros_like(p) for p in params),
                jnp.zeros_like(batch.points),
                jnp.zeros_like(batch.points),
                jnp.zeros_like(batch.projection),
                jnp.zeros_like(batch.projection),
            )
            state = device_training.initial_state(params)
            definitions["training_step"] = (
                self._training_step,
                (state, batch.points, batch.projection, boxes),
            )
        total = self._owned_device_bytes + sum(memory.values())
        started = time.perf_counter()
        for name in missing:
            function, operands = definitions[name]
            # Count an attempted trace/lower/compile even when it raises.
            self._compile_calls += 1
            executable = jax.jit(function).lower(*operands).compile()
            info = executable.memory_analysis()
            _require(info is not None, "executable memory analysis unavailable")
            count = (
                info.argument_size_in_bytes
                + info.output_size_in_bytes
                + info.temp_size_in_bytes
                - info.alias_size_in_bytes
            )
            _require(count >= 0, "negative executable memory accounting")
            total += count
            _require(
                total <= self.max_state_bytes,
                "combined executable state budget exceeded",
            )
            memory[name], compiled[name] = count, executable
        self._compiled = MappingProxyType(compiled)
        self._compiled_memory = MappingProxyType(memory)
        self._compiled_identity = tuple(
            (name, id(value), memory[name]) for name, value in compiled.items()
        )
        return {
            "cache_hit": False,
            "compile_s": time.perf_counter() - started,
            "compile_calls": self._compile_calls,
            "memory_bytes": dict(memory),
            "requested_paths": list(requests),
            "compiled_paths": sorted(compiled),
            "owned_constant_device_bytes": self._owned_device_bytes,
            "combined_executable_and_owned_constant_bytes": total,
            "structure_key": self.structure_key,
            "memory_scope": "conservative reported executable buffers plus owned source constants; compiler/global allocator/process peaks separate",
        }

    def _qualify_objective_only(self, inputs, tolerances):
        """Separate actual scalar graph, never VG[0] as a proxy for it."""
        if self._objective_only_graph is None:
            return {
                "applicable": False,
                "admissible": True,
                "scope": "standalone scalar path not exposed; full scalar objective is VG[0]",
            }
        values, meta = bounds.interpret(
            self._objective_only_graph,
            inputs,
            self.max_equations,
            self._objective_only_bindings,
            max_state_bytes=self.max_state_bytes,
        )
        _require(
            len(values) == 1 and values[0].lo.shape == (),
            "actual standalone scalar graph output inventory differs",
        )
        return {
            "applicable": True,
            **values[0].summary(tolerances.forward_atol, tolerances.forward_rtol),
            "graph": meta,
            "identity": json.loads(self._objective_only_identity_json),
            "scope": "actual separately emitted objective-only scalar graph",
        }

    def qualify(self, params, batch, domain, *, tolerances=None):
        self._validate_context()
        batch.validate()
        _require(batch._context is self, "prepared batch belongs to another context")
        self._check_args(params, batch.points, batch.projection)
        _require(
            isinstance(domain, planner.Domain) and type(domain.parameters) is tuple,
            "explicit CPU parameter/input Domain required",
        )
        self._validate_domain(domain, batch)
        tolerances = tolerances or planner.Tolerances()
        _require(
            asdict(tolerances) == asdict(planner.Tolerances()),
            "unchanged numerical contract required",
        )
        key = cache_manager.digest_record(
            {
                "structure": self.structure_key,
                "batch": batch.token,
                "domain": [
                    [cache_manager.digest_array(a) for a in (b.lo, b.hi, b.error)]
                    for b in (*domain.parameters, domain.inputs)
                ],
                "tolerances": asdict(tolerances),
                "equations": self.max_equations,
                "state_bytes": self.max_state_bytes,
            }
        )
        if key in self._certificates:
            certificate = self._certificates.pop(key)
            certificate.validate()
            _require(certificate._batch is batch, "certificate cache issuance differs")
            _require(
                bool(planner.runtime_guard(params, batch.points, *certificate.boxes[:4])),
                "actual point outside cached domain",
            )
            self._certificates[key] = certificate
            self._certificate_hits += 1
            return certificate
        # Conservative full-shape allowance before creating any owned domain or
        # runtime threshold copy. Compressed bounds may ultimately retain less.
        domain_entries = sum(math.prod(shape) for shape in self._parameter_shapes) + math.prod(
            self.batch_shape
        )
        projected_entries = math.prod(batch.projection.shape)
        item = batch.storage_report()
        preflight = (
            domain_entries * 24
            + (domain_entries + projected_entries) * 16
            + item["host_metadata_bytes"]
            + item["device_payload_bytes"]
        )
        _require(
            preflight <= self._certificate_budget,
            "one certificate domain/threshold/batch payload exceeds host retention budget",
        )
        domain = _owned_domain(domain)
        construction = batch.projection_bound
        lower, upper = construction.enlarged()
        _require(
            np.isfinite(lower).all() and np.isfinite(upper).all(),
            "finite enlarged constructed-Z0 thresholds required",
        )
        boxes = RuntimeBoxes(
            tuple(jax.device_put(b.lo, self._owner.base.device) for b in domain.parameters),
            tuple(jax.device_put(b.hi, self._owner.base.device) for b in domain.parameters),
            jax.device_put(domain.inputs.lo, self._owner.base.device),
            jax.device_put(domain.inputs.hi, self._owner.base.device),
            jax.device_put(lower, self._owner.base.device),
            jax.device_put(upper, self._owner.base.device),
        )
        # Only a scalar guard status is read, never full live parameters/inputs.
        _require(
            bool(planner.runtime_guard(params, batch.points, *boxes[:4])),
            "actual point outside requested domain",
        )
        inputs = [*domain.parameters, domain.inputs, construction]
        _require(
            len(inputs)
            == len(self._forward_graph.jaxpr.invars)
            == len(self._gradient_graph.jaxpr.invars),
            "explicit constructed-input arity mismatch",
        )
        report = {
            "contract": CONTRACT,
            "admissible": False,
            "structure_key": self.structure_key,
            "batch_token": batch.token,
            "base_token": batch.base_token,
            "input_token": batch.input_token,
            "scope": "actual three-argument forward/objective/complete first parameter gradient; point X, parameter box",
            "construction_error_preserved": True,
            "construction_max_error": float(np.max(construction.error)),
            "assumptions": "binary64 RN/gradual underflow, finite compatible compiler arithmetic, declared native transcendental errors and trusted exact owned transfers",
        }
        started = time.perf_counter()
        try:
            values, fmeta = bounds.interpret(
                self._forward_graph,
                inputs,
                self.max_equations,
                self._forward_bindings,
                max_state_bytes=self.max_state_bytes,
            )
            joint, gmeta = bounds.interpret(
                self._gradient_graph,
                inputs,
                self.max_equations,
                self._gradient_bindings,
                max_state_bytes=self.max_state_bytes,
            )
            _require(
                len(values) == len(self._operators) and len(joint) == 6,
                "interpreter omitted an output/gradient",
            )
            out = [
                value.summary(tolerances.forward_atol, tolerances.forward_rtol) for value in values
            ]
            objective = joint[0].summary(tolerances.forward_atol, tolerances.forward_rtol)
            gradients = [
                value.summary(tolerances.gradient_atol, tolerances.gradient_rtol)
                for value in joint[1:]
            ]
            scalar_only = self._qualify_objective_only(inputs, tolerances)
            report.update(
                outputs=out,
                objective=objective,
                gradients=gradients,
                standalone_objective=scalar_only,
                forward_graph=fmeta,
                gradient_graph=gmeta,
                admissible=all(
                    value["admissible"] for value in [*out, objective, *gradients, scalar_only]
                ),
            )
            if not report["admissible"]:
                report["reason"] = "accuracy bound inconclusive; no tolerance relaxation"
        except (ArithmeticError, ValueError, TypeError, NotImplementedError) as error:
            report.update(
                reason=str(error),
                exception=type(error).__name__,
                traceback=traceback.format_exc(),
            )
        report["qualification_s"] = time.perf_counter() - started
        certificate = Certificate(
            bool(report["admissible"]),
            batch.token,
            self.structure_key,
            domain,
            boxes,
            json.dumps(report, sort_keys=True, allow_nan=False),
            self,
            batch,
            _domain_origins(domain),
            tuple(_device_signature(a) for a in jax.tree.leaves(boxes)),
            key,
        )
        self._issued_certificates[id(certificate)] = certificate
        _require(
            self._certificate_payload_bytes(certificate) <= self._certificate_budget,
            "one certificate including proof report exceeds host retention budget",
        )
        if certificate.admissible:
            self._certificates[key] = certificate
            while len(self._certificates) > self.max_certificate_entries:
                self._certificates.popitem(last=False)
            while (
                self._certificates
                and self.storage_report()["certificate_retained_payload_bytes"]
                > self._certificate_budget
            ):
                self._certificates.popitem(last=False)
        return certificate

    def check_certificate(self, batch, certificate):
        self._validate_context()
        batch.validate()
        _require(
            isinstance(certificate, Certificate)
            and certificate._context is self
            and batch._context is self
            and certificate._batch is batch,
            "exact context/prepared-batch certificate required",
        )
        certificate.validate()
        _require(
            certificate.admissible and certificate.batch_token == batch.token,
            "certificate refused or belongs to another fresh batch",
        )
        return True

    @staticmethod
    def _contains(outer, inner):
        return (
            outer.lo.shape == inner.lo.shape
            and outer.hi.shape == inner.hi.shape
            and outer.error.shape == inner.error.shape
            and bool(
                np.all(inner.lo >= outer.lo)
                and np.all(inner.hi <= outer.hi)
                and np.all(inner.error <= outer.error)
            )
        )

    @staticmethod
    def _same_bound(left, right):
        return all(
            np.array_equal(a, b)
            for a, b in zip(
                (left.lo, left.hi, left.error),
                (right.lo, right.hi, right.error),
                strict=True,
            )
        )

    def _active_boxes(self, domain, construction, parameter_lower, parameter_upper):
        lower, upper = construction.enlarged()
        return RuntimeBoxes(
            parameter_lower,
            parameter_upper,
            jax.device_put(domain.inputs.lo, self._owner.base.device),
            jax.device_put(domain.inputs.hi, self._owner.base.device),
            jax.device_put(lower, self._owner.base.device),
            jax.device_put(upper, self._owner.base.device),
        )

    @staticmethod
    def _domain_report(domain, construction):
        return {
            "parameters": [
                {
                    "shape": list(b.lo.shape),
                    "range": [float(b.lo.min()), float(b.hi.max())],
                }
                for b in domain.parameters
            ],
            "X": {
                "shape": list(domain.inputs.lo.shape),
                "range": [float(domain.inputs.lo.min()), float(domain.inputs.hi.max())],
            },
            "Z": {
                "shape": list(construction.lo.shape),
                "range": [float(construction.lo.min()), float(construction.hi.max())],
                "maximum_error": float(construction.error.max()),
            },
            "scope": "range summaries only; complete bounds retained in immutable proof/receipt owners",
        }

    def _normalize_reusable_domain(self, domain, batch):
        _require(
            isinstance(domain, planner.Domain)
            and type(domain.parameters) is tuple
            and len(domain.parameters) == 5,
            "complete explicit CPU Domain required",
        )
        shapes = (*self._parameter_shapes, self.batch_shape)
        normalized = []
        for box, shape in zip((*domain.parameters, domain.inputs), shapes, strict=True):
            _require(
                isinstance(box, bounds.Bound)
                and all(isinstance(a, np.ndarray) for a in (box.lo, box.hi, box.error)),
                "CPU Bound arrays required; no implicit device readback",
            )
            _require(
                not np.any(box.error),
                "parameter/real-X Domain requires zero initial error",
            )
            item = bounds.Bound(*(np.broadcast_to(a, shape) for a in (box.lo, box.hi, box.error)))
            normalized.append(bounds.immutable_bound(item))
        result = planner.Domain(tuple(normalized[:5]), normalized[-1])
        points = batch.host_points
        _require(
            np.all(points >= result.inputs.lo) and np.all(points <= result.inputs.hi),
            "owned current host X is outside declared reusable X domain",
        )
        return result

    def _reusable_preflight(self, batch):
        entries = sum(math.prod(shape) for shape in self._parameter_shapes) + math.prod(
            self.batch_shape
        )
        projected = math.prod(batch.projection.shape)
        item = batch.storage_report()
        # Per full-shape entry: two host enclosure owners (2*24 bytes),
        # active thresholds (16) and retained proof thresholds (16). Shared
        # owners may be counted twice; equal-domain temporary p guards fit this
        # conservative allowance. Native allocator/compiler/process peaks separate.
        allowance = (
            entries * 80
            + projected * 80
            + item["host_metadata_bytes"]
            + item["device_payload_bytes"]
        )
        _require(
            allowance <= self._certificate_budget,
            "reusable proof/domain/threshold payload exceeds host retention budget",
        )
        _require(
            allowance + self._fixed_host_bytes <= self.max_state_bytes,
            "reusable construction/threshold allowance exceeds state budget",
        )

    def _retain_receipt(self, certificate):
        self._issued_certificates[id(certificate)] = certificate
        _require(
            self._certificate_payload_bytes(certificate) <= self._certificate_budget,
            "one reusable proof receipt exceeds retention budget",
        )
        if certificate.admissible:
            self._certificates[certificate._key] = certificate
            self._certificates.move_to_end(certificate._key)
            while len(self._certificates) > self.max_certificate_entries:
                self._certificates.popitem(last=False)
            while (
                self._certificates
                and self.storage_report()["certificate_retained_payload_bytes"]
                > self._certificate_budget
            ):
                self._certificates.popitem(last=False)
        return certificate

    @staticmethod
    def _reserve_call_timing(report):
        # Reserve more bytes than any finite Python float JSON encoding. The
        # authority payload is immutable after this private builder returns;
        # finalizing timing below can only shrink its charged report storage.
        report["qualification_call_s"] = "pending private timing finalization".ljust(128)
        report["rebinding_and_retention_s"] = "pending private timing finalization".ljust(128)
        report["qualification_call_scope"] = (
            "perf_counter wall through domain/ownership/construction/containment/guard/threshold/"
            "receipt/budget/LRU work; final timing JSON serialization and return excluded; "
            "caller outer wall/process measurement remains authoritative"
        )
        report["qualification_call_scope"] += (
            " Final post-sampling report budget verification is also excluded."
        )
        report["timing_fields_overlap"] = (
            "qualification_call_s contains the observed current phases; do not sum them or "
            "charge original_proof_analysis_s again on a reuse"
        )
        report["proof_analysis_scope"] = (
            "actual forward/VG bounds interpretation, output checks, summaries, qualification-report "
            "updates and refusal/exception formatting; current domain dot enclosure belongs to "
            "qualification_call_s/reuse_check_s"
        )

    def _finish_call_timing(self, certificate, call_started, binding_started):
        certificate = self._retain_receipt(certificate)
        report = certificate.report
        finished = time.perf_counter()
        report["qualification_call_s"] = finished - call_started
        report["rebinding_and_retention_s"] = finished - binding_started
        encoded = json.dumps(report, sort_keys=True, allow_nan=False)
        _require(
            len(encoded.encode()) <= len(certificate._report_json.encode()),
            "reserved timing metadata storage exhausted",
        )
        # Internal single-threaded construction finalization, before returning
        # the issued receipt. It changes timing metadata only, never authority.
        object.__setattr__(certificate, "_report_json", encoded)
        _require(
            self._certificate_payload_bytes(certificate) <= self._certificate_budget
            and self.storage_report()["certificate_retained_payload_bytes"]
            <= self._certificate_budget,
            "final report/retained cache payload exceeds budget",
        )
        return certificate

    def qualify_reusable_domain(self, params, batch, declaredDomain, *, tolerances=None):
        """Opt in to a broad input contract; initial proof may be inconclusive.

        Fresh ownership and separate ideal-range/error containment are checked
        on every request. Only actual graph bounds are reused, never old data.
        The default qualify() retains its exact-owned-X point policy.
        """
        started = time.perf_counter()
        self._validate_context()
        batch.validate()
        _require(batch._context is self, "prepared batch belongs to another context")
        self._check_args(params, batch.points, batch.projection)
        tolerances = tolerances or planner.Tolerances()
        _require(
            asdict(tolerances) == asdict(planner.Tolerances()),
            "unchanged numerical contract required",
        )
        self._reusable_preflight(batch)
        domain = self._normalize_reusable_domain(declaredDomain, batch)
        construction = bounds.immutable_bound(
            matrix_product(domain.inputs, bounds.Bound.point(self._owner.frozen.values["base"].T))
        )
        point_construction = batch.projection_bound
        _require(
            self._contains(construction, point_construction),
            "fresh projection ideal range or construction error exceeds declared reusable bound",
        )
        requested_lower = tuple(
            jax.device_put(b.lo, self._owner.base.device) for b in domain.parameters
        )
        requested_upper = tuple(
            jax.device_put(b.hi, self._owner.base.device) for b in domain.parameters
        )
        _require(
            bool(device_training.inside(params, requested_lower, requested_upper)),
            "current parameters outside requested reusable domain",
        )
        for old_key, previous in reversed(tuple(self._certificates.items())):
            proof = previous._reusable_proof
            if proof is None:
                continue
            previous.validate()
            if (
                all(
                    self._contains(outer, inner)
                    for outer, inner in zip(
                        (*proof.domain.parameters, proof.domain.inputs),
                        (*domain.parameters, domain.inputs),
                        strict=True,
                    )
                )
                and self._contains(proof.construction, construction)
                and self._contains(proof.construction, point_construction)
            ):
                self._certificates.move_to_end(old_key)
                same = all(
                    self._same_bound(left, right)
                    for left, right in zip(
                        (
                            *proof.domain.parameters,
                            proof.domain.inputs,
                            proof.construction,
                        ),
                        (*domain.parameters, domain.inputs, construction),
                        strict=True,
                    )
                )
                if same:
                    active_domain, active_construction, active_boxes = (
                        proof.domain,
                        proof.construction,
                        proof.boxes,
                    )
                else:
                    active_domain, active_construction = domain, construction
                    active_boxes = self._active_boxes(
                        domain, construction, requested_lower, requested_upper
                    )
                active_origins = (
                    *_domain_origins(active_domain),
                    *(
                        _Origin.capture(a)
                        for a in (
                            active_construction.lo,
                            active_construction.hi,
                            active_construction.error,
                        )
                    ),
                )
                active_box_signatures = tuple(
                    _device_signature(a) for a in jax.tree.leaves(active_boxes)
                )
                report = proof.report
                checked_s = time.perf_counter() - started
                report.update(
                    batch_token=batch.token,
                    base_token=batch.base_token,
                    input_token=batch.input_token,
                    proof_reused=True,
                    point_contained=True,
                    proof_origin_qualification_s=report["qualification_s"],
                    qualification_s=checked_s,
                    original_proof_analysis_s=report["qualification_s"],
                    this_call_proof_analysis_s=0.0,
                    reuse_check_s=checked_s,
                    timing_scope="ownership/domain/construction/point-guard/containment checks; caller times reporting/cache insertion too",
                    proof_domain=self._domain_report(proof.domain, proof.construction),
                    active_requested_domain=self._domain_report(active_domain, active_construction),
                    active_requested_domain_is_proof_domain=same,
                    receipt_scope="fresh owned batch; optimizer guards honor the current requested domain",
                )
                self._reserve_call_timing(report)
                key = "reusable:" + proof.identity + ":" + batch.token
                binding_started = time.perf_counter()
                certificate = Certificate(
                    True,
                    batch.token,
                    self.structure_key,
                    active_domain,
                    active_boxes,
                    json.dumps(report, sort_keys=True, allow_nan=False),
                    self,
                    batch,
                    active_origins,
                    active_box_signatures,
                    key,
                    proof,
                    active_construction,
                )
                self._certificate_hits += 1
                return self._finish_call_timing(certificate, started, binding_started)
        # Hash only immutable compact declared-domain owners on a new proof.
        # Fresh live device p/X/Z are never hashed or read back in this path.
        identity = cache_manager.digest_record(
            {
                "scope": "reusable-lowrank-domain/v1",
                "structure": self.structure_key,
                "tolerances": asdict(tolerances),
                "domains": [
                    [
                        list(b.lo.shape),
                        *[
                            cache_manager.digest_array(bounds.compact_bound_array(a))
                            for a in (b.lo, b.hi, b.error)
                        ],
                    ]
                    for b in (*domain.parameters, domain.inputs, construction)
                ],
            }
        )
        boxes = self._active_boxes(domain, construction, requested_lower, requested_upper)
        report = {
            "contract": CONTRACT,
            "admissible": False,
            "structure_key": self.structure_key,
            "proof_origin": identity,
            "proof_reused": False,
            "point_contained": True,
            "batch_token": batch.token,
            "base_token": batch.base_token,
            "input_token": batch.input_token,
            "scope": "actual three-argument forward/objective/full first gradient over declared p/X/Z contract",
            "construction_error_preserved": True,
            "construction_max_error": float(np.max(construction.error)),
            "reuse_check_s": time.perf_counter() - started,
            "proof_domain": self._domain_report(domain, construction),
            "active_requested_domain": self._domain_report(domain, construction),
            "active_requested_domain_is_proof_domain": True,
            "assumptions": "unchanged conditional binary64/native-dot/compiler arithmetic and trusted owned transfers",
        }
        analysis_started = time.perf_counter()
        try:
            inputs = [*domain.parameters, domain.inputs, construction]
            values, fmeta = bounds.interpret(
                self._forward_graph,
                inputs,
                self.max_equations,
                self._forward_bindings,
                max_state_bytes=self.max_state_bytes,
            )
            joint, gmeta = bounds.interpret(
                self._gradient_graph,
                inputs,
                self.max_equations,
                self._gradient_bindings,
                max_state_bytes=self.max_state_bytes,
            )
            _require(
                len(values) == len(self._operators) and len(joint) == 6,
                "incomplete reusable proof outputs",
            )
            out = [v.summary(tolerances.forward_atol, tolerances.forward_rtol) for v in values]
            objective = joint[0].summary(tolerances.forward_atol, tolerances.forward_rtol)
            gradients = [
                v.summary(tolerances.gradient_atol, tolerances.gradient_rtol) for v in joint[1:]
            ]
            scalar_only = self._qualify_objective_only(inputs, tolerances)
            report.update(
                outputs=out,
                objective=objective,
                gradients=gradients,
                standalone_objective=scalar_only,
                forward_graph=fmeta,
                gradient_graph=gmeta,
                admissible=all(v["admissible"] for v in [*out, objective, *gradients, scalar_only]),
            )
            if not report["admissible"]:
                report["reason"] = "reusable-domain accuracy inconclusive; no tolerance relaxation"
        except (ArithmeticError, ValueError, TypeError, NotImplementedError) as error:
            report.update(
                reason=str(error),
                exception=type(error).__name__,
                traceback=traceback.format_exc(),
            )
        report["qualification_s"] = time.perf_counter() - analysis_started
        report["original_proof_analysis_s"] = report["qualification_s"]
        report["this_call_proof_analysis_s"] = report["qualification_s"]
        proof = None
        box_signatures = tuple(_device_signature(a) for a in jax.tree.leaves(boxes))
        if report["admissible"]:
            origins = (
                *_domain_origins(domain),
                *(
                    _Origin.capture(a)
                    for a in (construction.lo, construction.hi, construction.error)
                ),
            )
            proof = _ReusableProof(
                domain,
                construction,
                boxes,
                identity,
                self,
                json.dumps(report, sort_keys=True, allow_nan=False),
                origins,
                box_signatures,
            )
            self._issued_reusable_proofs[id(proof)] = proof
        receipt_origins = (
            *_domain_origins(domain),
            *(_Origin.capture(a) for a in (construction.lo, construction.hi, construction.error)),
        )
        self._reserve_call_timing(report)
        binding_started = time.perf_counter()
        certificate = Certificate(
            bool(report["admissible"]),
            batch.token,
            self.structure_key,
            domain,
            boxes,
            json.dumps(report, sort_keys=True, allow_nan=False),
            self,
            batch,
            receipt_origins,
            box_signatures,
            "reusable:" + identity + ":" + batch.token,
            proof,
            construction,
        )
        return self._finish_call_timing(certificate, started, binding_started)

    def call_step(self, state, batch, certificate):
        self.check_certificate(batch, certificate)
        if self._compiled is None or "training_step" not in self._compiled:
            self.compile(state.params, batch, requests=("training_step",))
        return jax.block_until_ready(
            self._compiled["training_step"](
                state, batch.points, batch.projection, certificate.boxes
            )
        )

    def call_objective_only(self, params, batch, certificate):
        """Check the CURRENT issued domain before the actual scalar graph.

        Scalar guard readback is explicit overhead. This does not certify an
        optimizer update or manufacture the bounds of a different callback.
        """
        self.check_certificate(batch, certificate)
        _require(
            self._objective_only is not None,
            "standalone scalar path requires an explicit streaming objective",
        )
        self._check_args(params, batch.points, batch.projection)
        boxes = certificate.boxes
        valid = planner.runtime_guard(params, batch.points, *boxes[:4])
        valid = valid & jnp.all(
            jnp.isfinite(batch.projection)
            & (batch.projection >= boxes.projection_lower)
            & (batch.projection <= boxes.projection_upper)
        )
        _require(
            bool(valid),
            "current operands outside issued scalar domain; no objective evaluated",
        )
        if self._compiled is None or "objective_only" not in self._compiled:
            self.compile(params, batch, requests=("objective_only",))
        return jax.block_until_ready(
            self._compiled["objective_only"](params, batch.points, batch.projection)
        )

    def storage_report(self):
        self._validate_context()
        retained = sum(
            self._certificate_payload_bytes(certificate)
            for certificate in self._certificates.values()
        )
        return {
            "setup_s": self._setup_s,
            "graph_builds": self.graph_builds,
            "compile_calls": self.compile_calls,
            "certificate_entries": len(self._certificates),
            "compiled_paths": sorted(self._compiled or {}),
            "compiled_path_memory_bytes": dict(self._compiled_memory),
            "certificate_hits": self._certificate_hits,
            "certificate_retained_payload_bytes": retained,
            "fixed_context_host_payload_bytes": self._fixed_host_bytes,
            "owned_context_device_payload_bytes": self._owned_device_bytes,
            "retained_X_double_buffer_reserve_bytes": self._stream_host_reserve,
            "certificate_payload_budget_bytes": self._certificate_budget,
            "certificate_reuse_scope": "default exact-point; opt-in reusable p/X/Z containment with fresh issued receipts",
            "shared_base_and_requested_norms": self.owner.storage_report(),
            "geometry_demands": self._ridge_program.demands,
            "streaming_configuration": json.loads(self._streaming_config_json),
            "streaming_metadata_payload_bytes": self._streaming_metadata_bytes,
            "streaming_scope": "explicit scalar and full first-gradient paths only; separate full forward still stores all operator outputs",
            "binding_table_scope": "owned read-only map copies plus current key/Bound/array metadata identities; no numerical hashes or live parameter readback",
            "producer_compatibility": "issued RuntimeBatch uses the same X@A0.T producer/enclosure as standalone RidgeProjection; no duplicate projection packet/transfer",
            "objective_literal_data_scope": "captured finite binary64 data and exact-range bool/integer metadata are supplied objective literals; setup freezing/readback/hashing charged to setup_s",
            "public_integration": False,
            "new_final_source_confirmation_required": True,
            "scope": "conservative retained payload sum; caller-held evicted certificates and compiler/global allocator/process peaks are separate",
        }

    @staticmethod
    def _certificate_payload_bytes(certificate):
        item = certificate.storage_report()
        return (
            item["owned_domain_host_bytes"]
            + item["runtime_threshold_device_bytes"]
            + item["report_bytes"]
            + item["reusable_proof_payload_bytes"]
            + item["active_construction_host_bytes"]
            + item["batch"]["host_metadata_bytes"]
            + item["batch"]["device_payload_bytes"]
        )


def prepare(
    base,
    operators,
    params_template,
    batch_shape,
    *,
    provider="parity",
    scalar_objective=None,
    device=None,
    max_host_bytes=3 * 1024**3,
    max_state_bytes=3 * 1024**3,
    max_equations=100000,
    max_certificate_entries=16,
    streaming_objective_spec=None,
    chunk_rows=None,
):
    """Own one ridge base/requested-norm context; fresh batches do not enter executable identity."""
    return RuntimePreparedContext(
        base,
        operators,
        params_template,
        batch_shape,
        provider=provider,
        scalar_objective=scalar_objective,
        device=device,
        max_host_bytes=max_host_bytes,
        max_state_bytes=max_state_bytes,
        max_equations=max_equations,
        max_certificate_entries=max_certificate_entries,
        streaming_objective_spec=streaming_objective_spec,
        chunk_rows=chunk_rows,
    )

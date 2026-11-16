# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private whole-scan conditional proof reuse; numerical tests UNEXECUTED.

This file has a stdlib-only top level. Construction requires a real, already
loaded exact33 runtime, a pinned source contract, and a fresh all3-idle check.
It adds a separate current-domain device precheck before the unchanged scan.
"""

import hashlib
import json
import math
import os
import stat
import time
import types
import uuid
import weakref
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
CONTRACT = "owned-streaming-whole-block-domain-reuse/moment-NN/v2"
GATE = ROOT / "pending_candidate33/candidate_validation.py"
GATE_SHA256 = "0eb907696c707fd3f1a67068f6eb54ac4a751aca7a7146615323e01807567545"
IDLE = ROOT / "post_numerical_validation.py"
IDLE_SHA256 = "777afb2b0eac946d953284714c270569c3ba09b10a491893e50d53acd9648290"
BLOCK_SHA256 = "9abfb38fc2d456b1026f1df2cdcfe001d089bdc7e218cb7f1e1c42c3f49b9a9b"
SOURCE_PROPOSAL = ROOT / "pending_candidate33/proposal_manifest_v1.json"
SOURCE_PROPOSAL_SHA256 = "4322f3cd6bdc26be6d34132cb0f8833ebc5e9f0f0ddd44b09532137c905867eb"
SOURCE_RECIPE = ROOT / "pending_candidate33/source_recipe_v1.json"
SOURCE_RECIPE_SHA256 = "28997ab05da7b7d68e2a66258444d9236eee6f89ec7f67b738e4f517eefdbfdf"
RUNTIME_SHA256 = "1938c213049b83685d2309058cef3ce19d18b04faa19aea7117fd246e60dcd82"
STREAMING_SHA256 = "4407c2696b13105dba23529ef96797a937c3e512cc4f7207b964dfad6ff6ca30"
RUNTIME_CONTRACT = "runtime-prepared-ridge-streaming/first-gradient/v2"
STREAMING_CONTRACT = "lowrank-ridge-objective-streaming/source-proposal-v1"
MAX_BYTES = 3 * 1024**3
MAX_ENTRIES = 16
OUTCOME_JSON_CAP = 65536


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(len(raw.encode()) <= 8 * 1024**2, "bounded receipt metadata required")
    return raw


def _source(path, expected):
    path = Path(path)
    require(
        path.is_absolute() and ".." not in path.parts,
        "absolute nontraversing source path required",
    )
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    raw_fd = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            os.close(parent)
            parent = child
        raw_fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        with os.fdopen(raw_fd, "rb") as stream:
            raw_fd = None
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size <= 4 * 1024**2,
                "bounded regular source required",
            )
            raw = stream.read(4 * 1024**2 + 1)
            after = os.fstat(stream.fileno())
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)

            def signature(x):
                return x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns

            require(
                signature(before) == signature(after) == signature(current),
                "source changed during read",
            )
        require(
            len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == expected,
            "independently pinned source changed",
        )
        return raw
    finally:
        try:
            if raw_fd is not None:
                os.close(raw_fd)
        finally:
            os.close(parent)


def _module(name, path, raw):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def _cell_contains(outer, inner):
    """Ideal inclusion AND error dominance AND outward-enlarged inclusion."""
    ol, oh, oe = outer
    il, ih, ie = inner
    if not all(math.isfinite(v) for v in (*outer, *inner)):
        return False
    if not (ol <= oh and il <= ih and oe >= 0 and ie >= 0):
        return False

    def enlarged(lo, hi, error):
        return (
            (lo, hi)
            if error == 0
            else (
                math.nextafter(lo - error, -math.inf),
                math.nextafter(hi + error, math.inf),
            )
        )

    olo, ohi = enlarged(ol, oh, oe)
    ilo, ihi = enlarged(il, ih, ie)
    return (
        all(math.isfinite(v) for v in (olo, ohi, ilo, ihi))
        and il >= ol
        and ih <= oh
        and ie <= oe
        and ilo >= olo
        and ihi <= ohi
    )


def _counter_contract(counter, steps, offset, maximum):
    require(type(maximum) is int and maximum >= 1, "bounded exact maximum required")
    require(type(steps) is int and 1 <= steps <= 16, "fixed1..16 block length required")
    require(
        type(offset) is int and 0 <= offset <= maximum,
        "bounded exact source offset required",
    )
    require(
        type(counter) is int and 0 <= counter <= maximum - steps,
        "exact fixed counter with T-headroom required",
    )
    return counter, steps, offset


def _publication_contract(publication):
    """Full catalogue count is distinct from exact eagerly loaded module map."""
    required = {"gate_path", "gate_sha256", "recipe_path", "recipe_sha256",
                "proposal_path", "proposal_sha256", "selected_sources",
                "catalogue_source_count"}
    require(type(publication) is dict and set(publication) == required,
            "explicit future final-source publication required; old33 is not this authority")
    selected = publication["selected_sources"]
    require(type(selected) is dict and selected and all(
        type(name) is str and name.isidentifier() and type(digest) is str
        and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        for name, digest in selected.items()), "exact loaded module-source map required")
    count = publication["catalogue_source_count"]
    require(publication["gate_sha256"] != GATE_SHA256
            and type(count) is int and len(selected) <= count <= 256,
            "new bounded complete catalogue count required, separate from loaded map")


def _loaded_inventory(publication, provenance, modules):
    """Gate independently verifies ALL catalogue files; no eager-load inference."""
    require(type(provenance) is dict and type(provenance.get("candidate_contract")) is dict
            and provenance["candidate_contract"].get("source_count")
            == publication["catalogue_source_count"],
            "verified complete catalogue count disagrees with frozen publication")
    require(set(modules) == set(publication["selected_sources"]),
            "exact declared loaded module-source map required; no captured subset")


def _require_streaming_context(modules, context):
    """Typed known-builder boundary; never accepts a derivative callback."""
    context_api, streaming = modules["runtime_prepared"], modules["streaming_objective"]
    require(context_api.CONTRACT == RUNTIME_CONTRACT and streaming.CONTRACT == STREAMING_CONTRACT,
            "new streaming runtime/builder contract required")
    require(type(context) is context_api.RuntimePreparedContext,
            "actual selected streaming RuntimePreparedContext class required")
    program = context._streaming_program
    require(type(program) is streaming.StreamingProgram
            and type(program.objective_spec) is streaming.LeastSquares
            and program.owner is context.owner and program.provider == context.provider
            and context._objective_only is not None and context._objective_only_graph is not None,
            "explicit known-builder streaming descriptor/VG required; legacy full-output context refused")
    return program


@dataclass(frozen=True)
class BlockDomain:
    """Owned broad proof inputs. Projection is derived from points and fixed A0."""

    state: object
    points: object
    projection: object
    boxes: object


@dataclass(frozen=True)
class WholeBlockProof:
    domain: BlockDomain
    guard: object
    guard_bounds: object
    trace_bounds: object
    key: tuple
    token: str
    _owner_ref: object
    _guard_signature: tuple
    _bounds_signature: tuple
    _report_json: str
    _nn_output: object = None

    @property
    def report(self):
        return json.loads(self._report_json)

    def validate(self):
        owner = self._owner_ref()
        require(owner is not None, "originating reuse engine no longer exists")
        owner._validate_proof(self)
        return self


@dataclass(frozen=True)
class BlockReceipt:
    """Current authority, with current origin and narrow guard; no old receipt."""

    proof: WholeBlockProof
    active_guard: object
    active_guard_bounds: object
    _base: object
    _owner_ref: object
    _active_signature: tuple
    _bounds_signature: tuple
    _report_json: str

    @property
    def token(self):
        return self._base.token

    @property
    def packet(self):
        return self._base.packet

    @property
    def state_bounds(self):
        return self._base.state_bounds

    @property
    def trace_bounds(self):
        return self.proof.trace_bounds

    @property
    def report(self):
        return json.loads(self._report_json)

    def validate(self):
        owner = self._owner_ref()
        require(owner is not None, "originating reuse engine no longer exists")
        owner._validate_receipt(self)
        return self


class ActiveGuardRefusal(ValueError):
    def __init__(self, report):
        self.report = json.loads(canonical(report))
        super().__init__(report["reason"])


class PostDispatchFailure(ValueError):
    """The real returned trace remains inspectable; no automatic adoption/resume."""

    def __init__(self, report, outcome):
        self.report = json.loads(canonical(report))
        self.outcome = outcome
        super().__init__(report["reason"])


class WholeBlockReuse:
    """External private consumer of the sealed33 runtime and V2 RuntimeBlock.

    No arbitrary VG/proof boolean or prior32 authority is accepted. This fresh
    engine requires the exact33 explicit-streaming context and generated VG.
    """

    def __init__(
        self,
        runtime,
        block,
        *,
        manifest_path,
        expected_catalogue_sha256,
        expected_adapter_sha256,
        steps,
        max_entries=16,
        publication=None,
    ):
        require(type(steps) is int and 1 <= steps <= 16, "fixed block length required")
        require(
            type(max_entries) is int and 2 <= max_entries <= MAX_ENTRIES,
            "combined proof/receipt LRU hardcap2..16 required",
        )
        # The future gate validates the complete catalogue independently.
        # Only its exact eagerly loaded REQUIRED_MODULES map is bound here.
        _publication_contract(publication)
        helper = _module("_whole_block_idle", IDLE, _source(IDLE, IDLE_SHA256))
        idle = helper.require_idle(ROOT)
        gate_path = Path(publication["gate_path"])
        gate = _module("_whole_block_verified_gate", gate_path,
                       _source(gate_path, publication["gate_sha256"]))
        provenance = gate.verify_existing_runtime(
            runtime, manifest_path, expected_catalogue_sha256=expected_catalogue_sha256)
        _source(Path(__file__).absolute(), expected_adapter_sha256)
        modules = dict(runtime["modules"])
        _source(Path(publication["proposal_path"]), publication["proposal_sha256"])
        _source(Path(publication["recipe_path"]), publication["recipe_sha256"])
        _loaded_inventory(publication, provenance, modules)
        for name, digest in publication["selected_sources"].items():
            _source(Path(modules[name].__file__).absolute(), digest)
        context_api, streaming = modules["runtime_prepared"], modules["streaming_objective"]
        _source(Path(context_api.__file__).absolute(), RUNTIME_SHA256)
        _source(Path(streaming.__file__).absolute(), STREAMING_SHA256)
        _require_streaming_context(modules, block.context)
        base = modules["runtime_block"]
        require(base.CONTRACT == "owned-runtime-block/moment-carry-outward-radius/v1"
                and modules["_bound_carry_authority"].CONTRACT == "owned-emitted-moment-nonnegative/carry/v1",
                "new known owned-moment block+authority source contracts required")
        require(
            type(block) is base.RuntimeBlock,
            "actual sealed RuntimeBlock instance required",
        )
        require(
            block._scan.__func__ is base.RuntimeBlock._scan
            and block._entry_valid.__func__ is base.RuntimeBlock._entry_valid,
            "unchanged actual scan/guard source methods required",
        )
        block.validate()
        require(
            block.context.value_gradient is block.optimizer.value_gradient,
            "generated full objective/VG binding changed",
        )
        self.base, self.block, self.context = base, block, block.context
        self.jax, self.jnp, self.np = (runtime[key] for key in ("jax", "jnp", "np"))
        self.modules = MappingProxyType(modules)
        self.bounds, self.planner = modules["bounds"], modules["planner"]
        self.steps, self.max_entries = steps, max_entries
        self._source_json = canonical(
            {
                "adapter_sha256": expected_adapter_sha256,
                "gate_sha256": publication["gate_sha256"],
                "source_proposal_sha256": publication["proposal_sha256"],
                "source_recipe_sha256": publication["recipe_sha256"],
                "complete_publication": publication,
                "moment_NN_profile": block._moment_authority.profile(),
                "moment_NN_profile_id": block._moment_authority.profile_id(),
                "streaming_runtime_sha256": RUNTIME_SHA256,
                "streaming_builder_sha256": STREAMING_SHA256,
                "runtime_contract": RUNTIME_CONTRACT,
                "streaming_contract": STREAMING_CONTRACT,
                "streaming_config": json.loads(self.context._streaming_config_json),
                "streaming_scalar_graph": json.loads(self.context._objective_only_identity_json),
                "streaming_gradient_graph": self.planner._emitted_graph_identity(self.context._gradient_graph),
                "provenance": provenance,
                "idle": idle,
            }
        )
        self._source_id = hashlib.sha256(self._source_json.encode()).hexdigest()
        self._nonce, self._issued = uuid.uuid4().hex, 0
        self._cache = OrderedDict()
        self._proofs, self._receipts = (
            weakref.WeakValueDictionary(),
            weakref.WeakValueDictionary(),
        )
        self._domains = weakref.WeakValueDictionary()
        self._program = None
        self._program_origin = None
        self._pending_bytes, self._busy = 0, False
        self.graph_builds = self.compile_calls = self.compile_successes = (
            self.interpreter_calls
        ) = 0
        self._vg_origin = self._binding_signature(
            self.context.value_gradient.constant_bounds
        )
        self._vg_graph_origin = self._graph_signature(self.context._gradient_graph)
        self._vg_function_origin = self._function_metadata(self.context.value_gradient)
        self._context_bindings_origin = self._context_bindings_signature()
        self._streaming_origin = self._streaming_signature()
        self._identity = self._metadata()
        constructor_bytes = (
            self.block._base_storage(())
            + self.block._additional_storage()
            + len(self._source_json.encode())
        )
        require(
            constructor_bytes <= self.context.max_state_bytes,
            "reuse engine constructor context state budget exhausted",
        )
        self.block._budget(
            constructor_bytes,
            "reuse engine constructor source/context budget exhausted",
        )

    def _metadata(self):
        optimizer = self.block.optimizer
        return (
            id(self.block),
            id(self.context),
            id(self.jax),
            id(self.jnp),
            id(self.np),
            id(self.base),
            id(self.bounds),
            id(self.planner),
            self.steps,
            self.max_entries,
            self._source_id,
            self._source_json,
            self.context.structure_key,
            tuple((key, id(value)) for key, value in sorted(self.modules.items())),
            id(self.base.RuntimeBlock._scan),
            id(self.base.RuntimeBlock._entry_valid),
            id(self.block._scan.__func__),
            id(self.block._entry_valid.__func__),
            id(self.context.value_gradient),
            optimizer.max_accepted_steps,
            optimizer.learning_rate,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.epsilon,
            optimizer.bits,
            self.block.optimizer_profile(),
            self.block._moment_authority.profile_id(),
            canonical(asdict(self.block.limits)),
            canonical(asdict(self.block.tolerances)),
        )

    def validate(self):
        self.block.validate()
        require(
            self._metadata() == self._identity,
            "whole-block structural/source/optimizer/limit identity changed",
        )
        require(
            self._binding_signature(self.context.value_gradient.constant_bounds)
            == self._vg_origin
            and self._graph_signature(self.context._gradient_graph)
            == self._vg_graph_origin
            and self._function_metadata(self.context.value_gradient)
            == self._vg_function_origin,
            "generated VG graph/constant-bound mappings changed",
        )
        require(
            self._context_bindings_signature() == self._context_bindings_origin,
            "upstream exact33 mutable binding maps changed",
        )
        require(self._streaming_signature() == self._streaming_origin,
                "streaming descriptor/objective/VG/scalar graph source origins changed")
        if self._program is not None:
            require(
                self.base._signature(self._program["constants"])
                == self._program["constant_signature"],
                "frozen scan/precheck constants deleted or replaced",
            )
            require(
                self._program_signature(self._program) == self._program_origin,
                "owned frozen program/bindings/graphs/executables changed",
            )

    def _bound_signature(self, tree):
        self.base._validate_bounds(tree)
        return (
            str(self.jax.tree.structure(tree)),
            tuple(
                (
                    id(b),
                    tuple(
                        (
                            id(a),
                            a.shape,
                            a.dtype.str,
                            a.strides,
                            a.__array_interface__["data"][0],
                        )
                        for a in (b.lo, b.hi, b.error)
                    ),
                )
                for b in self.jax.tree.leaves(tree)
            ),
        )

    def _context_bindings_signature(self):
        return tuple(
            (name, self._binding_signature(getattr(self.context, name)))
            for name in ("_bindings", "_forward_bindings", "_gradient_bindings", "_objective_only_bindings")
        )

    def _streaming_signature(self):
        program = self.context._streaming_program
        require(type(program) is self.modules["streaming_objective"].StreamingProgram
                and type(program.objective_spec) is self.modules["streaming_objective"].LeastSquares,
                "originating selected streaming builder/descriptor changed")
        raw_functions = (program.objective, program.value_gradient, program._prepare, program._loss_chunk)
        raw_metadata = tuple((id(f), id(f.__code__), id(f.__defaults__), id(f.__kwdefaults__),
                              tuple(id(cell.cell_contents) for cell in (f.__closure__ or ())))
                             for f in raw_functions)
        return (id(program), id(program.owner), id(program._runtime), program.provider,
                canonical(asdict(program.objective_spec)), canonical(dict(program.layout)),
                program.max_state_bytes, program._row_orders, id(program._np), raw_metadata,
                self.context._streaming_config_json, self.context._objective_only_identity_json,
                self._function_metadata(self.context._objective_only),
                self._graph_signature(self.context._objective_only_graph),
                self._binding_signature(self.context._objective_only.constant_bounds),
                self._binding_signature(program.objective.constant_bounds),
                self._binding_signature(program.value_gradient.constant_bounds))

    def _array_metadata(self, value):
        if isinstance(value, self.jax.Array):
            return ("device", self.base._signature((value,)))
        if isinstance(value, self.np.ndarray):
            return (
                "host",
                id(value),
                value.shape,
                value.dtype.str,
                value.strides,
                value.__array_interface__["data"][0],
                value.flags.writeable,
            )
        return ("other", type(value).__module__, type(value).__name__, id(value))

    def _binding_signature(self, binding):
        require(
            isinstance(binding, self.bounds.ConstantBindings),
            "originating ConstantBindings required",
        )
        return (
            id(binding),
            tuple(
                (name, id(getattr(binding, name)))
                for name in (
                    "slots",
                    "objects",
                    "legacy",
                    "owned_transfers",
                    "owned_transfer_origins",
                )
            ),
            tuple(
                (key, self._bound_signature(bound))
                for key, bound in binding.slots.items()
            ),
            tuple(
                (
                    key,
                    self._array_metadata(row[0]),
                    self._bound_signature(row[1]),
                    row[2],
                )
                for key, row in binding.objects.items()
            ),
            tuple(
                (key, self._bound_signature(bound))
                for key, bound in binding.legacy.items()
            ),
            tuple(
                (key, self._array_metadata(value))
                for key, value in binding.owned_transfers.items()
            ),
            tuple(
                (
                    key,
                    tuple(
                        id(v)
                        if hasattr(v, "shape") or isinstance(v, bytes)
                        else repr(v)
                        for v in row
                    ),
                )
                for key, row in binding.owned_transfer_origins.items()
            ),
        )

    def _seal_bindings(self, binding):
        for name in (
            "slots",
            "objects",
            "legacy",
            "owned_transfers",
            "owned_transfer_origins",
        ):
            setattr(binding, name, MappingProxyType(dict(getattr(binding, name))))
        return binding

    def _graph_signature(self, value):
        # Metadata only: no GPU readback or value hash on reuse/dispatch.
        if hasattr(value, "consts") and hasattr(value, "jaxpr"):
            return (
                "closed",
                id(value),
                tuple(self._array_metadata(a) for a in value.consts),
                self._graph_signature(value.jaxpr),
            )
        if hasattr(value, "eqns"):

            def variable(a):
                literal = self._graph_signature(a.val) if hasattr(a, "val") else None
                return (id(a), repr(a.aval), literal)

            return (
                "jaxpr",
                id(value),
                tuple(
                    variable(a)
                    for a in (*value.constvars, *value.invars, *value.outvars)
                ),
                tuple(
                    (
                        id(eq),
                        id(eq.primitive),
                        tuple(variable(a) for a in (*eq.invars, *eq.outvars)),
                        self._graph_signature(eq.params),
                    )
                    for eq in value.eqns
                ),
            )
        if type(value) is dict:
            return (
                "dict",
                id(value),
                tuple((key, self._graph_signature(v)) for key, v in value.items()),
            )
        if type(value) in (tuple, list):
            return (
                type(value).__name__,
                id(value),
                tuple(self._graph_signature(v) for v in value),
            )
        if isinstance(value, (self.jax.Array, self.np.ndarray)):
            return self._array_metadata(value)
        return (type(value).__module__, type(value).__name__, id(value), repr(value))

    def _program_signature(self, record):
        functions = (record["frozen"], record["active_frozen"])
        return (
            id(record),
            tuple(
                (
                    self._function_metadata(f),
                    tuple(id(cell.cell_contents) for cell in (f.__closure__ or ())),
                    self._graph_signature(f.closed_jaxpr),
                    self._binding_signature(f.constant_bounds),
                )
                for f in functions
            ),
            self._binding_signature(record["slots"]),
            id(record["executable"]),
            id(record["active_executable"]),
            record["retained_bytes"],
            record["shape_key"],
            record["scan_id"],
            record["active_id"],
            str(self.jax.tree.structure(record["shaped"])),
            tuple(
                (id(a), tuple(a.shape), str(a.dtype))
                for a in self.jax.tree.leaves(record["shaped"])
            ),
            canonical((record["scan_graph"], record["active_graph"])),
        )

    @staticmethod
    def _function_metadata(function):
        return (
            id(function),
            id(function.__code__),
            id(function.__defaults__),
            id(function.__kwdefaults__),
            id(function.closed_jaxpr),
            tuple(id(cell.cell_contents) for cell in (function.__closure__ or ())),
        )

    def _publish_program(self, record):
        self._program = MappingProxyType(dict(record))
        self._program_origin = self._program_signature(self._program)

    def _room(self, required):
        # Pop cached-only owners BEFORE testing live weak-registry counts. An
        # external owner stays counted even after its cache reference is gone.
        while (
            len(self._proofs) + len(self._receipts) + required > self.max_entries
            and self._cache
        ):
            self._cache.popitem(last=False)
        require(
            len(self._proofs) + len(self._receipts) + required <= self.max_entries,
            "caller-held proof/receipt hardcap exhausted after LRU eviction",
        )

    def _find_proof(self, key, domain):
        for candidate in list(self._proofs.values()):
            candidate.validate()
            if candidate.key == key and all(
                self._tree_contains(a, b)
                for a, b in zip(
                    self._inputs(candidate.domain), self._inputs(domain), strict=True
                )
            ):
                cached = ("proof", candidate.token)
                if cached in self._cache:
                    self._cache.move_to_end(cached)
                return candidate
        return None

    def _token(self, kind, payload):
        require(
            type(self._issued) is int and self._issued < (1 << 63) - 1,
            "monotonic proof/receipt ID exhausted",
        )
        self._issued += 1
        return self.planner.digest_record(
            {
                "contract": CONTRACT,
                "engine": self.block.engine_id,
                "nonce": self._nonce,
                "sequence": self._issued,
                "kind": kind,
                "payload": payload,
            }
        )

    def _bound_bytes(self, tree):
        return self.base._bound_bytes(tree)

    def _proof_bytes(self, proof):
        return (
            self._bound_bytes(
                (
                    proof.domain.state,
                    proof.domain.points,
                    proof.domain.projection,
                    proof.domain.boxes,
                    proof.guard_bounds,
                    proof.trace_bounds,
                )
            )
            + self.base._bytes(proof.guard)
            + len(proof._report_json.encode())
        )

    def _receipt_extra_bytes(self, receipt):
        return (
            self.base._bytes(receipt.active_guard)
            + self._bound_bytes(receipt.active_guard_bounds)
            + len(receipt._report_json.encode())
        )

    def _storage(self, packet, *, extra=0):
        # Base storage charges base certificates/packets/outcomes/carries, including
        # caller-held weakly issued objects. Extra proof storage is overcounted.
        return (
            self.block._base_storage(packet.entries)
            + self.block._additional_storage()
            + sum(self._proof_bytes(p) for p in list(self._proofs.values()))
            + sum(self._receipt_extra_bytes(r) for r in list(self._receipts.values()))
            + sum(
                self._bound_bytes(self._inputs(d)) for d in list(self._domains.values())
            )
            + (0 if self._program is None else self._program["retained_bytes"])
            + len(self._source_json.encode())
            + self._pending_bytes
            + extra
        )

    def _budget(self, packet, extra=0):
        total = self._storage(packet, extra=extra)
        require(
            total <= self.context.max_state_bytes,
            "whole-block context state budget exhausted",
        )
        self.block._budget(
            total,
            "whole-block proof/receipt/compiled/caller-held storage budget exhausted",
        )

    def _retain(self, kind, value):
        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
        self._cache[(kind, value.token)] = value

    def _normalize(self, bound, shape, *, zero_error=False):
        require(isinstance(bound, self.bounds.Bound), "explicit CPU Bound required")
        require(
            all(
                isinstance(a, self.np.ndarray)
                for a in (bound.lo, bound.hi, bound.error)
            ),
            "no implicit device readback for proof domains",
        )
        if zero_error:
            require(
                not self.np.any(bound.error),
                "exact supplied-X/threshold domain requires zero error",
            )
        return self.bounds.immutable_bound(
            self.bounds.Bound(
                *(
                    self.np.broadcast_to(a, shape)
                    for a in (bound.lo, bound.hi, bound.error)
                )
            )
        )

    def make_domain(self, state_domain, points_domain, boxes_domain, packet):
        require(not self._busy, "single-writer domain preparation required")
        self._busy = True
        try:
            return self._make_domain(state_domain, points_domain, boxes_domain, packet)
        finally:
            self._busy = False

    def _make_domain(self, state_domain, points_domain, boxes_domain, packet):
        """Own full broadcast-normalized CPU domains; derive Z per slot with E."""
        self.validate()
        packet.validate()
        require(
            packet._engine is self.block and packet.steps == self.steps,
            "same engine/fixed T packet required",
        )
        require(
            isinstance(state_domain, self.modules["device_training"].TrainingState),
            "full16 state domain required",
        )
        require(
            isinstance(boxes_domain, self.modules["bounded_optimizer"].TraceBoxes),
            "complete14 stacked threshold domain required",
        )
        require(
            len(self._domains) < MAX_ENTRIES,
            "live owned domain declaration hardcap exhausted",
        )
        shapes = self.context._parameter_shapes
        require(
            len(shapes) == 5
            and all(
                type(getattr(state_domain, name)) is tuple
                and len(getattr(state_domain, name)) == 5
                for name in ("params", "first_moment", "second_moment")
            ),
            "complete tuple5 state parameter/moment domains required",
        )
        p = sum(math.prod(shape) for shape in shapes)
        total = 3 * p + 1
        total += (
            math.prod(packet.points.shape)
            + math.prod(packet.projection.shape)
            + sum(
                math.prod(b.lo.shape) for b in self.jax.tree.leaves(packet.box_bounds)
            )
        )
        self._budget(packet, 192 * total)
        # Actual target shapes are checked again against a real initial state at
        # qualification. This operation performs only explicit CPU preparation.
        trees = [
            tuple(
                self._normalize(b, shape)
                for b, shape in zip(getattr(state_domain, name), shapes, strict=True)
            )
            for name in ("params", "first_moment", "second_moment")
        ]
        state = self.modules["device_training"].TrainingState(
            *trees, self._normalize(state_domain.accepted_steps, ())
        )
        x = self._normalize(points_domain, packet.points.shape, zero_error=True)
        boxes = self.jax.tree.map(
            lambda b, target: self._normalize(b, target.lo.shape, zero_error=True),
            boxes_domain,
            packet.box_bounds,
        )
        fixed = self.bounds.Bound.point(self.context.owner.frozen.values["base"].T)
        z = []
        for i in range(self.steps):
            row = self.bounds.Bound(x.lo[i], x.hi[i], x.error[i])
            z.append(self.modules["runtime_prepared"].matrix_product(row, fixed))
        projection = self.bounds.immutable_bound(
            self.bounds.Bound(
                *(
                    self.np.stack([getattr(row, edge) for row in z])
                    for edge in ("lo", "hi", "error")
                )
            )
        )
        domain = BlockDomain(state, x, projection, boxes)
        self.base._validate_bounds((state, x, projection, boxes))
        self._domains[id(domain)] = domain
        self._budget(packet)
        return domain

    def _contains(self, outer, inner):
        self.base._validate_bounds((outer, inner))
        return outer.lo.shape == inner.lo.shape and all(
            _cell_contains(a, b)
            for a, b in zip(
                zip(outer.lo.flat, outer.hi.flat, outer.error.flat, strict=True),
                zip(inner.lo.flat, inner.hi.flat, inner.error.flat, strict=True),
                strict=True,
            )
        )

    def _tree_contains(self, outer, inner):
        aa, at = self.jax.tree.flatten(outer)
        bb, bt = self.jax.tree.flatten(inner)
        return (
            at == bt
            and len(aa) == len(bb)
            and all(self._contains(a, b) for a, b in zip(aa, bb, strict=True))
        )

    def _inputs(self, domain):
        return domain.state, domain.points, domain.projection, domain.boxes

    def _domain_id(self, domain):
        return self.base._bound_identity(self._inputs(domain))

    def _counter(self, domain, packet):
        counter = domain.state.accepted_steps
        require(
            counter.lo.shape == ()
            and not self.np.any(counter.error)
            and self.np.array_equal(counter.lo, counter.hi),
            "exact point counter domain required",
        )
        value = float(counter.lo)
        require(value.is_integer(), "exact integer counter domain required")
        return _counter_contract(
            int(value),
            self.steps,
            packet.source_offset,
            self.block.optimizer.max_accepted_steps,
        )

    def _active(self, state, packet, declared, carry):
        packet.validate()
        require(
            packet._engine is self.block and packet.steps == self.steps,
            "same engine/fixed T packet required",
        )
        require(
            carry is not None or packet.source_offset == 0,
            "nonroot packet requires originating carry; no error reset",
        )
        if carry is not None:
            require(
                isinstance(carry, (self.base.Carry, self.base.Resume)),
                "actual issued carry required",
            )
            carry.validate()
            require(
                carry._engine is self.block
                and carry.state is state
                and carry.packet is packet
                and carry.state_bounds is declared,
                "EXACT current carry/state/packet/Bound origin required",
            )
        owned = self.block._own_initial(
            state, declared, packet.entries[0][0], carried=carry is not None, carry=carry
        )
        return BlockDomain(
            owned, packet.point_bound, packet.projection_bound, packet.box_bounds
        )

    def _guard(self, state, domain):
        lower = self.jax.tree.map(lambda b: b.enlarged()[0], domain.state)
        upper = self.jax.tree.map(lambda b: b.enlarged()[1], domain.state)
        lower = lower._replace(
            accepted_steps=self.np.asarray(
                lower.accepted_steps, dtype=state.accepted_steps.dtype
            )
        )
        upper = upper._replace(
            accepted_steps=self.np.asarray(
                upper.accepted_steps, dtype=state.accepted_steps.dtype
            )
        )
        xlo, xhi = domain.points.enlarged()
        zlo, zhi = domain.projection.enlarged()
        blo = self.jax.tree.map(lambda b: b.lo, domain.boxes)
        bhi = self.jax.tree.map(lambda b: b.hi, domain.boxes)
        host = self.base.EntryGuard(lower, upper, xlo, xhi, zlo, zhi, blo, bhi)
        guard = self.jax.tree.map(
            lambda a: self.jax.device_put(a, self.context.owner.base.device), host
        )
        return guard, self.base._owned(self.base._point(host))

    def _shape_key(self, args):
        return (
            str(self.jax.tree.structure(args)),
            tuple(
                (tuple(a.shape), str(a.dtype), str(a.device))
                for a in self.jax.tree.leaves(args)
            ),
        )

    def _constant_arrays(self, value):
        result = {}

        def walk(item):
            if hasattr(item, "consts") and hasattr(item, "jaxpr"):
                result.update((id(a), a) for a in item.consts)
                walk(item.jaxpr)
            elif hasattr(item, "eqns"):
                for eq in item.eqns:
                    walk(eq.params)
            elif type(item) is dict:
                for child in item.values():
                    walk(child)
            elif type(item) in (tuple, list):
                for child in item:
                    walk(child)

        walk(value)
        return tuple(result.values())

    def _binding_bytes(self, binding):
        values = [
            *binding.slots.values(),
            *binding.legacy.values(),
            *(row[1] for row in binding.objects.values()),
        ]
        return self._bound_bytes(tuple(values)) + sum(
            a.nbytes for a in binding.owned_transfers.values()
        )

    def _program_for(self, args, packet):
        key = self._shape_key(args)
        if self._program is not None:
            require(
                self._program["shape_key"] == key,
                "new executable shape/dtype/device structure needs a separate reuse engine",
            )
            return self._program, True
        self._budget(packet)

        def scan(*values):
            return self.block._scan(*values)

        scan.constant_bounds = self.block.constant_bounds()

        def active(*values):
            return self.block._entry_valid(*values, self.steps)

        active.constant_bounds = self.bounds.ConstantBindings()
        frozen = self.planner.freeze_callable(scan, args)
        guard = self.planner.freeze_callable(active, args)
        self.graph_builds += 2
        shaped = self.jax.eval_shape(frozen, *args)
        boolean = self.jax.eval_shape(guard, *args)
        require(
            boolean.shape == () and self.np.dtype(boolean.dtype).kind == "b",
            "actual active precheck must return one boolean",
        )
        scan_identity = self.planner._emitted_graph_identity(frozen.closed_jaxpr)
        guard_identity = self.planner._emitted_graph_identity(guard.closed_jaxpr)
        slots = self.modules["bounded_optimizer"]._strict_slots(
            frozen.closed_jaxpr, frozen.constant_bounds
        )
        constants = self._constant_arrays((frozen.closed_jaxpr, guard.closed_jaxpr))
        retained = (
            self.base._bytes(constants)
            + self._bound_bytes(tuple(slots.slots.values()))
            + self._binding_bytes(frozen.constant_bounds)
            + self._binding_bytes(guard.constant_bounds)
            + len(canonical((scan_identity, guard_identity)).encode())
        )
        record = {
            "shape_key": key,
            "frozen": frozen,
            "active_frozen": guard,
            "shaped": shaped,
            "slots": slots,
            "constants": constants,
            "constant_signature": self.base._signature(constants),
            "scan_graph": scan_identity,
            "active_graph": guard_identity,
            "scan_id": self.planner.digest_record(scan_identity),
            "active_id": self.planner.digest_record(guard_identity),
            "executable": None,
            "active_executable": None,
            "retained_bytes": retained,
        }
        self._seal_bindings(frozen.constant_bounds)
        self._seal_bindings(guard.constant_bounds)
        self._seal_bindings(slots)
        self._publish_program(record)
        self._budget(packet)
        return self._program, False

    def _compile(self, record, args, packet):
        for function, entry in (
            ("frozen", "executable"),
            ("active_frozen", "active_executable"),
        ):
            record = self._program
            if record[entry] is None:
                self._budget(packet)
                self.compile_calls += 1
                executable = self.jax.jit(record[function]).lower(*args).compile()
                size = self.base.RuntimeBlock._executable_bytes(executable)
                self._budget(packet, size)
                updated = dict(record)
                updated[entry], updated["retained_bytes"] = (
                    executable,
                    record["retained_bytes"] + size,
                )
                self._publish_program(updated)
                self.compile_successes += 1

    def _summaries(self, output, shaped):
        rows = []

        def add(field, physical, target, gradient=False):
            for index, (bound, actual) in enumerate(
                zip(
                    self.jax.tree.leaves(physical),
                    self.jax.tree.leaves(target),
                    strict=True,
                )
            ):
                if self.np.dtype(actual.dtype).kind in "biu":
                    info = {
                        "admissible": bool(
                            self.np.array_equal(bound.lo, bound.hi)
                            and not self.np.any(bound.error)
                            and self.np.all(bound.lo == self.np.floor(bound.lo))
                        ),
                        "discrete": True,
                    }
                else:
                    tol = self.block.tolerances
                    info = bound.summary(
                        tol.gradient_atol if gradient else tol.forward_atol,
                        tol.gradient_rtol if gradient else tol.forward_rtol,
                    )
                rows.append({"field": field, "leaf": index, **info})

        add("final_state", output.final_state, shaped.final_state)
        for field in self.modules["bounded_optimizer"].TraceStep._fields:
            add(
                "history/" + field,
                getattr(output.history, field),
                getattr(shaped.history, field),
                field == "gradient",
            )
        for field in ("failed_index", "status", "entry_valid"):
            add(field, getattr(output, field), getattr(shaped, field))
        require(len(rows) == 64, "complete64 actual BlockTrace bounds required")
        return rows

    def _validate_proof(self, proof):
        self.validate()
        require(
            self._proofs.get(proof.token) is proof,
            "whole-scan proof was not issued by this engine",
        )
        require(
            self._domains.get(id(proof.domain)) is proof.domain,
            "proof domain was not constructed by this engine",
        )
        self.base._validate_bounds(
            (*self._inputs(proof.domain), proof.guard_bounds, proof.trace_bounds)
        )
        self.block._moment_authority.validate_output(proof._nn_output, proof.trace_bounds)
        require(
            self._bound_signature(
                (*self._inputs(proof.domain), proof.guard_bounds, proof.trace_bounds)
            )
            == proof._bounds_signature,
            "originating complete proof Bound layouts changed",
        )
        require(
            self.base._signature(proof.guard) == proof._guard_signature,
            "shared proof guard device owner changed",
        )
        require(
            proof.key[0] == self._program["scan_id"],
            "shared proof executable graph identity changed",
        )

    def _validate_receipt(self, receipt):
        self.validate()
        receipt.proof.validate()
        receipt._base.validate()
        carry = receipt._base.carry
        if carry is not None:
            self.block._moment_authority.active(carry.state, receipt.state_bounds, carry)
        require(
            self._receipts.get(receipt.token) is receipt,
            "fresh block-domain receipt not issued here",
        )
        self.base._validate_bounds(receipt.active_guard_bounds)
        require(
            self._bound_signature((receipt.state_bounds, receipt.active_guard_bounds))
            == receipt._bounds_signature,
            "originating current receipt Bound layouts changed",
        )
        require(
            self.base._signature(receipt.active_guard) == receipt._active_signature,
            "current active guard changed",
        )
        require(
            receipt._base.executable is self._program["executable"]
            and receipt._base.trace_bounds is receipt.proof.trace_bounds
            and receipt._base.guard is receipt.proof.guard,
            "proof/executable/output/guard authority changed",
        )

    def qualify(self, state, packet, state_bounds, domain, *, carry=None):
        """Fresh current authority; whole proof reuse iff every input is contained."""
        self.validate()
        require(not self._busy, "single-writer qualification required")
        self._busy = True
        report = {
            "contract": CONTRACT,
            "source_contract_sha256": self._source_id,
            "certificate_issued": False,
            "admissible": False,
            "proof_reused": False,
            "runtime_input_containment_checked": False,
            "public_integration": False,
            "performance_evidence": False,
            "new_science_auditor_required": True,
        }
        proof = receipt = core = None
        try:
            packet.validate()
            require(
                packet._engine is self.block and packet.steps == self.steps,
                "same engine/fixed T packet required",
            )
            require(
                isinstance(domain, BlockDomain)
                and self._domains.get(id(domain)) is domain,
                "whole-block domain must be constructed and owned by this engine",
            )
            counter = self._counter(domain, packet)
            if carry is not None:
                carry.validate()
                self.block._moment_authority.validate_permit(carry._nn_permit, state, state_bounds)
            else:
                require(packet.source_offset == 0 and all(
                    self.np.all(b.lo >= b.error) for b in state_bounds.second_moment),
                    "root current all5 NN enclosure required before any cache lookup")
            if self._program is not None:
                key = (
                    self._program["scan_id"],
                    *counter,
                    canonical(asdict(self.block.tolerances)),
                    canonical(asdict(self.block.limits)),
                    self.block._moment_authority.profile_id(),
                )
                proof = self._find_proof(key, domain)
                report["proof_reused"] = proof is not None
            self._room(2 if proof is None else 1)
            # Preflight all current state copies/guards BEFORE allocating them.
            # First-proof work additionally reserves broad guard+two full output
            # Bound owners. Frozen program constants are copies of generated VG
            # constants; no supplied packet/state/input is a closed constant.
            state_entries = sum(math.prod(a.shape) for a in self.jax.tree.leaves(state))
            p = sum(math.prod(shape) for shape in self.context._parameter_shapes)
            entries = (
                state_entries
                + math.prod(packet.points.shape)
                + math.prod(packet.projection.shape)
            )
            entries += sum(
                math.prod(b.lo.shape) for b in self.jax.tree.leaves(packet.box_bounds)
            )
            trace_entries = (
                state_entries + self.steps * (2 * state_entries + 2 * p + 3) + 3
            )
            self._pending_bytes = 24 * state_entries + 80 * entries + 16 * 1024**2
            if proof is None:
                self._pending_bytes += 80 * entries + 48 * trace_entries
            if self._program is None:
                self._pending_bytes += 4 * self.base._bytes(
                    self._constant_arrays(self.context._gradient_graph)
                )
                self._pending_bytes += 4 * self._binding_bytes(
                    self.context.value_gradient.constant_bounds
                )
            self._budget(packet)
            active = self._active(state, packet, state_bounds, carry)
            self.base._validate_bounds(self._inputs(domain))
            require(
                all(
                    self._tree_contains(a, b)
                    for a, b in zip(
                        self._inputs(domain), self._inputs(active), strict=True
                    )
                ),
                "fresh ideal/error/enlarged inputs outside declared whole-block domain",
            )
            require(
                all(
                    not self.np.any(b.error)
                    for b in (domain.points, *self.jax.tree.leaves(domain.boxes))
                ),
                "proof real-X/threshold errors must be zero",
            )
            # A broad box may have negative loose lower endpoints only as
            # box-intersect-NN theorem. The CURRENT active state must prove all
            # five NN predicates before EACH proof lookup/use and precheck.
            current_nn = self.block._moment_authority.active(state, active.state, carry)
            report["moment_NN_profile_id"] = self.block._moment_authority.profile_id()
            report["current_moment_NN_origin"] = current_nn
            require(
                self._counter(active, packet) == counter,
                "counter/T/source offset proof mismatch",
            )
            active_guard, active_guard_bounds = self._guard(state, active)
            args = (state, packet.points, packet.projection, packet.boxes, active_guard)
            record, compiled_structure_hit = self._program_for(args, packet)
            key = (
                record["scan_id"],
                *counter,
                canonical(asdict(self.block.tolerances)),
                canonical(asdict(self.block.limits)),
                self.block._moment_authority.profile_id(),
            )
            if proof is None:
                self._room(2)
                proof_guard, proof_guard_bounds = self._guard(state, domain)
                inputs = (*self._inputs(domain), proof_guard_bounds)
                # Pending reserves include outside owners and owned output-copy
                # allowance; interpreter's own aggregate bound storage is capped.
                outside = self._storage(packet)
                budget = (
                    min(
                        self.block.limits.state_bytes,
                        self.context.max_state_bytes,
                        self.context.max_host_bytes,
                    )
                    - outside
                )
                require(
                    budget >= self._bound_bytes(inputs),
                    "actual whole-scan interpreter preflight exhausted",
                )
                self.interpreter_calls += 1
                started = time.perf_counter()
                request = self.block._moment_authority.prepare(
                    inputs, record["shaped"], state=state, active_bounds=active.state,
                    carry=carry, graph=record["scan_id"], closed=record["frozen"].closed_jaxpr)
                values, meta, nn_output = self.bounds._interpret_owned(
                    record["frozen"].closed_jaxpr, self.jax.tree.leaves(inputs),
                    self.block.limits.equations, record["slots"], budget,
                    authority=self.block._moment_authority, request=request)
                require(
                    len(values) == len(self.jax.tree.leaves(record["shaped"])),
                    "truncated actual fulltrace bound output",
                )
                output = self.base._owned(
                    self.jax.tree.unflatten(
                        self.jax.tree.structure(record["shaped"]), values
                    )
                )
                del values
                rows = self._summaries(output, record["shaped"])
                proof_report = {
                    "contract": CONTRACT,
                    "source_contract_sha256": self._source_id,
                    "admissible": all(row["admissible"] for row in rows),
                    "certificate_issued": False,
                    "output_bounds": rows,
                    "actual_scan_graph": record["scan_graph"],
                    "emitted_graph_id": record["scan_id"],
                    "active_precheck_graph": record["active_graph"],
                    "active_precheck_graph_id": record["active_id"],
                    "domain_id": self._domain_id(domain),
                    "counter_T_offset": counter,
                    "moment_NN_profile": self.block._moment_authority.profile(),
                    "moment_NN_profile_id": self.block._moment_authority.profile_id(),
                    "moment_NN_output_token": nn_output.token,
                    "proof_domain_is_box_intersect_all5_ideal_and_actual_NN": True,
                    "branch_semantics": "exact certified predicates only; uncertain branch/sign/index refuses; no branch union",
                    "assumptions": "unchanged R9 interpreter RN/gradual-underflow/finite-compatible-transform/native-transcendental and trusted generated-VG/owned-transfer contracts; separate actual active precheck source trusted",
                    "interpreter": meta,
                    "analysis_s": time.perf_counter() - started,
                    "analysis_scope": "actual fullscan interpretation plus output ownership/complete64 summary and graph/domain report construction before final timing serialization; no numerical acceptance",
                }
                report["conditional_analysis"] = proof_report
                require(
                    proof_report["admissible"]
                    and self.np.any(output.history.evaluated.lo == 1),
                    "actual whole-scan proof inconclusive or vacuous; preserve refusal",
                )
                token = self._token(
                    "proof", {"key": key, "domain": proof_report["domain_id"]}
                )
                proof = WholeBlockProof(
                    domain,
                    proof_guard,
                    proof_guard_bounds,
                    output,
                    key,
                    token,
                    weakref.ref(self),
                    self.base._signature(proof_guard),
                    self._bound_signature(
                        (*self._inputs(domain), proof_guard_bounds, output)
                    ),
                    canonical(proof_report),
                    nn_output,
                )
                self._budget(packet, self._proof_bytes(proof))
                self._proofs[token] = proof
                self._retain("proof", proof)
            self._room(1)
            # The proof is now owned/charged; pending memory is only the new
            # current active receipt and its complete initial state snapshot.
            self._pending_bytes = (
                self.base._bytes(active_guard)
                + self._bound_bytes(active_guard_bounds)
                + self._bound_bytes(active.state)
                + 16 * 1024**2
            )
            self._compile(
                record,
                (state, packet.points, packet.projection, packet.boxes, proof.guard),
                packet,
            )
            record = self._program
            self._budget(packet)
            started = time.perf_counter()
            valid = bool(self.jax.block_until_ready(record["active_executable"](*args)))
            precheck_s = time.perf_counter() - started
            require(
                valid,
                "current narrow active state/input/threshold guard refused before issuance",
            )
            origin = {
                "scope": "issued_carried_trajectory"
                if carry is not None
                else "external_zero_error_state_domain",
                "carry_token": None if carry is None else carry.token,
                "moment_NN_permit": None if carry is None else carry._nn_permit.token,
                "bound_identity": self.base._bound_identity(active.state),
                "source_offset": packet.source_offset,
            }
            token = self.block._next_token(
                "certificate",
                {
                    "packet": packet.token,
                    "graph": record["scan_id"],
                    "state_origin": origin,
                    "whole_proof": proof.token,
                    "active_guard_graph": record["active_id"],
                },
            )
            report.pop("conditional_analysis", None)
            report.update(
                admissible=True,
                certificate_issued=True,
                certificate_token=token,
                nonvacuous_evaluated_case=True,
                runtime_input_containment_checked=True,
                packet_token=packet.token,
                engine_id=self.block.engine_id,
                source_offset=packet.source_offset,
                steps=self.steps,
                source_receipts=packet.report["receipt_ids"],
                state_origin=origin,
                proof_origin=proof.token,
                proof_domain_id=proof.report["domain_id"],
                active_domain_id=self._domain_id(active),
                active_guard_graph=record["active_graph"],
                active_guard_graph_id=record["active_id"],
                emitted_graph_identity=record["scan_graph"],
                emitted_graph_id=record["scan_id"],
                output_bounds=proof.report["output_bounds"],
                construction_error_preserved=True,
                current_active_guard_before_every_scan=True,
                moment_NN_output_token=proof._nn_output.token,
                compile_structure_hit=compiled_structure_hit,
                executable_bytes=self.base.RuntimeBlock._executable_bytes(
                    record["executable"]
                ),
                active_precheck_executable_bytes=self.base.RuntimeBlock._executable_bytes(
                    record["active_executable"]
                ),
                qualification_active_precheck_s=precheck_s,
                proof_analysis_calls_this_request=0 if report["proof_reused"] else 1,
                scope="current active request/carry authority, separate shared broad fullscan theorem; scalar precheck and all lookup/containment/binding/storage/compile costs charged by outer caller; no free phase or matched timing claim",
            )
            report["retained_payload_and_executable_bytes"] = 0
            initial_raw = canonical(report)
            self._pending_bytes = (
                self.base._bytes(active_guard)
                + self._bound_bytes(active_guard_bounds)
                + self._bound_bytes(active.state)
                + 2 * (len(initial_raw.encode()) + 256)
            )
            report["retained_payload_and_executable_bytes"] = self._storage(packet)
            raw = canonical(report)
            require(
                len(raw.encode()) <= len(initial_raw.encode()) + 256,
                "final resource report exceeded reserved metadata allowance",
            )
            self._budget(packet)
            self.base.positive(
                len(self.block._certificates) + 1, 16, "live block certificates"
            )
            core = self.base.BlockCertificate(
                packet,
                active.state,
                proof.guard,
                proof.trace_bounds,
                record["executable"],
                token,
                carry,
                () if carry is None else self.base._signature(carry.state),
                self.block,
                self.base._signature(proof.guard),
                raw,
                proof._nn_output,
            )
            receipt = BlockReceipt(
                proof,
                active_guard,
                active_guard_bounds,
                core,
                weakref.ref(self),
                self.base._signature(active_guard),
                self._bound_signature((active.state, active_guard_bounds)),
                raw,
            )
            self.block._certificates[id(core)] = core
            self._receipts[token] = receipt
            self._retain("receipt", receipt)
            self._pending_bytes = 0
            receipt.validate()
            self._budget(packet)
            return receipt, receipt.report
        except (
            ArithmeticError,
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            NotImplementedError,
            MemoryError,
        ) as error:
            if receipt is not None:
                self._receipts.pop(receipt.token, None)
                self._cache.pop(("receipt", receipt.token), None)
            if core is not None:
                self.block._certificates.pop(id(core), None)
            report.update(
                certificate_issued=False,
                admissible=False,
                reason=str(error),
                exception=type(error).__name__,
            )
            return None, json.loads(canonical(report))
        finally:
            self._pending_bytes = 0
            self._busy = False

    def run(self, state, receipt):
        require(not self._busy, "single-writer block dispatch required")
        self._busy = True
        try:
            return self._run_checked(state, receipt)
        finally:
            self._busy = False

    def _run_checked(self, state, receipt):
        receipt.validate()
        require(receipt._owner_ref() is self, "same whole-block reuse engine required")
        cached = ("receipt", receipt.token)
        if cached in self._cache:
            self._cache.move_to_end(cached)
        core, packet = receipt._base, receipt.packet
        if core.carry is not None:
            require(
                state is core.carry.state
                and self.base._signature(state) == core.carried_state_signature,
                "exact current carried state required before active precheck",
            )
        self._budget(
            packet,
            2 * packet.report["logical_payload_estimate"]["trace_output_bytes"]
            + 2 * OUTCOME_JSON_CAP,
        )
        started = time.perf_counter()
        valid = bool(
            self.jax.block_until_ready(
                self._program["active_executable"](
                    state,
                    packet.points,
                    packet.projection,
                    packet.boxes,
                    receipt.active_guard,
                )
            )
        )
        precheck_s = time.perf_counter() - started
        if not valid:
            raise ActiveGuardRefusal(
                {
                    "contract": CONTRACT,
                    "reason": "current narrow active-entry guard refused before scan/VG",
                    "evaluated": False,
                    "accepted_state_unchanged": True,
                    "certificate_token": receipt.token,
                    "proof_origin": receipt.proof.token,
                    "active_guard_graph_id": self._program["active_id"],
                    "active_precheck_s": precheck_s,
                    "performance_evidence": False,
                }
            )
        outcome = self.block.run(state, core)
        try:
            report = outcome.report
            report.update(
                contract=CONTRACT,
                base_scan_contract=self.base.CONTRACT,
                proof_origin=receipt.proof.token,
                active_guard_graph_id=self._program["active_id"],
                actual_active_precheck_executed=True,
                active_precheck_s=precheck_s,
                new_science_auditor_required=True,
                timing_scope="separate scalar active precheck plus actual full five-argument guarded scan; all phases and traffic externally charged, no matched acceptance",
            )
            raw = canonical(report)
            require(
                len(raw.encode()) <= OUTCOME_JSON_CAP,
                "outcome report exceeds preflight cap; preserve actual trace",
            )
            self._budget(packet, 2 * len(raw.encode()))
            updated = self.base.Outcome(
                outcome.trace,
                core,
                self.block,
                self.base._signature(outcome.trace),
                raw,
            )
            self.block._outcomes.pop(id(outcome), None)
            self.block._outcomes[id(updated)] = updated
            try:
                return updated.validate()
            except BaseException:
                self.block._outcomes.pop(id(updated), None)
                self.block._outcomes[id(outcome)] = outcome
                raise
        except Exception as error:
            raise PostDispatchFailure(
                {
                    "contract": CONTRACT,
                    "reason": str(error),
                    "exception": type(error).__name__,
                    "actual_scan_dispatched": True,
                    "preserve_actual_outcome": True,
                    "automatic_state_adoption": False,
                    "certificate_token": receipt.token,
                    "proof_origin": receipt.proof.token,
                },
                outcome,
            ) from error

    def qualify_carry(self, carry, domain):
        carry.validate()
        require(carry.packet is not None, "terminal Resume has no block to qualify")
        return self.qualify(
            carry.state, carry.packet, carry.state_bounds, domain, carry=carry
        )

    def storage_report(self):
        self.validate()
        return {
            "contract": CONTRACT,
            "cache_entries": len(self._cache),
            "combined_cache_hardcap": self.max_entries,
            "live_proofs_including_caller_held": len(self._proofs),
            "live_receipts_including_caller_held": len(self._receipts),
            "shared_proof_payload_bytes": sum(
                self._proof_bytes(p) for p in list(self._proofs.values())
            ),
            "active_receipt_extra_bytes": sum(
                self._receipt_extra_bytes(r) for r in list(self._receipts.values())
            ),
            "compiled_program_payload_bytes": 0
            if self._program is None
            else self._program["retained_bytes"],
            "graph_builds": self.graph_builds,
            "compile_calls": self.compile_calls,
            "interpreter_calls": self.interpreter_calls,
            "source_contract_sha256": self._source_id,
            "source_contract": json.loads(self._source_json),
            "scope": "conservative shared-buffer/category overcount; live base engine/context/packet/outcome/Carry/Resume storage additionally charged at every operation; Python/compiler/global allocator and process peaks separate",
            "numerical_acceptance": False,
            "performance_evidence": False,
            "public_integration": False,
        }

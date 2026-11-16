# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Immutable environment fingerprints and bounded, distinct research caches."""

import hashlib
import importlib.util
import json
import os
import platform
import stat
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from weakref import WeakValueDictionary

import jax
import jaxlib
import numpy as np


def digest_array(value):
    array = np.ascontiguousarray(value)
    header = str((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def digest_record(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


# Complete explicit inventories are substituted by the source preparation script.
ENVIRONMENT_INVENTORY_RECIPE_SHA256 = (
    "3e46cffefd059bbd3f78a4790ab6fe06f00d2f48c4245779d46561a76e5c9e15"
)
BACKEND_SOURCE_NAMES = (
    "__init__.py",
    "_adam_bias_schedule_math.py",
    "_bound_carry_authority.py",
    "_bound_sign_facts.py",
    "bounded_optimizer.py",
    "bounds.py",
    "bundles.py",
    "cache_manager.py",
    "certificates.py",
    "device_training.py",
    "execution_measurements.py",
    "explicit_fft.py",
    "feature_models.py",
    "frozen_batch.py",
    "frozen_previous.py",
    "generic_programs.py",
    "kernels.py",
    "legacy.py",
    "low_rank_operators.py",
    "operators.py",
    "pde_math.py",
    "planner.py",
    "prefetch.py",
    "providers.py",
    "qualification_session.py",
    "reduction.py",
    "reuse.py",
    "ridge_diagonal_geometry.py",
    "runtime_block.py",
    "runtime_prepared.py",
    "scheduled_optimizer.py",
    "series_fourier.py",
    "shared_models.py",
    "solve_certificate.py",
    "streaming.py",
    "streaming_objective.py",
    "variable_projection.py",
)
CORE_SOURCE_NAMES = (
    "__init__.py",
    "_chebyshev_calculus.py",
    "_rational_linear.py",
    "apolar.py",
    "band_embed.py",
    "bell.py",
    "block_search.py",
    "composed_curvature.py",
    "confluence.py",
    "contraction.py",
    "cubature.py",
    "exp_polynomial.py",
    "frames.py",
    "ftc.py",
    "gaussian_bias_jet.py",
    "gaussian_critical.py",
    "gaussian_jet.py",
    "implicit.py",
    "integral_kernel.py",
    "jet_algebra.py",
    "jet_series.py",
    "jet_systems.py",
    "jets.py",
    "line_search.py",
    "locus.py",
    "mollifier.py",
    "multi_index.py",
    "multipack.py",
    "operator_polynomial.py",
    "operator_quotient.py",
    "operator_spec.py",
    "parameter_jets.py",
    "periodic_heat.py",
    "polynomials.py",
    "profile_discovery.py",
    "profile_system.py",
    "proof/__init__.py",
    "proof/catalog.py",
    "proof/certificate.py",
    "proof/condition.py",
    "proof/discovery.py",
    "proof/inequality.py",
    "proof/lean_check.py",
    "proof/lean_lock.py",
    "proof/lift.py",
    "proof/obligations/__init__.py",
    "proof/obligations/rational_stencil.py",
    "proof/observe.py",
    "proof/realization_algebra_replay.py",
    "proof/realization_replay.py",
    "proof/replay.py",
    "rational_field.py",
    "realization/__init__.py",
    "realization/algebraic.py",
    "realization/membership.py",
    "realization/polynomial.py",
    "realization/rank.py",
    "realization/schema.py",
    "realization/transition.py",
    "realization/witness.py",
    "refine.py",
    "ridge_system.py",
    "scale.py",
    "scan.py",
    "spec.py",
    "spectral_design.py",
    "tanh_method.py",
    "transfer.py",
    "transforms_pde.py",
    "uncertainty.py",
    "verified/__init__.py",
    "verified/_gaussian_bias_response.py",
    "verified/_gaussian_moments.py",
    "verified/affine.py",
    "verified/asymptotic_jet.py",
    "verified/banded.py",
    "verified/clamped_biharmonic.py",
    "verified/coeffs.py",
    "verified/complex_interval.py",
    "verified/complex_ode.py",
    "verified/complex_rootfind.py",
    "verified/conditioning.py",
    "verified/dfinite_continuation.py",
    "verified/eig.py",
    "verified/eig_operator.py",
    "verified/enclosure_collapse.py",
    "verified/euler_maclaurin.py",
    "verified/exp_polynomial.py",
    "verified/fourier.py",
    "verified/frobenius.py",
    "verified/ftc.py",
    "verified/gaussian_affine.py",
    "verified/gaussian_affine_adjoint.py",
    "verified/gaussian_affine_aging.py",
    "verified/gaussian_affine_approximation.py",
    "verified/gaussian_affine_evolution.py",
    "verified/gaussian_affine_parameters.py",
    "verified/gaussian_affine_sensitivity.py",
    "verified/gaussian_affine_stationarity.py",
    "verified/gaussian_aging.py",
    "verified/gaussian_aging_derivative.py",
    "verified/gaussian_analytic.py",
    "verified/gaussian_bias_analytic.py",
    "verified/gaussian_chebyshev.py",
    "verified/gaussian_chebyshev_derivative.py",
    "verified/gaussian_collinear_minimum.py",
    "verified/gaussian_collinear_reduction.py",
    "verified/gaussian_critical.py",
    "verified/gaussian_entropy_bias.py",
    "verified/gaussian_entropy_bias_face.py",
    "verified/gaussian_entropy_bias_region.py",
    "verified/gaussian_entropy_face.py",
    "verified/gaussian_entropy_joint_face.py",
    "verified/gaussian_entropy_joint_region.py",
    "verified/gaussian_entropy_minimum.py",
    "verified/gaussian_entropy_residual.py",
    "verified/gaussian_entropy_sensitivity.py",
    "verified/gaussian_entropy_solver.py",
    "verified/gaussian_entropy_unique.py",
    "verified/gaussian_evolution.py",
    "verified/gaussian_grouped_directional.py",
    "verified/gaussian_grouped_equilibrium.py",
    "verified/gaussian_heat.py",
    "verified/gaussian_joint_directional.py",
    "verified/gaussian_polyhedral.py",
    "verified/gaussian_polyhedral_directional.py",
    "verified/gaussian_polyhedral_equilibrium.py",
    "verified/gaussian_polyhedral_path.py",
    "verified/gaussian_polyhedral_radius.py",
    "verified/gaussian_polyhedral_reduced_minimum.py",
    "verified/gaussian_polyhedral_reduced_sensitivity.py",
    "verified/gaussian_polyhedral_reduction.py",
    "verified/gaussian_polyhedral_unique.py",
    "verified/gaussian_posterior_domain.py",
    "verified/gaussian_predictor.py",
    "verified/gaussian_projected_equilibrium.py",
    "verified/gaussian_projection_minimum.py",
    "verified/gaussian_sensitivity.py",
    "verified/gaussian_tube.py",
    "verified/gaussian_two_center.py",
    "verified/global_min.py",
    "verified/hardy_line.py",
    "verified/hermite_basis.py",
    "verified/interval.py",
    "verified/interval_array.py",
    "verified/invariant_subspace.py",
    "verified/inverse_gaussian_affine.py",
    "verified/inverse_gaussian_heat.py",
    "verified/inverse_mean_map.py",
    "verified/jet.py",
    "verified/jet_flow.py",
    "verified/jet_mv.py",
    "verified/kantorovich.py",
    "verified/laguerre_basis.py",
    "verified/linalg.py",
    "verified/linalg_array.py",
    "verified/line.py",
    "verified/locality.py",
    "verified/lohner.py",
    "verified/mean_map.py",
    "verified/mlp_objective.py",
    "verified/ode.py",
    "verified/pde_certificate.py",
    "verified/periodic_heat.py",
    "verified/polyhedral_projection.py",
    "verified/quadrature.py",
    "verified/radii_series.py",
    "verified/radii_spectral.py",
    "verified/riesz.py",
    "verified/rootfind.py",
    "verified/sampled.py",
    "verified/sequence_space.py",
    "verified/series.py",
    "verified/sigma.py",
    "verified/spectral.py",
    "verified/sqg.py",
    "verified/taylor_model.py",
    "verified/taylor_model_mv.py",
    "verified/tm_neuron.py",
    "verified/transcend.py",
    "weight_loss_jet.py",
)
_CORE_IMPORTS = (
    "polynomials",
    "operator_spec",
    "jet_algebra",
    "jet_series",
    "multi_index",
    "verified.coeffs",
    "verified.interval",
    "verified.transcend",
)
_HASH_CHUNK_BYTES = 65536
_environment = None
_environment_builds = 0
_environment_epoch = 0
_environment_pid = os.getpid()
_environment_state = "empty"
_environment_device = None
_environment_root = None
_environment_error = None
_environment_last_hash_work = None
_environment_lock = threading.RLock()
_refresh_hooks = []


def _require_environment_pid():
    # This check deliberately precedes acquisition of a possibly inherited lock.
    if os.getpid() != _environment_pid:
        raise RuntimeError("inherited environment identity; use a fresh exec")


def _freeze_environment(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze_environment(item) for key, item in value.items()})
    if type(value) in (tuple, list):
        return tuple(_freeze_environment(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unsupported environment identity metadata")


def _file_stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _sha256_file(path, work):
    """Hash a stable required regular file without materializing its full payload.

    API byte counts describe these explicit hash reads, not physical disk I/O or
    all imports/discovery reads. Immutable dependency ownership remains required.
    """
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("environment dependency is not a regular file")
        total = before.st_size
        digest = hashlib.sha256()
        offset = 0
        work["files"] += 1
        while offset < total:
            block = os.read(descriptor, min(_HASH_CHUNK_BYTES, total - offset))
            work["read_calls"] += 1
            work["api_bytes"] += len(block)
            work["maximum_chunk_bytes"] = max(work["maximum_chunk_bytes"], len(block))
            if not block:
                raise ValueError("environment dependency truncated while hashing")
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if _file_stamp(before) != _file_stamp(after) or _file_stamp(before) != _file_stamp(current):
            raise ValueError("environment dependency changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


_CPU_IDENTITY_FIELDS = (
    "vendor_id",
    "cpu family",
    "model",
    "model name",
    "stepping",
    "microcode",
    "flags",
    "physical id",
    "siblings",
    "core id",
    "cpu cores",
)


def _cpu_identity(text):
    """Supported x86 identity fields, excluding frequency/runtime observations."""
    fields = {}
    for line in text.split("\n\n", 1)[0].splitlines():
        key, separator, value = line.partition(":")
        key = key.strip()
        if separator and key in _CPU_IDENTITY_FIELDS:
            if key in fields:
                raise ValueError("duplicate CPU identity field")
            fields[key] = " ".join(value.split())
    if set(fields) != set(_CPU_IDENTITY_FIELDS) or not all(fields.values()):
        raise NotImplementedError("complete supported x86 CPU identity unavailable")
    return fields


def _collect_environment(work):
    root = Path(__file__).resolve().parent
    sources = {name: _sha256_file(root / name, work) for name in BACKEND_SOURCE_NAMES}
    # Preserve the existing eight core imports, but hash the complete declared
    # core inventory once instead of discovering/importing a current source tree.

    core_modules = {
        name: importlib.import_module("omnibias.core." + name) for name in _CORE_IMPORTS
    }
    core_root = Path(core_modules["operator_spec"].__file__).resolve().parent
    core_sources = {name: _sha256_file(core_root / name, work) for name in CORE_SOURCE_NAMES}
    for name, module in core_modules.items():
        relative = name.replace(".", "/") + ".py"
        if Path(module.__file__).resolve() != core_root / relative or relative not in core_sources:
            raise ValueError("core import does not match declared source inventory")
        sources["omnibias.core." + name] = core_sources[relative]
    devices = tuple(jax.devices())
    if len(devices) != 1 or devices[0].platform not in ("cpu", "gpu"):
        raise NotImplementedError("one owned CPU or GPU device required")
    device = devices[0]
    binary_root = Path(jaxlib.__file__).parent
    binaries = {
        str(p.relative_to(binary_root)): _sha256_file(p, work) for p in binary_root.glob("*.so")
    }
    jax_root = Path(jax.__file__).parent
    compiler_sources = {
        str(p.relative_to(jax_root)): _sha256_file(p, work) for p in jax_root.rglob("*.py")
    }
    plugins = {}
    if device.platform == "gpu":
        for name in ("jax_cuda13_plugin", "jax_cuda12_plugin"):
            spec = importlib.util.find_spec(name)
            if spec is not None and spec.origin:
                folder = Path(spec.origin).parent
                plugins[name] = {
                    str(p.relative_to(folder)): _sha256_file(p, work) for p in folder.rglob("*.so")
                }
        if not plugins:
            raise NotImplementedError("GPU compiler identity unavailable")
    driver = Path("/proc/driver/nvidia/version")
    cpu = Path("/proc/cpuinfo")
    record = {
        "sources": sources,
        "core_sources": core_sources,
        "inventory_recipe": ENVIRONMENT_INVENTORY_RECIPE_SHA256,
        "binaries": binaries,
        "compiler_sources": compiler_sources,
        "plugins": plugins,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "device": str(device),
        "device_kind": device.device_kind,
        "platform": device.platform,
        "machine": platform.machine(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu": _cpu_identity(cpu.read_text()),
        "driver": driver.read_text() if driver.exists() else None,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "arithmetic_assumptions": "RN binary64; gradual underflow; finite intermediates; compatible compiler arithmetic; native tanh/exp 8u*abs(f)+eta assumed",
    }
    return record, device, root


def _build_environment(epoch, state):
    global _environment, _environment_device, _environment_root, _environment_builds
    global _environment_state, _environment_error, _environment_last_hash_work
    work = {"files": 0, "read_calls": 0, "api_bytes": 0, "maximum_chunk_bytes": 0}
    try:
        record, device, root = _collect_environment(work)
        dependency_digest = digest_record(record)
        # Epoch is explicit invalidation, not an observed dependency change.
        digest = digest_record({"dependency_digest": dependency_digest, "refresh_epoch": epoch})
        snapshot = _freeze_environment(
            {
                "digest": digest,
                "dependency_digest": dependency_digest,
                "generation": epoch,
                "pid": _environment_pid,
                "record": record,
                "hash_work": work,
            }
        )
        _require_environment_pid()
        with _environment_lock:
            if _environment_epoch != epoch or _environment_state != state:
                raise RuntimeError("environment generation changed during collection")
            _environment_device = device
            _environment_root = root
            _environment = snapshot
            _environment_builds += 1
            _environment_last_hash_work = snapshot["hash_work"]
            _environment_error = None
            _environment_state = "ready"
            return snapshot
    except BaseException as exc:
        _require_environment_pid()
        with _environment_lock:
            _environment = None
            _environment_device = None
            _environment_root = None
            _environment_state = "failed"
            _environment_error = type(exc).__name__ + ": " + str(exc)[:512]
            _environment_last_hash_work = _freeze_environment(work)
        raise


def register_refresh(callback):
    _require_environment_pid()
    if not callable(callback):
        raise TypeError("refresh hook must be callable")
    with _environment_lock:
        if _environment_state in ("building", "refreshing"):
            raise RuntimeError("cannot change hooks during identity collection/refresh")
        if callback not in _refresh_hooks:
            _refresh_hooks.append(callback)


def environment_snapshot():
    """One immutable identity per PID/explicit epoch; no evaluation hashing.

    Source/compiler/plugin files and device/environment configuration must stay
    immutable until a quiescent explicit refresh. Concurrent initial collection
    and refresh are refused instead of exposing a partially issued identity.
    """
    global _environment_state
    _require_environment_pid()
    with _environment_lock:
        if _environment_state == "ready":
            return _environment
        if _environment_state != "empty":
            raise RuntimeError("environment identity unavailable; finish or explicitly refresh")
        _environment_state = "building"
        epoch = _environment_epoch
    return _build_environment(epoch, "building")


def _require_environment_current(snapshot):
    _require_environment_pid()
    with _environment_lock:
        if (
            _environment_state != "ready"
            or _environment is not snapshot
            or snapshot["generation"] != _environment_epoch
        ):
            raise RuntimeError("environment refreshed; prepare a new runtime context")


def _environment_source(snapshot, name, source_file):
    _require_environment_current(snapshot)
    with _environment_lock:
        _require_environment_current(snapshot)
        if (
            name not in BACKEND_SOURCE_NAMES
            or Path(source_file).absolute() != _environment_root / name
        ):
            raise ValueError("source is outside the declared backend inventory")
        return snapshot["record"]["sources"][name]


def _environment_device_for(snapshot, selected):
    _require_environment_current(snapshot)
    # Retain actual ownership separately from portable serialized device text.
    # The supported single-device snapshot is never a multi-device guarantee.
    with _environment_lock:
        if _environment is not snapshot or _environment_state != "ready":
            raise RuntimeError("environment changed during device selection")
        device = _environment_device
        if selected is not None and selected is not device:
            raise ValueError("selected device is not the owned environment device")
        record = snapshot["record"]
        if (
            device.platform != record["platform"]
            or device.device_kind != record["device_kind"]
            or str(device) != record["device"]
        ):
            raise ValueError("owned device metadata changed")
        return device


def refresh_environment():
    """Invalidate before clearing caches; failures require another explicit refresh.

    Refresh is a quiescent administrative operation, not cancellation of any
    already running kernel. Old Program digests expire even for unchanged bytes.
    """
    global \
        _environment, \
        _environment_device, \
        _environment_root, \
        _environment_epoch, \
        _environment_state
    global _environment_error
    _require_environment_pid()
    with _environment_lock:
        if _environment_state in ("building", "refreshing"):
            raise RuntimeError("nested or concurrent environment refresh")
        _environment_epoch += 1
        epoch = _environment_epoch
        _environment = None
        _environment_device = None
        _environment_root = None
        _environment_state = "refreshing"
        callbacks = tuple(_refresh_hooks)
    try:
        for callback in callbacks:
            callback()
    except BaseException as exc:
        _require_environment_pid()
        with _environment_lock:
            _environment_state = "failed"
            _environment_error = type(exc).__name__ + ": " + str(exc)[:512]
        raise
    return _build_environment(epoch, "refreshing")


def environment_info():
    _require_environment_pid()
    with _environment_lock:
        return {
            "fingerprint_builds": _environment_builds,
            "initialized": _environment_state == "ready",
            "refresh_required_after_source_compiler_device_or_flag_change": True,
            "refresh_epoch": _environment_epoch,
            "state": _environment_state,
            "owner_pid": _environment_pid,
            "last_hash_api_bytes": 0
            if _environment_last_hash_work is None
            else _environment_last_hash_work["api_bytes"],
            "last_hash_files": 0
            if _environment_last_hash_work is None
            else _environment_last_hash_work["files"],
            "last_error": _environment_error,
            "hash_work_scope": "explicit stable regular-file SHA API reads only; imports/discovery/text metadata and full Python heap/RSS are separate",
        }


@dataclass
class Artifact:
    value: object
    reported_bytes: int = 0


class PlanCaches:
    """At most 16 complete plans; orphaned artifacts are pruned on eviction."""

    def __init__(self):
        self.plans = OrderedDict()
        self.executables = {}
        self.certificates = {}
        self.selections = {}
        self.programs = WeakValueDictionary()
        self.invalidated_plans = 0

    def clear(self):
        self.plans.clear()
        self.executables.clear()
        self.certificates.clear()
        self.selections.clear()

    def register_program(self, program):
        try:
            self.programs[id(program)] = program
        except TypeError:
            # Generic cache test objects may not support weak references.
            # Real Program objects do; this does not retain external owners.
            pass

    def get_plan(self, key):
        if key not in self.plans:
            return None
        item = self.plans.pop(key)
        from .bounds import is_owned_bound

        domain = getattr(item["program"], "domain", None)
        try:
            if domain is not None:
                for bound in (*domain.parameters, domain.inputs):
                    is_owned_bound(bound)
        except ValueError:
            self.invalidated_plans += 1
            self.prune()
            return None
        self.plans[key] = item
        return item["program"]

    def put_plan(
        self,
        key,
        program,
        *,
        executable_keys,
        certificate_keys,
        selection_key,
        limit=16,
        memory_limit=3 * 1024**3,
    ):
        self.register_program(program)
        self.plans.pop(key, None)
        self.plans[key] = {
            "program": program,
            "executables": tuple(executable_keys),
            "certificates": tuple(certificate_keys),
            "selection": selection_key,
        }
        self.prune()
        while len(self.plans) > limit or self.info()["reported_artifact_bytes"] > memory_limit:
            self.plans.popitem(last=False)
            self.prune()
        self.prune()

    def prune(self):
        used_e = {key for p in self.plans.values() for key in p["executables"]}
        used_c = {key for p in self.plans.values() for key in p["certificates"]}
        used_s = {p["selection"] for p in self.plans.values()}
        for mapping, used in (
            (self.executables, used_e),
            (self.certificates, used_c),
            (self.selections, used_s),
        ):
            for key in tuple(mapping):
                if key not in used:
                    del mapping[key]

    def info(self):
        executable = sum((v.reported_bytes for v in self.executables.values()))
        certificate = sum((v.reported_bytes for v in self.certificates.values()))
        selection = sum((v.reported_bytes for v in self.selections.values()))
        from .bounds import _array_storage_owner, bound_storage_owners

        invalid_domains = []
        unknown_payloads = [0]

        def domain_owners(domain, category):
            owners = {}
            if domain is not None and hasattr(domain, "parameters") and hasattr(domain, "inputs"):
                for bound in (*domain.parameters, domain.inputs):
                    try:
                        owners.update(bound_storage_owners(bound))
                    except ValueError:
                        # Counting immutable payload bytes does not reuse a
                        # numerical certificate. Invalid external metadata is
                        # visible and cannot block unrelated new plans.
                        invalid_domains.append(category)
                        for array in (bound.lo, bound.hi, bound.error):
                            if isinstance(array, np.ndarray):
                                owner, size = _array_storage_owner(array)
                                owners[id(owner)] = (owner, size)
                            else:
                                unknown_payloads[0] += 1
            return owners

        certificate_owners = {}
        for artifact in self.certificates.values():
            if isinstance(artifact.value, dict):
                certificate_owners.update(
                    domain_owners(artifact.value.get("domain"), "certificate")
                )
        cached_programs = {id(item["program"]): item["program"] for item in self.plans.values()}
        cache_host, cache_guard = {}, {}
        for program in cached_programs.values():
            cache_host.update(domain_owners(getattr(program, "domain", None), "cached_program"))
            cache_guard.update(
                {
                    id(v): int(v.nbytes)
                    for v in jax.tree.leaves(getattr(program, "runtime_boxes", ()))
                }
            )
        uncovered = sum(
            size for key, (_, size) in cache_host.items() if key not in certificate_owners
        )
        certificate_payload = sum(size for _, size in certificate_owners.values())
        deficit = max(0, certificate_payload - certificate)
        outside_programs = [p for key, p in self.programs.items() if key not in cached_programs]
        outside_host, outside_guard = {}, {}
        for program in outside_programs:
            outside_host.update(domain_owners(getattr(program, "domain", None), "outside_program"))
            outside_guard.update(
                {
                    id(v): int(v.nbytes)
                    for v in jax.tree.leaves(getattr(program, "runtime_boxes", ()))
                }
            )
        covered_host = set(certificate_owners) | set(cache_host)
        outside_host_bytes = sum(
            size for key, (_, size) in outside_host.items() if key not in covered_host
        )
        outside_guard_bytes = sum(
            size for key, size in outside_guard.items() if key not in cache_guard
        )
        guards = sum(cache_guard.values())
        return {
            "entries": len(self.plans),
            "executable_entries": len(self.executables),
            "certificate_entries": len(self.certificates),
            "selection_entries": len(self.selections),
            "reported_executable_bytes": executable,
            "reported_certificate_bytes": certificate,
            "reported_plan_runtime_guard_bytes_upper": guards,
            "reported_plan_host_domain_bytes_not_in_certificates": uncovered,
            "reported_certificate_payload_deficit_bytes": deficit,
            "reported_artifact_bytes": executable
            + certificate
            + selection
            + guards
            + uncovered
            + deficit,
            "invalidated_plan_count": self.invalidated_plans,
            "invalid_domain_metadata_records": len(invalid_domains),
            "invalid_cached_domain_metadata_records": sum(
                v != "outside_program" for v in invalid_domains
            ),
            "invalid_outside_domain_metadata_records": sum(
                v == "outside_program" for v in invalid_domains
            ),
            "unknown_domain_payload_arrays": unknown_payloads[0],
            "outside_cache_live_program_count": len(outside_programs),
            "outside_cache_runtime_guard_bytes_upper": outside_guard_bytes,
            "outside_cache_domain_host_payload_bytes": outside_host_bytes,
            "scope": "known cached artifact/payload accounting; guards and uncovered domain owners participate in cache admission/eviction",
            "conservative_overcount": "certificate logical bound bytes can overcount shared compact domains; logical device guard bytes can alias host payload on CPU",
            "outside_cache_scope": "weak caller-held/evicted Program guard/domain payloads only; caller executables, Python metadata, JAX global caches, allocator reservations and RSS remain separate and unmeasured here",
        }


CACHES = PlanCaches()
register_refresh(CACHES.clear)


def configure_persistent_cache(directory):
    """Opt-in executable caching; directory ownership is the caller's responsibility."""
    jax.config.update(
        "jax_compilation_cache_dir", None if directory is None else str(Path(directory).resolve())
    )
    return {
        "directory": None if directory is None else str(Path(directory).resolve()),
        "numerical_certificate_reused_by_this_setting": False,
    }

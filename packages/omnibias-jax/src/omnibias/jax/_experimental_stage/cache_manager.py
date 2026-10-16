"""Immutable environment fingerprints and bounded, distinct research caches."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import threading
import jax
import jaxlib
import numpy as np

def digest_array(value):
    array = np.ascontiguousarray(value)
    header = str((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()

def digest_record(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
_environment = None
_environment_builds = 0
_environment_lock = threading.Lock()
_refresh_hooks = []

def register_refresh(callback):
    if callback not in _refresh_hooks:
        _refresh_hooks.append(callback)

def environment_snapshot():
    """Sources/binaries are immutable until explicit refresh; no evaluation hashing."""
    global _environment, _environment_builds
    with _environment_lock:
        if _environment is not None:
            return _environment
        root = Path(__file__).resolve().parent
        source_names = ('bundles.py', 'providers.py', 'bounds.py', 'planner.py', 'reuse.py', 'certificates.py', 'cache_manager.py', 'streaming.py', 'device_training.py', 'prefetch.py', 'reduction.py', 'variable_projection.py', 'feature_models.py', 'explicit_fft.py', 'low_rank_operators.py', 'shared_models.py', 'qualification_session.py', 'solve_certificate.py', 'frozen_previous.py', 'legacy.py', 'operators.py', 'kernels.py', 'generic_programs.py', 'series_fourier.py', 'execution_measurements.py', 'frozen_batch.py')
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names if (root / name).exists()}
        import importlib
        for name in ('polynomials', 'operator_spec', 'jet_algebra', 'jet_series', 'multi_index', 'verified.coeffs', 'verified.interval', 'verified.transcend'):
            module = importlib.import_module('omnibias.core.' + name)
            sources['omnibias.core.' + name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        device = jax.devices()[0]
        binary_root = Path(jaxlib.__file__).parent
        binaries = {str(p.relative_to(binary_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in binary_root.glob('*.so')}
        jax_root = Path(jax.__file__).parent
        compiler_sources = {str(p.relative_to(jax_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in jax_root.rglob('*.py')}
        plugins = {}
        if device.platform == 'gpu':
            for name in ('jax_cuda13_plugin', 'jax_cuda12_plugin'):
                spec = importlib.util.find_spec(name)
                if spec is not None and spec.origin:
                    folder = Path(spec.origin).parent
                    plugins[name] = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*.so')}
            if not plugins:
                raise NotImplementedError('GPU compiler identity unavailable')
        driver = Path('/proc/driver/nvidia/version')
        cpu = Path('/proc/cpuinfo')
        record = {'sources': sources, 'binaries': binaries, 'compiler_sources': compiler_sources, 'plugins': plugins, 'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'device': str(device), 'device_kind': device.device_kind, 'platform': device.platform, 'machine': platform.machine(), 'affinity': sorted(os.sched_getaffinity(0)), 'cpu': cpu.read_text().split('\n\n')[0] if cpu.exists() else None, 'driver': driver.read_text() if driver.exists() else None, 'environment': {name: os.environ.get(name) for name in ('XLA_FLAGS', 'JAX_PLATFORMS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, 'arithmetic_assumptions': 'RN binary64; gradual underflow; finite intermediates; compatible compiler arithmetic; native tanh/exp 8u*abs(f)+eta assumed'}
        _environment_builds += 1
        _environment = {'digest': digest_record(record), 'record': record}
        return _environment

def refresh_environment():
    global _environment
    with _environment_lock:
        _environment = None
    for callback in tuple(_refresh_hooks):
        callback()
    return environment_snapshot()

def environment_info():
    return {'fingerprint_builds': _environment_builds, 'initialized': _environment is not None, 'refresh_required_after_source_compiler_device_or_flag_change': True}

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

    def clear(self):
        self.plans.clear()
        self.executables.clear()
        self.certificates.clear()
        self.selections.clear()

    def get_plan(self, key):
        if key not in self.plans:
            return None
        item = self.plans.pop(key)
        self.plans[key] = item
        return item['program']

    def put_plan(self, key, program, *, executable_keys, certificate_keys, selection_key, limit=16, memory_limit=3 * 1024 ** 3):
        self.plans.pop(key, None)
        self.plans[key] = {'program': program, 'executables': tuple(executable_keys), 'certificates': tuple(certificate_keys), 'selection': selection_key}
        self.prune()
        while len(self.plans) > limit or self.info()['reported_artifact_bytes'] > memory_limit:
            self.plans.popitem(last=False)
            self.prune()
        self.prune()

    def prune(self):
        used_e = {key for p in self.plans.values() for key in p['executables']}
        used_c = {key for p in self.plans.values() for key in p['certificates']}
        used_s = {p['selection'] for p in self.plans.values()}
        for mapping, used in ((self.executables, used_e), (self.certificates, used_c), (self.selections, used_s)):
            for key in tuple(mapping):
                if key not in used:
                    del mapping[key]

    def info(self):
        executable = sum((v.reported_bytes for v in self.executables.values()))
        certificate = sum((v.reported_bytes for v in self.certificates.values()))
        selection = sum((v.reported_bytes for v in self.selections.values()))
        return {'entries': len(self.plans), 'executable_entries': len(self.executables), 'certificate_entries': len(self.certificates), 'selection_entries': len(self.selections), 'reported_executable_bytes': executable, 'reported_certificate_bytes': certificate, 'reported_artifact_bytes': executable + certificate + selection, 'scope': 'reported artifacts; excludes JAX global caches, allocator reservations and process RSS'}
CACHES = PlanCaches()
register_refresh(CACHES.clear)

def configure_persistent_cache(directory):
    """Opt-in executable caching; directory ownership is the caller's responsibility."""
    jax.config.update('jax_compilation_cache_dir', None if directory is None else str(Path(directory).resolve()))
    return {'directory': None if directory is None else str(Path(directory).resolve()), 'numerical_certificate_reused_by_this_setting': False}

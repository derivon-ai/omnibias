"""Bounded next-batch preparation for explicitly immutable dependencies.

Only input/frozen-data work belongs here. Parameter-dependent projections must
remain in the differentiated training step. Async dispatch is not a measured
claim of transfer/compute overlap.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import time
import threading
from types import MappingProxyType
import jax
import numpy as np

def _snapshot(value):
    array = np.array(value, copy=True, order='C')
    if array.dtype.kind not in 'biufc':
        raise TypeError('frozen dependencies and batches must be numerical arrays')
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

def _hash_tree(tree):
    leaves, definition = jax.tree.flatten(tree)
    digest = hashlib.sha256(str(definition).encode())
    for array in leaves:
        a = np.asarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()

@dataclass(frozen=True, init=False)
class FrozenDependencies:
    _values: dict
    token: str
    nbytes: int

    def __init__(self, values):
        if not isinstance(values, dict) or not all((isinstance(k, str) for k in values)):
            raise TypeError('named frozen numerical dependencies required')
        owned = {name: jax.tree.map(_snapshot, value) for name, value in sorted(values.items())}
        object.__setattr__(self, '_values', owned)
        object.__setattr__(self, 'token', _hash_tree(owned))
        object.__setattr__(self, 'nbytes', sum((a.nbytes for a in jax.tree.leaves(owned))))

    @property
    def values(self):
        return MappingProxyType(jax.tree.map(lambda a: a, self._values))

@dataclass(frozen=True)
class HostPrepared:
    """Producer output with numerical annotations retained solely on the host.

    DoubleBuffer owns immutable snapshots of both parts. Only payload is sent
    to the device. Metadata is a named numerical pytree, with separately
    accounted bytes, for such things as saved-arithmetic error enclosures.
    """
    payload: object
    metadata: dict

@dataclass(frozen=True)
class PreparedBatch:
    index: int
    dependency_token: str
    input_token: str
    payload: object
    host_preparation_s: float
    transfer_dispatch_s: float
    host_start_ns: int = 0
    host_end_ns: int = 0
    transfer_start_ns: int = 0
    transfer_end_ns: int = 0
    host_thread_id: int = 0
    host_metadata: object = None

    def require_dependencies(self, frozen):
        if self.dependency_token != frozen.token:
            raise ValueError('prepared batch belongs to different immutable dependencies')
        return self.payload

    def wait(self):
        start = time.perf_counter()
        jax.block_until_ready(self.payload)
        return {'consumer_wait_s': time.perf_counter() - start}

    def timeline(self):
        """Monotonic host timestamps, not CUDA transfer completion events."""
        result = {name: getattr(self, name) for name in ('index', 'dependency_token', 'input_token', 'host_start_ns', 'host_end_ns', 'transfer_start_ns', 'transfer_end_ns', 'host_thread_id', 'host_preparation_s', 'transfer_dispatch_s')}
        result['host_metadata_bytes'] = 0 if self.host_metadata is None else self.host_metadata.nbytes
        result['host_metadata_token'] = None if self.host_metadata is None else self.host_metadata.token
        return result

class DoubleBuffer:
    """One worker and at most one future next batch; context-manager ownership.

    prepare_fn(batch, frozen_mapping) may only use its arguments plus immutable
    scalar/structural constants. Captured arrays are rejected; arbitrary hidden
    Python state cannot be proved pure, so callers must honor the explicit
    frozen-only contract. All trainable values remain outside this interface.
    """

    def __init__(self, batches, prepare_fn, frozen, *, device=None, max_host_bytes=3 * 1024 ** 3, trace_annotations=False, _prefetched=True):
        if not isinstance(frozen, FrozenDependencies):
            raise TypeError('owned FrozenDependencies required')
        if type(max_host_bytes) is not int or max_host_bytes < 1 or frozen.nbytes > max_host_bytes:
            raise ValueError('immutable dependencies exceed host-memory budget')
        if inspect.isfunction(prepare_fn):
            captured = inspect.getclosurevars(prepare_fn)
            captured_values = (*captured.nonlocals.values(), *captured.globals.values())
            if any((isinstance(v, (np.ndarray, jax.Array, dict, list, set)) for v in jax.tree.leaves(captured_values))) or any((isinstance(v, (dict, list, set)) for v in captured_values)):
                raise ValueError('numerical closure captures must be explicit frozen dependencies')
        self._iterator = iter(batches)
        self._prepare = prepare_fn
        self.frozen = frozen
        self.device = device
        self.max_host_bytes = max_host_bytes
        self.trace_annotations = bool(trace_annotations)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='omnibias-prefetch') if _prefetched else None
        self._future = None
        self._closed = False
        self._next_index = 0

    def _produce(self, index):
        start = time.perf_counter_ns()
        scope = jax.profiler.TraceAnnotation('omnibias_host_prepare', batch_index=index) if self.trace_annotations else nullcontext()
        with scope:
            prepared = self._prepare_one()
        if prepared is None:
            return None
        payload, token, metadata = prepared
        host_end = time.perf_counter_ns()
        transfer_start = time.perf_counter_ns()
        scope = jax.profiler.TraceAnnotation('omnibias_transfer_dispatch', batch_index=index) if self.trace_annotations else nullcontext()
        with scope:
            placed = jax.device_put(payload, self.device)
        transfer_end = time.perf_counter_ns()
        return PreparedBatch(index, self.frozen.token, token, placed, (host_end - start) * 1e-09, (transfer_end - transfer_start) * 1e-09, start, host_end, transfer_start, transfer_end, threading.get_ident(), metadata)

    def _prepare_one(self):
        try:
            original = next(self._iterator)
        except StopIteration:
            return None
        input_bytes = sum((np.asarray(a).nbytes for a in jax.tree.leaves(original)))
        if self.frozen.nbytes + 2 * input_bytes > self.max_host_bytes:
            raise MemoryError('input double buffer exceeds declared limit')
        batch = jax.tree.map(_snapshot, original)
        token = _hash_tree(batch)
        prepared = self._prepare(batch, self.frozen.values)
        if isinstance(prepared, HostPrepared):
            metadata = FrozenDependencies(prepared.metadata)
            payload = prepared.payload
        else:
            metadata = None
            payload = prepared
        payload = jax.tree.map(_snapshot, payload)
        working_bytes = sum((a.nbytes for a in jax.tree.leaves((batch, payload))))
        if metadata is not None:
            working_bytes += metadata.nbytes
        if self.frozen.nbytes + 2 * working_bytes > self.max_host_bytes:
            raise MemoryError('double-buffer host storage exceeds declared limit')
        return (payload, token, metadata)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._executor is None:
            try:
                result = self._produce(self._next_index)
            except BaseException:
                self.close()
                raise
            if result is None:
                self.close()
                raise StopIteration
            self._next_index += 1
            return result
        if self._future is None:
            self._future = self._executor.submit(self._produce, self._next_index)
        try:
            result = self._future.result()
        except BaseException:
            self.close()
            raise
        if result is None:
            self.close()
            raise StopIteration
        self._next_index += 1
        self._future = self._executor.submit(self._produce, self._next_index)
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            if self._future is not None:
                self._future.cancel()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            close = getattr(self._iterator, 'close', None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

class SynchronousBuffer(DoubleBuffer):
    """Identical ownership, preparation, hashing and transfer without overlap."""

    def __init__(self, batches, prepare_fn, frozen, **options):
        super().__init__(batches, prepare_fn, frozen, _prefetched=False, **options)

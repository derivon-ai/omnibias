"""Deterministic training input pipelines and separate profiler diagnostics.

The synchronous and prefetched paths execute identical numerical preparation,
immutable ownership, hashes and transfers. No trainable dependency is cached.
Host dispatch timestamps cannot establish CUDA transfer/compute overlap.
"""
from dataclasses import replace
import json
from pathlib import Path
import time
import jax
import numpy as np
from .prefetch import DoubleBuffer, FrozenDependencies, SynchronousBuffer

def _identity_prepare(batch, frozen):
    return batch

def _dimensions(shape):
    shape = tuple(shape)
    if len(shape) != 2 or any((type(n) is not int or n < 1 for n in shape)):
        raise ValueError('a positive two-dimensional input shape is required')
    return shape

def _bounds(bounds, shape):
    lower, upper = (bounds.lo, bounds.hi) if hasattr(bounds, 'lo') else bounds
    lower, upper = (np.array(a, dtype=np.float64, copy=True) for a in (lower, upper))
    lo_view, hi_view = (np.broadcast_to(lower, shape), np.broadcast_to(upper, shape))
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(lo_view <= hi_view)):
        raise ValueError('finite ordered input bounds required')
    return (lower, upper)

def indexed_points(shape, seed, count=100, bounds=(-0.2, 0.2)):
    """Batch i is exactly default_rng(seed+i).uniform(lower, upper, shape)."""
    shape = _dimensions(shape)
    if type(count) is not int or count < 0 or type(seed) is not int or (seed < 0):
        raise ValueError('nonnegative integer seed and count required')
    lower, upper = _bounds(bounds, shape)
    return _indexed_validated(shape, seed, count, lower, upper)

def _indexed_validated(shape, seed, count, lower, upper):
    for index in range(count):
        yield np.random.default_rng(seed + index).uniform(lower, upper, shape)

class _FixedResident:

    def __init__(self, batch, count, device, max_host_bytes, trace_annotations):
        self._source = SynchronousBuffer((batch,), _identity_prepare, FrozenDependencies({}), device=device, max_host_bytes=max_host_bytes, trace_annotations=trace_annotations)
        self._prepared = None
        self._count, self._index, self._closed = (count, 0, False)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._index >= self._count:
            self.close()
            raise StopIteration
        if self._prepared is None:
            self._prepared = next(self._source)
        index = self._index
        self._index += 1
        if index == 0:
            return self._prepared
        return replace(self._prepared, index=index, host_preparation_s=0.0, transfer_dispatch_s=0.0, host_start_ns=0, host_end_ns=0, transfer_start_ns=0, transfer_end_ns=0)

    def close(self):
        self._closed = True
        self._source.close()

class InputStream:
    """Own an iterator; consumers retain pending batches across guarded stops.

    Only the consumer's accepted step count identifies consumed input. Iteration
    never consumes or changes optimizer state. The caller must retain any block
    suffix that was fetched but not evaluated; replay that suffix before asking
    for later batches. At most one future batch is owned by this iterator.
    """

    def __init__(self, iterator, report):
        self._iterator, self.report = (iterator, report)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self._iterator.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

def input_stream(shape, pattern, seed, count=100, bounds=(-0.2, 0.2), device=None, prefetched=False, *, resident=None, max_host_bytes=3 * 1024 ** 3, trace_annotations=False):
    """Return indexed PreparedBatch records with immutable ownership tokens.

    ``fresh`` uses one independent seed per index. ``fixed`` snapshots resident
    data once (or generates seed zero's batch once); prefetch is not applicable.
    Bounds affect data generation, not numerical-certificate authority.
    """
    shape = _dimensions(shape)
    if pattern not in ('fresh', 'fixed'):
        raise ValueError('pattern must be fresh or fixed')
    if type(count) is not int or count < 0 or type(seed) is not int or (seed < 0):
        raise ValueError('nonnegative integer seed and count required')
    if not jax.config.x64_enabled:
        raise ValueError('binary64 input pipeline requires JAX_ENABLE_X64=1')
    lower, upper = _bounds(bounds, shape)
    bound_bytes = lower.nbytes + upper.nbytes
    if type(max_host_bytes) is not int or max_host_bytes <= bound_bytes:
        raise ValueError('input bounds exceed the host-memory budget')
    if resident is not None and pattern != 'fixed':
        raise ValueError('resident data is exclusive to the fixed pattern')
    if pattern == 'fixed':
        batch = next(_indexed_validated(shape, seed, 1, lower, upper)) if resident is None else np.array(resident, copy=True)
        if batch.shape != shape or batch.dtype != np.dtype('float64'):
            raise ValueError('resident input must have the declared shape and float64 dtype')
        if not np.all(np.isfinite(batch) & (batch >= lower) & (batch <= upper)):
            raise ValueError('resident input outside generation bounds')
        iterator = _FixedResident(batch, count, device, max_host_bytes - bound_bytes, trace_annotations)
    else:
        cls = DoubleBuffer if prefetched else SynchronousBuffer
        iterator = cls(_indexed_validated(shape, seed, count, lower, upper), _identity_prepare, FrozenDependencies({}), device=device, max_host_bytes=max_host_bytes - bound_bytes, trace_annotations=trace_annotations)
    report = {'pattern': pattern, 'seed': seed, 'count': count, 'shape': list(shape), 'prefetch_requested': bool(prefetched), 'prefetch_applicable': pattern == 'fresh', 'prefetch_active': bool(prefetched and pattern == 'fresh'), 'maximum_future_batches': int(bool(prefetched and pattern == 'fresh')), 'host_memory_limit_bytes': max_host_bytes, 'owned_generation_bound_bytes': bound_bytes, 'fixed_scope': 'one owned snapshot; reuse resident device payload' if pattern == 'fixed' else None, 'preparation': 'owned float64 input snapshot, input/dependency tokens, identity map, device_put', 'trainable_dependencies': [], 'timeline_clock': 'perf_counter_ns, monotonic host', 'cuda_overlap': 'unavailable: no CUDA device completion events measured', 'consumer_pending_blocks': 'caller-owned; retain unused suffix after guarded block stop'}
    return InputStream(iterator, report)

def profile_training(step, state, boxes, *, directory, shape, seed, bounds=(-0.2, 0.2), device=None, prefetched=False, steps=2):
    """Trace an actual small guarded train loop, separately from matched data.

    Compilation is performed before tracing. This does not requalify on exits:
    the first rejected proposal remains rolled back and ends the diagnostic.
    No speed ratio or inference of GPU overlap is produced.
    """
    if type(steps) is not int or not 1 <= steps <= 8:
        raise ValueError('diagnostic trace must contain one to eight steps')
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    if device is None:
        device = jax.devices()[0]
    sample = jax.device_put(next(indexed_points(shape, seed, 1, bounds)), device)
    compiled = jax.jit(step).lower(state, sample, boxes).compile()
    report = {'kind': 'diagnostic_only', 'device': str(device), 'platform': device.platform, 'requested_steps': steps, 'steps': [], 'cuda_overlap_established': False, 'cuda_completion_events': 'unavailable; host dispatch timestamps are insufficient', 'profiling_status': 'pending', 'performance_claim': None}
    try:
        with jax.profiler.trace(str(destination), create_perfetto_link=False):
            with input_stream(shape, 'fresh', seed, count=steps, bounds=bounds, device=device, prefetched=prefetched, trace_annotations=True) as batches:
                report['pipeline'] = batches.report
                for batch in batches:
                    start = time.perf_counter_ns()
                    with jax.profiler.StepTraceAnnotation('omnibias_training', step_num=batch.index):
                        result = jax.block_until_ready(compiled(state, batch.payload, boxes))
                    end = time.perf_counter_ns()
                    state = result.accepted
                    report['steps'].append({'input': batch.timeline(), 'step_start_ns': start, 'step_end_ns': end, 'status': int(result.status), 'evaluated': bool(result.evaluated), 'accepted_steps': int(state.accepted_steps)})
                    if int(result.status) != 0:
                        break
        report['profiling_status'] = 'trace_collected'
    except Exception as error:
        report['profiling_status'] = 'failed_preserved'
        report['failure'] = repr(error)
    report['trace_files'] = [str(path.relative_to(destination)) for path in sorted(destination.rglob('*')) if path.is_file()]
    (destination / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

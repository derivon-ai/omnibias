"""Next-batch projection using only an explicitly frozen low-rank base.

The host producer computes X@A0.T while the consumer may evaluate the current
batch. A0@A0.T is prepared once. U,V and all trainable values remain runtime
arguments to LowRankAdapter; no parameter-dependent value enters this worker.
Actual transfer/compute overlap requires separate device-event evidence.
"""
from dataclasses import dataclass
import hashlib
import time
import jax
import numpy as np
from .bounds import Bound, ConstantBindings
from .certificates import matrix_product
from .prefetch import DoubleBuffer, FrozenDependencies, HostPrepared, SynchronousBuffer
from .reuse import LowRankAdapter

def _fixed_metric(base):
    with np.errstate(over='raise', invalid='raise'):
        metric = base @ base.T
    if not np.all(np.isfinite(metric)):
        raise ArithmeticError('frozen base metric is nonfinite')
    return metric

@dataclass(frozen=True)
class _Project:
    shape: tuple

    def __call__(self, batch, frozen):
        if not isinstance(batch, np.ndarray) or batch.shape != self.shape or batch.dtype != np.float64:
            raise ValueError('next batch must match declared binary64 (rows,dimension) shape')
        if not np.all(np.isfinite(batch)):
            raise ValueError('finite next-batch input required')
        with np.errstate(over='raise', invalid='raise'):
            projection = batch @ frozen['base'].T
        if not np.all(np.isfinite(projection)):
            raise ArithmeticError('saved next-batch projection is nonfinite')
        enclosure = matrix_product(batch, frozen['base'].T)
        return HostPrepared((batch, projection), {'projection': projection, 'lo': enclosure.lo, 'hi': enclosure.hi, 'error': enclosure.error})

@dataclass(frozen=True)
class ProjectedBatch:
    adapter: LowRankAdapter
    preparation: object
    base_token: str
    fixed_metadata: FrozenDependencies

    @property
    def index(self):
        return self.preparation.index

    @property
    def payload(self):
        """Points for worker input plumbing; .adapter retains cached geometry."""
        return self.adapter.points

    @property
    def dependency_token(self):
        return self.base_token

    @property
    def input_token(self):
        return self.preparation.input_token

    @property
    def host_preparation_s(self):
        return self.preparation.host_preparation_s

    @property
    def transfer_dispatch_s(self):
        return self.preparation.transfer_dispatch_s

    def require_dependencies(self, frozen):
        if not isinstance(frozen, FrozenDependencies) or frozen.token != self.base_token:
            raise ValueError('prepared projection belongs to a different immutable base')
        return self.adapter

    def constant_bounds(self):
        """Bind producer-enclosed saved dots without reading device arrays.

        The same owned-transfer path produces each device value and its CPU
        snapshot. Metric enclosures were constructed once at base preparation;
        projection enclosures were constructed in the batch producer. Native
        dot hypotheses are unchanged. Binding metadata is still charged.
        """
        if self.preparation.host_metadata is None:
            raise ValueError('saved projection enclosure metadata required')
        projection = self.preparation.host_metadata.values
        metric = self.fixed_metadata.values
        z_bound = Bound(projection['lo'], projection['hi'], projection['error'])
        g_bound = Bound(metric['lo'], metric['hi'], metric['error'])
        bindings = ConstantBindings()
        for device_value, host_value, bound in ((self.adapter.projection, projection['projection'], z_bound), (self.adapter.metric, metric['metric'], g_bound)):
            bindings.bind_owned_transfer(device_value, host_value, bound).add_legacy(host_value, bound)
        diagonal = Bound(np.diag(g_bound.lo), np.diag(g_bound.hi), np.diag(g_bound.error))
        bindings.add_legacy(np.diag(metric['metric']), diagonal)
        return bindings

    def wait(self):
        return self.preparation.wait()

    def timeline(self):
        return {**self.preparation.timeline(), 'precomputation': 'X@A0.T on host; metric/base shared', 'source_digest': self.adapter.source_digest}

class _ProjectionStream:

    def __init__(self, owner, batches, batch_shape, *, prefetched, trace_annotations):
        cls = DoubleBuffer if prefetched else SynchronousBuffer
        self._owner = owner
        self._stream = cls(batches, _Project(batch_shape), owner.frozen, device=owner.device, max_host_bytes=owner.max_host_bytes - owner.fixed_metadata.nbytes, trace_annotations=trace_annotations)
        self.report = {**owner.report, 'batch_shape': list(batch_shape), 'prefetched': bool(prefetched), 'maximum_future_batches': int(prefetched), 'same_synchronous_and_prefetched_math': True, 'pending_consumer_blocks': 'caller-owned, separately accounted and retained on guard exit'}

    def __iter__(self):
        return self

    def __next__(self):
        record = next(self._stream)
        points, projection = record.require_dependencies(self._owner.frozen)
        digest = hashlib.sha256(('frozen-base-projection/v1:' + record.dependency_token + ':' + record.input_token).encode()).hexdigest()
        adapter = LowRankAdapter(self._owner.base, points, projection, self._owner.metric, digest)
        return ProjectedBatch(adapter, record, self._owner.frozen.token, self._owner.fixed_metadata)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

@dataclass(frozen=True)
class FrozenBase:
    frozen: FrozenDependencies
    base: object
    metric: object
    device: object
    max_host_bytes: int
    metric_bytes: int
    setup_s: float
    fixed_metadata: FrozenDependencies

    @property
    def report(self):
        return {'frozen_dependency_names': ['base'], 'base_token': self.frozen.token, 'base_shape': list(self.base.shape), 'base_host_bytes': self.frozen.nbytes, 'shared_metric_bytes': self.metric_bytes, 'fixed_host_enclosure_bytes': self.fixed_metadata.nbytes, 'shared_device_bytes': int(self.base.nbytes + self.metric.nbytes), 'host_memory_limit_bytes': self.max_host_bytes, 'setup_s': self.setup_s, 'precompute': ['A0@A0.T and its enclosure once per base owner', 'X@A0.T and its enclosure once per batch in producer'], 'live_gradients': 'all supplied U,V,bias,readout,offset remain runtime values', 'gradient_exclusions': ['frozen base', 'frozen input batch'], 'source_digest_scheme': 'SHA256(version, immutable dependency token, input token)', 'saved_arithmetic': 'NumPy binary64 BLAS dot; same conditional RN/gradual-underflow/compatible-dot hypotheses as low_rank_constants', 'certificate_work': 'saved-dot construction charged to producer/base setup; consumer binds immutable CPU enclosures without device readback', 'cuda_overlap': 'unproven without device completion events'}

    def stream(self, batches, *, batch_shape, prefetched=False, trace_annotations=False):
        shape = tuple(batch_shape)
        if len(shape) != 2 or any((type(n) is not int or n < 1 for n in shape)) or shape[1] != self.base.shape[1]:
            raise ValueError('positive fixed batch shape with the frozen base input dimension required')
        needed = self.frozen.nbytes + self.fixed_metadata.nbytes + 2 * 8 * shape[0] * (2 * shape[1] + 5 * self.base.shape[0])
        if needed > self.max_host_bytes:
            raise MemoryError('frozen geometry and projected batch buffers exceed host budget')
        return _ProjectionStream(self, batches, shape, prefetched=prefetched, trace_annotations=trace_annotations)

def prepare_frozen_base(frozen, *, device=None, max_host_bytes=3 * 1024 ** 3):
    """Own fixed geometry from FrozenDependencies({'base': A0}), without U,V."""
    if not jax.config.x64_enabled:
        raise ValueError('binary64 configuration required')
    if not isinstance(frozen, FrozenDependencies) or tuple(frozen.values) != ('base',):
        raise ValueError('exactly one explicit frozen dependency named base is required; trainable factors are prohibited')
    base = frozen.values['base']
    if base.ndim != 2 or min(base.shape) < 1 or base.dtype != np.float64 or (not np.all(np.isfinite(base))):
        raise ValueError('finite binary64 base(H,D) required')
    metric_bytes = 8 * base.shape[0] ** 2
    if type(max_host_bytes) is not int or max_host_bytes <= frozen.nbytes + 4 * metric_bytes:
        raise MemoryError('fixed base, metric and enclosure exceed the declared host budget')
    started = time.perf_counter()
    metric = _fixed_metric(base)
    enclosure = matrix_product(base, base.T)
    metadata = FrozenDependencies({'metric': metric, 'lo': enclosure.lo, 'hi': enclosure.hi, 'error': enclosure.error})
    placed_base, placed_metric = jax.device_put((base, metric), device)
    jax.block_until_ready((placed_base, placed_metric))
    return FrozenBase(frozen, placed_base, placed_metric, placed_base.device, max_host_bytes, metric_bytes, time.perf_counter() - started, metadata)

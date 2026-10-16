"""Bounded per-program certificate reuse with frozen transformable graphs.

This owns a separate maximum-16 LRU, reported independently of planner CACHES.
Only host domain metadata is hashed during qualification; runtime evaluation
does not hash live parameter/input arrays or transfer them to the host.
"""
from collections import OrderedDict, deque
from dataclasses import asdict
import json
import os
import time
import jax
import numpy as np
from .bounds import Bound, ConstantBindings
from .cache_manager import digest_array, digest_record, environment_snapshot, environment_info
from .planner import Domain, Limits, Tolerances, _domain_contains, _emitted_graph_identity, freeze_callable, qualify

def _copy_bound(bound):
    arrays = [np.array(value, copy=True) for value in (bound.lo, bound.hi, bound.error)]
    for array in arrays:
        array.flags.writeable = False
    return Bound(*arrays)

def _copy_domain(domain):
    return Domain(tuple((_copy_bound(bound) for bound in domain.parameters)), _copy_bound(domain.inputs))

def _freeze_bindings(bindings, graphs):
    """Conservatively bind saved constant values after graph constant copying."""
    frozen = ConstantBindings()
    if bindings is None:
        return frozen
    if not isinstance(bindings, ConstantBindings):
        for key, bound in bindings.items():
            frozen.legacy[key] = _copy_bound(bound)
        return frozen
    for key, bound in bindings.legacy.items():
        frozen.legacy[key] = _copy_bound(bound)
    for value, bound, digest in bindings.objects.values():
        source = bindings.owned_host_snapshot(value) if id(value) in bindings.owned_transfers else value
        frozen.add_legacy(source, _copy_bound(bound))
    if bindings.slots:
        raise NotImplementedError('session freezing requires object/value provenance; graph-local slot bindings cannot be transported implicitly')
    return frozen

class QualificationSession:

    def __init__(self, forward, value_gradient, objective, params, points, *, limits=Limits(), tolerances=Tolerances(), constant_bounds=None, max_entries=16, max_bytes=None):
        if type(max_entries) is not int or not 1 <= max_entries <= 16:
            raise ValueError('certificate session requires1..16 entries')
        self.max_entries = max_entries
        self.max_bytes = limits.state_bytes if max_bytes is None else max_bytes
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError('positive certificate byte budget required')
        args = (params, points)
        self.forward = freeze_callable(forward, args)
        self.value_gradient = freeze_callable(value_gradient, args)
        self.objective = freeze_callable(objective, (params, points, self.forward(*args)))
        bindings = constant_bounds if constant_bounds is not None else getattr(forward, 'constant_bounds', None)
        self.constant_bounds = _freeze_bindings(bindings, (self.forward, self.value_gradient))
        self.limits, self.tolerances = (limits, tolerances)
        self.structure = jax.tree.structure(args)
        self.signature = tuple(((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(args)))
        self.environment = environment_snapshot()['digest']
        self.generation = environment_info()['fingerprint_builds']
        self.identity = digest_record({'environment': self.environment, 'pid': os.getpid(), 'forward': _emitted_graph_identity(self.forward.closed_jaxpr), 'value_gradient': _emitted_graph_identity(self.value_gradient.closed_jaxpr), 'objective': _emitted_graph_identity(self.objective.closed_jaxpr), 'constant_bounds': {key: [digest_array(v) for v in (b.lo, b.hi, b.error)] for key, b in self.constant_bounds.legacy.items()}})
        self._certificates = OrderedDict()
        self.hits, self.misses, self.evictions = (0, 0, 0)
        self.events = deque(maxlen=256)

    def _key(self, domain, tolerances, limits):
        return digest_record({'graph_identity': self.identity, 'domain': [[digest_array(a) for a in (b.lo, b.hi, b.error)] for b in (*domain.parameters, domain.inputs)], 'tolerances': asdict(tolerances), 'limits': asdict(limits)})

    def qualify_reusing(self, params, points, domain, *, tolerances=None, limits=None):
        if environment_snapshot()['digest'] != self.environment or environment_info()['fingerprint_builds'] != self.generation:
            self._certificates.clear()
            raise ValueError('environment refreshed; rebuild qualification session')
        args = (params, points)
        if jax.tree.structure(args) != self.structure or tuple(((v.shape, v.dtype) for v in jax.tree.leaves(args))) != self.signature:
            raise ValueError('frozen graph shape/dtype/parameter structure changed')
        tolerances, limits = (tolerances or self.tolerances, limits or self.limits)
        key, started = (self._key(domain, tolerances, limits), time.perf_counter())
        reusable = None
        for previous, item in reversed(self._certificates.items()):
            weaker = all((getattr(tolerances, name) >= getattr(item['tolerances'], name) for name in asdict(tolerances)))
            if weaker and item['limits'] == limits and _domain_contains(item['domain'], domain):
                reusable = previous
                break
        if reusable is not None:
            item = self._certificates.pop(reusable)
            self._certificates[reusable] = item
            self.hits += 1
            event = {'lookup': 'exact' if reusable == key else 'contained_domain', 'certificate_key': reusable, 'lookup_s': time.perf_counter() - started}
            self.events.append(event)
            return {**item['report'], 'session': event}
        self.misses += 1
        report = qualify(self.forward, args, domain, self.objective, tolerances, limits, constant_bounds=self.constant_bounds, emitted_value_and_grad=self.value_gradient)
        event = {'lookup': 'analyzed', 'certificate_key': key, 'analysis_s': time.perf_counter() - started}
        if report.get('admissible'):
            snapshot = _copy_domain(domain)
            byte_count = sum((b.lo.nbytes + b.hi.nbytes + b.error.nbytes for b in (*snapshot.parameters, snapshot.inputs)))
            byte_count += len(json.dumps(report, default=str))
            if byte_count <= self.max_bytes:
                self._certificates[key] = {'report': report, 'domain': snapshot, 'tolerances': tolerances, 'limits': limits, 'bytes': byte_count}
                while len(self._certificates) > self.max_entries or self.info()['reported_bytes'] > self.max_bytes:
                    self._certificates.popitem(last=False)
                    self.evictions += 1
            else:
                event['cache_admission'] = 'refused_certificate_byte_budget'
        self.events.append(event)
        return {**report, 'session': event}

    def clear(self):
        self._certificates.clear()

    def info(self):
        return {'entries': len(self._certificates), 'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions, 'reported_bytes': sum((v['bytes'] for v in self._certificates.values())), 'max_entries': self.max_entries, 'max_bytes': self.max_bytes, 'graph_identity': self.identity, 'environment': self.environment, 'environment_generation': self.generation, 'event_history_limit': 256, 'scope': 'certificate session only; excludes planner/global JAX caches and runtime arrays'}

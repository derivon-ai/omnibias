"""Conditional binary64 forward-error interpreter for actual JAX operation DAGs.

No empirical error is used as a proof bound. Transcendental real intervals use
the existing outward core backend; native primitive errors are hypotheses.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import hashlib
import numpy as np
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import tanh_iv, sigmoid_iv, exp_iv, set_strict_backend
set_strict_backend(True)
U = 2.0 ** (-53)
ETA = np.nextafter(0.0, 1.0)

def up(x):
    return np.nextafter(np.asarray(x, dtype=float), np.inf)

def down(x):
    return np.nextafter(np.asarray(x, dtype=float), -np.inf)

def plus(a, b):
    return up(np.asarray(a) + np.asarray(b))

def times(a, b):
    return up(np.asarray(a) * np.asarray(b))

def magnitude(lo, hi):
    return np.maximum(np.abs(lo), np.abs(hi))

@dataclass
class Bound:
    lo: object
    hi: object
    error: object

    def __post_init__(self):
        if any((np.iscomplexobj(v) for v in (self.lo, self.hi, self.error))):
            raise NotImplementedError('complex values require explicit real-pair bounds')
        self.lo, self.hi, self.error = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (self.lo, self.hi, self.error)])
        if not all((np.all(np.isfinite(v)) for v in (self.lo, self.hi, self.error))):
            raise ArithmeticError('nonfinite interval/error: overflow or inconclusive')
        if np.any(self.lo > self.hi) or np.any(self.error < 0):
            raise ArithmeticError('invalid enclosure')
        if not np.all(np.isfinite(self.lo - self.error)) or not np.all(np.isfinite(self.hi + self.error)):
            raise ArithmeticError('possible overflow of enlarged interval')

    @classmethod
    def point(cls, x):
        x = np.asarray(x)
        if np.iscomplexobj(x):
            raise NotImplementedError('complex values require explicit real-pair bounds')
        if x.dtype.kind in 'iu' and x.size and (np.any(x > 2 ** 53) or (x.dtype.kind == 'i' and np.any(x < -2 ** 53))):
            raise NotImplementedError('integer constant outside exact binary64 analysis range')
        if x.ndim and x.size and np.all(x == x.flat[0]):
            x = np.broadcast_to(np.asarray(x.flat[0]).reshape((1,) * x.ndim), x.shape)
        return cls(x, x, 0.0)

    def enlarged(self):
        return (np.where(self.error == 0, self.lo, down(self.lo - self.error)), np.where(self.error == 0, self.hi, up(self.hi + self.error)))

    def summary(self, atol, rtol):
        lower_abs = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, np.minimum(np.abs(self.lo), np.abs(self.hi)))
        tolerance = down(atol + down(rtol * lower_abs))
        ratio = self.error / tolerance
        return {'max_error_bound': float(np.max(self.error)), 'true_range': [float(self.lo.min()), float(self.hi.max())], 'minimum_tolerance': float(tolerance.min()), 'max_gate_ratio': float(ratio.max()), 'admissible': bool(np.all(self.error <= tolerance))}

def compress_repeated_domains(bound):
    """Share identical enclosure axes without asserting equality of variables.

    Only representation of independent elementwise ranges changes. Contractions
    and reductions still count every original axis element in their bounds.
    This handles a row domain repeated over a whole batch, even if the caller
    materialized the rows instead of providing a broadcast view.
    """
    arrays = [bound.lo, bound.hi, bound.error]
    for axis, size in enumerate(bound.lo.shape):
        if size <= 1 or all((v.strides[axis] == 0 for v in arrays)):
            continue
        first = [np.take(v, [0], axis=axis) for v in arrays]
        if all((np.array_equal(v, np.broadcast_to(f, v.shape)) for v, f in zip(arrays, first))):
            arrays = [np.broadcast_to(f, v.shape) for v, f in zip(arrays, first)]
    return Bound(*arrays)

def join_bounds(a, b):
    """Union of alternative ideal meanings for an identical saved value."""
    if np.shape(a.lo) != np.shape(b.lo):
        raise ValueError('constant binding shape conflict')
    return Bound(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi), np.maximum(a.error, b.error))

def immutable_bound(bound):
    arrays = []
    for source in (bound.lo, bound.hi, bound.error):
        value = np.array(source, dtype=float, copy=True)
        value.setflags(write=False)
        arrays.append(value)
    return Bound(*arrays)

def constant_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()

class ConstantBindings:
    """Provenance-first bindings for saved quantities and bounded constants.

    Slots address a closed-jaxpr constant as (nested_path, constvar_index).
    Object bindings address an owned captured array, and legacy value hashes
    are conservatively joined so equal numerical arrays cannot erase bounds.
    Registering objects retains references, preventing Python id reuse.
    """

    def __init__(self):
        self.slots, self.objects, self.legacy = ({}, {}, {})
        self.owned_transfers = {}

    def bind_slot(self, path, index, bound):
        key = (tuple(path), int(index))
        self.slots[key] = immutable_bound(join_bounds(self.slots[key], bound) if key in self.slots else bound)
        return self

    def bind_object(self, value, bound):
        old = self.objects.get(id(value))
        if old is not None and constant_digest(value) != old[2]:
            raise ValueError('mutated bound constant; rebuild its numerical bindings')
        self.objects[id(value)] = (value, immutable_bound(join_bounds(old[1], bound) if old else bound), constant_digest(value))
        self.owned_transfers.pop(id(value), None)
        return self

    @staticmethod
    def _owned_transfer_metadata(value, snapshot):
        import jax
        if not isinstance(value, jax.Array):
            raise TypeError('owned transfer binding requires a JAX device array')
        if value.is_deleted():
            raise ValueError('owned transferred buffer was deleted or donated')
        if not isinstance(snapshot, np.ndarray) or snapshot.flags.writeable:
            raise ValueError('owned transfer requires an immutable bytes-backed CPU snapshot')
        storage = snapshot
        while isinstance(storage, np.ndarray) and storage.base is not None:
            storage = storage.base
        if not isinstance(storage, bytes):
            raise ValueError('snapshot storage must be immutable bytes, not a read-only view of mutable memory')
        if snapshot.shape != value.shape or snapshot.dtype != value.dtype:
            raise ValueError('owned transfer snapshot shape/dtype mismatch')

    def bind_owned_transfer(self, value, snapshot, bound):
        """Bind an owner-created device copy using its immutable CPU snapshot.

        Trusted-transfer contract: the caller must have created ``value`` by
        transferring these exact snapshot bytes with no intervening numerical
        operation. Metadata alone cannot prove equality of arbitrary arrays.
        This route retains the object and bytes; it neither hashes nor reads
        the device value. Immutable JAX array identity supplies provenance until
        deletion/donation. Other trace/freezing code may still read constants.
        Ordinary bind_object and fallback collision joins remain unchanged.
        """
        self._owned_transfer_metadata(value, snapshot)
        if bound.lo.shape != snapshot.shape:
            raise ValueError('owned transfer bound shape mismatch')
        digest = constant_digest(snapshot)
        old = self.objects.get(id(value))
        if old is not None and old[2] != digest:
            raise ValueError('owned device object cannot acquire a different transfer snapshot')
        self.objects[id(value)] = (value, immutable_bound(join_bounds(old[1], bound) if old else bound), digest)
        self.owned_transfers[id(value)] = snapshot
        return self

    def owned_host_snapshot(self, value):
        obj = self.objects.get(id(value))
        snapshot = self.owned_transfers.get(id(value))
        if obj is None or obj[0] is not value or snapshot is None:
            raise ValueError('no trusted transfer provenance for this object')
        self._owned_transfer_metadata(value, snapshot)
        return snapshot

    def add_legacy(self, value, bound):
        key = constant_digest(value)
        self.legacy[key] = immutable_bound(join_bounds(self.legacy[key], bound) if key in self.legacy else bound)
        return self

    def resolve(self, value, path, index):
        key = (tuple(path), index)
        if key in self.slots:
            return self.slots[key]
        obj = self.objects.get(id(value))
        if obj is not None and obj[0] is value:
            if id(value) in self.owned_transfers:
                self._owned_transfer_metadata(value, self.owned_transfers[id(value)])
            elif constant_digest(value) != obj[2]:
                raise ValueError('mutated bound constant; rebuild its numerical bindings')
            return obj[1]
        return self.legacy.get(constant_digest(value))

    def extend(self, other):
        if other is None:
            return self
        if not isinstance(other, ConstantBindings):
            for key, bound in other.items():
                self.legacy[key] = immutable_bound(join_bounds(self.legacy[key], bound) if key in self.legacy else bound)
            return self
        for (path, index), bound in other.slots.items():
            self.bind_slot(path, index, bound)
        for value, bound, digest in other.objects.values():
            if id(value) in other.owned_transfers:
                self.bind_owned_transfer(value, other.owned_transfers[id(value)], bound)
            else:
                self.bind_object(value, bound)
        for key, bound in other.legacy.items():
            self.legacy[key] = immutable_bound(join_bounds(self.legacy[key], bound) if key in self.legacy else bound)
        return self

def sqrt_bound(a):
    if np.any(a.lo < a.error):
        raise ArithmeticError('sqrt enlarged argument may be negative')
    lo, hi = (down(np.sqrt(a.lo)), up(np.sqrt(a.hi)))
    lo = np.maximum(lo, 0.0)
    lower = np.maximum(down(a.lo - a.error), 0.0)
    denominator = down(np.maximum(down(np.sqrt(a.lo)), 0.0) + np.maximum(down(np.sqrt(lower)), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        propagated = np.where(denominator > 0, up(a.error / denominator), up(np.sqrt(a.error)))
    propagated = np.where(a.error == 0, 0.0, propagated)
    error = plus(propagated, plus(times(U, plus(hi, propagated)), ETA))
    return Bound(lo, hi, error)

def binary(a, b, kind):
    shape = np.broadcast_shapes(a.lo.shape, b.lo.shape)
    arrays = [np.broadcast_to(v, shape) for o in (a, b) for v in (o.lo, o.hi, o.error)]
    collapse = [i for i, s in enumerate(shape) if s > 1 and all((v.strides[i] == 0 for v in arrays))]
    index = tuple((slice(0, 1) if i in collapse else slice(None) for i in range(len(shape))))
    if collapse:
        result = _binary(Bound(*[v[index] for v in arrays[:3]]), Bound(*[v[index] for v in arrays[3:]]), kind)
        return Bound(*[np.broadcast_to(v, shape) for v in (result.lo, result.hi, result.error)])
    return _binary(a, b, kind)

def _binary(a, b, kind):
    if kind == 'add':
        lo, hi = (down(a.lo + b.lo), up(a.hi + b.hi))
        e = plus(a.error, b.error)
    elif kind == 'sub':
        lo, hi = (down(a.lo - b.hi), up(a.hi - b.lo))
        e = plus(a.error, b.error)
    elif kind == 'mul':
        products = np.stack(np.broadcast_arrays(a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi))
        lo, hi = (down(products.min(0)), up(products.max(0)))
        e = plus(plus(times(magnitude(a.lo, a.hi), b.error), times(magnitude(b.lo, b.hi), a.error)), times(a.error, b.error))
    elif kind == 'div':
        bl, bh = b.enlarged()
        if np.any((bl <= 0) & (bh >= 0)):
            raise ArithmeticError('division denominator may contain zero')
        denom = down(np.minimum(np.abs(b.lo), np.abs(b.hi)) * np.minimum(np.abs(bl), np.abs(bh)))
        if np.any(denom <= 0):
            raise ArithmeticError('reciprocal error denominator underflow')
        rec = Bound(down(1 / b.hi), up(1 / b.lo), up(b.error / denom))
        return binary(a, rec, 'mul')
    else:
        raise NotImplementedError(kind)
    e = plus(e, plus(times(U, plus(magnitude(lo, hi), e)), ETA))
    return Bound(lo, hi, e)

def reduction(a, axes):
    axes = tuple((i % a.lo.ndim for i in axes))
    n = math.prod((a.lo.shape[i] for i in axes))
    if n == 0:
        return Bound.point(np.sum(a.lo, axis=axes))
    repeated = [i for i in axes if all((v.strides[i] == 0 for v in (a.lo, a.hi, a.error)))]
    repeat = math.prod((a.lo.shape[i] for i in repeated))
    if repeated:
        slices = tuple((slice(0, 1) if i in repeated else slice(None) for i in range(a.lo.ndim)))
        a = Bound(*[v[slices] for v in (a.lo, a.hi, a.error)])
    compact_n = math.prod((a.lo.shape[i] for i in axes))
    keep = [i for i in range(a.lo.ndim) if i not in axes]
    perm = keep + list(axes)
    shape = tuple((a.lo.shape[i] for i in keep))
    lo = np.transpose(a.lo, perm).reshape(*shape, compact_n)
    hi = np.transpose(a.hi, perm).reshape(*shape, compact_n)
    er = np.transpose(a.error, perm).reshape(*shape, compact_n)
    if n * U >= 1:
        raise ArithmeticError('reduction rounding budget exceeded')
    gamma = up(n * U / (1 - n * U))

    def positive_sum(v):
        return plus(up(np.sum(v, axis=-1) * repeat / float(down(1 - float(gamma)))), up(n * ETA))
    lpad = times(gamma, positive_sum(np.abs(lo)))
    hpad = times(gamma, positive_sum(np.abs(hi)))
    L = down(np.sum(lo, axis=-1) * repeat - lpad)
    H = up(np.sum(hi, axis=-1) * repeat + hpad)
    E = positive_sum(er)
    M = positive_sum(magnitude(lo, hi))
    E = plus(E, plus(times(gamma, plus(M, E)), times(n, ETA)))
    return Bound(L, H, E)

@lru_cache(maxsize=8192)
def real_interval(name, lo, hi):
    fn = {'tanh': tanh_iv, 'logistic': sigmoid_iv, 'exp': exp_iv}[name]
    q = fn(Interval(lo, hi))
    return (q.lo, q.hi)

def unary(a, fn):
    axes = [i for i, s in enumerate(a.lo.shape) if s > 1 and all((v.strides[i] == 0 for v in (a.lo, a.hi, a.error)))]
    idx = tuple((slice(0, 1) if i in axes else slice(None) for i in range(a.lo.ndim)))
    result = fn(Bound(*[v[idx] for v in (a.lo, a.hi, a.error)])) if axes else fn(a)
    return Bound(*[np.broadcast_to(v, a.lo.shape) for v in (result.lo, result.hi, result.error)])

def transcend(a, name):
    return unary(a, lambda v: _transcend(v, name))

def _transcend(a, name):
    enlarged_lo, enlarged_hi = a.enlarged()
    domain = 40.0 if name == 'tanh' else 80.0
    if np.any(magnitude(enlarged_lo, enlarged_hi) > domain):
        raise ArithmeticError('transcendental assumption domain exceeded')
    pairs, inverse = np.unique(np.stack((a.lo.ravel(), a.hi.ravel()), axis=-1), axis=0, return_inverse=True)
    ranges = np.asarray([real_interval(name, float(l), float(h)) for l, h in pairs])
    lo = ranges[inverse, 0].reshape(a.lo.shape)
    hi = ranges[inverse, 1].reshape(a.hi.shape)
    if name == 'tanh':
        lip = 1.0
    elif name == 'logistic':
        lip = 0.25
    else:
        _, eh = a.enlarged()
        lip = np.array([real_interval('exp', float(x), float(x))[1] for x in eh.flat]).reshape(eh.shape)
    propagated = times(lip, a.error)
    e = plus(propagated, plus(times(8 * U, plus(magnitude(lo, hi), propagated)), ETA))
    return Bound(lo, hi, e)

def dot(a, b, dims):
    (ac, bc), (ab, bb) = dims
    af = [i for i in range(a.lo.ndim) if i not in ac and i not in ab]
    bf = [i for i in range(b.lo.ndim) if i not in bc and i not in bb]
    original_shape = tuple((a.lo.shape[i] for i in ab)) + tuple((a.lo.shape[i] for i in af)) + tuple((b.lo.shape[i] for i in bf))
    k = math.prod((a.lo.shape[i] for i in ac))
    if k == 0:
        return Bound.point(np.zeros(original_shape))

    def uniform(o, i):
        return all((v.strides[i] == 0 for v in (o.lo, o.hi, o.error)))
    ca = [i for i in af if uniform(a, i)]
    cb = [i for i in bf if uniform(b, i)]
    for i, j in zip(ac, bc):
        if uniform(a, i) and uniform(b, j):
            ca.append(i)
            cb.append(j)
    for i, j in zip(ab, bb):
        if uniform(a, i) and uniform(b, j):
            ca.append(i)
            cb.append(j)

    def compressed(o, axes):
        idx = tuple((slice(0, 1) if i in axes else slice(None) for i in range(o.lo.ndim)))
        return Bound(*[v[idx] for v in (o.lo, o.hi, o.error)])
    a = compressed(a, ca)
    b = compressed(b, cb)
    bs = tuple((a.lo.shape[i] for i in ab))
    as_ = tuple((a.lo.shape[i] for i in af))
    bs_ = tuple((b.lo.shape[i] for i in bf))
    compact_k = math.prod((a.lo.shape[i] for i in ac))
    repeat = k // compact_k

    def convert(v, side):
        return np.transpose(v, list(ab) + af + list(ac)).reshape(bs + (math.prod(as_), compact_k)) if side == 0 else np.transpose(v, list(bb) + list(bc) + bf).reshape(bs + (compact_k, math.prod(bs_)))
    aa = [convert(v, 0) for v in (a.lo, a.hi, a.error)]
    ba = [convert(v, 1) for v in (b.lo, b.hi, b.error)]
    if 2 * k * U >= 1:
        raise ArithmeticError('dot rounding budget exceeded')
    gamma = float(up(2 * k * U / (1 - 2 * k * U)))

    def positive(A, B):
        return plus(up(np.matmul(A, B) * repeat / float(down(1 - gamma))), up(k * ETA))

    def center(lo, hi):
        m = lo * 0.5 + hi * 0.5
        return (m, up(np.maximum(up(m - lo), up(hi - m))))
    am, ar = center(aa[0], aa[1])
    bm, br = center(ba[0], ba[1])
    middle = np.matmul(am, bm) * repeat
    padding = plus(times(gamma, positive(np.abs(am), np.abs(bm))), up(k * ETA))
    radius = plus(plus(positive(np.abs(am), br), positive(ar, np.abs(bm))), positive(ar, br))
    lower = down(middle - plus(padding, radius))
    upper = up(middle + plus(padding, radius))
    amag = magnitude(aa[0], aa[1])
    bmag = magnitude(ba[0], ba[1])
    e = plus(plus(positive(amag, ba[2]), positive(aa[2], bmag)), positive(aa[2], ba[2]))
    e = plus(e, plus(times(gamma, plus(positive(amag, bmag), e)), up(k * ETA)))
    result = Bound(lower, upper, e)
    return Bound(*[np.broadcast_to(v.reshape(bs + as_ + bs_), original_shape) for v in (result.lo, result.hi, result.error)])

def interpret(closed, inputs, max_equations=100000, constant_bounds=None, max_state_bytes=None):
    """Interpret every node; unknown primitives cause refusal, never omission."""
    from jax._src import core
    env = {}
    count = [0]
    primitives = {}
    peak = [0]
    scan_steps = [0]
    storage = {}
    live_bytes = [0]
    peak_live = [0]
    peak_projected = [0]

    def owner_of(array):
        owner = np.asarray(array)
        seen = set()
        while id(owner) not in seen:
            seen.add(id(owner))
            base = getattr(owner, 'base', None)
            if base is None and isinstance(owner, memoryview):
                base = owner.obj
            if base is None:
                break
            owner = base
        try:
            size = int(owner.nbytes)
        except AttributeError:
            try:
                size = int(memoryview(owner).nbytes)
            except TypeError:
                size = int(np.asarray(array).nbytes)
        return (owner, size)

    def track(values, scope):
        for value in values:
            for array in (value.lo, value.hi, value.error):
                owner, size = owner_of(array)
                key = id(owner)
                if key in scope:
                    continue
                scope.add(key)
                if key in storage:
                    storage[key][2] += 1
                else:
                    storage[key] = [owner, size, 1]
                    live_bytes[0] += size
        peak_live[0] = max(peak_live[0], live_bytes[0])
        if max_state_bytes is not None and live_bytes[0] > max_state_bytes:
            raise ArithmeticError('aggregate live bound-storage budget exhausted')

    def release(scope):
        for key in scope:
            row = storage[key]
            row[2] -= 1
            if not row[2]:
                live_bytes[0] -= row[1]
                del storage[key]
        scope.clear()

    def project(amount):
        peak_projected[0] = max(peak_projected[0], live_bytes[0] + amount)
        if max_state_bytes is not None and live_bytes[0] + amount > max_state_bytes:
            raise ArithmeticError('aggregate projected bound-storage budget exhausted before allocation')

    def exact(v, kind='index'):
        if np.any(v.error) or not np.array_equal(v.lo, v.hi):
            raise ArithmeticError('uncertain ' + kind)
        if kind == 'index' and (np.any(v.lo != np.floor(v.lo)) or np.any(np.abs(v.lo) > 2 ** 53)):
            raise ArithmeticError('index outside exact integer range')
        return v.lo

    def accounted(out, scope):
        size = sum((int(v.lo.size) * 24 for v in out))
        peak[0] = max(peak[0], size)
        track(out, scope)
        return out

    def walk_impl(jaxpr, consts, args, path, scope):

        def constant(index, value):
            found = None
            if isinstance(constant_bounds, ConstantBindings):
                found = constant_bounds.resolve(value, path, index)
            elif constant_bounds:
                found = constant_bounds.get(constant_digest(value))
            if found is not None:
                if np.shape(found.lo) != np.shape(value):
                    raise ValueError('constant binding shape does not match constvar')
                return found
            return Bound.point(value)
        local = dict(zip(jaxpr.constvars, [constant(i, v) for i, v in enumerate(consts)]))
        local.update(zip(jaxpr.invars, args))
        track(local.values(), scope)

        def read(v):
            return Bound.point(v.val) if isinstance(v, core.Literal) else local[v]
        for equation_index, eq in enumerate(jaxpr.eqns):
            count[0] += 1
            if count[0] > max_equations:
                raise ArithmeticError('analysis equation budget exhausted')
            name = eq.primitive.name
            p = eq.params
            vs = [read(v) for v in eq.invars]
            projected = sum((math.prod(v.aval.shape) * 24 for v in eq.outvars if hasattr(v.aval, 'shape')))
            views = {'reshape', 'transpose', 'broadcast_in_dim', 'slice', 'squeeze', 'stop_gradient', 'copy', 'split', 'unstack', 'rev', 'optimization_barrier'}
            nested = {'jit', 'pjit', 'custom_jvp_call', 'custom_vjp_call', 'closed_call', 'remat2', 'cond'}
            if name in views or name in nested:
                projected = 0
            elif name == 'convert_element_type' and np.dtype(p['new_dtype']).kind != 'b':
                projected = 0
            elif name == 'scan':
                projected = sum((math.prod(v.aval.shape) * 24 for v in eq.outvars[p['num_carry']:]))
            project(projected)
            primitives[name] = primitives.get(name, 0) + 1
            if name in ('jit', 'pjit', 'custom_jvp_call', 'custom_vjp_call', 'closed_call', 'remat2'):
                sub = p.get('jaxpr', p.get('call_jaxpr'))
                out = walk(getattr(sub, 'jaxpr', sub), getattr(sub, 'consts', ()), vs, path + (f'{equation_index}:{name}',))
            elif name == 'optimization_barrier':
                out = vs
            elif name == 'scan':
                sub = p['jaxpr']
                nc = p['num_consts']
                ncarry = p['num_carry']
                length = p['length']
                if not isinstance(length, int) or length < 0:
                    raise NotImplementedError('finite static scan length required')
                if length > max_equations - count[0]:
                    raise ArithmeticError('scan iteration budget exceeds analysis equation budget')
                constants = vs[:nc]
                carry = vs[nc:nc + ncarry]
                sequences = vs[nc + ncarry:]
                stacked = [None] * length
                carry_scope = set()
                try:
                    indices = range(length - 1, -1, -1) if p.get('reverse', False) else range(length)
                    for i in indices:
                        scan_steps[0] += 1
                        count[0] += 1
                        if count[0] > max_equations:
                            raise ArithmeticError('analysis equation budget exhausted')
                        row = [Bound(v.lo[i], v.hi[i], v.error[i]) for v in sequences]
                        step = walk(sub.jaxpr, sub.consts, [*constants, *carry, *row], path + (f'{equation_index}:scan',))
                        release(carry_scope)
                        carry = step[:ncarry]
                        stacked[i] = step[ncarry:]
                        track(carry, carry_scope)
                        track(stacked[i], scope)
                    if length:
                        project(projected)
                        ys = [Bound(*[np.stack([getattr(step[j], attr) for step in stacked], axis=0) for attr in ('lo', 'hi', 'error')]) for j in range(len(stacked[0]))]
                    else:
                        ys = [Bound.point(np.zeros(v.aval.shape, dtype=float)) for v in eq.outvars[ncarry:]]
                    out = [*carry, *ys]
                    track(out, scope)
                finally:
                    release(carry_scope)
            elif name == 'cond':
                index = int(exact(vs[0]).item())
                if not 0 <= index < len(p['branches']):
                    raise ArithmeticError('conditional branch index outside bounds')
                sub = p['branches'][index]
                out = walk(sub.jaxpr, sub.consts, vs[1:], path + (f'{equation_index}:cond:{index}',))
            elif name in ('add', 'add_any', 'sub', 'mul', 'div'):
                kind = eq.outvars[0].aval.dtype.kind
                if kind in 'iu':
                    av, bv = [exact(v).astype(np.int64).astype(object) for v in vs]
                    if name == 'div':
                        if np.any(bv == 0):
                            raise ArithmeticError('integer division by zero')
                        quotient = np.frompyfunc(lambda a, b: abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1), 2, 1)
                        value = quotient(av, bv)
                    else:
                        operation = {'add': np.add, 'add_any': np.add, 'sub': np.subtract, 'mul': np.multiply}[name]
                        value = operation(av, bv)
                    info = np.iinfo(eq.outvars[0].aval.dtype)
                    if np.any(value < info.min) or np.any(value > info.max) or np.any(np.abs(value) > 2 ** 53):
                        raise ArithmeticError('integer operation overflow/exactness refusal')
                    out = [Bound.point(value)]
                else:
                    out = [binary(*vs, 'add' if name == 'add_any' else name)]
            elif name in ('and', 'or', 'xor', 'not'):
                if eq.outvars[0].aval.dtype.kind == 'b':
                    values = [exact(v, 'boolean').astype(bool) for v in vs]
                    fn = {'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor, 'not': np.logical_not}[name]
                else:
                    values = [exact(v).astype(eq.outvars[0].aval.dtype) for v in vs]
                    fn = {'and': np.bitwise_and, 'or': np.bitwise_or, 'xor': np.bitwise_xor, 'not': np.bitwise_not}[name]
                out = [Bound.point(fn(*values))]
            elif name == 'is_finite':
                out = [Bound.point(np.ones(vs[0].lo.shape, dtype=bool))]
            elif name in ('reduce_and', 'reduce_or'):
                if eq.outvars[0].aval.dtype.kind == 'b':
                    value = exact(vs[0], 'boolean').astype(bool)
                    result = (np.all if name == 'reduce_and' else np.any)(value, axis=p['axes'])
                else:
                    value = exact(vs[0]).astype(eq.outvars[0].aval.dtype)
                    result = (np.bitwise_and if name == 'reduce_and' else np.bitwise_or).reduce(value, axis=p['axes'])
                out = [Bound.point(result)]
            elif name == 'sqrt':
                out = [unary(vs[0], sqrt_bound)]
            elif name in ('min', 'max'):
                fn = np.minimum if name == 'min' else np.maximum
                out = [Bound(fn(vs[0].lo, vs[1].lo), fn(vs[0].hi, vs[1].hi), np.maximum(vs[0].error, vs[1].error))]
            elif name in ('reduce_max', 'reduce_min'):
                fn = np.max if name == 'reduce_max' else np.min
                out = [Bound(fn(vs[0].lo, axis=p['axes']), fn(vs[0].hi, axis=p['axes']), np.max(vs[0].error, axis=p['axes']))]
            elif name == 'neg':
                out = [unary(vs[0], lambda a: Bound(-a.hi, -a.lo, a.error))]
            elif name == 'abs':

                def absolute(a):
                    lo = np.where((a.lo <= 0) & (a.hi >= 0), 0.0, np.minimum(np.abs(a.lo), np.abs(a.hi)))
                    return Bound(lo, magnitude(a.lo, a.hi), a.error)
                out = [unary(vs[0], absolute)]
            elif name == 'iota':
                shape = p['shape']
                axis = p['dimension']
                v = np.arange(shape[axis], dtype=p['dtype'])
                s = [1] * len(shape)
                s[axis] = shape[axis]
                out = [Bound.point(np.broadcast_to(v.reshape(s), shape))]
            elif name in ('lt', 'le', 'gt', 'ge', 'eq', 'ne'):
                al, ah = vs[0].enlarged()
                bl, bh = vs[1].enlarged()
                true = {'lt': ah < bl, 'le': ah <= bl, 'gt': al > bh, 'ge': al >= bh, 'eq': (al == ah) & (bl == bh) & (al == bl), 'ne': (ah < bl) | (al > bh)}[name]
                false = {'lt': al >= bh, 'le': al > bh, 'gt': ah <= bl, 'ge': ah < bl, 'eq': (ah < bl) | (al > bh), 'ne': (al == ah) & (bl == bh) & (al == bl)}[name]
                points = (vs[0].lo == vs[0].hi) & (vs[1].lo == vs[1].hi) & (vs[0].error == 0) & (vs[1].error == 0)
                compare = {'lt': np.less, 'le': np.less_equal, 'gt': np.greater, 'ge': np.greater_equal, 'eq': np.equal, 'ne': np.not_equal}[name]
                exact_true = compare(vs[0].lo, vs[1].lo)
                true = np.where(points, exact_true, true)
                false = np.where(points, ~exact_true, false)
                if not np.all(true | false):
                    raise ArithmeticError('branch sign cannot be certified on this domain')
                out = [Bound.point(true)]
            elif name == 'select_n':
                condition = vs[0]
                if not np.array_equal(condition.lo, condition.hi) or np.any(condition.error):
                    raise ArithmeticError('uncertain select condition')
                which = exact(condition).astype(int)
                if np.any(which < 0) or np.any(which >= len(vs) - 1):
                    raise ArithmeticError('select index outside range')
                out = [Bound(*[np.choose(which, [getattr(v, a) for v in vs[1:]]) for a in ('lo', 'hi', 'error')])]
            elif name in ('tanh', 'logistic', 'exp'):
                out = [transcend(vs[0], name)]
            elif name == 'integer_pow':
                n = p['y']
                out = [Bound.point(np.broadcast_to(np.asarray(1.0), vs[0].lo.shape))]
                for _ in range(abs(n)):
                    out = [binary(out[0], vs[0], 'mul')]
                if n < 0:
                    out = [binary(Bound.point(np.broadcast_to(np.asarray(1.0), vs[0].lo.shape)), out[0], 'div')]
            elif name == 'reduce_sum':
                out = [reduction(vs[0], p['axes'])]
            elif name == 'dot_general':
                out = [dot(*vs, p['dimension_numbers'])]
            elif name in ('reshape', 'transpose', 'broadcast_in_dim', 'slice', 'squeeze', 'convert_element_type', 'stop_gradient', 'copy'):
                if name == 'convert_element_type':
                    source = np.dtype(eq.invars[0].aval.dtype)
                    target = np.dtype(p['new_dtype'])
                    if target.kind == 'f' and target != np.dtype('float64'):
                        raise NotImplementedError('non-binary64 floating conversion')
                    if source.kind == 'f' and source != np.dtype('float64'):
                        raise NotImplementedError('non-binary64 floating input')
                    if source.kind in 'iu' and target.kind == 'f' and np.any(magnitude(vs[0].lo, vs[0].hi) > 2 ** 53):
                        raise NotImplementedError('integer narrowing requires a conversion-error bound')
                    if target.kind in 'iu' and (not (np.array_equal(vs[0].lo, vs[0].hi) and np.all(vs[0].lo == np.floor(vs[0].lo)) and (not np.any(vs[0].error)))):
                        raise NotImplementedError('nonconstant integer conversion')
                    if target.kind in 'iu':
                        limits = np.iinfo(target)
                        if np.any(vs[0].lo < limits.min) or np.any(vs[0].hi > limits.max):
                            raise NotImplementedError('narrowing integer conversion outside target range')
                    if target.kind == 'b' and (not (np.array_equal(vs[0].lo, vs[0].hi) and (not np.any(vs[0].error)))):
                        raise NotImplementedError('uncertain conversion to boolean')

                def transform(v):
                    if name == 'reshape':
                        if p.get('dimensions') is not None:
                            v = np.transpose(v, p['dimensions'])
                        return v.reshape(p['new_sizes'])
                    if name == 'transpose':
                        return np.transpose(v, p['permutation'])
                    if name == 'broadcast_in_dim':
                        shape = [1] * len(p['shape'])
                        for i, d in enumerate(p['broadcast_dimensions']):
                            shape[d] = v.shape[i]
                        return np.broadcast_to(v.reshape(shape), p['shape'])
                    if name == 'slice':
                        return v[tuple((slice(a, b, c) for a, b, c in zip(p['start_indices'], p['limit_indices'], p.get('strides') or [1] * v.ndim)))]
                    if name == 'squeeze':
                        return np.squeeze(v, axis=p['dimensions'])
                    return v
                if name == 'convert_element_type' and np.dtype(p['new_dtype']).kind == 'b':
                    out = [Bound.point(vs[0].lo.astype(bool))]
                else:
                    out = [Bound(*[transform(v) for v in (vs[0].lo, vs[0].hi, vs[0].error)])]
            elif name in ('dynamic_slice', 'dynamic_update_slice'):
                operand = vs[0]
                starts = vs[1:] if name == 'dynamic_slice' else vs[2:]
                sizes = tuple(p['slice_sizes']) if name == 'dynamic_slice' else vs[1].lo.shape
                indices = tuple((int(exact(v).item()) for v in starts))
                indices = tuple((max(0, min(i, size - width)) for i, size, width in zip(indices, operand.lo.shape, sizes)))
                slices = tuple((slice(i, i + size) for i, size in zip(indices, sizes)))
                if name == 'dynamic_slice':
                    out = [Bound(*[getattr(operand, a)[slices] for a in ('lo', 'hi', 'error')])]
                else:
                    arrays = []
                    for a in ('lo', 'hi', 'error'):
                        ar = getattr(operand, a).copy()
                        ar[slices] = getattr(vs[1], a)
                        arrays.append(ar)
                    out = [Bound(*arrays)]
            elif name == 'gather':
                operand, indices = vs
                indices = exact(indices).astype(np.int64)
                dims = p['dimension_numbers']
                sizes = p['slice_sizes']
                shape = eq.outvars[0].aval.shape
                offset = tuple(dims.offset_dims)
                batch = tuple((i for i in range(len(shape)) if i not in offset))
                window = tuple((i for i in range(operand.lo.ndim) if i not in dims.collapsed_slice_dims and i not in dims.operand_batching_dims))
                arrays = [np.empty(shape, dtype=float) for _ in range(3)]
                mode = getattr(p['mode'], 'name', str(p['mode']))
                for out_index in np.ndindex(shape):
                    batch_index = tuple((out_index[i] for i in batch))
                    vector = indices[batch_index] if indices.ndim > 1 else indices
                    starts = [0] * operand.lo.ndim
                    for axis, start in zip(dims.start_index_map, np.ravel(vector)):
                        starts[axis] = int(start)
                    for axis, index_axis in zip(dims.operand_batching_dims, dims.start_indices_batching_dims):
                        starts[axis] = batch_index[index_axis]
                    valid = all((0 <= start <= extent - width for start, extent, width in zip(starts, operand.lo.shape, sizes)))
                    if not valid and mode == 'CLIP':
                        starts = [max(0, min(start, extent - width)) for start, extent, width in zip(starts, operand.lo.shape, sizes)]
                    elif not valid:
                        fill = p.get('fill_value')
                        if mode not in ('FILL_OR_DROP', 'FILL') or fill is None or (not np.isfinite(fill)):
                            raise ArithmeticError('gather index outside proved finite region')
                        for array, value in zip(arrays, (fill, fill, 0.0)):
                            array[out_index] = value
                        continue
                    for axis, offset_axis in zip(window, offset):
                        starts[axis] += out_index[offset_axis]
                    for array, attr in zip(arrays, ('lo', 'hi', 'error')):
                        array[out_index] = getattr(operand, attr)[tuple(starts)]
                out = [Bound(*arrays)]
            elif name in ('scatter', 'scatter-add'):
                if name == 'scatter' and p.get('update_jaxpr') is not None:
                    raise NotImplementedError('custom scatter combiner requires its own proof')
                operand, indices, updates = vs
                indices = exact(indices).astype(np.int64)
                dims = p['dimension_numbers']
                window = tuple(dims.update_window_dims)
                batch = tuple((i for i in range(updates.lo.ndim) if i not in window))
                operand_window = tuple((i for i in range(operand.lo.ndim) if i not in dims.inserted_window_dims and i not in dims.operand_batching_dims))
                widths = [1] * operand.lo.ndim
                for axis, update_axis in zip(operand_window, window):
                    widths[axis] = updates.lo.shape[update_axis]
                mode = getattr(p['mode'], 'name', str(p['mode']))
                destinations = {}
                for update_index in np.ndindex(updates.lo.shape):
                    batch_index = tuple((update_index[i] for i in batch))
                    vector = indices[batch_index] if indices.ndim > 1 else indices
                    starts = [0] * operand.lo.ndim
                    for axis, start in zip(dims.scatter_dims_to_operand_dims, np.ravel(vector)):
                        starts[axis] = int(start)
                    for axis, index_axis in zip(dims.operand_batching_dims, dims.scatter_indices_batching_dims):
                        starts[axis] = batch_index[index_axis]
                    valid = all((0 <= start <= extent - width for start, extent, width in zip(starts, operand.lo.shape, widths)))
                    if not valid and mode == 'CLIP':
                        starts = [max(0, min(start, extent - width)) for start, extent, width in zip(starts, operand.lo.shape, widths)]
                    elif not valid:
                        if mode in ('FILL_OR_DROP', 'DROP'):
                            continue
                        raise ArithmeticError('scatter index outside proved region')
                    for axis, update_axis in zip(operand_window, window):
                        starts[axis] += update_index[update_axis]
                    destination = tuple(starts)
                    destinations.setdefault(destination, []).append(update_index)
                arrays = [getattr(operand, attr).copy() for attr in ('lo', 'hi', 'error')]
                for destination, source_indices in destinations.items():
                    if name == 'scatter':
                        if len(source_indices) != 1:
                            raise NotImplementedError('overlapping scatter assignment is not a unique ideal operator')
                        for array, attr in zip(arrays, ('lo', 'hi', 'error')):
                            array[destination] = getattr(updates, attr)[source_indices[0]]
                    else:
                        terms = []
                        for attr in ('lo', 'hi', 'error'):
                            terms.append(np.asarray([getattr(operand, attr)[destination]] + [getattr(updates, attr)[index] for index in source_indices]))
                        summed = reduction(Bound(*terms), (0,))
                        for array, attr in zip(arrays, ('lo', 'hi', 'error')):
                            array[destination] = getattr(summed, attr)
                out = [Bound(*arrays)]
            elif name == 'unstack':
                axis = p['axis']
                out = []
                for i in range(vs[0].lo.shape[axis]):
                    out.append(Bound(*[np.take(getattr(vs[0], a), i, axis=axis) for a in ('lo', 'hi', 'error')]))
            elif name == 'rev':
                out = [Bound(*[np.flip(getattr(vs[0], a), axis=p['dimensions']) for a in ('lo', 'hi', 'error')])]
            elif name == 'concatenate':
                out = [Bound(*[np.concatenate([getattr(v, a) for v in vs], axis=p['dimension']) for a in ('lo', 'hi', 'error')])]
            elif name == 'stack':
                out = [Bound(*[np.stack([getattr(v, a) for v in vs], axis=p['axis']) for a in ('lo', 'hi', 'error')])]
            elif name == 'pad':
                configuration = p['padding_config']
                old = vs[0]
                original_shape = tuple((l + h + s + (s - 1) * gap for s, (l, h, gap) in zip(old.lo.shape, configuration)))
                collapse = [i for i, c in enumerate(configuration) if c == (0, 0, 0) and all((v.strides[i] == 0 for v in (old.lo, old.hi, old.error)))]
                compact = tuple((slice(0, 1) if i in collapse else slice(None) for i in range(old.lo.ndim)))
                old = Bound(*[v[compact] for v in (old.lo, old.hi, old.error)])
                shape = tuple((l + h + s + (s - 1) * gap for s, (l, h, gap) in zip(old.lo.shape, configuration)))
                source = []
                dest = []
                for s, (l, h, gap), end in zip(old.lo.shape, configuration, shape):
                    d = l + np.arange(s) * (gap + 1)
                    valid = (d >= 0) & (d < end)
                    source.append(np.nonzero(valid)[0])
                    dest.append(d[valid])
                arrays = []
                for attr in ('lo', 'hi', 'error'):
                    ar = np.full(shape, float(getattr(vs[1], attr)))
                    ar[np.ix_(*dest)] = getattr(old, attr)[np.ix_(*source)]
                    arrays.append(np.broadcast_to(ar, original_shape))
                out = [Bound(*arrays)]
            elif name == 'split':
                axis = p['axis']
                sizes = p['sizes']
                out = []
                offset = 0
                for size in sizes:
                    slices = [slice(None)] * vs[0].lo.ndim
                    slices[axis] = slice(offset, offset + size)
                    offset += size
                    out.append(Bound(*[v[tuple(slices)] for v in (vs[0].lo, vs[0].hi, vs[0].error)]))
            else:
                raise NotImplementedError('unbounded primitive ' + name)
            accounted(out, scope)
            local.update(zip(eq.outvars, out))
        return [read(v) for v in jaxpr.outvars]

    def walk(jaxpr, consts, args, path=()):
        scope = set()
        try:
            return walk_impl(jaxpr, consts, args, path, scope)
        finally:
            release(scope)
    input_scope = set()
    try:
        track(inputs, input_scope)
        compressed_inputs = [compress_repeated_domains(v) for v in inputs]
        outputs = walk(closed.jaxpr, closed.consts, compressed_inputs)
    finally:
        release(input_scope)
    return (outputs, {'equations': count[0], 'primitives': primitives, 'scan_steps': scan_steps[0], 'largest_equation_bound_bytes': peak[0], 'estimated_numpy_bound_live_peak_bytes': peak_live[0], 'projected_numpy_bound_peak_upper_bytes': peak_projected[0], 'storage_accounting': 'deduplicated ultimate NumPy owners across active nested scopes; includes inputs/constants, excludes interpreter arithmetic temporaries, Python/JAX memory and external RSS', 'constant_binding_contract': 'slot/object provenance preferred; legacy collisions must be joined'})

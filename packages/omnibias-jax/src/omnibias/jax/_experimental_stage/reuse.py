"""Frozen readouts and exact frozen-base adapters. Real binary64 research API."""
from dataclasses import dataclass
import hashlib
import time
import jax
import jax.numpy as jnp
import numpy as np

def digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()

def snapshot(array, device=None):
    if np.iscomplexobj(array):
        raise ValueError('real frozen dependencies required')
    value = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite frozen dependency')
    return jax.device_put(value, device)

@dataclass(frozen=True)
class FrozenReadout:
    factor: object
    target: object
    constant: object
    source_digest: str
    report: dict
    statistics: object = None

    def objective(self, readout):
        if readout.dtype != jnp.float64 or readout.shape != (self.factor.shape[1],):
            raise ValueError('binary64 readout of declared shape required')
        residual = self.factor @ readout - self.target
        return 0.5 * (jnp.sum(residual ** 2) + self.constant)

    def value_and_grad(self, readout):
        return jax.value_and_grad(self.objective)(readout)

    def solve(self, rcond=None, *, tolerances=None):
        from .solve_certificate import certify_solution
        solution, residuals, rank, singular = jnp.linalg.lstsq(self.factor, self.target, rcond=rcond)
        jax.block_until_ready(solution)
        info = {'rank': rank, 'singular_values': singular, 'residuals': residuals, 'qualification': 'pending unique least-squares a-posteriori certificate'}
        retained = np.asarray(singular)[:int(rank)]
        info['condition'] = float(retained[0] / retained[-1]) if len(retained) else 1.0
        if self.statistics is None:
            info.update(status='inconclusive_missing_statistics', qualified=False, reason='preparation did not retain bounded sufficient statistics')
            return (None, info)
        if self.statistics.source_digest != self.source_digest:
            info.update(status='inconclusive_source_mismatch', qualified=False, reason='solve statistics belong to different frozen dependencies')
            return (None, info)
        certificate = certify_solution(self.statistics, np.asarray(solution), numerical_rank=int(rank), tolerances=tolerances)
        info.update(certificate)
        info['qualification'] = 'conditional coefficient, objective-excess and stationarity bounds'
        return (solution if certificate['qualified'] else None, info)

def prepare_frozen_readout(design, target, *, placement='host', device=None):
    if not jax.config.x64_enabled:
        raise ValueError('binary64 configuration required')
    start = time.perf_counter()
    if np.iscomplexobj(design) or np.iscomplexobj(target):
        raise ValueError('real design/target required')
    f, y = (np.array(design, dtype=np.float64, copy=True), np.array(target, dtype=np.float64, copy=True))
    if f.ndim != 2 or y.shape != (f.shape[0],) or min(f.shape) < 1:
        raise ValueError('real nonempty F(M,H), y(M) required')
    if not np.isfinite(f).all() or not np.isfinite(y).all():
        raise ValueError('finite frozen data required')
    from .solve_certificate import prepare_statistics
    statistics_started = time.perf_counter()
    statistics = prepare_statistics(f, y)
    statistics_s = time.perf_counter() - statistics_started
    augmented = np.column_stack((f, y))
    transfer = time.perf_counter()
    if placement == 'host':
        factor = np.linalg.qr(augmented, mode='r')
    elif placement == 'device':
        factor = jnp.linalg.qr(snapshot(augmented, device), mode='r')
        jax.block_until_ready(factor)
    else:
        raise ValueError('placement is host or device')
    h = f.shape[1]
    rows = min(f.shape[0], h)
    if placement == 'device':
        r, d = (factor[:rows, :h], factor[:rows, h])
        rho = factor[h, h] ** 2 if f.shape[0] > h else jnp.asarray(0.0, dtype=jnp.float64)
    else:
        r = snapshot(factor[:rows, :h], device)
        d = snapshot(factor[:rows, h], device)
        rho = snapshot(factor[h, h] ** 2 if f.shape[0] > h else np.asarray(0.0), device)
    jax.block_until_ready((r, d, rho))
    if not np.isfinite(np.asarray(factor)).all() or not np.isfinite(np.asarray(rho)):
        raise ValueError('factorization or residual constant overflow')
    return FrozenReadout(r, d, rho, digest(f, y), {'placement': placement, 'setup_s': time.perf_counter() - start, 'factor_and_transfer_s': time.perf_counter() - transfer, 'scope': 'readout-only; immutable owned factors; replacement creates new plan', 'rows': f.shape[0], 'columns': h, 'device_factor_bytes': r.nbytes + d.nbytes + rho.nbytes, 'host_statistics_bytes': statistics.host_bytes, 'statistics_setup_s': statistics_s, 'persistent_bytes': r.nbytes + d.nbytes + rho.nbytes + statistics.host_bytes, 'qualification': 'pending independent factor and emitted-graph bounds; solve has separate a-posteriori certificate'}, statistics)

@dataclass(frozen=True)
class LowRankAdapter:
    base: object
    points: object
    projection: object
    metric: object
    source_digest: str

    def geometry(self, factors):
        u, v = factors
        if u.ndim != 2 or v.ndim != 2 or u.shape[0] != self.base.shape[0] or (v.shape[0] != self.base.shape[1]) or (u.shape[1] != v.shape[1]) or (u.dtype != jnp.float64) or (v.dtype != jnp.float64):
            raise ValueError('live U(H,r), V(D,r) binary64 factors required')
        av = self.base @ v
        projection = self.projection + self.points @ v @ u.T
        metric = self.metric + av @ u.T + u @ av.T + u @ (v.T @ v) @ u.T
        return (projection, metric)

def prepare_low_rank(base, points, *, device=None):
    if not jax.config.x64_enabled:
        raise ValueError('binary64 configuration required')
    a, x = (snapshot(base, device), snapshot(points, device))
    if a.ndim != 2 or x.ndim != 2 or a.shape[1] != x.shape[1]:
        raise ValueError('base(H,D), points(B,D) required')
    z, g = (x @ a.T, a @ a.T)
    jax.block_until_ready((z, g))
    return LowRankAdapter(a, x, z, g, digest(base, points))

def low_rank_objective(geometry, factors):
    z, metric = geometry(factors)
    return jnp.mean(jnp.tanh(z) ** 2) + 0.001 * jnp.sum(metric ** 2)

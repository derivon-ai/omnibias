"""Supplemental operator composition over a frozen-base factorization.

Research only; no promotion is implied by the geometry benchmark. The callable
accepts live (U,V,bias,readout,offset), and owns its frozen input points through
the supplied adapter. No new points/base gradients are accepted.
"""
from fractions import Fraction
import math
import jax.numpy as jnp
from .providers import tower

def build_low_rank_ridge(adapter, operators, provider='parity'):
    dimension = adapter.base.shape[1]
    for op in operators:
        op.check_dimension(dimension)
        if op.order > 16 or op.mode == 'directional':
            raise NotImplementedError('constant-coordinate operators through16 required')
    rows = []
    orders = set()
    for op in operators:
        if op.mode == 'laplacian':
            rows.append(('laplacian', op.power))
            orders.add(op.order)
        else:
            terms = []
            for alpha, coefficient in op.resolve_terms(dimension):
                value = float(coefficient)
                if not math.isfinite(value) or Fraction(value) != coefficient:
                    raise NotImplementedError('coefficient conversion obligation')
                terms.append((alpha, value))
                orders.add(sum(alpha))
            rows.append(('polynomial', terms))
    orders = tuple(sorted(orders))
    derivatives = tower(orders, provider)
    base_norm = jnp.diag(adapter.metric)

    def function(params):
        u, v, bias, readout, offset = params
        projection = adapter.projection + adapter.points @ v @ u.T + bias
        av = adapter.base @ v
        norms = base_norm + 2 * jnp.sum(av * u, axis=1) + jnp.sum(u @ (v.T @ v) * u, axis=1)
        values = dict(zip(orders, derivatives(projection)))
        answers = []
        for kind, spec in rows:
            if kind == 'laplacian':
                answer = values[2 * spec] @ (readout * norms ** spec)
                if spec == 0:
                    answer = answer + offset
            else:
                grouped = {}
                zero = 0.0
                for alpha, coefficient in spec:
                    factor = jnp.ones_like(readout) * coefficient
                    for axis, n in enumerate(alpha):
                        if n:
                            factor = factor * (adapter.base[:, axis] + u @ v[axis]) ** n
                    order = sum(alpha)
                    grouped[order] = grouped.get(order, 0) + factor
                    if order == 0:
                        zero += coefficient
                answer = sum((values[n] @ (readout * factor) for n, factor in grouped.items())) + zero * offset
            answers.append(answer)
        return tuple(answers)
    function.demands = {'activation_orders': list(orders), 'gradient_orders': sorted(set(orders) | {n + 1 for n in orders}), 'geometry': 'projection and diagonal metric only; live factors and operator coefficients', 'frozen': 'base and input points; first gradients only for supplied runtime parameters', 'qualification': 'pending saved-projection and diagonal-metric relational bounds', 'performance': 'not part of frozen geometry benchmark; no integration claim'}
    return function

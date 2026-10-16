"""Objective-only streaming with explicit row-local sufficient statistics.

This research implementation makes no performance or numerical-certificate claim.
Qualification must analyze the actual value and first-gradient callables below.
The row callback receives one row, not a batch; global objectives must use the
explicit statistic finalizer or the existing full-output path.
"""
from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from .bundles import _bottleneck, _ridge, _separable, coefficient_bounds, metadata

def _zero_regularizer(params):
    return jnp.asarray(0.0, dtype=jax.tree.leaves(params)[0].dtype)

class _Objective:

    def full_objective(self, params, points, outputs):
        statistics = jax.vmap(self.row_statistics, in_axes=(None, 0, 0))(params, points, outputs)
        return self.finalize(params, jnp.sum(statistics, axis=0), points.shape[0])

@dataclass(frozen=True)
class StatisticObjective(_Objective):
    """row_statistics(params, point, operator_values) -> fixed-shape array.

    finalize(params, sum_of_row_statistics, row_count) -> scalar. Parameters may
    occur in both callbacks. The first derivative uses two bounded-memory passes.
    """
    row_statistics: Callable
    finalize: Callable
    name: str = 'sufficient_statistics'

@dataclass(frozen=True)
class SumObjective(_Objective):
    row_loss: Callable
    regularizer: Callable = _zero_regularizer
    name: str = 'sum'

    def row_statistics(self, params, point, values):
        return jnp.reshape(self.row_loss(params, point, values), (1,))

    def finalize(self, params, totals, row_count):
        return totals[0] + self.regularizer(params)

@dataclass(frozen=True)
class WeightedMeanObjective(_Objective):
    """Sum(weight * row_loss) / sum(weight), plus a global regularizer.

    The user declares nonnegative weights and a strictly positive total weight.
    Parameters in weight_fn are differentiated. Qualification must establish a
    nonzero denominator; runtime domain/finiteness checks belong to the plan.
    """
    row_loss: Callable
    weight_fn: Callable
    regularizer: Callable = _zero_regularizer
    name: str = 'weighted_mean'

    def row_statistics(self, params, point, values):
        weight = self.weight_fn(params, point)
        return jnp.stack((weight * self.row_loss(params, point, values), weight))

    def finalize(self, params, totals, row_count):
        return totals[0] / totals[1] + self.regularizer(params)

def _zeros(tree):
    return jax.tree.map(jnp.zeros_like, tree)

def _add(a, b):
    return jax.tree.map(jnp.add, a, b)

def _chunks(points, chunk_size):
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError('nonempty rank-two points required')
    length = points.shape[0]
    size = min(chunk_size, length)
    count = (length + size - 1) // size
    pad = count * size - length
    padded = jnp.concatenate((points, jnp.broadcast_to(points[0], (pad, points.shape[1]))))
    masks = (jnp.arange(count * size) < length).reshape(count, size)
    return (padded.reshape(count, size, points.shape[1]), masks)

class StreamingProgram:

    def __init__(self, prepare, evaluate, objective, chunk_size, demands, rematerialize):
        self.prepare = prepare
        self.evaluate = jax.checkpoint(evaluate) if rematerialize else evaluate
        self.objective = objective
        self.chunk_size = chunk_size
        self.report = {'execution': 'objective-only lax.scan', 'chunk_size': chunk_size, 'objective': objective.name, 'requested_orders': list(demands), 'gradient_orders': sorted(set(demands) | {n + 1 for n in demands}), 'prepare_pullbacks_per_gradient': 1, 'gradient_passes': 1 if isinstance(objective, SumObjective) else 2, 'qualification': 'requires actual emitted graph qualification', 'working_state': 'chunk intermediates plus parameter/prepared adjoints; full input points remain resident', 'rematerialize': rematerialize, 'higher_parameter_derivatives': 'unsupported'}

    def _statistics(self, params, prepared, chunk, mask):
        outputs = self.evaluate(params, chunk, prepared)
        stats = jax.vmap(self.objective.row_statistics, in_axes=(None, 0, 0))(params, chunk, outputs)
        if stats.ndim != 2:
            raise ValueError('row statistics must return a one-dimensional array')
        return jnp.sum(jnp.where(mask[:, None], stats, jnp.zeros_like(stats)), axis=0)

    def _totals(self, params, points, prepared):
        chunks, masks = _chunks(points, self.chunk_size)
        shape = jax.eval_shape(self._statistics, params, prepared, chunks[0], masks[0])

        def body(total, item):
            chunk, mask = item
            return (total + self._statistics(params, prepared, chunk, mask), None)
        return jax.lax.scan(body, jnp.zeros(shape.shape, shape.dtype), (chunks, masks))[0]

    def value(self, params, points):
        prepared = self.prepare(params)
        totals = self._totals(params, points, prepared)
        return self.objective.finalize(params, totals, points.shape[0])

    def full_objective(self, params, points, outputs):
        return self.objective.full_objective(params, points, outputs)

    def value_and_grad(self, params, points):
        """First parameter gradient; no batch-sized activation tape is retained."""
        prepared, pullback = jax.vjp(self.prepare, params)
        chunks, masks = _chunks(points, self.chunk_size)
        if isinstance(self.objective, SumObjective):

            def chunk_loss(p, s, chunk, mask):
                return self._statistics(p, s, chunk, mask)[0]
            chunk_vg = jax.value_and_grad(chunk_loss, argnums=(0, 1))

            def body(carry, item):
                loss, direct, prepared_bar = carry
                value, (gp, gs) = chunk_vg(params, prepared, *item)
                return ((loss + value, _add(direct, gp), _add(prepared_bar, gs)), None)
            zero = jnp.asarray(0.0, dtype=points.dtype)
            (value, direct, prepared_bar), _ = jax.lax.scan(body, (zero, _zeros(params), _zeros(prepared)), (chunks, masks))
            regularizer, regularizer_bar = jax.value_and_grad(self.objective.regularizer)(params)
            gradient = _add(_add(direct, pullback(prepared_bar)[0]), regularizer_bar)
            return (value + regularizer, gradient)
        totals = self._totals(params, points, prepared)
        finalizer = lambda p, s: self.objective.finalize(p, s, points.shape[0])
        value, (direct, stat_bar) = jax.value_and_grad(finalizer, argnums=(0, 1))(params, totals)

        def body(carry, item):
            gp, gs = carry
            _, backward = jax.vjp(lambda p, s: self._statistics(p, s, *item), params, prepared)
            local_p, local_s = backward(stat_bar)
            return ((_add(gp, local_p), _add(gs, local_s)), None)
        (direct, prepared_bar), _ = jax.lax.scan(body, (direct, _zeros(prepared)), (chunks, masks))
        return (value, _add(direct, pullback(prepared_bar)[0]))

    def full_outputs(self, params, points):
        prepared = self.prepare(params)
        chunks, masks = _chunks(points, self.chunk_size)

        def body(carry, chunk):
            return (carry, self.evaluate(params, chunk, prepared))
        _, outputs = jax.lax.scan(body, (), chunks)
        return tuple((value.reshape((-1,) + value.shape[2:])[:points.shape[0]] for value in outputs))

def build_streaming(model, operators, objective, *, chunk_size=256, provider='parity', rematerialize=False):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('positive integer chunk size required')
    if not isinstance(objective, (SumObjective, WeightedMeanObjective, StatisticObjective)):
        raise TypeError('streaming requires an explicit row-statistic objective descriptor')
    from .shared_models import SharedModel
    descriptor = model if isinstance(model, SharedModel) else None
    model = descriptor.model if descriptor else model
    operators = metadata(model, operators)
    factory = {'ridge': _ridge, 'separable': _separable, 'bottleneck': _bottleneck}[model.family]
    prepare, evaluate, demands = factory(model, operators, provider)
    if descriptor:
        base_prepare, base_evaluate = (prepare, evaluate)

        def roles(params):
            if len(params) != descriptor.parameter_count:
                raise ValueError('declared unique runtime leaves required')
            return tuple((params[i] for i in descriptor.role_bindings))
        prepare = lambda params: base_prepare(roles(params))
        evaluate = lambda params, points, prepared: base_evaluate(roles(params), points, prepared)
    program = StreamingProgram(prepare, evaluate, objective, chunk_size, demands, rematerialize)
    program.constant_bounds = coefficient_bounds(operators, model.dimension)
    return program

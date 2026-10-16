"""Explicit supported formulas, shared operator demands, and complete-step AD."""
from dataclasses import dataclass
from fractions import Fraction
import math
import jax
import jax.numpy as jnp
from omnibias.core.operator_spec import OperatorSpec
from .providers import tower

@dataclass(frozen=True)
class Model:
    family: str
    dimension: int
    hidden: int = 16
    latent: int = 2

    def __post_init__(self):
        if self.family not in ('ridge', 'separable', 'bottleneck'):
            raise ValueError('unsupported explicit model family')
        if self.dimension < 1 or self.hidden < 1 or self.latent != 2:
            raise ValueError('positive dimensions and latent width two required')

def exact(coefficient):
    value = float(coefficient)
    if not math.isfinite(value) or Fraction(value) != coefficient:
        raise NotImplementedError('coefficient conversion needs an error bound')
    return value

def balanced(values):
    values = list(values)
    if not values:
        raise ValueError('empty product')
    while len(values) > 1:
        values = [values[i] * values[i + 1] if i + 1 < len(values) else values[i] for i in range(0, len(values), 2)]
    return values[0]

def metadata(model, operators):
    for op in operators:
        op.check_dimension(model.dimension)
        if op.order > 16:
            raise NotImplementedError('order >16 is experimental only')
        if op.mode == 'directional':
            raise NotImplementedError('runtime direction box adapter is required')
    if not operators:
        raise ValueError('nonempty operator bundle required')
    return tuple(operators)

def ridge_symbols(operators, weights):
    """Keep operator symbols live, including zero-order terms."""
    symbols = []
    for op in operators:
        if op.mode == 'laplacian':
            symbols.append({op.order: jnp.sum(weights * weights, axis=1) ** op.power})
        else:
            terms = {}
            for alpha, coefficient in op.resolve_terms(weights.shape[1]):
                factor = jnp.ones(weights.shape[0], dtype=weights.dtype) * exact(coefficient)
                for axis, power in enumerate(alpha):
                    if power:
                        factor = factor * weights[:, axis] ** power
                order = sum(alpha)
                terms[order] = terms.get(order, 0) + factor
            symbols.append(terms)
    return symbols

def _ridge(model, operators, provider):
    orders = tuple(sorted({op.order for op in operators if op.mode == 'laplacian'} | {sum(alpha) for op in operators if op.mode != 'laplacian' for alpha, _ in op.resolve_terms(model.dimension)}))
    derivative = tower(orders, provider)

    def prepare(params):
        return ridge_symbols(operators, params[0])

    def evaluate(params, points, symbols):
        weights, bias, readout, offset = params
        rows = dict(zip(orders, derivative(points @ weights.T + bias)))
        answers = []
        for terms in symbols:
            answer = sum((rows[order] @ (readout * symbol) for order, symbol in terms.items()))
            if 0 in terms:
                answer = answer + offset * terms[0][0]
            answers.append(answer)
        return tuple(answers)
    return (prepare, evaluate, orders)

def _separable(model, operators, provider):
    terms = [op.resolve_terms(model.dimension) for op in operators]
    if any((any((n > 1 for n in alpha)) for row in terms for alpha, _ in row)):
        raise NotImplementedError('terminal scalar-cut path requires square-free multi-indices')
    orders = tuple(sorted({sum(alpha) for row in terms for alpha, _ in row}))
    inner, outer = (tower((0, 1), provider), tower(orders, provider))

    def prepare(params):
        return ()

    def evaluate(params, points, prepared):
        weights, bias = params
        values, slopes = inner(points * weights + bias)
        rows = dict(zip(orders, outer(jnp.sum(values, axis=1))))
        factors = slopes * weights
        results = []
        for row in terms:
            answer = jnp.zeros(points.shape[0], dtype=points.dtype)
            for alpha, coefficient in row:
                used = [factors[:, i] for i, n in enumerate(alpha) if n]
                product = balanced(used) if used else jnp.ones(points.shape[0], dtype=points.dtype)
                answer = answer + exact(coefficient) * rows[sum(alpha)] * product
            results.append(answer)
        return tuple(results)
    return (prepare, evaluate, orders)

def _bottleneck(model, operators, provider):
    if any((op.mode != 'laplacian' or op.power > 2 for op in operators)):
        raise NotImplementedError('metric bundle supports value, Laplacian and biharmonic')
    order = max((op.order for op in operators))
    inner = tower(tuple(range(order + 1)), provider)
    outer = tower(tuple(range(order + 1)), provider)

    def prepare(params):
        return params[0] @ params[0].T

    def evaluate(params, points, gram):
        affine, bias, weights, hidden_bias, readout, offset = params
        raw = inner(points @ affine.T + bias)
        values = outer(raw[0] @ weights.T + hidden_bias)
        outputs = {0: values[0] @ readout + offset}
        if order:
            first = raw[1][:, None, :] * weights
            second = raw[2][:, None, :] * weights
            g0 = first[..., 0] * gram[0, 0] + first[..., 1] * gram[0, 1]
            g1 = first[..., 0] * gram[0, 1] + first[..., 1] * gram[1, 1]
            q = first[..., 0] * g0 + first[..., 1] * g1
            ell = second[..., 0] * gram[0, 0] + second[..., 1] * gram[1, 1]
            outputs[1] = (values[2] * q + values[1] * ell) @ readout
            if order == 4:
                third = raw[3][:, None, :] * weights
                fourth = raw[4][:, None, :] * weights
                ghg = second[..., 0] * g0 ** 2 + second[..., 1] * g1 ** 2
                grad_l = g0 * third[..., 0] * gram[0, 0] + g1 * third[..., 1] * gram[1, 1]
                hn = second[..., 0] ** 2 * gram[0, 0] ** 2 + 2 * second[..., 0] * second[..., 1] * gram[0, 1] ** 2 + second[..., 1] ** 2 * gram[1, 1] ** 2
                f4 = fourth[..., 0] * gram[0, 0] ** 2 + fourth[..., 1] * gram[1, 1] ** 2
                outputs[2] = (values[4] * q ** 2 + values[3] * (2 * ell * q + 4 * ghg) + values[2] * (ell ** 2 + 4 * grad_l + 2 * hn) + values[1] * f4) @ readout
        return tuple((outputs[op.power] for op in operators))
    return (prepare, evaluate, tuple(range(order + 1)))

def build(model, operators, provider='parity', *, chunk_size=None, rematerialize=False):
    operators = metadata(model, operators)
    prepare, evaluate, orders = {'ridge': _ridge, 'separable': _separable, 'bottleneck': _bottleneck}[model.family](model, operators, provider)
    if rematerialize:
        evaluate = jax.checkpoint(evaluate)

    def function(params, points):
        prepared = prepare(params)
        if chunk_size is None or chunk_size >= points.shape[0]:
            return evaluate(params, points, prepared)
        if chunk_size < 1:
            raise ValueError('positive chunk size required')
        pieces = [evaluate(params, points[i:i + chunk_size], prepared) for i in range(0, points.shape[0], chunk_size)]
        return tuple((jnp.concatenate([piece[j] for piece in pieces], axis=0) for j in range(len(operators))))
    function.demands = {'requested': list(orders), 'gradient_union': sorted(set(orders) | {k + 1 for k in orders}), 'provider': provider, 'chunk_size': chunk_size, 'rematerialize': rematerialize, 'metric_lifetime': 'one differentiated step'}
    return function

def objective(params, points, outputs):
    return sum((jnp.mean((value - 0.1 * points[:, 0]) ** 2) for value in outputs))

def joint(function, scalar_objective=objective):

    def loss(params, points):
        outputs = function(params, points)
        return (scalar_objective(params, points, outputs), outputs)
    return jax.value_and_grad(loss, has_aux=True)

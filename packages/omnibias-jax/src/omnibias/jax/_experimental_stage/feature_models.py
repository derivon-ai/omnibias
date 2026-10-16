"""Explicit linear-readout feature families for operator residuals.

All hidden arrays remain live arguments.  The separable bank is an explicit
new model, not a reinterpretation of the old single separable unit.  Features
are exact structural formulas; their floating-point accuracy is qualified by
the emitted-graph interpreter in variable_projection.py.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import jax
import jax.numpy as jnp
from omnibias.core.operator_spec import OperatorSpec
from .providers import tower
from .bundles import exact, coefficient_bounds

def _coefficient(value):
    return exact(value)

def _offset_coefficient(op, dimension):
    if op.mode == 'laplacian':
        return Fraction(int(op.power == 0))
    return sum((c for a, c in op.resolve_terms(dimension) if not any(a)), Fraction(0))

def _offset(op, dimension):
    return _coefficient(_offset_coefficient(op, dimension))

def _balanced_product(values, shape, dtype):
    values = list(values)
    if not values:
        return jnp.ones(shape, dtype=dtype)
    while len(values) > 1:
        values = [values[i] * values[i + 1] if i + 1 < len(values) else values[i] for i in range(0, len(values), 2)]
    return values[0]

def _affine(points, weights, bias):
    return jax.lax.optimization_barrier(points @ weights.T) + bias

@dataclass(frozen=True)
class FeatureModel:
    """A field is ``feature_matrix(hidden, x) @ readout``.

    Hidden tuples: ridge (W,b); bottleneck (A,b,W,h); separable_bank (W,b).
    ``include_offset`` adds one linear readout coordinate with the correct
    zero-order operator coefficient. It does not create a hidden parameter.
    """
    family: str
    dimension: int
    features: int
    include_offset: bool = True
    latent: int = 2
    provider: str = 'parity'

    def __post_init__(self):
        if self.family not in ('ridge', 'bottleneck', 'separable_bank'):
            raise ValueError('explicit ridge, bottleneck or separable_bank required')
        if self.dimension < 1 or self.features < 1:
            raise ValueError('positive feature count and input dimension required')
        if self.family == 'bottleneck' and self.latent != 2:
            raise NotImplementedError('bottleneck contractions require latent dimension two')

    @property
    def readout_size(self):
        return self.features + int(self.include_offset)

    @property
    def parameter_shapes(self):
        h, d = (self.features, self.dimension)
        if self.family == 'bottleneck':
            return ((2, d), (2,), (h, 2), (h,))
        return ((h, d), (h,) if self.family == 'ridge' else (h, d))

    def check_parameters(self, hidden):
        if not isinstance(hidden, (tuple, list)) or len(hidden) != len(self.parameter_shapes):
            raise ValueError('hidden parameter tuple does not match explicit model')
        for value, shape in zip(hidden, self.parameter_shapes):
            if value.shape != shape or value.dtype != jnp.float64:
                raise ValueError('live hidden parameters require declared shape and binary64')

    def build(self, operators):
        operators = tuple(operators)
        if not operators:
            raise ValueError('at least one operator required')
        for op in operators:
            op.check_dimension(self.dimension)
            if op.order > 16:
                raise NotImplementedError('supported spatial orders end at 16')
            if op.mode == 'directional':
                raise NotImplementedError('directional features need an explicit direction domain')
        if self.family == 'ridge':
            kernel = self._ridge(operators)
        elif self.family == 'bottleneck':
            kernel = self._bottleneck(operators)
        else:
            kernel = self._separable(operators)
        offsets = tuple((_offset(op, self.dimension) for op in operators))

        def evaluate_prepared(hidden, points, prepared):
            self.check_parameters(hidden)
            if points.ndim != 2 or points.shape[1] != self.dimension or points.dtype != jnp.float64:
                raise ValueError('points require (rows,dimension) binary64')
            matrices = kernel(hidden, points, prepared)
            if self.include_offset:
                return tuple((jnp.concatenate((v, jnp.full((points.shape[0], 1), c, dtype=points.dtype)), axis=1) for v, c in zip(matrices, offsets)))
            return matrices

        def evaluate(hidden, points):
            return evaluate_prepared(hidden, points, kernel.prepare(hidden))
        evaluate.model = self
        evaluate.operators = operators
        evaluate.prepare = kernel.prepare
        evaluate.prepared = evaluate_prepared
        bindings = coefficient_bounds(operators, self.dimension)
        conversions = []
        for op in operators:
            if op.mode == 'laplacian':
                continue
            for alpha, coefficient in op.resolve_terms(self.dimension):
                represented = Fraction(float(coefficient))
                if represented != coefficient:
                    conversions.append({'multi_index': list(alpha), 'ideal': str(coefficient), 'binary64': float(coefficient), 'absolute_conversion_error_exact': str(abs(represented - coefficient))})
        conversion_report = {'scope': 'exact rational operator coefficients; binary64 fixed row data', 'nonexact_conversions': conversions, 'constant_binding_method': 'captured arrays; conservative joined value bindings'}
        for function in (evaluate, evaluate_prepared, kernel.prepare):
            function.constant_bounds = bindings
            function.coefficient_conversion = conversion_report
        return evaluate

    def _ridge(self, operators):
        resolved = tuple((None if op.mode == 'laplacian' else op.resolve_terms(self.dimension) for op in operators))
        orders = tuple(sorted({op.order for op in operators if op.mode == 'laplacian'} | {sum(a) for terms in resolved if terms is not None for a, _ in terms}))
        derivatives = tower(orders, self.provider)
        symbol_orders = tuple(((op.order,) if terms is None else tuple(sorted({sum(a) for a, _ in terms})) for op, terms in zip(operators, resolved)))

        def prepare(hidden):
            weights = hidden[0]
            norm = jnp.sum(weights * weights, axis=1)
            symbols = []
            for op, terms in zip(operators, resolved):
                if terms is None:
                    symbols.append((norm ** op.power,))
                    continue
                terms_by_order = {}
                for alpha, coefficient in terms:
                    symbol = jnp.full((self.features,), _coefficient(coefficient), dtype=weights.dtype)
                    for axis, power in enumerate(alpha):
                        if power:
                            symbol = symbol * weights[:, axis] ** power
                    n = sum(alpha)
                    terms_by_order[n] = terms_by_order.get(n, 0) + symbol
                symbols.append(tuple((terms_by_order[n] for n in sorted(terms_by_order))))
            return tuple(symbols)

        def evaluate(hidden, points, prepared):
            weights, bias = hidden
            rows = dict(zip(orders, derivatives(_affine(points, weights, bias))))
            outputs = []
            for used_orders, symbols in zip(symbol_orders, prepared):
                outputs.append(sum((rows[n] * symbol for n, symbol in zip(used_orders, symbols))))
            return tuple(outputs)
        evaluate.prepare = prepare
        return evaluate

    def _separable(self, operators):
        terms = tuple((op.resolve_terms(self.dimension) for op in operators))
        if any((any((n > 1 for n in a)) for row in terms for a, _ in row)):
            raise NotImplementedError('separable scalar cut requires square-free multi-indices')
        orders = tuple(sorted({sum(a) for row in terms for a, _ in row}))
        inner, outer = (tower((0, 1), self.provider), tower(orders, self.provider))

        def evaluate(hidden, points, prepared):
            weights, bias = hidden
            values, slopes = inner(points[:, None, :] * weights[None, :, :] + bias[None, :, :])
            rows = dict(zip(orders, outer(jnp.sum(values, axis=2))))
            factors = slopes * weights[None, :, :]
            results = []
            for row in terms:
                answer = jnp.zeros((points.shape[0], self.features), dtype=points.dtype)
                for alpha, coefficient in row:
                    factors_used = [factors[:, :, axis] for axis, n in enumerate(alpha) if n]
                    product = _balanced_product(factors_used, answer.shape, answer.dtype)
                    answer = answer + _coefficient(coefficient) * rows[sum(alpha)] * product
                results.append(answer)
            return tuple(results)
        evaluate.prepare = lambda hidden: ()
        return evaluate

    def _bottleneck(self, operators):
        if any((op.mode != 'laplacian' or op.power > 2 for op in operators)):
            raise NotImplementedError('bottleneck features support value/Laplacian/biharmonic')
        order = max((op.order for op in operators))
        inner = tower(tuple(range(order + 1)), self.provider)
        outer = tower(tuple(range(order + 1)), self.provider)

        def evaluate(hidden, points, metric):
            affine, bias, weights, hidden_bias = hidden
            raw = inner(_affine(points, affine, bias))
            values = outer(_affine(raw[0], weights, hidden_bias))
            outputs = {0: values[0]}
            if order:
                first = raw[1][:, None, :] * weights
                second = raw[2][:, None, :] * weights
                g0 = first[..., 0] * metric[0, 0] + first[..., 1] * metric[0, 1]
                g1 = first[..., 0] * metric[0, 1] + first[..., 1] * metric[1, 1]
                q = first[..., 0] * g0 + first[..., 1] * g1
                ell = second[..., 0] * metric[0, 0] + second[..., 1] * metric[1, 1]
                outputs[1] = values[2] * q + values[1] * ell
                if order == 4:
                    third = raw[3][:, None, :] * weights
                    fourth = raw[4][:, None, :] * weights
                    ghg = second[..., 0] * g0 ** 2 + second[..., 1] * g1 ** 2
                    grad_l = g0 * third[..., 0] * metric[0, 0] + g1 * third[..., 1] * metric[1, 1]
                    hn = second[..., 0] ** 2 * metric[0, 0] ** 2 + 2 * second[..., 0] * second[..., 1] * metric[0, 1] ** 2 + second[..., 1] ** 2 * metric[1, 1] ** 2
                    f4 = fourth[..., 0] * metric[0, 0] ** 2 + fourth[..., 1] * metric[1, 1] ** 2
                    outputs[2] = values[4] * q ** 2 + values[3] * (2 * ell * q + 4 * ghg) + values[2] * (ell ** 2 + 4 * grad_l + 2 * hn) + values[1] * f4
            return tuple((outputs[op.power] for op in operators))
        evaluate.prepare = lambda hidden: hidden[0] @ hidden[0].T
        return evaluate

def heat_residual(model, x, t, diffusivity, *, weights=None, regularization=0.0, hidden_domain=None):
    """Trainable periodic ridge features for u=u0+t*phi(sin x,cos x,t)c.

    Initial data u0=sin(x)+sin(2x)/2 are built in. The supplied binary64
    transformed collocation inputs define the certified algebraic objective;
    trigonometric embedding and PDE discretization are separate error sources.
    """
    import numpy as np
    from .variable_projection import LinearResidual
    if model.family != 'ridge' or model.dimension != 3:
        raise ValueError('heat helper requires ridge features on (sin x,cos x,t)')
    x, t = (np.asarray(x, float), np.asarray(t, float))
    if x.shape != t.shape or x.ndim != 1 or np.any(t < 0) or (diffusivity <= 0):
        raise ValueError('matched nonnegative time samples and positive diffusivity required')
    s, c = (np.sin(x), np.cos(x))
    operators = tuple((OperatorSpec.mixed(a) for a in ((0, 0, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (1, 1, 0), (1, 0, 0), (0, 1, 0))))
    coefficients = np.column_stack((np.ones_like(t), t, -diffusivity * t * c * c, -diffusivity * t * s * s, 2 * diffusivity * t * s * c, diffusivity * t * s, diffusivity * t * c))
    target = diffusivity * (-s - 2 * np.sin(2 * x))
    return LinearResidual(model, operators, np.column_stack((s, c, t)), target, row_weights=weights, coefficients=coefficients, regularization=regularization, hidden_domain=hidden_domain)

def heat_solution(model, hidden, readout, x, t):
    import jax.numpy as jnp
    x, t = (jnp.asarray(x), jnp.asarray(t))
    points = jnp.stack((jnp.sin(x), jnp.cos(x), t), axis=1)
    values = model.build((OperatorSpec.laplacian(0),))(hidden, points)[0]
    return jnp.sin(x) + 0.5 * jnp.sin(2 * x) + t * (values @ readout)

def exact_heat(x, t, diffusivity):
    import numpy as np
    x, t = (np.asarray(x), np.asarray(t))
    return np.exp(-diffusivity * t) * np.sin(x) + 0.5 * np.exp(-4 * diffusivity * t) * np.sin(2 * x)

def train_heat_features(model, hidden, train_x, train_t, evaluation_x, evaluation_t, diffusivity, *, steps, learning_rate, chunk_size=None, row_weights=None, regularization=0.0, hidden_domain=None):
    """Bounded host-orchestrated variable-projection heat experiment.

    Each accepted update has its own solve/value/gradient certificate. No
    moving-domain theorem is inferred from a point certificate. Returned
    solution/residual errors are independent Fourier-reference diagnostics,
    not certified PDE error bounds. Fixed-step gradient descent is deliberately
    explicit so a benchmark can freeze the same trajectory for controls.
    """
    import time
    import jax
    import numpy as np
    if not isinstance(steps, int) or steps < 0 or (not math.isfinite(learning_rate)) or (learning_rate <= 0):
        raise ValueError('nonnegative steps and positive finite learning rate required')
    problem = heat_residual(model, train_x, train_t, diffusivity, weights=row_weights, regularization=regularization, hidden_domain=hidden_domain)
    history = []
    current = tuple((jnp.asarray(v) for v in hidden))
    started = time.perf_counter()
    result, certificate = problem.certified_value_and_grad(current, chunk_size=chunk_size)
    qualification_seconds = time.perf_counter() - started
    if result is None:
        return {'status': 'initial_qualification_failed', 'accepted_steps': 0, 'certificate': certificate, 'history': history, 'qualification_and_evaluation_s': qualification_seconds}
    status = 'completed'
    refusal = None
    history.append({'step': 0, 'objective': float(result[0]), 'certificate': certificate})
    for step in range(1, steps + 1):
        proposed = tuple((value - learning_rate * gradient for value, gradient in zip(current, result[1])))
        started = time.perf_counter()
        candidate, certificate = problem.certified_value_and_grad(proposed, chunk_size=chunk_size)
        qualification_seconds += time.perf_counter() - started
        if candidate is None:
            status = 'proposed_update_refused'
            refusal = {'step': step, 'certificate': certificate}
            break
        current, result = (proposed, candidate)
        history.append({'step': step, 'objective': float(result[0]), 'certificate': certificate})
    ex, et = (np.asarray(evaluation_x, float), np.asarray(evaluation_t, float))
    if ex.shape != et.shape or ex.ndim != 1 or (not len(ex)):
        raise ValueError('matching nonempty held-out x/t vectors required')
    readout = result[2]
    solution = np.asarray(jax.block_until_ready(heat_solution(model, current, readout, ex, et)))
    reference = exact_heat(ex, et, diffusivity)
    evaluation_problem = heat_residual(model, ex, et, diffusivity)
    f, y = evaluation_problem.design(current)
    physical_residual = np.asarray(jax.block_until_ready(f @ readout - y))
    exact_ut = -diffusivity * np.exp(-diffusivity * et) * np.sin(ex) - 2 * diffusivity * np.exp(-4 * diffusivity * et) * np.sin(2 * ex)
    initial = np.asarray(heat_solution(model, current, readout, ex, np.zeros_like(et)))
    endpoints = np.asarray(heat_solution(model, current, readout, np.r_[np.zeros_like(et), np.full_like(et, 2 * np.pi)], np.r_[et, et]))
    quality = {'solution_relative_L2': float(np.linalg.norm(solution - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny)), 'normalized_residual': float(np.linalg.norm(physical_residual) / max(np.linalg.norm(exact_ut), 1.0)), 'initial_error_max': float(np.max(np.abs(initial - exact_heat(ex, np.zeros_like(et), diffusivity)))), 'periodic_error_max': float(np.max(np.abs(endpoints[:len(et)] - endpoints[len(et):])))}
    quality['targets'] = {str(tol): all((quality[k] <= tol for k in ('solution_relative_L2', 'normalized_residual', 'initial_error_max', 'periodic_error_max'))) for tol in (0.001, 1e-06)}
    return {'status': status, 'accepted_steps': len(history) - 1, 'history': history, 'refusal': refusal, 'hidden': current, 'readout': readout, 'quality': quality, 'solution': solution, 'reference': reference, 'qualification_and_evaluation_s': qualification_seconds, 'certificate_scope': 'a fresh point certificate at every accepted hidden update; quality is measured separately'}

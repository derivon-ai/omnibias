"""Conditional first-order variable projection for explicit linear residuals.

QR/SVD are opaque numerical solvers: their answers are accepted only through
an a-posteriori stationarity and verified positive-Gram certificate. No QR
backward-error guarantee or empirical probe is substituted for this check.
The available derivative is the reduced scalar objective's first hidden
gradient. Readout argmin Jacobians and higher derivatives are unsupported.
"""
from dataclasses import dataclass, replace
import hashlib
import math
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
import numpy as np
from .bounds import Bound, binary, dot, interpret, reduction, up, down
from .feature_models import FeatureModel, _offset_coefficient

@jax.custom_jvp
def _first_order_output(value):
    return value

@_first_order_output.defjvp
def _no_higher_derivative(primals, tangents):
    raise NotImplementedError('use returned first hidden gradient; differentiating varpro outputs is unsupported')

def _owned(value):
    if np.iscomplexobj(value):
        raise ValueError('real fixed data required; imaginary parts cannot be discarded')
    value = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError('finite real binary64 fixed data required')
    return jnp.asarray(value)

def _matmul(a, b):
    return dot(a, b, (((1,), (0,)), ((), ())))

def _transpose(a):
    return Bound(a.lo.T, a.hi.T, a.error.T)

def _ideal(a):
    return Bound(a.lo, a.hi, 0.0)

def _norm_upper(a):
    """Outward Frobenius/Euclidean norm of a true interval array."""
    v = np.maximum(np.abs(a.lo), np.abs(a.hi)).reshape(-1, 1)
    square = _matmul(Bound.point(v.T), Bound.point(v)).hi[0, 0]
    return float(up(np.sqrt(square)))

def _positive_sum(a, b):
    return float(up(a + b))

def _positive_product(a, b):
    return float(up(a * b))

def _positive_quotient(a, b):
    if not b > 0:
        raise ArithmeticError('positive lower denominator required')
    return float(up(a / b))

def certify_stationarity(design, target, readout):
    """Exact-real design/target enclosures; solve by QR elsewhere, never Gram.

    An arbitrary finite inverse proposal B is verified by ||I-BH||_F<1.
    Then lambda_min(H)>=(1-rho)/||B||_F. The native inverse is not trusted;
    its residual is checked with outward arithmetic. H=F.T F is used only
    for this certificate. Input ``Bound.error`` is not added to the true
    intervals: it records emitted feature arithmetic, handled separately.
    """
    f, y = (_ideal(design), _ideal(target))
    c = np.asarray(readout, dtype=np.float64)
    if f.lo.ndim != 2 or y.lo.shape != (f.lo.shape[0],) or c.shape != (f.lo.shape[1],):
        raise ValueError('stationarity shapes do not match')
    if not np.isfinite(c).all():
        return {'admissible_rank': False, 'reason': 'nonfinite readout'}
    residual = binary(_matmul(f, Bound.point(c[:, None])), Bound(y.lo[:, None], y.hi[:, None], 0), 'sub')
    stationary = _matmul(_transpose(f), _ideal(residual))
    gram = _matmul(_transpose(f), f)
    midpoint = 0.5 * gram.lo + 0.5 * gram.hi
    try:
        inverse = np.linalg.inv(midpoint)
        if not np.isfinite(inverse).all():
            raise ArithmeticError('nonfinite inverse proposal')
        inverse_bound = Bound.point(inverse)
        remainder = binary(Bound.point(np.eye(c.size)), _matmul(inverse_bound, _ideal(gram)), 'sub')
        rho = _norm_upper(remainder)
        inverse_norm = _norm_upper(inverse_bound)
        gap = float(down(1.0 - rho))
        lambda_lower = float(down(gap / inverse_norm)) if gap > 0 and inverse_norm > 0 else 0.0
    except (np.linalg.LinAlgError, ArithmeticError, FloatingPointError):
        rho, lambda_lower = (math.inf, 0.0)
    result = {'admissible_rank': lambda_lower > 0, 'gram_inverse_residual_norm_bound': rho, 'lambda_min_lower_bound': lambda_lower, 'rank_method': 'verified inverse residual of interval Gram; no normal-equation solve'}
    if lambda_lower <= 0:
        return {**result, 'reason': 'full column rank unresolved; numerical rank is not a certificate'}
    eta = _norm_upper(stationary)
    delta = _positive_quotient(eta, lambda_lower)
    excess = _positive_quotient(_positive_product(eta, eta), float(down(2 * lambda_lower)))
    result.update(stationarity_norm_bound=eta, readout_error_norm_bound=delta, objective_excess_bound=excess, sigma_min_lower_bound=float(down(np.sqrt(lambda_lower))), design_norm_bound=_norm_upper(f), residual_norm_bound=_norm_upper(residual), readout_norm_bound=_norm_upper(Bound.point(c)))
    return result

@dataclass(frozen=True)
class LinearResidual:
    """Owned fixed rows of sum_j alpha_ij L_j phi(theta,x_i)c - y_i.

    The only trainable arguments are hidden feature parameters. Fixed row
    weights are nonnegative and regularization is an explicit model choice.
    Offset/readout coordinates are independent of hidden parameters. All
    fixed row data are binary64 snapshots; operator coefficients retain their
    exact rational meanings and bounded conversions. Replacing data creates a
    new descriptor. A rank-deficient solve may be observed via SVD but does not
    qualify first gradients without a structural nullspace proof.
    """
    model: FeatureModel
    operators: tuple
    points: object
    target: object
    row_weights: object = None
    coefficients: object = None
    regularization: float = 0.0
    hidden_domain: object = None
    input_domain: object = None
    nonlinear_in_readout: bool = False
    tied_hidden_readout: bool = False

    def __post_init__(self):
        if not jax.config.x64_enabled:
            raise ValueError('enable binary64 before constructing the descriptor')
        if self.nonlinear_in_readout or self.tied_hidden_readout:
            raise NotImplementedError('variable projection requires an independent linear readout')
        if not isinstance(self.model, FeatureModel):
            raise TypeError('explicit FeatureModel required; arbitrary callables are unsupported')
        for value in (self.points, self.target, self.row_weights, self.coefficients):
            if value is not None and np.iscomplexobj(value):
                raise ValueError('real residual data required')
        operators = tuple(self.operators)
        point = _owned(self.points)
        target = _owned(self.target)
        rows = point.shape[0] if point.ndim == 2 else 0
        if rows < 1 or point.shape[1] != self.model.dimension or target.shape != (rows,):
            raise ValueError('points(rows,dimension) and target(rows) required')
        weights = _owned(np.ones(rows) if self.row_weights is None else self.row_weights)
        coefficients = _owned(np.ones((rows, len(operators))) if self.coefficients is None else self.coefficients)
        if weights.shape != (rows,) or np.any(np.asarray(weights) < 0):
            raise ValueError('nonnegative fixed row weights required')
        if coefficients.shape != (rows, len(operators)):
            raise ValueError('one fixed coefficient for every row/operator required')
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError('finite nonnegative explicit regularization required')
        domain = self.hidden_domain
        if domain is not None:
            domain = tuple((Bound(np.array(b.lo, copy=True), np.array(b.hi, copy=True), 0) for b in domain))
            if len(domain) != len(self.model.parameter_shapes) or any((b.lo.shape != shape for b, shape in zip(domain, self.model.parameter_shapes))):
                raise ValueError('hidden boxes must match hidden parameter tuple')
            for bound in domain:
                bound.lo.flags.writeable = False
                bound.hi.flags.writeable = False
        input_domain = self.input_domain
        input_valid = True
        if input_domain is not None:
            lo = np.broadcast_to(np.asarray(input_domain.lo), point.shape).copy()
            hi = np.broadcast_to(np.asarray(input_domain.hi), point.shape).copy()
            input_domain = Bound(lo, hi, 0)
            input_domain.lo.flags.writeable = False
            input_domain.hi.flags.writeable = False
            input_valid = bool(np.all(np.asarray(point) >= lo) and np.all(np.asarray(point) <= hi))
        for name, value in (('points', point), ('target', target), ('operators', operators), ('row_weights', weights), ('coefficients', coefficients), ('hidden_domain', domain), ('input_domain', input_domain), ('_input_domain_valid', input_valid), ('_feature', self.model.build(operators))):
            object.__setattr__(self, name, value)
        digest = hashlib.sha256()
        for value in (point, target, weights, coefficients):
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        digest.update(repr((self.model, [o.to_config() for o in operators], self.regularization)).encode())
        for bound in (() if domain is None else domain) + (() if input_domain is None else (input_domain,)):
            digest.update(np.ascontiguousarray(bound.lo).tobytes())
            digest.update(np.ascontiguousarray(bound.hi).tobytes())
        object.__setattr__(self, 'source_digest', digest.hexdigest())
        zero_offset = self.model.include_offset and self.regularization == 0 and all((_offset_coefficient(op, self.model.dimension) == 0 for op in operators))
        object.__setattr__(self, 'structural_rank_obstruction', 'derivative-only operators annihilate the offset; use include_offset=False or an explicitly regularized model' if zero_offset else None)

    @property
    def columns(self):
        return self.model.readout_size

    def replace_batch(self, points, target, weights=None):
        """Return a new owned preparation for the same operator row schema.

        The row count is fixed because each row's operator mixing
        coefficients remain part of this descriptor. Omitted weights retain
        the current row weights. The original descriptor and its digest are
        unchanged; the replacement rechecks domains and snapshots all data.
        """
        if np.shape(points) != self.points.shape:
            raise ValueError('replacement points must retain the operator row schema')
        return replace(self, points=points, target=target, row_weights=self.row_weights if weights is None else weights)

    def domain_valid(self, hidden):
        self.model.check_parameters(hidden)
        valid = jnp.asarray(self._input_domain_valid)
        for index, value in enumerate(hidden):
            valid = valid & jnp.all(jnp.isfinite(value))
            if self.hidden_domain is not None:
                b = self.hidden_domain[index]
                valid = valid & jnp.all((value >= jnp.asarray(b.lo)) & (value <= jnp.asarray(b.hi)))
        return valid

    def _rows(self, hidden, points, target, coefficients, weights, prepared=None):
        features = self._feature(hidden, points) if prepared is None else self._feature.prepared(hidden, points, prepared)
        design = sum((value * coefficients[:, i:i + 1] for i, value in enumerate(features)))
        scale = jnp.sqrt(weights)
        return (design * scale[:, None], target * scale)

    def design(self, hidden, prepared=None):
        return self._rows(hidden, self.points, self.target, self.coefficients, self.row_weights, prepared)

    def augmented_design(self, hidden, prepared=None):
        design, target = self.design(hidden, prepared)
        regularizer = jnp.sqrt(jnp.asarray(self.regularization, dtype=jnp.float64)) * jnp.eye(self.columns)
        return (jnp.concatenate((design, regularizer)), jnp.concatenate((target, jnp.zeros(self.columns))))

    def _chunks(self, chunk_size):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('positive integer chunk size required')
        rows = self.points.shape[0]
        padding = -rows % chunk_size
        points = jnp.concatenate((self.points, jnp.repeat(self.points[-1:], padding, axis=0)))
        reshape = lambda v: v.reshape((-1, chunk_size) + v.shape[1:])
        return tuple((reshape(v) for v in (points, jnp.pad(self.target, (0, padding)), jnp.pad(self.coefficients, ((0, padding), (0, 0))), jnp.pad(self.row_weights, (0, padding)))))

    def _factor(self, hidden, chunk_size=None, prepared=None):
        prepared = self._feature.prepare(hidden) if prepared is None else prepared
        if chunk_size is None:
            f, y = self.augmented_design(hidden, prepared)
            return jnp.linalg.qr(jnp.concatenate((f, y[:, None]), axis=1), mode='r')
        initial = jnp.zeros((self.columns + 1, self.columns + 1), dtype=jnp.float64)
        initial = initial.at[:self.columns, :self.columns].set(jnp.sqrt(jnp.asarray(self.regularization)) * jnp.eye(self.columns))

        def body(factor, chunk):
            f, y = self._rows(hidden, *chunk, prepared=prepared)
            block = jnp.concatenate((f, y[:, None]), axis=1)
            return (jnp.linalg.qr(jnp.concatenate((factor, block)), mode='r'), None)
        return jax.lax.scan(body, initial, self._chunks(chunk_size))[0]

    def _solve(self, hidden, chunk_size=None, solver='qr', rcond=None, prepared=None):
        factor = self._factor(hidden, chunk_size, prepared)
        r, target = (factor[:self.columns, :self.columns], factor[:self.columns, self.columns])
        if solver == 'qr':
            c = solve_triangular(r, target, lower=False)
        elif solver == 'svd':
            c = jnp.linalg.lstsq(r, target, rcond=rcond)[0]
        else:
            raise ValueError('solver is qr or svd; fallback choice is explicit')
        return c

    def _envelope(self, hidden, readout, chunk_size=None):
        """Actual emitted first gradient. A forward scan sums chunk VJPs.

        This deliberately avoids reverse AD of an outer scan (which could
        retain an all-chunk tape). The factorization and this second pass may
        recompute features; no full collocation outputs are retained.
        """
        prepared, pullback = jax.vjp(self._feature.prepare, hidden)
        return self._envelope_prepared(hidden, readout, prepared, pullback, chunk_size)

    def _envelope_prepared(self, hidden, readout, prepared, pullback, chunk_size):
        if chunk_size is None:

            def loss(params, shared):
                f, y = self.augmented_design(params, shared)
                residual = f @ readout - y
                return 0.5 * jnp.sum(residual * residual)
            value, (direct, shared) = jax.value_and_grad(loss, argnums=(0, 1))(hidden, prepared)
            return (value, tuple((a + b for a, b in zip(direct, pullback(shared)[0]))))
        zero = tuple((jnp.zeros_like(v) for v in hidden))
        zero_prepared = jax.tree_util.tree_map(jnp.zeros_like, prepared)

        def body(carry, chunk):

            def local(params, shared):
                f, y = self._rows(params, *chunk, prepared=shared)
                r = f @ readout - y
                return 0.5 * jnp.sum(r * r)
            value, (gradient, shared_gradient) = jax.value_and_grad(local, argnums=(0, 1))(hidden, prepared)
            return ((carry[0] + value, tuple((a + b for a, b in zip(carry[1], gradient))), jax.tree_util.tree_map(lambda a, b: a + b, carry[2], shared_gradient)), None)
        value, direct, shared = jax.lax.scan(body, (jnp.asarray(0.0, dtype=jnp.float64), zero, zero_prepared), self._chunks(chunk_size))[0]
        gradient = tuple((a + b for a, b in zip(direct, pullback(shared)[0])))
        value = value + 0.5 * self.regularization * jnp.sum(readout * readout)
        return (value, gradient)

    def value_and_grad(self, hidden, *, chunk_size=None, solver='qr', rcond=None, derivative_order=1, solution_override=None):
        """JIT-compatible first-order result; use qualify before accepting it.

        Returns (objective, hidden_gradient, readout, domain_valid). Returned
        objective/gradient/readout explicitly refuse another AD transform.
        ``solution_override`` is a diagnostic injection, checked identically
        by the a-posteriori certificate.
        """
        if derivative_order != 1:
            raise NotImplementedError('only the reduced scalar first hidden gradient is supported')
        self.model.check_parameters(hidden)
        valid = self.domain_valid(hidden)

        def evaluate(_):
            prepared, pullback = jax.vjp(self._feature.prepare, hidden)
            c = self._solve(hidden, chunk_size, solver, rcond, prepared) if solution_override is None else jnp.asarray(solution_override)
            if c.shape != (self.columns,) or c.dtype != jnp.float64:
                raise ValueError('readout requires the declared binary64 shape')
            c = jax.lax.stop_gradient(c)
            value, gradient = self._envelope_prepared(hidden, c, prepared, pullback, chunk_size)
            return (value, gradient, c)

        def refused(_):
            return (jnp.asarray(jnp.nan, dtype=jnp.float64), tuple((jnp.full_like(v, jnp.nan) for v in hidden)), jnp.full((self.columns,), jnp.nan, dtype=jnp.float64))
        value, gradient, c = jax.lax.cond(valid, evaluate, refused, operand=None)
        return (_first_order_output(value), tuple((_first_order_output(v) for v in gradient)), _first_order_output(c), valid)

    def certified_value_and_grad(self, hidden, *, evaluator=None, **options):
        """Host orchestrated evaluation with point qualification on every call.

        Returns ``(result, report)`` only if accepted, otherwise ``(None,
        report)``. Rank/solve/gradient certification is repeated after every
        update and must be charged to training cost. Domain guards establish
        membership only; they never turn a point certificate into a box
        certificate. This method is deliberately not a device training step.
        """
        if options.get('derivative_order', 1) != 1:
            raise NotImplementedError('only the reduced scalar first hidden gradient is supported')
        if not bool(np.asarray(self.domain_valid(hidden))):
            return (None, {'admissible': False, 'status': 'domain_violation', 'scope': 'pointwise; input/hidden guard failed before evaluation'})
        if self.structural_rank_obstruction is not None:
            return (None, {'admissible': False, 'status': 'inconclusive_rank', 'reason': self.structural_rank_obstruction})
        evaluation_keys = ('chunk_size', 'solver', 'rcond', 'derivative_order', 'solution_override')
        evaluation_options = {k: v for k, v in options.items() if k in evaluation_keys}
        result = jax.block_until_ready(self.value_and_grad(hidden, **evaluation_options) if evaluator is None else evaluator(hidden))
        qualification_options = {k: v for k, v in options.items() if k != 'derivative_order'}
        qualification_options.pop('solution_override', None)
        report = self.qualify_result(hidden, result, **qualification_options)
        return (result if report['admissible'] else None, report)

    def qualify_result(self, hidden, result, **options):
        """Qualify this exact (value, gradient, readout, domain) packet.

        No numerical solve or envelope re-evaluation is performed. The packet
        may come from the compiled value_and_grad callable; it is checked
        against the prescribed envelope graph and its ideal intervals. A
        certificate is never transferred to a later QR/SVD execution.
        """
        if 'solution_override' in options or '_evaluated_result' in options:
            raise ValueError('the evaluated packet supplies its own readout')
        return self.qualify(hidden, _evaluated_result=result, **options)

    def qualify(self, hidden, *, chunk_size=None, solver='qr', rcond=None, solution_override=None, max_equations=100000, max_state_bytes=268435456, atol=1e-09, rtol=2e-10, gradient_atol=1e-08, gradient_rtol=2e-09, _evaluated_result=None):
        """Pointwise conditional certificate, rerun after every hidden update.

        Native tanh/exp assumptions and compatible binary64 transformations
        are inherited from bounds.interpret. This is not a domain-wide rank
        certificate, a vendor guarantee, or a proof of PDE solution accuracy.
        The analysis may materialize small-case feature Jacobians, bounded by
        max_state_bytes. Runtime TSQR/streamed-gradient storage is separate.
        """
        report = {'admissible': False, 'scope': 'pointwise reduced scalar value and first hidden gradient', 'source_digest': self.source_digest, 'solver': solver, 'streaming': chunk_size is not None, 'chunk_size': chunk_size, 'hidden_gradient_order': 1, 'regularization': self.regularization, 'result_scope': 'only the inspected value/gradient/readout packet; no future solver execution', 'qualification_reexecutes_solver': _evaluated_result is None, 'coefficient_conversion': self._feature.coefficient_conversion, 'assumptions': 'binary64 RN, gradual underflow, finite intermediates; compatible compiler transforms; bounds.py native-transcendental hypotheses; QR/SVD answer checked a posteriori'}
        try:
            self.model.check_parameters(hidden)
            if not bool(np.asarray(self.domain_valid(hidden))):
                return {**report, 'status': 'domain_violation', 'reason': 'input or hidden parameters outside declared domain'}
            if self.structural_rank_obstruction is not None:
                return {**report, 'status': 'inconclusive_rank', 'reason': self.structural_rank_obstruction}
            p = tuple((jnp.asarray(v) for v in hidden))
            flat, unflatten = ravel_pytree(p)
            rows = self.points.shape[0] + self.columns
            required = 3 * 8 * rows * self.columns * (flat.size + 1)
            if required > max_state_bytes:
                return {**report, 'status': 'analysis_budget', 'required_jacobian_bound_bytes': required}
            if max_equations < 1:
                raise ArithmeticError('analysis equation budget exhausted')
            result = self.value_and_grad(p, chunk_size=chunk_size, solver=solver, rcond=rcond, solution_override=solution_override) if _evaluated_result is None else _evaluated_result
            if not isinstance(result, (tuple, list)) or len(result) != 4:
                raise ValueError('value/gradient/readout/domain packet required')
            value, gradients, readout, valid = jax.block_until_ready(result)
            if np.shape(valid) != () or np.asarray(valid).dtype.kind != 'b' or (not bool(valid)):
                return {**report, 'status': 'domain_violation', 'reason': 'packet domain guard failed'}
            if not isinstance(gradients, (tuple, list)) or len(gradients) != len(p):
                raise ValueError('complete hidden-gradient packet required')
            for output, shape in ((value, ()), (readout, (self.columns,)), *zip(gradients, self.model.parameter_shapes)):
                if np.shape(output) != shape or np.asarray(output).dtype != np.float64:
                    raise ValueError('packet requires declared shapes and binary64 values')
                if not np.isfinite(np.asarray(output)).all():
                    raise ArithmeticError('nonfinite inspected result packet')
            readout = np.asarray(readout)
            if not np.isfinite(readout).all():
                return {**report, 'status': 'inconclusive_rank', 'reason': 'nonfinite QR solution; explicit SVD may provide observational output'}
            inputs = [Bound.point(v) for v in p]
            constant_bounds = self._feature.constant_bounds
            traced = jax.make_jaxpr(self.augmented_design)(p)
            (f, y), graph = interpret(traced, inputs, max_equations=max_equations, constant_bounds=constant_bounds, max_state_bytes=max_state_bytes)
            remaining_equations = max_equations - graph['equations']
            stationary = certify_stationarity(f, y, readout)
            report['solve_certificate'] = stationary
            report['feature_graph'] = graph
            if not stationary['admissible_rank']:
                return {**report, 'status': 'inconclusive_rank', 'reason': stationary['reason']}
            emitted = lambda pp, cc: self._envelope(pp, cc, chunk_size)
            emitted_graph = jax.make_jaxpr(emitted)(p, jnp.asarray(readout))
            arithmetic, graph_metadata = interpret(emitted_graph, inputs + [Bound.point(readout)], max_equations=remaining_equations, constant_bounds=constant_bounds, max_state_bytes=max_state_bytes)
            packet_checks = []
            inspected = [value, *gradients]
            for index, (bound, actual) in enumerate(zip(arithmetic, inspected)):
                actual = np.asarray(actual)
                lo, hi = bound.enlarged()
                consistent = bool(np.all(actual >= lo) and np.all(actual <= hi))
                discrepancy = np.maximum(up(np.abs(actual - bound.lo)), up(np.abs(actual - bound.hi)))
                arithmetic[index] = Bound(bound.lo, bound.hi, np.maximum(bound.error, discrepancy))
                packet_checks.append({'inside_emitted_arithmetic_enclosure': consistent, 'actual_to_ideal_interval_error_bound': float(np.max(discrepancy))})
            report['inspected_packet'] = {'checked': True, 'outputs': packet_checks, 'output_sha256': [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in inspected], 'output_shapes': [list(np.shape(v)) for v in inspected], 'output_dtypes': [np.asarray(v).dtype.str for v in inspected], 'valid': bool(valid), 'readout_sha256': hashlib.sha256(np.ascontiguousarray(readout).tobytes()).hexdigest(), 'readout_shape': list(readout.shape), 'readout_dtype': readout.dtype.str}
            if not all((row['inside_emitted_arithmetic_enclosure'] for row in packet_checks)):
                return {**report, 'status': 'inconclusive_packet', 'reason': 'actual returned value or gradient lies outside its emitted arithmetic enclosure'}
            remaining_equations -= graph_metadata['equations']
            derivative = jax.jacfwd(lambda pp: self.augmented_design(unflatten(pp))[0])
            derivative_graph = jax.make_jaxpr(derivative)(flat)
            derivatives, derivative_metadata = interpret(derivative_graph, [Bound.point(flat)], max_equations=remaining_equations, constant_bounds=constant_bounds, max_state_bytes=max_state_bytes)
            jf = derivatives[0]
            fnorm, residual_norm, cnorm, delta = (stationary[k] for k in ('design_norm_bound', 'residual_norm_bound', 'readout_norm_bound', 'readout_error_norm_bound'))
            parenthesis = _positive_sum(residual_norm, _positive_sum(_positive_product(fnorm, cnorm), _positive_product(fnorm, delta)))
            per_parameter = []
            for index in range(flat.size):
                sensitivity = _norm_upper(Bound(jf.lo[..., index], jf.hi[..., index], 0))
                per_parameter.append(_positive_product(sensitivity, _positive_product(delta, parenthesis)))
            additions = np.asarray(per_parameter)
            gradient_additions = jax.tree_util.tree_leaves(unflatten(jnp.asarray(additions)))
            objective = arithmetic[0]
            excess = stationary['objective_excess_bound']
            objective = Bound(down(objective.lo - excess), objective.hi, up(objective.error + excess))
            value_summary = objective.summary(atol, rtol)
            readout_summary = Bound(down(readout - delta), up(readout + delta), delta).summary(atol, rtol)
            gradient_summaries = []
            for bound, extra in zip(arithmetic[1:], gradient_additions):
                extra = np.asarray(extra)
                combined = Bound(down(bound.lo - extra), up(bound.hi + extra), up(bound.error + extra))
                gradient_summaries.append(combined.summary(gradient_atol, gradient_rtol))
            admissible = value_summary['admissible'] and readout_summary['admissible'] and all((v['admissible'] for v in gradient_summaries))
            report.update(admissible=admissible, status='qualified' if admissible else 'inconclusive_accuracy', objective=value_summary, readout=readout_summary, gradients=gradient_summaries, emitted_graph=graph_metadata, sensitivity_graph=derivative_metadata, required_jacobian_bound_bytes=required, value=float(value), gradient_norm=float(np.linalg.norm(np.concatenate([np.asarray(v).ravel() for v in gradients]))), runtime_factor_bytes=(self.columns + 1) ** 2 * 8 if chunk_size else None, storage_scope='factor storage excludes input data, parameter gradients, chunk work and compiler workspace')
            return report
        except (ArithmeticError, ValueError, NotImplementedError, TypeError) as error:
            return {**report, 'status': 'inconclusive_analysis', 'reason': str(error)}

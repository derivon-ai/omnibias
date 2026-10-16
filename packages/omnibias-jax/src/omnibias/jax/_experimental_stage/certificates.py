"""Relational certificates for frozen designs and exact low-rank operators.

Saved arrays represent rounded computations, not exact ideal precomputations.
Each error contract combines the relation to its ideal input with the actual
emitted arithmetic graph. Conditional binary64 assumptions remain explicit.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .bounds import Bound, ConstantBindings, dot, up
from .planner import Domain, Limits, Tolerances, qualify

def matrix_product(a, b):
    a = a if isinstance(a, Bound) else Bound.point(a)
    b = b if isinstance(b, Bound) else Bound.point(b)
    return dot(a, b, (((1,), (0,)), ((), ())))

def _check_feature(value, bound, name):
    if bound is None:
        return Bound.point(value)
    if np.shape(value) != bound.lo.shape:
        raise ValueError(name + ' bound shape mismatch')
    lo, hi = bound.enlarged()
    if np.any(value < lo) or np.any(value > hi):
        raise ValueError('saved ' + name + ' is outside its declared construction bound')
    return Bound(bound.lo, bound.hi, 0.0)

def certify_readout(plan, design, target, lower, upper, *, feature_bounds=None, target_bounds=None, tolerances=Tolerances(), limits=Limits(cells=1)):
    """Compose supplied feature enclosures, QR relation and emitted gradient.

    Without feature_bounds/target_bounds the ideal is exactly the binary64
    design/target supplied to preparation. Bounds must enclose the ideal
    frozen features and justify the saved values through their error fields.
    This does not establish hidden-parameter derivatives of frozen factors.
    """
    f, y = (np.asarray(design), np.asarray(target))
    lower, upper = (np.asarray(lower), np.asarray(upper))
    if f.ndim != 2 or y.shape != (f.shape[0],) or lower.shape != (f.shape[1],) or (upper.shape != lower.shape):
        raise ValueError('design, target and readout box shapes must match')
    if np.any(lower > upper):
        raise ValueError('invalid readout domain')
    fi = _check_feature(f, feature_bounds, 'feature')
    yi = _check_feature(y, target_bounds, 'target')
    r, d = (np.asarray(plan.factor), np.asarray(plan.target))
    ideal = Bound(np.column_stack((fi.lo, -yi.hi)), np.column_stack((fi.hi, -yi.lo)), 0.0)
    k_transpose = Bound(ideal.lo.T, ideal.hi.T, 0.0)
    s = np.column_stack((r, -d))
    original = matrix_product(k_transpose, ideal)
    compressed = matrix_product(s.T, s)
    lo, hi = (compressed.lo.copy(), compressed.hi.copy())
    lo[-1, -1] = np.nextafter(lo[-1, -1] + float(plan.constant), -np.inf)
    hi[-1, -1] = np.nextafter(hi[-1, -1] + float(plan.constant), np.inf)
    error = up(np.maximum(np.abs(original.lo - hi), np.abs(original.hi - lo)))
    v = np.r_[np.maximum(np.abs(lower), np.abs(upper)), 1.0]
    row = matrix_product(error, v[:, None])
    loss = matrix_product(v[None, :], row.hi)
    gradient = matrix_product(error[:-1], v[:, None])
    model_loss = float(up(0.5 * loss.hi[0, 0]))
    model_gradient = float(np.max(gradient.hi))
    p = (jnp.asarray(lower * 0.5 + upper * 0.5),)
    x = jnp.zeros((1, 1), dtype=jnp.float64)
    domain = Domain((Bound(lower, upper, 0),), Bound(np.zeros((1, 1)), np.zeros((1, 1)), 0))
    fn = lambda params, points: (plan.objective(params[0]),)
    arithmetic = qualify(fn, (p, x), domain, lambda params, points, out: out[0], tolerances, limits)
    report = {'scope': 'frozen ideal residual design plus actual emitted readout objective/gradient', 'feature_construction_included': feature_bounds is not None or target_bounds is not None, 'model_loss_error_bound': model_loss, 'model_gradient_error_bound': model_gradient, 'arithmetic': arithmetic, 'admissible': False, 'acceptance': 'absolute-only sufficient test of supplied abs+relative tolerances', 'assumptions': 'binary64 RN compatible dot arithmetic, finite intermediates; caller-provided feature/target enclosures must be justified'}
    if arithmetic['admissible']:
        fbound = max((cell['objective']['max_error_bound'] for cell in arithmetic['cells'])) + model_loss
        gbound = max((v['max_error_bound'] for cell in arithmetic['cells'] for v in cell['gradients'])) + model_gradient
        report.update(total_loss_error_bound=float(up(fbound)), total_gradient_error_bound=float(up(gbound)), admissible=bool(up(fbound) <= tolerances.forward_atol and up(gbound) <= tolerances.gradient_atol))
    return report

def low_rank_constants(adapter, *, function=None, args=None):
    """Bind full and diagonal cached geometry with provenance and safe fallback."""
    a, x = (np.asarray(adapter.base), np.asarray(adapter.points))
    z_bound = matrix_product(x, a.T)
    g_bound = matrix_product(a, a.T)
    bindings = ConstantBindings()
    for value, bound in ((adapter.projection, z_bound), (adapter.metric, g_bound)):
        bindings.bind_object(value, bound).add_legacy(value, bound)
    diagonal = Bound(np.diag(g_bound.lo), np.diag(g_bound.hi), np.diag(g_bound.error))
    bindings.add_legacy(np.diag(np.asarray(adapter.metric)), diagonal)
    return bindings

def certify_low_rank(adapter, factors, radius=1e-05, *, tolerances=Tolerances(), limits=Limits(cells=1)):

    def fn(params, points):
        return adapter.geometry(params)

    def objective(params, points, outputs):
        z, g = outputs
        return jnp.mean(jnp.tanh(z) ** 2) + 0.001 * jnp.sum(g ** 2)
    radii = radius if isinstance(radius, tuple) else (radius,) * len(factors)
    boxes = tuple((Bound(np.nextafter(np.asarray(v) - np.asarray(r), -np.inf), np.nextafter(np.asarray(v) + np.asarray(r), np.inf), 0) for v, r in zip(factors, radii)))
    zero = Bound(np.zeros((1, 1)), np.zeros((1, 1)), 0)
    report = qualify(fn, (factors, jnp.zeros((1, 1), dtype=jnp.float64)), Domain(boxes, zero), objective, tolerances, limits, constant_bounds=low_rank_constants(adapter))
    report['frozen_precomputation'] = 'projection and metric include initial dot arithmetic; provenance-safe constant bindings'
    return report

def certify_low_rank_operator(adapter, operators, parameters, *, provider='parity', scalar_objective=None, radii=1e-05, tolerances=Tolerances(), limits=Limits(cells=1)):
    """Qualify the actual derivative bundle, including all live factor gradients."""
    from .low_rank_operators import build_low_rank_ridge
    from .bundles import objective as default_objective
    function = build_low_rank_ridge(adapter, operators, provider)
    fn = lambda params, points: function(params)
    args = (parameters, jnp.zeros((1, 1), dtype=jnp.float64))
    radii = radii if isinstance(radii, tuple) else (radii,) * len(parameters)
    if len(radii) != len(parameters):
        raise ValueError('one radius per live parameter leaf required')
    boxes = tuple((Bound(np.nextafter(np.asarray(v) - np.asarray(r), -np.inf), np.nextafter(np.asarray(v) + np.asarray(r), np.inf), 0.0) for v, r in zip(parameters, radii)))
    domain = Domain(boxes, Bound(np.zeros((1, 1)), np.zeros((1, 1)), 0.0))
    report = qualify(fn, args, domain, scalar_objective or default_objective, tolerances, limits, constant_bounds=low_rank_constants(adapter))
    report['scope'] = 'ideal A0+U V^T operators; complete live U,V,bias,readout,offset first gradients'
    report['frozen_precomputation'] = 'saved projection and diagonal metric include initial dot arithmetic bounds'
    report['demands'] = function.demands
    return report

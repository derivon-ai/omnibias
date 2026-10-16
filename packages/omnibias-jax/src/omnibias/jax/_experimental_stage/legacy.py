"""Research-only cost/error selection. Existing distribution APIs are untouched."""
import os
import functools
import hashlib
import itertools
import math
import time
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.jet import jet
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.jet_algebra import JetSpec, JetBudget
from omnibias.core.jet_series import activation, multiply
from omnibias.core.polynomials import tanh_polynomial_coeffs
from omnibias.core.verified.coeffs import tanh_poly_coeffs_exact
from .bounds import Bound, interpret
OPS = SimpleNamespace(zeros_like=jnp.zeros_like, ones_like=jnp.ones_like, tanh=jnp.tanh, sigmoid=jax.nn.sigmoid)
TOLERANCES = {'forward': (1e-09, 2e-10), 'gradient': (1e-08, 2e-09)}

def raw_tower(z, orders, provider):
    orders = tuple(sorted(set(orders)))
    n = max(orders)
    if provider == 'polynomial':
        t = jnp.tanh(z)
        out = []
        for k in orders:
            if k == 0:
                out.append(t)
                continue
            coefficients = tanh_poly_coeffs_exact(k)
            if k <= 17 and any((int(float(c)) != c for c in coefficients)):
                raise ArithmeticError('coefficient narrowing needs a bound')
            value = jnp.ones_like(z) * float(coefficients[-1])
            for c in coefficients[-2::-1]:
                value = value * t + float(c)
            out.append(value)
        return tuple(out)
    if provider == 'jax_jet':
        if n == 0:
            return (jnp.tanh(z),)
        primal, rows = jet(jnp.tanh, (z,), ((jnp.ones_like(z),) + (jnp.zeros_like(z),) * (n - 1),), factorial_scaled=False)
        return tuple((primal if k == 0 else rows[k - 1] * float(math.factorial(k)) for k in orders))
    if provider != 'recurrence':
        raise ValueError(provider)
    rows = activation([z] + ([jnp.ones_like(z)] if n else []) + [jnp.zeros_like(z)] * max(0, n - 1), JetSpec.directional(n), 'tanh', OPS)
    return tuple((rows[k] * float(math.factorial(k)) for k in orders))

@functools.lru_cache(maxsize=128)
def derivative_function(orders, provider):
    """One shared tower; JVP uses a jointly selected derivative provider."""
    orders = tuple(orders)

    @jax.custom_jvp
    def evaluate(z):
        return raw_tower(z, orders, provider)

    @evaluate.defjvp
    def rule(primals, tangents):
        z, = primals
        v, = tangents
        demands = tuple(sorted(set(orders) | {k + 1 for k in orders}))
        all_rows = raw_tower(z, demands, provider)
        lookup = dict(zip(demands, all_rows))
        return (tuple((lookup[k] for k in orders)), tuple((lookup[k + 1] * v for k in orders)))
    return evaluate

def request(config):
    if config['family'] == 'separable':
        return OperatorSpec.mixed((1,) * config['dim'])
    if config['operator'] == 'sparse':
        return OperatorSpec.polynomial({(n,) + (0,) * (config['dim'] - 1): 1 for n in (0, 2, 4, 8)})
    return OperatorSpec.laplacian(config['power'])

def build(config, providers, representation='structural'):
    family = config['family']
    D = config['dim']
    op = request(config)
    if op.order > 16:
        raise NotImplementedError('order exceeds prototype support (32 is diagnostic only)')
    orders = (0, 2, 4, 8) if config['operator'] == 'sparse' else (op.order,)
    derivative = derivative_function(orders, providers[0]) if providers[0] != 'jax_jet' else lambda z: raw_tower(z, orders, 'jax_jet')

    def primal(p, x):
        if family == 'separable':
            return jnp.tanh(jnp.sum(jnp.tanh(x * p[0] + p[1]), axis=-1))
        if family == 'ridge':
            return jnp.tanh(x @ p[0].T + p[1]) @ p[2] + p[3]
        return jnp.tanh(jnp.tanh(x @ p[0].T + p[1]) @ p[2].T + p[3]) @ p[4] + p[5]
    if representation == 'ambient_taylor':
        spec = op.jet_spec(D)
        work = sum((len(q) for q in spec.pairs)) * 8
        op.planning_budget.check_graph(0, work)
        if work > op.planning_budget.max_graph_nodes:
            raise ArithmeticError('ambient Taylor emission exceeds conservative graph-work gate')

        def function(p, x):
            rows = []
            for alpha in spec.indices:
                if not any(alpha):
                    rows.append(x)
                elif sum(alpha) == 1:
                    rows.append(jnp.broadcast_to(jnp.eye(D, dtype=x.dtype)[alpha.index(1)], x.shape))
                else:
                    rows.append(jnp.zeros_like(x))
            if family == 'separable':
                values = [row * p[0] + (p[1] if i == 0 else 0) for i, row in enumerate(rows)]
                branches = activation(values, spec, 'tanh', OPS)
                result = activation([jnp.sum(v, axis=-1) for v in branches], spec, 'tanh', OPS)
            else:
                for i in range(0, len(p), 2):
                    weight, bias = p[i:i + 2]
                    values = [(row @ weight.T if weight.ndim == 2 else row @ weight) + (bias if k == 0 else 0) for k, row in enumerate(rows)]
                    rows = activation(values, spec, 'tanh', OPS) if i < len(p) - 2 else values
                result = rows
            return sum((result[i] * float(weight) for i, weight in op.jet_readout(spec)))
        return (function, primal)
    if family == 'ridge':

        def function(p, x):
            w, b, c, a = p
            z = x @ w.T + b
            rows = derivative(z)
            if config['operator'] == 'sparse':
                value = sum((d * w[:, 0] ** n for n, d in zip(orders, rows)))
                return value @ c + a
            return rows[0] @ (c * jnp.sum(w * w, axis=-1) ** config['power'])
    elif family == 'separable':
        branch = derivative_function((0, 1), providers[0]) if providers[0] != 'jax_jet' else lambda z: raw_tower(z, (0, 1), 'jax_jet')
        outer = derivative_function((D,), providers[1]) if providers[1] != 'jax_jet' else lambda z: raw_tower(z, (D,), 'jax_jet')

        def function(p, x):
            values, slopes = branch(x * p[0] + p[1])
            product = jnp.ones(x.shape[:-1], dtype=x.dtype)
            for i in range(D):
                product = product * (slopes[..., i] * p[0][i])
            return outer(jnp.sum(values, axis=-1))[0] * product
    else:
        k = config['power']
        N = 2 * k
        spec = JetSpec.mixed([(i, N - i) for i in range(N + 1)])
        outer = derivative_function(tuple(range(N + 1)), providers[1]) if providers[1] != 'jax_jet' else lambda z: raw_tower(z, tuple(range(N + 1)), 'jax_jet')
        inner = derivative_function(tuple(range(N + 1)), providers[0]) if providers[0] != 'jax_jet' else lambda z: raw_tower(z, tuple(range(N + 1)), 'jax_jet')
        if representation == 'metric':

            def function(p, x):
                A, b, V, d, c, a = p
                z = x @ A.T + b
                rows = inner(z)
                G = A @ A.T
                u = rows[0] @ V.T + d
                tower = outer(u)
                f1 = rows[1][..., None, :] * V
                f2 = rows[2][..., None, :] * V
                g0 = f1[..., 0] * G[0, 0] + f1[..., 1] * G[0, 1]
                g1 = f1[..., 0] * G[1, 0] + f1[..., 1] * G[1, 1]
                Q = f1[..., 0] * g0 + f1[..., 1] * g1
                L = f2[..., 0] * G[0, 0] + f2[..., 1] * G[1, 1]
                if k == 1:
                    result = tower[2] * Q + tower[1] * L
                else:
                    f3 = rows[3][..., None, :] * V
                    f4 = rows[4][..., None, :] * V
                    ghg = f2[..., 0] * g0 * g0 + f2[..., 1] * g1 * g1
                    gradL = g0 * f3[..., 0] * G[0, 0] + g1 * f3[..., 1] * G[1, 1]
                    hnorm = f2[..., 0] * f2[..., 0] * G[0, 0] * G[0, 0] + 2 * f2[..., 0] * f2[..., 1] * G[0, 1] * G[0, 1] + f2[..., 1] * f2[..., 1] * G[1, 1] * G[1, 1]
                    fourth = f4[..., 0] * G[0, 0] * G[0, 0] + f4[..., 1] * G[1, 1] * G[1, 1]
                    result = tower[4] * Q * Q + tower[3] * (2 * L * Q + 4 * ghg) + tower[2] * (L * L + 4 * gradL + 2 * hnorm) + tower[1] * fourth
                return result @ c
            return (function, primal)

        def function(p, x):
            A, b, V, d, c, a = p
            z = x @ A.T + b
            rows = {}
            for r in range(2):
                tower = inner(z[..., r])
                rows[r] = [tower[alpha[r]] / float(math.factorial(alpha[r])) if alpha[1 - r] == 0 else jnp.zeros_like(z[..., r]) for alpha in spec.indices]
            u = []
            for idx, alpha in enumerate(spec.indices):
                val = rows[0][idx][..., None] * V[:, 0] + rows[1][idx][..., None] * V[:, 1]
                if sum(alpha) == 0:
                    val = val + d
                u.append(val)
            if representation == 'latent_recurrence':
                out = activation(u, spec, 'tanh', OPS)
            else:
                tower = outer(u[0])
                zero = jnp.zeros_like(u[0])
                v = [zero] + u[1:]
                power = [jnp.ones_like(u[0])] + [zero] * (spec.size - 1)
                out = [tower[0]] + [zero] * (spec.size - 1)
                for degree in range(1, N + 1):
                    power = multiply(power, v, spec, OPS)
                    out = [q + tower[degree] / float(math.factorial(degree)) * r for q, r in zip(out, power)]
            G = A @ A.T
            symbol = {(0, 0): jnp.asarray(1.0, dtype=x.dtype)}
            for _ in range(k):
                fresh = {}
                for alpha, coefficient in symbol.items():
                    for delta, weight in (((2, 0), G[0, 0]), ((1, 1), 2 * G[0, 1]), ((0, 2), G[1, 1])):
                        beta = tuple((i + j for i, j in zip(alpha, delta)))
                        fresh[beta] = fresh.get(beta, 0) + coefficient * weight
                symbol = fresh
            result = jnp.zeros(x.shape[:-1], dtype=x.dtype)
            lookup = {alpha: i for i, alpha in enumerate(spec.indices)}
            for alpha, weight in symbol.items():
                result = result + weight * float(math.prod((math.factorial(i) for i in alpha))) * (out[lookup[alpha]] @ c)
            return result
    return (function, primal)

def parameter_data(config, seed):
    rng = np.random.default_rng(seed)
    D = config['dim']
    H = config.get('hidden', 16)
    if config['family'] == 'separable':
        return (rng.uniform(0.04, 0.08, D), rng.uniform(-0.025, 0.025, D))
    if config['family'] == 'ridge':
        return (rng.normal(0, 0.075, (H, D)), rng.uniform(-0.03, 0.03, H), rng.normal(0, 0.1, H), np.asarray(0.03))
    return (rng.normal(0, 0.075, (2, D)), rng.uniform(-0.03, 0.03, 2), rng.normal(0, 0.12, (H, 2)), rng.uniform(-0.03, 0.03, H), rng.normal(0, 0.1, H), np.asarray(0.03))

def parameter_box(p, radius=1e-05):
    return tuple((Bound(np.nextafter(v - radius, -np.inf), np.nextafter(v + radius, np.inf), 0.0) for v in p))

def input_box(D):
    return Bound(np.full((1, D), -0.25), np.full((1, D), 0.25), 0.0)

def loss_function(fn):
    return lambda p, x: jnp.mean((fn(p, x) / 1000.0) ** 2)

def graph_cost(closed, weights):
    total = 0.0
    state = 0
    count = 0

    def walk(graph):
        nonlocal total, state, count
        for e in graph.eqns:
            count += 1
            size = sum((math.prod(getattr(v.aval, 'shape', ())) for v in e.outvars))
            state = max(state, size * 8)
            factor = 1
            if e.primitive.name == 'dot_general':
                contractions = e.params['dimension_numbers'][0][0]
                factor = math.prod((e.invars[0].aval.shape[i] for i in contractions))
            total += weights.get(e.primitive.name, 1.0) * size * factor
            sub = e.params.get('jaxpr', e.params.get('call_jaxpr'))
            if sub is not None:
                walk(getattr(sub, 'jaxpr', sub))
    walk(closed.jaxpr)
    return {'estimated_cost': total, 'largest_value_bytes': state, 'equations': count}

def enumerate_candidates(config):
    regions = 1 if config['family'] == 'ridge' else 2
    representations = ('metric', 'structural', 'latent_recurrence') if config['family'] == 'bottleneck' else ('structural',)
    all_candidates = [(p, r) for r in representations for p in itertools.product(('polynomial', 'recurrence'), repeat=regions) if r != 'latent_recurrence' or p[-1] == 'recurrence']
    all_candidates.append((('recurrence',) * regions, 'ambient_taylor'))
    return (all_candidates[:256], len(all_candidates) > 256)

def analyze_cells(forward, gradient, boxes, max_equations, max_cells):
    if not 1 <= max_cells <= 4096:
        raise ValueError('analysis cell budget must be 1..4096')
    pending = [boxes]
    accepted = []
    leaves = 1
    analyses = 0
    while pending:
        current = pending.pop(0)
        analyses += 1
        fb, fd = interpret(forward, current, max_equations)
        gb, gd = interpret(gradient, current, max_equations)
        ok = fb[0].summary(*TOLERANCES['forward'])['admissible'] and all((v.summary(*TOLERANCES['gradient'])['admissible'] for v in gb[1:]))
        if ok:
            accepted.append((fb, gb))
            continue
        if leaves >= max_cells:
            return (fb, gb, fd, gd, {'leaf_cells': leaves, 'analyzed_cells': analyses, 'complete_cover': False, 'reason': 'accuracy subdivision budget exhausted'})
        widths = current[-1].hi - current[-1].lo
        idx = np.unravel_index(np.argmax(widths), widths.shape)
        if widths[idx] == 0:
            return (fb, gb, fd, gd, {'leaf_cells': leaves, 'analyzed_cells': analyses, 'complete_cover': False, 'reason': 'point domain remains inconclusive'})
        midpoint = float((current[-1].hi[idx] + current[-1].lo[idx]) / 2)
        left_hi = current[-1].hi.copy()
        left_hi[idx] = midpoint
        right_lo = current[-1].lo.copy()
        right_lo[idx] = midpoint
        pending.extend([current[:-1] + [Bound(current[-1].lo, left_hi, 0.0)], current[:-1] + [Bound(right_lo, current[-1].hi, 0.0)]])
        leaves += 1

    def merge(column):
        return [Bound(np.minimum.reduce([row[column][i].lo for row in accepted]), np.maximum.reduce([row[column][i].hi for row in accepted]), np.maximum.reduce([row[column][i].error for row in accepted])) for i in range(len(accepted[0][column]))]
    return (merge(0), merge(1), fd, gd, {'leaf_cells': leaves, 'analyzed_cells': analyses, 'complete_cover': True})

def select(config, p, weights, max_equations=100000, batch=1, max_cells=1):
    start = time.perf_counter()
    records = []
    chosen = []
    configs, truncated = enumerate_candidates(config)
    args = (tuple((jnp.asarray(v) for v in p)), jnp.zeros((batch, config['dim']), dtype=jnp.float64))
    box = input_box(config['dim'])
    boxes = [*parameter_box(p), Bound(np.broadcast_to(box.lo, (batch, config['dim'])), np.broadcast_to(box.hi, (batch, config['dim'])), 0.0)]
    for providers, representation in configs:
        identifier = representation + ':' + ','.join(providers)
        row = {'candidate': identifier, 'providers': providers, 'representation': representation, 'algebraic_identity': 'proved under documented structural prerequisites'}
        try:
            fn, _ = build(config, providers, representation)
            forward = jax.make_jaxpr(fn)(*args)
            vg = jax.make_jaxpr(jax.value_and_grad(loss_function(fn)))(*args)
            fb, gb, fd, gd, cells = analyze_cells(forward, vg, boxes, min(max_equations, request(config).planning_budget.max_graph_nodes), max_cells)
            row['domain_analysis'] = cells
            row['forward_bound'] = fb[0].summary(*TOLERANCES['forward'])
            row['gradient_bounds'] = [v.summary(*TOLERANCES['gradient']) for v in gb[1:]]
            row['loss_bound'] = gb[0].summary(*TOLERANCES['forward'])
            row['graph_analysis'] = {'forward': fd, 'value_gradient': gd}
            row.update(graph_cost(vg, weights))
            row['admissible'] = row['forward_bound']['admissible'] and all((v['admissible'] for v in row['gradient_bounds']))
            row['admissible'] = row['admissible'] and cells['complete_cover']
            request(config).planning_budget.check_graph(row['equations'])
            if row['largest_value_bytes'] * (request(config).order + 1) > request(config).budget.max_tensor_bytes:
                raise ArithmeticError('conservative forward state exceeds tensor budget')
            row['status'] = 'conditionally_admissible' if row['admissible'] else 'inconclusive_accuracy'
            if row['admissible']:
                chosen.append((row['estimated_cost'], row['largest_value_bytes'], identifier, fn))
        except (ArithmeticError, NotImplementedError, ValueError) as e:
            row.update(status='inconclusive', admissible=False, reason=str(e))
        records.append(row)
    chosen.sort(key=lambda t: t[:3])
    report = {'config': config, 'batch': batch, 'candidates': records, 'truncated': truncated, 'selected': chosen[0][2] if chosen else None, 'status': 'selected' if chosen else 'inconclusive', 'planning_ms': (time.perf_counter() - start) * 1000, 'domain': 'Each input coordinate [-.25,.25], every parameter +/-1e-5 about frozen binary64 center', 'optimality': 'minimum calibrated graph estimate among admissible enumerated candidates', 'guarantee': 'conditional operation-model bounds; compiled native implementation independently tested', 'declined_alternative': 'Apolar lowering of transported latent operators requires a fixed exact symbol; runtime Gram coefficients are retained, so this prototype does not freeze an apolar basis from current parameter values.' if config['family'] == 'bottleneck' else None}
    return (chosen[0][3] if chosen else None, report)

def capture_model(config, p):
    """Use existing Keras capture; recognition is graph-based, never fitted rank."""
    import keras
    from omnibias.keras.experimental.operator import _capture
    x = keras.Input((config['dim'],), dtype='float64')
    family = config['family']
    if family == 'separable':
        total = None
        branches = []
        for i in range(config['dim']):
            layer = keras.layers.Dense(1, activation='tanh', dtype='float64', name='branch_' + str(i))
            value = layer(x[:, i:i + 1])
            layer.set_weights([np.asarray([[p[0][i]]]), np.asarray([p[1][i]])])
            branches.append(layer)
            total = value if total is None else total + value
        y = keras.layers.Activation('tanh', dtype='float64')(total)
        model = keras.Model(x, y)
        lookup = {id(v): (leaf, i, tuple(v.shape)) for i, layer in enumerate(branches) for leaf, v in enumerate((layer.kernel, layer.bias))}
        object.__setattr__(model, '_research_binding', [lookup[id(v)] for v in model.trainable_variables])
    else:
        z = x
        widths = (len(p[1]), 1) if family == 'ridge' else (2, len(p[3]), 1)
        for i, width in enumerate(widths):
            z = keras.layers.Dense(width, activation='tanh' if i < len(widths) - 1 else None, dtype='float64')(z)
        model = keras.Model(x, z)
        values = []
        for i in range(0, len(p), 2):
            values.extend([p[i].T if p[i].ndim == 2 else p[i].reshape(-1, 1), np.atleast_1d(p[i + 1])])
        model.set_weights(values)
    nodes, output = _capture(model, request(config), 'strict')
    from omnibias.keras.experimental.operator import _ridge, _bottleneck, _separable_cut, _scalar_cut, _read_once
    dimension = config['dim']
    op = request(config)
    if family == 'ridge':
        proved = _ridge(nodes, output)
    elif family == 'bottleneck':
        proved = _bottleneck(nodes, output, dimension) == 2
    else:
        proved = _separable_cut(nodes, output) is not None or _scalar_cut(nodes, output, dimension) is not None or _read_once(nodes, output, dimension)
    if not proved:
        raise NotImplementedError('captured graph did not prove requested family')
    object.__setattr__(model, '_research_contract', (dict(config), tuple((hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in p))))
    return (model, {'nodes': len(nodes), 'family_proved': family, 'rules': [n.rule or n.activation or type(n.source).__name__ for n in nodes], 'recognition': 'declared architecture and captured graph; no parameter-value sparsity or rank tests'})

def existing_compiler(config, p):
    from omnibias.keras.experimental import operator_model
    source, report = capture_model(config, p)
    model = operator_model(source, operator=request(config), activation_method='recurrence', precision='strict')
    model(jnp.zeros((1, config['dim']), dtype=jnp.float64))

    def fn(params, x):
        if config['family'] == 'separable':
            variables = [params[leaf][i].reshape(shape) for leaf, i, shape in source._research_binding]
        else:
            variables = []
            for i in range(0, len(params), 2):
                variables.extend([params[i].T if params[i].ndim == 2 else params[i].reshape(-1, 1), params[i + 1].reshape(-1)])
        y, _ = model.stateless_call(variables, [], x, training=False)
        return y.squeeze(-1)
    return (fn, {'capture': report, 'compiler': model.explain()})

def domain_check(p, x, centers):
    return bool(np.all(np.isfinite(x)) and np.all(x >= -0.25) and np.all(x <= 0.25) and all((np.all(np.isfinite(v)) and np.all(v >= np.nextafter(c - 1e-05, -np.inf)) and np.all(v <= np.nextafter(c + 1e-05, np.inf)) for v, c in zip(p, centers))))

def compile_research(captured_model, config, p, weights):
    """Captured model accepted only through the three supported factory patterns."""
    from omnibias.keras.experimental.operator import _capture, _ridge, _bottleneck, _separable_cut, _scalar_cut, _read_once
    contract = (dict(config), tuple((hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in p)))
    if getattr(captured_model, '_research_contract', None) != contract:
        return (None, {'status': 'unsupported', 'reason': 'model adapter is limited to the captured factory and declared parameter centers'})
    nodes, out = _capture(captured_model, request(config), 'strict')
    D = config['dim']
    family = config['family']
    valid = _ridge(nodes, out) if family == 'ridge' else _bottleneck(nodes, out, D) == 2 if family == 'bottleneck' else _separable_cut(nodes, out) is not None or _scalar_cut(nodes, out, D) is not None or _read_once(nodes, out, D)
    if not valid:
        return (None, {'status': 'unsupported', 'reason': 'capture did not prove structure'})
    result, report = select(config, p, weights)
    report['capture_nodes'] = len(nodes)
    report['adapter_scope'] = 'models produced by capture_model with this config and centers only'
    return (result, report)

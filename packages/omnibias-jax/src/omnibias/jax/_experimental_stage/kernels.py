"""Operator demand reductions and contracted metric adjoints."""
import functools
import jax
import jax.numpy as jnp
from .providers import tower
from . import legacy

def requested(c):
    if 'requested_operator' in c:
        return legacy.OperatorSpec.from_config(c['requested_operator'])
    return legacy.request(c)

def balanced(values):
    values = list(values)
    if not values:
        raise ValueError('empty product')
    while len(values) > 1:
        values = [values[i] * values[i + 1] if i + 1 < len(values) else values[i] for i in range(0, len(values), 2)]
    return values[0]

def metric_local(r0, r1, v0, v1, d, g00, g01, g11, c, power, providers):
    N = 2 * power
    outer = tower(tuple(range(1, N + 1)), providers[1])
    u = v0 * r0[0] + v1 * r1[0] + d
    f = outer(u)
    a0 = v0 * r0[1]
    a1 = v1 * r1[1]
    h0 = v0 * r0[2]
    h1 = v1 * r1[2]
    g0 = a0 * g00 + a1 * g01
    g1 = a0 * g01 + a1 * g11
    Q = a0 * g0 + a1 * g1
    L = h0 * g00 + h1 * g11
    if power == 1:
        return c * (f[1] * Q + f[0] * L)
    j0 = v0 * r0[3]
    j1 = v1 * r1[3]
    k0 = v0 * r0[4]
    k1 = v1 * r1[4]
    ghg = h0 * g0 * g0 + h1 * g1 * g1
    gradL = g0 * j0 * g00 + g1 * j1 * g11
    hnorm = h0 * h0 * g00 * g00 + 2 * h0 * h1 * g01 * g01 + h1 * h1 * g11 * g11
    fourth = k0 * g00 * g00 + k1 * g11 * g11
    return c * (f[3] * Q * Q + f[2] * (2 * L * Q + 4 * ghg) + f[1] * (L * L + 4 * gradL + 2 * hnorm) + f[0] * fourth)

def local_args(p, x, power, providers):
    A, b, V, d, c, a = p
    z = x @ A.T + b
    shape = (x.shape[0], c.shape[0])
    G = A @ A.T
    bcast = lambda v: jnp.broadcast_to(v, shape)
    inner = tower(tuple(range(2 * power + 1)), providers[0])
    r0 = inner(z[:, 0])
    r1 = inner(z[:, 1])
    return (*tuple((bcast(v[:, None]) for v in r0)), *tuple((bcast(v[:, None]) for v in r1)), *tuple(map(bcast, (V[None, :, 0], V[None, :, 1], d[None, :], G[0, 0], G[0, 1], G[1, 1], c[None, :]))))

def metric(config, providers, adjoint):
    count = 2 * config['power'] + 1

    def local(*args):
        return metric_local(args[:count], args[count:2 * count], *args[2 * count:], power=config['power'], providers=providers)

    def plain(p, x):
        return jnp.sum(local(*local_args(p, x, config['power'], providers)), axis=1)
    if not adjoint:
        return plain

    @jax.custom_vjp
    def fn(p, x):
        return plain(p, x)

    def forward(p, x):
        return (plain(p, x), (p, x))

    def backward(residual, bar):
        p, x = residual
        A, b, V, d, c, a = p
        args = local_args(p, x, config['power'], providers)
        _, pull = jax.vjp(local, *args)
        bars = pull(jnp.broadcast_to(bar[:, None], args[0].shape))
        z = x @ A.T + b
        inner = tower(tuple(range(count)), providers[0])
        zbars = []
        for axis in range(2):
            _, inner_pull = jax.vjp(inner, z[:, axis])
            cot = tuple((jnp.sum(bars[axis * count + n], axis=1) for n in range(count)))
            zbars.append(inner_pull(cot)[0])
        zbar = jnp.stack(zbars, axis=1)
        offset = 2 * count
        Vbar = jnp.stack((jnp.sum(bars[offset], axis=0), jnp.sum(bars[offset + 1], axis=0)), axis=1)
        g00, g01, g11 = (jnp.sum(bars[offset + i]) for i in (3, 4, 5))
        metricbar = jnp.stack((2 * g00 * A[0] + g01 * A[1], g01 * A[0] + 2 * g11 * A[1]))
        pbar = (zbar.T @ x + metricbar, jnp.sum(zbar, axis=0), Vbar, jnp.sum(bars[offset + 2], axis=0), jnp.sum(bars[offset + 6], axis=0), jnp.zeros_like(a))
        return (pbar, zbar @ A)
    fn.defvjp(forward, backward)
    return fn

def build(config, providers=('polynomial',), representation='direct'):
    family = config['family']
    D = config['dim']
    op = requested(config)
    if op.order > 16:
        raise NotImplementedError('order 32 is a refusal/stress case')
    if 'requested_operator' in config:
        from .operators import supplied_build
        return supplied_build(config, op, providers, representation)
    if family == 'bottleneck':
        if representation in ('metric_ad', 'metric_vjp'):
            return metric(config, providers, representation == 'metric_vjp')
        if representation in ('latent', 'latent_recurrence'):
            previous = legacy.derivative_function
            try:
                legacy.derivative_function = lambda orders, provider: tower(tuple(orders), provider)
                return legacy.build(config, providers, 'structural' if representation == 'latent' else representation)[0]
            finally:
                legacy.derivative_function = previous
        raise ValueError(representation)
    if family == 'ridge':
        orders = (0, 2, 4, 8) if config['operator'] == 'sparse' else (op.order,)
        dt = tower(orders, providers[0])

        def fn(p, x):
            W, b, c, a = p
            rows = dt(x @ W.T + b)
            if config['operator'] == 'sparse':
                return sum((r * W[:, 0] ** n for n, r in zip(orders, rows))) @ c + a
            return rows[0] @ (c * jnp.sum(W * W, axis=1) ** config['power'])
        return fn
    branch = tower((0, 1), providers[0])
    outer = tower((D,), providers[1])

    def fn(p, x):
        values, slopes = branch(x * p[0] + p[1])
        factors = slopes * p[0]
        return outer(jnp.sum(values, axis=1))[0] * balanced((factors[:, i] for i in range(D)))
    return fn

def selected_hessian_ridge(p, x, axes, provider='polynomial'):
    W, b, c, a = p
    axes = tuple(axes)
    if len(set(axes)) != len(axes) or any((i < 0 or i >= W.shape[1] for i in axes)):
        raise ValueError('invalid axes')
    weights = jnp.stack([W[:, i] for i in axes], axis=1)
    second = tower((2,), provider)(x @ W.T + b)[0] * c
    return jnp.einsum('bh,hi,hj->bij', second, weights, weights)

def demands(config, providers, representation):
    N = requested(config).order
    if 'requested_operator' in config:
        terms = requested(config).resolve_terms(config['dim'])
        if config['family'] == 'ridge':
            regions = [tuple(sorted({sum(a) for a, v in terms}))]
        elif config['family'] == 'separable':
            regions = [tuple(range(max((max(a) for a, v in terms)) + 1)), tuple(range(N + 1))]
        else:
            regions = [tuple(range(N + 1)), tuple(range(N + 1))]
    elif config['family'] == 'ridge':
        wanted = (0, 2, 4, 8) if config['operator'] == 'sparse' else (N,)
        regions = [wanted]
    elif config['family'] == 'separable':
        regions = [(0, 1), (N,)]
    else:
        regions = [tuple(range(N + 1)), tuple(range(1, N + 1))]
    return {'family': config['family'], 'representation': representation, 'regions': [{'graph_identity': f'activation_region_{i}', 'requested': list(r), 'gradient_union': sorted(set(r) | {n + 1 for n in r}), 'computed': list(range(max(r) + 2)) if provider in ('recurrence', 'jax_native', 'jax_analytic') else sorted(set(r) | {n + 1 for n in r}), 'base_activation_carrier_order': 0, 'carrier_note': 'source evaluates tanh(z) as the polynomial carrier; optimized graphs may eliminate unused carriers in tail evaluation', 'provider': provider} for i, (r, provider) in enumerate(zip(regions, providers))], 'structural_prerequisite': 'captured graph proves ridge, disjoint scalar branches, or affine latent width two; no numerical-zero pruning'}

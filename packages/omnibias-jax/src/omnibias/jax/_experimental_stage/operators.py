"""Bounded supplied constant-coefficient operators for the captured factories."""
import math
from fractions import Fraction
import jax.numpy as jnp
from .legacy import request, OPS
from .kernels import build as build_kernel
from .providers import tower

def build(config, providers, representation):
    config = dict(config)
    config.pop('requested_operator', None)
    return (build_kernel(config, providers, representation), None)

def derivative_function(orders, provider):
    return tower(tuple(orders), provider)
from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import activation, multiply

def exact_weight(q):
    q = Fraction(q)
    f = float(q)
    if not math.isfinite(f) or Fraction(f) != q:
        raise NotImplementedError('nonrepresentable supplied coefficient needs an additional narrowing-error node')
    return f

def equivalent(a, b):
    return (a.mode, a.order, a.alpha, a.power, a.terms, a.dim) == (b.mode, b.order, b.alpha, b.power, b.terms, b.dim)

def supplied_build(config, operator, providers, representation):
    if equivalent(operator, request(config)):
        return build(config, providers, representation)[0]
    if operator.mode == 'directional':
        raise NotImplementedError('supplied runtime direction and its domain are not implemented')
    if operator.order > 16:
        raise NotImplementedError('order exceeds16; order17 only for first gradients')
    terms = operator.resolve_terms(config['dim'])
    terms = tuple(((a, exact_weight(c)) for a, c in terms))
    if representation in ('metric', 'metric_ad', 'metric_vjp', 'ambient_taylor'):
        raise NotImplementedError('this supplied operator has no implemented metric/ambient lowering')
    family = config['family']
    N = operator.order
    if family == 'ridge':
        orders = tuple(sorted({sum(a) for a, c in terms}))
        derivatives = derivative_function(orders, providers[0])

        def fn(p, x):
            W, b, c, a = p
            z = x @ W.T + b
            tower = dict(zip(orders, derivatives(z)))
            result = jnp.zeros(x.shape[:-1], dtype=x.dtype)
            for alpha, weight in terms:
                factor = jnp.ones_like(c)
                for i, power in enumerate(alpha):
                    if power:
                        factor = factor * W[:, i] ** power
                result = result + weight * (tower[sum(alpha)] @ (c * factor))
                if not any(alpha):
                    result = result + weight * a
            return result
        return fn
    if family == 'separable':
        top_branch = max((max(alpha) for alpha, c in terms))
        branches = derivative_function(tuple(range(top_branch + 1)), providers[0])
        outer = derivative_function(tuple(range(N + 1)), providers[1])

        def fn(p, x):
            W, b = p
            raw = branches(x * W + b)
            phi = outer(jnp.sum(raw[0], axis=-1))
            zero = jnp.zeros(x.shape[:-1], dtype=x.dtype)
            one = jnp.ones_like(zero)
            answer = zero
            for alpha, weight in terms:
                packet = [one]
                for coordinate, n in enumerate(alpha):
                    if not n:
                        continue
                    bell = {(0, 0): one}
                    for i in range(1, n + 1):
                        bell[i, 0] = zero
                        for k in range(1, i + 1):
                            value = zero
                            for j in range(1, i - k + 2):
                                value = value + math.comb(i - 1, j - 1) * raw[j][..., coordinate] * W[coordinate] ** j * bell.get((i - j, k - 1), zero)
                            bell[i, k] = value
                    current = [bell[n, k] for k in range(n + 1)]
                    nextpacket = [zero] * (len(packet) + n)
                    for i, a in enumerate(packet):
                        for j, bv in enumerate(current):
                            nextpacket[i + j] = nextpacket[i + j] + a * bv
                    packet = nextpacket
                answer = answer + weight * sum((v * phi[i] for i, v in enumerate(packet)))
            return answer
        return fn
    targets = tuple(((i, n - i) for n in {sum(a) for a, c in terms} for i in range(n + 1)))
    spec = JetSpec.mixed(targets, budget=operator.budget)
    if sum(map(len, spec.pairs)) * 8 > operator.planning_budget.max_graph_nodes:
        raise ArithmeticError('latent composition exceeds conservative graph-work gate')
    inner = derivative_function(tuple(range(N + 1)), providers[0])
    outer = derivative_function(tuple(range(N + 1)), providers[1])

    def fn(p, x):
        A, b, V, d, c, a = p
        z = x @ A.T + b
        u = []
        rows = [inner(z[..., r]) for r in range(2)]
        for alpha in spec.indices:
            value = jnp.zeros(x.shape[:-1] + (V.shape[0],), dtype=x.dtype)
            for r in range(2):
                if alpha[1 - r] == 0:
                    value = value + rows[r][alpha[r]][..., None] * V[:, r] / float(math.factorial(alpha[r]))
            u.append(value + d if not any(alpha) else value)
        if representation == 'latent_recurrence':
            values = activation(u, spec, 'tanh', OPS)
        else:
            tower = outer(u[0])
            zero = jnp.zeros_like(u[0])
            power = [jnp.ones_like(zero)] + [zero] * (spec.size - 1)
            incoming = [zero] + u[1:]
            values = [tower[0]] + [zero] * (spec.size - 1)
            for n in range(1, N + 1):
                power = multiply(power, incoming, spec, OPS)
                values = [v + tower[n] / float(math.factorial(n)) * q for v, q in zip(values, power)]
        index = {a: i for i, a in enumerate(spec.indices)}
        answer = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for alpha, coefficient in terms:
            symbol = {(0, 0): jnp.asarray(coefficient, dtype=x.dtype)}
            for coordinate, n in enumerate(alpha):
                for _ in range(n):
                    fresh = {}
                    for beta, weight in symbol.items():
                        for r in range(2):
                            target = tuple((beta[i] + int(i == r) for i in range(2)))
                            fresh[target] = fresh.get(target, 0) + weight * A[r, coordinate]
                    symbol = fresh
            for beta, weight in symbol.items():
                answer = answer + weight * float(math.prod((math.factorial(i) for i in beta))) * (values[index[beta]] @ c)
            if not any(alpha):
                answer = answer + coefficient * a
        return answer
    return fn

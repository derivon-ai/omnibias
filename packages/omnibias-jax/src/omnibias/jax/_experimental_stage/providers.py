"""Equivalent derivative representations; research-only analytic first JVPs."""
import functools
import math
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from omnibias.core.verified.coeffs import tanh_poly_coeffs_exact

def deriv(c):
    return tuple((i * c[i] for i in range(1, len(c)))) or (0,)

def add(*terms):
    out = [0] * max((len(c) for c in terms))
    for c in terms:
        for i, v in enumerate(c):
            out[i] += v
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)

def scale(c, k, shift=0):
    return (0,) * shift + tuple((k * v for v in c))

@functools.lru_cache(maxsize=128)
def reduced(n):
    if n < 1:
        raise ValueError('positive order required')
    r = (1,)
    for m in range((n + 1) // 2):
        q = scale(add(r, scale(deriv(r), 1, 1)), -2)
        if n == 2 * m + 1:
            return r
        if n == 2 * m + 2:
            return q
        qp = deriv(q)
        r = add(scale(q, 3, 1), scale(q, -2), scale(qp, -2, 1), scale(qp, 2, 2))

@functools.lru_cache(maxsize=128)
def eulerian(n):
    if n == 0:
        return (1,)
    a = (1,)
    for m in range(2, n + 1):
        a = tuple(((k + 1) * (a[k] if k < len(a) else 0) + (m - k) * (a[k - 1] if k else 0) for k in range(m)))
    return a

def horner(c, z):
    y = jnp.full_like(z, float(c[-1]))
    for v in c[-2::-1]:
        y = y * z + float(v)
    return y

@functools.lru_cache(maxsize=128)
def exact_coefficients(n, provider):
    c = reduced(n) if provider == 'sech' else eulerian(n) if provider == 'tail' else tanh_poly_coeffs_exact(n)
    return all((int(float(v)) == v for v in c))

def raw(z, orders, provider):
    orders = tuple(orders)
    n = max(orders, default=0)
    if any((not isinstance(k, int) or k < 0 for k in orders)):
        raise ValueError('nonnegative integer derivative orders required')
    if n > 33:
        raise NotImplementedError('activation order exceeds stress limit')
    if n <= 17 and provider in ('polynomial', 'parity', 'sech', 'tail') and any((not exact_coefficients(k, provider) for k in orders if k)):
        raise ArithmeticError('coefficient conversion requires an additional bound')
    t = jnp.tanh(z)
    if provider in ('jax_native', 'jax_analytic'):
        if not n:
            return (t,)
        p, rows = jet(jnp.tanh, (z,), ((jnp.ones_like(z),) + (jnp.zeros_like(z),) * (n - 1),), factorial_scaled=False)
        return tuple((p if k == 0 else rows[k - 1] * float(math.factorial(k)) for k in orders))
    if provider in ('sech', 'recurrence', 'tail'):
        q = jnp.exp(-2 * jnp.abs(z))
        s = 4 * q / ((1 + q) * (1 + q))
    if provider == 'recurrence':
        rows = [t, s]
        for k in range(1, n):
            rows.append(-sum((float(math.comb(k, j)) * rows[j] * rows[k - j] for j in range(k + 1))))
        return tuple((rows[k] for k in orders))
    out = []
    for k in orders:
        if k == 0:
            out.append(t)
            continue
        if provider == 'polynomial':
            v = horner(tanh_poly_coeffs_exact(k), t)
        elif provider == 'parity':
            c = tanh_poly_coeffs_exact(k)
            parity = (k + 1) % 2
            v = horner(c[parity::2], t * t) * (t if parity else 1)
        elif provider == 'sech':
            v = s * horner(reduced(k), s) * (t if k % 2 == 0 else 1)
        elif provider == 'tail':
            a = eulerian(k)
            v = (-1) ** (k + 1) * 2.0 ** (k + 1) * q * horner(a, -q) / (1 + q) ** (k + 1)
            if k % 2 == 0:
                v = jnp.where(z < 0, -v, v)
        else:
            raise ValueError(provider)
        out.append(v)
    return tuple(out)

@functools.lru_cache(maxsize=128)
def tower(orders, provider):
    orders = tuple(sorted(set(orders)))
    if provider == 'jax_native':
        return lambda z: raw(z, orders, provider)

    @jax.custom_jvp
    def fn(z):
        return raw(z, orders, provider)

    @fn.defjvp
    def jvp(primals, tangents):
        z, = primals
        v, = tangents
        demand = tuple(sorted(set(orders) | {k + 1 for k in orders}))
        d = dict(zip(demand, raw(z, demand, provider)))
        return (tuple((d[k] for k in orders)), tuple((d[k + 1] * v for k in orders)))
    return fn

def trig_tower(z, orders, kind='sin'):
    if any((not isinstance(n, int) or n < 0 for n in orders)):
        raise ValueError('nonnegative integer derivative orders required')
    s, c = (jnp.sin(z), jnp.cos(z))
    cycle = (s, c, -s, -c) if kind == 'sin' else (c, -s, -c, s)
    return tuple((cycle[n % 4] for n in orders))

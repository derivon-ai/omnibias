"""Research adapter: same Python expression, native JAX or public core jet rules.
This is not an Omnibias arbitrary-callable JAX transformation API.
"""
import math
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from omnibias.core.jet_algebra import JetSpec
from omnibias.core import jet_series

class Ops:
    zeros_like = staticmethod(jnp.zeros_like)
    ones_like = staticmethod(jnp.ones_like)
    tanh = staticmethod(jnp.tanh)
    sigmoid = staticmethod(jax.nn.sigmoid)
    exp = staticmethod(jnp.exp)
    sin = staticmethod(jnp.sin)
    cos = staticmethod(jnp.cos)

class Series:

    def __init__(self, coefficients, spec):
        self.c, self.spec = (coefficients, spec)

    def lift(self, other):
        if isinstance(other, Series):
            return other
        return Series([jnp.ones_like(self.c[0]) * other] + [jnp.zeros_like(self.c[0])] * (self.spec.size - 1), self.spec)

    def __add__(self, other):
        other = self.lift(other)
        return Series([a + b for a, b in zip(self.c, other.c)], self.spec)
    __radd__ = __add__

    def __neg__(self):
        return Series([-a for a in self.c], self.spec)

    def __sub__(self, other):
        return self + -self.lift(other)

    def __rsub__(self, other):
        return self.lift(other) + -self

    def __mul__(self, other):
        if not isinstance(other, Series):
            return Series([a * other for a in self.c], self.spec)
        return Series(jet_series.multiply(self.c, other.c, self.spec, Ops), self.spec)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Series):
            return Series([a / other for a in self.c], self.spec)
        return self * Series(jet_series.reciprocal(other.c, self.spec, Ops), self.spec)

class SeriesOps:

    @staticmethod
    def apply(name, x):
        return Series(jet_series.activation(x.c, x.spec, name, Ops), x.spec)

    @staticmethod
    def tanh(x):
        return SeriesOps.apply('tanh', x)

    @staticmethod
    def sin(x):
        return SeriesOps.apply('sin', x)

    @staticmethod
    def exp(x):
        return SeriesOps.apply('exp', x)

    @staticmethod
    def cos(x):
        return SeriesOps.apply('cos', x)

def expression(name, op, x, p):
    if name == 'nested':
        return op.tanh(x * p[0] + op.tanh(x * p[2] + p[3]) * p[1])
    if name == 'rational':
        return op.tanh(x * x * p[0] + p[1]) / (1 + x * x * p[2]) + op.sin(x) * p[3] * op.exp(-x * x)
    if name == 'oscillatory':
        return op.exp(-x * x * p[0]) * op.sin(x * p[1] + p[2]) + op.cos(x * x) * p[3]
    raise ValueError(name)

def build(name, method, order):
    spec = JetSpec.directional(order)
    factorial = math.factorial(order)
    if method == 'omnibias':

        def forward(p, x):
            coefficients = [x, jnp.ones_like(x)] + [jnp.zeros_like(x)] * (order - 1)
            y = expression(name, SeriesOps, Series(coefficients, spec), p)
            return y.c[-1] * factorial
    elif method == 'jax_taylor':

        def forward(p, x):
            terms = (jnp.ones_like(x),) + tuple((jnp.zeros_like(x) for _ in range(order - 1)))
            _, out = jet(lambda z: expression(name, jnp, z, p), (x,), (terms,), factorial_scaled=False)
            return out[-1] * factorial
    else:
        raise ValueError(method)

    def objective(p, x):
        y = forward(p, x)
        return jnp.mean((y - 0.1 * x) ** 2)
    return (forward, jax.value_and_grad(objective))

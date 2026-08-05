# Limits & asymptotics (the jet `lim` operator)

A limit such as `lim_{t->0} f(t)/g(t)` of a `0/0` form is usually computer-algebra
territory. But omnibias already carries the **exact Taylor jet** of any deep
network along a ray (`mlp_jet`), so the same L'Hopital rule that a symbolic engine
applies is just a division of two leading Taylor coefficients -- and that division
lives *inside* the autodiff graph. The result is a `lim` operator that is

- **differentiable** (you can backpropagate through the limit),
- **batched / `jit`-able** (it is a plain elementwise op),
- **applicable to learned functions** (the limit of a neural network, which a
  symbolic engine cannot take), and
- optionally **certified** (an interval enclosure of the true limit).

## Differentiable L'Hopital

With the Taylor convention `jet[k] = f^(k)(0)/k!`, a `0/0` ratio whose numerator
and denominator both vanish to order `order - 1` has limit
`num_jet[order] / den_jet[order]`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.jax import lhopital_ratio, limit_of_ratio

# sin(t)/t -> 1 (both vanish at order 1)
sin_jet = jnp.array([0.0, 1.0, 0.0, -1.0 / 6.0])
t_jet   = jnp.array([0.0, 1.0, 0.0, 0.0])
lhopital_ratio(sin_jet, t_jet, order=1)      # -> 1.0

# (1 - cos t)/t^2 -> 1/2 (both vanish at order 2)
omc_jet = jnp.array([0.0, 0.0, 0.5, 0.0])
t2_jet  = jnp.array([0.0, 0.0, 1.0, 0.0])
lhopital_ratio(omc_jet, t2_jet, order=2)     # -> 0.5
```

`lhopital_ratio` is the differentiable entry point. When the vanishing order is
not known ahead of time, `limit_of_ratio` auto-detects the lowest non-vanishing
order (returning `0` when the numerator vanishes faster and `+/- inf` at a pole);
it is a forward-only convenience and is not differentiable at order transitions.

## The limit of a *learned* field (shipped PINN losses)

This is the capability a symbolic engine structurally lacks: the limit of a
*neural network* along a ray. `omnibias-pinn` ships it as trainable losses in
`omnibias.pinn.jax.losses` (and the bit-identical `omnibias.pinn.torch.losses`),
built directly on `mlp_jet` + `lhopital_ratio`:

```python
from omnibias.pinn.jax.losses import (
    asymptotic_ratio,
    asymptotic_bc_loss,
    far_field_decay_loss,
)

D = 2
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
W1, b1 = 0.5 * jax.random.normal(k1, (8, D)), jnp.zeros(8)
W2, b2 = 0.5 * jax.random.normal(k2, (1, 8)), jnp.zeros(1)

layers = [(W1, b1, "tanh"), (W2, b2, None)]   # the mlp_jet layer stack
x0 = jnp.zeros(D)                              # singular point r = 0
v = jnp.eye(D)[0]                              # ray direction

# removable regularity: lim_{r->0} N(x0 + r v) / r**rate
slope = asymptotic_ratio(layers, x0, v, rate=1)   # = N'(0) along v
```

`asymptotic_ratio` is the differentiable primitive: `rate=0` returns the value
`N(x0)`, `rate=1` the directional slope, and a higher `rate` resolves a deeper
removable singularity. Wrap it as a boundary condition with `asymptotic_bc_loss`:

```python
# impose lim_{r->0} N(r)/r = c as a trainable loss term
c = 0.0
loss_reg = asymptotic_bc_loss(layers, x0, v, target=c, rate=1)
```

For a **far-field** condition, evaluate at a far base point and drive the value
together with its directional derivatives toward zero:

```python
x_far = 8.0 * v
loss_decay = far_field_decay_loss(layers, x_far, v, order=2)  # flatten N far out
```

No quadrature, no finite differences; the limit is exact at the base point,
backpropagates into the weights, and the target / exponent is learnable.

### Optional: surface the limit through the field-ops registry

The jet limit is a *model-level* operator -- it needs the layer stack, not an
evaluated `FieldState` -- so it is deliberately not a built-in `FieldState` op.
For users who prefer the attribute-DSL, an opt-in registration exposes it as
`state.<component>.lim_along`:

<!-- docs-test: skip reason="needs the reader's own FieldState factory `field`" -->
```python
from omnibias.pinn.extensions import register_lim_along

register_lim_along()                       # opt-in; leaves the v0.1 ops surface intact
state = field(coords)
state.extra["lim_along"] = {               # per-state, per-component closures
    "u": lambda: asymptotic_ratio(layers, x0, v, rate=1),
}
state.u.lim_along                          # -> the differentiable limit of u
```

## Saturation metadata

Every saturating activation records its `z -> +/- inf` limit on its
`ActivationSpec`, so the `beta -> inf` behaviour reused by the binary surrogates
is queryable:

```python
from omnibias.core.spec import saturation_limit
from omnibias.jax.activations import get_activation

get_activation("tanh").limit_pos_inf          # 1.0
saturation_limit(get_activation("arctan"), -1) # -pi/2
saturation_limit(get_activation("exp"), +1)    # None (diverges)
```

The metadata is shared pure-Python data, so torch and jax report identical values.

## Certified limits

The interval twin returns a rigorous enclosure of the limit; if the leading
denominator coefficient straddles zero the limit is *not* certified finite and a
`ZeroDivisionError` is raised:

```python
from fractions import Fraction
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet import lhopital_ratio_iv

num = [Interval.from_rational(Fraction(0)), Interval.from_rational(Fraction(1))]
den = [Interval.from_rational(Fraction(0)), Interval.from_rational(Fraction(1))]
lhopital_ratio_iv(num, den, order=1).contains(1.0)   # True
```

## Symbolic asymptotes

On the explicit-expression side, a fitted `RationalExpression` exposes its
horizontal asymptote (leading-degree ratio), and the discovery recognizer now
labels the logistic saturation law `dy = y - y^2`:

```python
import numpy as np
from omnibias.symbolic import RationalExpression

expr = RationalExpression(np.array([1.0, 2.0]), np.array([3.0]), 0.0, 1.0, variable="t")
expr.horizontal_asymptote()   # 2/3 for (1 + 2t)/(1 + 3t)
```

## Backends

`omnibias.torch.lhopital_ratio` / `limit_of_ratio` / `removable_value` are
bit-identical twins of the jax functions (float64 cross-backend parity tests).
The PINN losses `omnibias.pinn.torch.losses.asymptotic_ratio` /
`asymptotic_bc_loss` / `far_field_decay_loss` are likewise bit-identical twins of
their jax counterparts. The verified path is pure-Python in `omnibias-core`.

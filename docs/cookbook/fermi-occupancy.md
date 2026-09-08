# Fermi occupancy and thermodynamic potentials

With `z = -beta (e - mu)`, the Fermi-Dirac occupancy is the sigmoid, so
every quantity below reads off the closed-form tower already shipped in
`omnibias.core.polynomials` -- no finite difference, one sigmoid or
softplus evaluation regardless of derivative order. See
[the API reference](../api/occupancy.md) for the full surface and the
[theory spec](https://github.com/derivon-ai/omnibias/blob/main/theory/04-bridges/03-fermi-occupancy-and-thermodynamic-potentials.md).

## Occupancy, entropy, and the grand potential

`FermiModel(beta, mu)` fixes the inverse temperature and chemical
potential; every function below takes it plus an energy.

```python
from omnibias.core.occupancy import (
    FermiModel,
    entropy_per_state,
    grand_potential_density,
    occupancy,
    occupancy_derivatives,
)

model = FermiModel(beta=2.0, mu=0.5)
f = occupancy(model, energy=0.3)
assert abs(f - 0.598687660112452) < 1e-12

# The entropy identity s(z) = softplus(z) - z*sigma(z) matches the
# textbook -f*ln(f) - (1-f)*ln(1-f) to machine precision.
s = entropy_per_state(model, energy=0.3)
textbook_s = -f * __import__("math").log(f) - (1.0 - f) * __import__("math").log(1.0 - f)
assert abs(s - textbook_s) < 1e-12

# d^n f / de^n, n = 0..2, one sigmoid evaluation regardless of n.
f_tower = occupancy_derivatives(model, energy=0.3, order=2)
assert len(f_tower) == 3
assert abs(f_tower[0] - f) < 1e-15

# d(omega)/d(mu) = -f exactly, reproducing dOmega/dmu = -N.
omega = grand_potential_density(model, energy=0.3)
assert omega < 0.0
```

## The `band` role: an exact electron count window

`occupancy_window` is a closed-form antiderivative of `f` over `[e_lo,
e_hi]` for a constant density of states -- no quadrature grid needed for
this special case.

```python
from omnibias.core.occupancy import occupancy_window

g0 = 2.0
window = occupancy_window(model, e_lo=-3.0, e_hi=3.0)
electron_count = g0 * window
assert electron_count > 0.0

import scipy.integrate

ref, _ = scipy.integrate.quad(lambda e: g0 * occupancy(model, e), -3.0, 3.0)
assert abs(electron_count - ref) < 1e-9
```

## Certified electron count and chemical potential

The verified twin composes the same tower with the repo's certified
quadrature and Kantorovich substrate, so the derivative and Lipschitz
bounds the solver needs are read straight off the tower rather than
supplied by the caller.

```python
from omnibias.core.verified.occupancy import (
    certified_chemical_potential,
    constant_density_of_states,
    electron_count_enclosure,
)

dos = constant_density_of_states(g0)
enc = electron_count_enclosure(2.0, 0.5, dos, -3.0, 3.0)
assert enc.lo <= electron_count <= enc.hi

# Newton on N(mu) - n_target, gated by a Kantorovich unique-zero ball.
decision = certified_chemical_potential(
    2.0, dos, -3.0, 3.0, electron_count, mu_bar=0.51, r_max=0.1
)
assert decision.accepted
assert decision.certificate is not None
radius = decision.certificate.radius
assert (0.51 - radius) <= 0.5 <= (0.51 + radius)
```

An unreasonably tight ball is a reported halt, not an exception and not
a silently widened result:

```python
halt = certified_chemical_potential(
    2.0, dos, -3.0, 3.0, electron_count, mu_bar=0.51, r_max=1e-6
)
assert not halt.accepted
assert halt.reason == "empty"
assert halt.certificate is None
```

## Closed-form Sommerfeld coefficients

The classical low-temperature expansion coefficients are exact moments
of the thermal broadening kernel, read off `zeta_even` rather than
truncated numerically.

```python
import math

from omnibias.core.occupancy import sommerfeld_coefficient
from omnibias.core.verified.occupancy import sommerfeld_coefficient_enclosure

a1 = sommerfeld_coefficient(1)
a2 = sommerfeld_coefficient(2)
assert abs(a1 - math.pi**2 / 6.0) < 1e-12
assert abs(a2 - 7.0 * math.pi**4 / 360.0) < 1e-12

enc1 = sommerfeld_coefficient_enclosure(1)
assert enc1.lo <= a1 <= enc1.hi
assert enc1.width < 1e-9
```

## Differentiable twins: gradients in `mu` and `beta`

`omnibias.torch.occupancy` and `omnibias.jax.occupancy` are
bit-identical differentiable twins built from one native `sigmoid` /
`softplus` call plus Horner over the same shared coefficients -- never
re-derived per backend. Because the package-level `occupancy` re-export
shadows the submodule (see the API page's note), import the functions
directly from the leaf module.

```python
import torch

from omnibias.torch.occupancy import occupancy as torch_occupancy

mu = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
beta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
energy = torch.tensor(0.3, dtype=torch.float64)
f = torch_occupancy(energy, beta, mu)
f.backward()
assert mu.grad is not None
assert beta.grad is not None
# df/dmu = +beta * sigma'(z) > 0: raising mu raises the occupancy.
assert mu.grad.item() > 0.0
```

## Honest scope

- Non-interacting fermions in a single band with an externally supplied
  (or absent) density of states -- not a many-body solve, not
  density-functional theory, no thermodynamic limit taken, and no
  phase-transition claim. See `honesty_payload()` on both
  `omnibias.core.occupancy` and `omnibias.core.verified.occupancy`.
- `beta -> inf` is the founding **temperature collapse**
  (`omnibias.core.collapse.schema`'s founding spec), evaluated here only
  as a named external reference value
  (`zero_temperature_occupancy`); this module never requests a new
  collapse-registry slot for it. The founding **bias collapse**
  (`delta -> 0`) never appears in this module at all -- every identity
  above is a finite-order derivative read, not a limit.
- `certified_chemical_potential` requires the caller to pick `r_max`; an
  overly generous value can make the tower-derived Lipschitz bound
  needlessly pessimistic and return `reason="empty"` even when a tighter
  ball would succeed. No automatic radius search is attempted.

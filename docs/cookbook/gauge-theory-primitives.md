# Gauge-theory primitives (`omnibias.geometry.gauge`)

`omnibias.geometry.gauge` makes a **non-abelian gauge field expressible** in omnibias. It
adds the connection-of-a-bundle objects that `omnibias-geometry` (purely
Riemannian, abelian, scalar-valued forms) does not have: Lie algebras with
structure constants, Lie-algebra-valued forms, the field strength
`F = dA + g [A, A]`, the gauge-covariant derivative, the Yang-Mills operator
`D_mu F^{mu nu}`, the Bianchi identity, a flat signature-aware Hodge star, the
action, the topological charge, and gauge transformations.

Field derivatives come from the omnibias **closed-form** activation-derivative
tower through a `FieldState` -- not autodiff, not finite differences. The torch
and jax operator surfaces share one set of backend-agnostic kernels, so they are
bit-identical twins.

## Lie algebras

```python
from omnibias.geometry.gauge._core.lie_algebra import as_lie_algebra

su2 = as_lie_algebra("su(2)")        # Pauli generators / 2
su3 = as_lie_algebra("su(3)")        # Gell-Mann generators / 2
f = su3.structure_constants()        # totally antisymmetric f^{abc}
d = su3.symmetric_constants()        # symmetric d^{abc}
# tr(T^a T^b) = 1/2 delta^{ab}; f^{123} = 1, f^{458} = sqrt(3)/2
```

`su(N)` for any `N`, plus `u(1)` (the abelian limit), are supported.

## Field strength of a connection

The field strength from an explicit connection `A` and its derivatives `dA`
(`A[B, mu, a]`, `dA[B, rho, nu, a]`, batched over sample points):

```python
import torch
from omnibias.geometry.gauge._core.lie_algebra import as_lie_algebra
from omnibias.geometry.gauge.torch.ops.connection import field_strength_from_arrays

algebra = as_lie_algebra("su(2)")
A = torch.randn(8, 4, 3, dtype=torch.float64)        # 8 points, 4 dims, su(2) adjoint
dA = torch.randn(8, 4, 4, 3, dtype=torch.float64)    # d_rho A_nu^a
F = field_strength_from_arrays(A, dA, algebra=algebra, coupling=0.7)
# F[B, mu, nu, a] is antisymmetric in (mu, nu); coupling -> 0 gives dA - (dA)^T
```

For the **closed-form** path, put the connection components in a `FieldState`
and call `field_strength(state, conn)`; the partials `d_rho A_nu^a` (and the
second partials needed for `D_mu F^{mu nu}`) come from the exact sigma-tower.

## The BPST instanton (validation gold standard)

The package is validated against the analytic BPST instanton (`su(2)`, regular
gauge), the exact self-dual solution of the Yang-Mills equations on `R^4`:

| property | continuum value | omnibias.geometry.gauge |
| --- | --- | --- |
| self-duality defect `F - *F` | `0` | machine precision |
| topological charge `Q` | `1` | `-> 1` (finite-box Riemann sum) |
| action `S` | `8 pi^2 / g^2` | `-> 8 pi^2 / g^2` |
| Yang-Mills EOM `D_mu F^{mu nu}` | `0` | machine precision |
| Bianchi `D_mu *F^{mu nu}` | `0` (any connection) | machine precision |

The numpy reimplementation in `omnibias.pinn.certified.yang_mills` is
cross-checked against these torch/jax ops to `atol = 1e-10`, and the torch and
jax backends agree to `rtol = 1e-9` in float64.

!!! note "Scope"
    `omnibias.geometry.gauge` ships the full **continuum** primitive set **and** an SU(2)
    lattice Monte-Carlo engine (`omnibias.geometry.gauge.lattice`, torch + JAX): heat-bath,
    over-relaxation, a DeTurck-gauged Langevin updater, Wilson loops, Creutz
    ratios, string tension, and an APE-smeared `0++` glueball correlator with a
    GEVP plateau scan. See the
    [`omnibias.geometry.gauge` API](../api/geometry-gauge.md) for the full surface.

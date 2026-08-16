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

Each row is asserted in the package's own test suite against an analytic numpy
BPST reference built directly from the closed-form instanton field strength, and
the torch and jax backends agree with each other to `rtol = 1e-9` in float64.

## Gauge-covariant jet (no coordinate trap)

Symbolic regression must not see `partial_rho A_nu^a`. Build a
`GaugeCovariantJet` from the connection arrays; the public fibers are `F` and
`D F` only, and STLSQ is restricted to the Weyl-singlet allowlist. This is a
classical local identity, not a Wilson-loop language and not a mass-gap claim.

```python
import numpy as np
from omnibias.geometry.gauge import (
    LEGAL_SINGLET_ATOMS,
    GaugeCovariantJet,
    bpst_instanton_arrays,
    su,
)

rng = np.random.default_rng(0)
x = rng.uniform(-1.2, 1.2, size=(16, 4))
A, dA, ddA = bpst_instanton_arrays(x)
jet = GaugeCovariantJet.from_arrays(
    A, dA, ddA, algebra=su(2), coupling=1.0, signature=(1, 1, 1, 1)
)
assert set(jet.singlets()) == LEGAL_SINGLET_ATOMS
assert not hasattr(jet, "A")
```

```python
from omnibias.symbolic.gauge_discovery import (
    discover_yang_mills_singlet_law,
    make_yang_mills_bpst_split,
)

train, val, test, conns, _pts = make_yang_mills_bpst_split(seed=0, counts=(24, 12, 12))
out = discover_yang_mills_singlet_law(train, val, test, conns)
assert out["diagnostics"]["gauge_equivariance"]["passed"]
```

The search space is the generated singlet dictionary, not the index count of
`A` or `D^k F`. In 4D SU(3) the coordinate 2-jet has 480 components; the
default mass-dimension-4 dictionary has two searchable singlets.

```python
from omnibias.geometry.gauge import (
    SU3_4D_COORDINATE_2JET,
    GaugeInvariantDictionary,
    raw_connection_jet_dimension,
    su,
)

assert raw_connection_jet_dimension(4, 8, 2) == SU3_4D_COORDINATE_2JET == 480
dictionary = GaugeInvariantDictionary.build(
    mass_dimension=4, max_cov_order=1, algebra=su(3)
)
assert len(dictionary.legal_names) == 2
```

The legal continuum path is analytic arrays or a weak residual against
adjoint test 1-forms, never a random-feature interpolant of lattice links.

```python
from omnibias.geometry.gauge import evaluate_weak_ym_identity

report = evaluate_weak_ym_identity(jet, x, atol=1e-7, n_tests=6)
assert report["passed"]
assert report["yang_mills_claim"] is False
assert report["continuum_claim"] is False
raised = False
try:
    GaugeCovariantJet.from_lattice_links(None)
except ValueError as exc:
    text = str(exc).lower()
    raised = "lattice" in text
assert raised
```

Holonomy is a second language: evaluate Wilson / Polyakov on links, never
by enlarging the local jet.

```python
from omnibias.geometry.gauge import (
    LatticeLinkField,
    evaluate_loop_atoms,
    identity_numpy_links,
    refuse_jet_as_loop_source,
)

table = evaluate_loop_atoms(LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2))))
assert table.values["Polyakov"].item() == 1.0
assert table.values["W(1,1)"].item() == 1.0
loop_raised = False
try:
    refuse_jet_as_loop_source(jet)
except ValueError as exc:
    loop_raised = "holonomy" in str(exc).lower()
assert loop_raised
```

Path B changes the object again: ensemble statistics, not a jet and not a
per-config holonomy table. The planted order-parameter slope is a finite
identity, not a continuum exponent.

```python
from omnibias.geometry.gauge import refuse_jet_as_ensemble_source
from omnibias.symbolic import discover_planted_order_parameter_scaling

scaling = discover_planted_order_parameter_scaling()
assert scaling["passed"]
assert scaling["yang_mills_claim"] is False
assert scaling["continuum_claim"] is False
ens_raised = False
try:
    refuse_jet_as_ensemble_source(jet)
except ValueError as exc:
    ens_raised = "ensemble" in str(exc).lower()
assert ens_raised
```

Named IR / Wilson families are parametric fits, not extra STLSQ atoms.
`ContinuumFitResult.continuum_claim` is fit-earned on a finite `a²` table;
sealed transfer certificates stay `continuum_claim=False`.

```python
from omnibias.symbolic import NamedFamilyDiscoverer, planted_decoupling_table
from omnibias.geometry.gauge import extrapolate_in_a2
import numpy as np

ir = NamedFamilyDiscoverer().fit(planted_decoupling_table(), family="decoupling")
assert ir.passed
assert ir.yang_mills_claim is False
a = np.array([0.20, 0.18, 0.16, 0.14, 0.12, 0.10])
fit = extrapolate_in_a2(a, 1.2 + 0.3 * a * a)
assert fit.continuum_claim is True
assert fit.yang_mills_claim is False
```

Wilson rectangles become a Path B table (area, Creutz, `V(r)`). That transform
is the integral analogue; a jet of `A` is still refused.

```python
from omnibias.geometry.gauge import sommer_r0, wilson_loops_to_ensemble_table
import numpy as np

loops = {}
for r in range(1, 5):
    for t in range(1, 5):
        loops[f"{r}x{t}"] = {"value": float(np.exp(-0.2 * r * t)), "err": 0.0}
table = wilson_loops_to_ensemble_table(loops, lattice_shape=(8, 8, 8, 8))
assert "V_r" in table.values
assert "creutz_chi" in table.values
scale = sommer_r0(np.unique(table.values["r"]), np.full(4, 0.2))
assert scale.yang_mills_claim is False
assert scale.value > 0.0
```

!!! note "Scope"
    `omnibias.geometry.gauge` ships the full **continuum** primitive set **and** an SU(2)
    lattice Monte-Carlo engine (`omnibias.geometry.gauge.lattice`, torch + JAX): heat-bath,
    over-relaxation, a DeTurck-gauged Langevin updater, Wilson loops, Creutz
    ratios, string tension, and an APE-smeared `0++` glueball correlator with a
    GEVP plateau scan. See the
    [`omnibias.geometry.gauge` API](../api/geometry-gauge.md) for the full surface.

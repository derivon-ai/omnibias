# Poincare compactification and finite Dulac cyclicity

`omnibias.dynamics.compactify` constructs the two charts at infinity of a
planar polynomial vector field by exact rational coefficient remapping.
`omnibias.dynamics.dulac` then certifies nonoscillation of a **supplied finite
Dulac expansion**.

These are separate statements. Compactifying a field does not derive its
singular return map, and a zero bound for a supplied expansion does not prove
that the expansion is the physical return displacement.

## Exact compactification

For a degree-\(n\) field \(x'=P(x,y), y'=Q(x,y)\), chart \(U_1\) uses
\(u=y/x,\ v=1/x\). After the standard positive time desingularization,

\[
\dot u=v^n\left[Q(1/v,u/v)-uP(1/v,u/v)\right],\qquad
\dot v=-v^{n+1}P(1/v,u/v).
\]

Every negative power cancels by the degree bound. The implementation performs
the cancellation as an index remap over \(\mathbb Q\); it never substitutes a
small floating value for \(v\). The normal component has an explicit factor
\(v\), proving invariance of the equator.

```python
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.dynamics import (
    PlanarPolynomialField,
    certify_poincare_compactification,
    verify_poincare_compactification,
)

x, y = (P.variable(2, axis) for axis in range(2))
field = PlanarPolynomialField(-2 * x + x**2, y + y**2, degree=2)
compact = certify_poincare_compactification(field)

assert all(chart.equator_invariant for chart in compact.charts[:2])
assert verify_poincare_compactification(compact)
```

Rational equator singularities are found exactly. Nonrational simple roots are
isolated with interval Newton and checked against an exact Sturm distinct-root
count. Multiple nonrational roots remain unsupported and cause an explicit
failure rather than an incomplete list.

## Rational and resonant Dulac models

A finite expansion consists of terms
\[
c_{\alpha,j}x^\alpha(\log x)^j.
\]
With \(x=e^{-\kappa}\), each term becomes
\[
(-1)^j c_{\alpha,j}\kappa^j e^{-\alpha\kappa}.
\]
For rational \(\alpha\), this is an exact
`ConfluentExponentialPolynomial`. Derivation-division terminates after its
finite dimension and gives a global isolated-zero bound.

```python
from fractions import Fraction

from omnibias.dynamics import (
    DulacExpansion,
    certify_dulac_cyclicity,
    displacement_expansion,
)

return_map = DulacExpansion.create(
    (
        (1, 0, 1),
        (2, 0, 2),
        (2, 1, 3),
        (3, 0, Fraction(1, 2)),
    ),
    truncation_order=3,
    remainder_bound=Fraction(1, 1000),
)
model = certify_dulac_cyclicity(displacement_expansion(return_map))

assert model.upper_bound == 2
assert len(model.leading_terms) == 3
assert model.formal_seal["honesty"]["physical_return_membership_proved"] is False
```

The \(x^2\log x\) term is represented by a degree-one polynomial multiplying
\(e^{-2\kappa}\); resonance therefore does not require division by a vanishing
exponent difference.

## Irrational ratio enclosures

An irrational hyperbolicity ratio is not represented as an exact floating
number. It is enclosed between rational endpoints. On an exponent box,
`power_compensator` soundly encloses
\[
\frac{x^a-x^b}{a-b},\qquad C(a,a;x)=x^a\log x,
\]
without division by \(a-b\).

The uniform cover checks the transformed coefficients of successive
\(\kappa\)-derivatives. If every coefficient has one strict sign, the positive
kernels \(\kappa^j e^{-\alpha\kappa}\) make that derivative globally nonzero;
Rolle gives the reported bound. Blocked boxes are bisected, and an exhausted
depth remains `BLOCKED`.

```python
from omnibias.dynamics import (
    certify_graphic_cyclicity,
    named_irrational_hyperbolic_graphic,
)

irrational = certify_graphic_cyclicity(
    named_irrational_hyperbolic_graphic()
)

assert irrational.status == "PROVED_MODEL"
assert irrational.upper_bound == 0
assert irrational.uniform is not None
assert irrational.uniform.seal["honesty"]["uniform_remainder_proved"] is False
```

## Open saddle-node at infinity

The named open target records the existing coalescing-root and central-capture
obstructions:

```python
from omnibias.dynamics import (
    certify_graphic_cyclicity,
    named_open_saddle_node_infinity_graphic,
)

open_case = certify_graphic_cyclicity(
    named_open_saddle_node_infinity_graphic()
)

assert open_case.status == "BLOCKED"
assert open_case.upper_bound is None
assert open_case.seal["honesty"]["drr_case_closed"] is False
assert open_case.seal["honesty"]["full_hilbert16_solved"] is False
```

The compactification coefficients, derivation identities, coefficient signs,
and exact box tiling can pass the Mathlib-free Lean kernel. Lean does not prove
the positivity of the transcendental kernels, the analytic Rolle implication,
the physical return-map expansion, its uniform remainder, or completeness of
the graphic itinerary.

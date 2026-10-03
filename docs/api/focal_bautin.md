# Focal/Bautin engine, derived singular return maps, collar membership, H16 ledger

Four modules that turn the previously *declared* pieces of the Hilbert-16
Dulac-cyclicity pipeline into *derived*, exact-Q ones, plus a machine-checked
obligation ledger that keeps the parent claim honest.

- `omnibias.dynamics.focal`: the Poincare-Lyapunov homological-equation
  focal-value engine.
- `omnibias.dynamics.bautin`: the Bautin (focal-value) ideal, its Groebner
  basis (`omnibias.holonomic._core.groebner`), and center-variety
  membership.
- `omnibias.dynamics.saddle_normal_form`: exact Poincare-Dulac resonant
  normal forms at a hyperbolic saddle, and a first-order **derived** Dulac
  corner expansion (no longer a hand-declared one).
- `omnibias.dynamics.membership`: sound collar-membership agreement between
  a declared and a derived Dulac model, including a genuine unique-cycle
  proof when the collar enclosure allows it.
- `omnibias.dynamics.hilbert16_ledger`: a machine-checked Hilbert-16
  obligation ledger whose parent flags (`full_hilbert16_solved` and
  friends) are **derived**, never asserted.

None of this proves physical return-map membership on an arbitrary graphic,
graphic-wide finite cyclicity, or Hilbert's 16th problem. Read every module
docstring's honesty note before quoting a single number.

## Focal values

```python
from fractions import Fraction

from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.dynamics.bautin import nonzero_focus_example
from omnibias.dynamics.compactify import PlanarPolynomialField
from omnibias.dynamics.focal import certify_focal_values, focal_order, verify_focal_values

x, y = (P.variable(2, axis) for axis in range(2))
p_pert, q_pert = nonzero_focus_example()
p = -y + P(2, dict(p_pert.terms))
q = x + P(2, dict(q_pert.terms))
field = PlanarPolynomialField(p, q, degree=3)

certificate = certify_focal_values(field, order=4)
v1 = dict(certificate.quantities.quantities)[4]

assert focal_order(certificate.quantities) == 4
assert v1.terms[()] == Fraction(3, 4)
assert verify_focal_values(certificate)
assert certificate.formal_seal["honesty"]["bautin_ideal_stabilization_proved"] is False
```

## The Bautin ideal

```python
from omnibias.dynamics.bautin import bautin_basis, bautin_quadratic_family
from omnibias.dynamics.focal import lyapunov_quantities

p, q = bautin_quadratic_family()
quantities = lyapunov_quantities(p, q, order=8)
basis = bautin_basis(quantities)

assert basis.basis_length == 3
assert basis.cyclicity_bound == 2
assert basis.bautin_ideal_stabilization_proved is False
```

`cyclicity_bound = basis_length - 1` is a **conditional** number: it equals
the true cyclicity bound only if the supplied generators already span the
full Bautin ideal, which this module never asserts on your behalf.
`verify_bautin_stabilization` may set
`finite_order_stabilization_verified`, but it keeps the all-orders
`bautin_ideal_stabilization_proved` flag false. The exact counterexample to
inferring an infinite tail from a finite prefix is recorded in
`packages/omnibias-dynamics/HILBERT16-BAUTIN-STABILIZATION-BARRIER.md`.

## Resonant normal forms and a derived corner expansion

```python
from fractions import Fraction as Q

from omnibias.dynamics.graphic import named_rational_hyperbolic_graphic
from omnibias.dynamics.saddle_normal_form import (
    diagonalize_saddle,
    dulac_corner_expansion,
    resonant_normal_form,
)

target = named_rational_hyperbolic_graphic()
diag = diagonalize_saddle(target.field, target.saddle)
normal_form = resonant_normal_form(diag, order=3)
derived = dulac_corner_expansion(normal_form, Q(2), order=2)

# xi^2 and eta^2 are both non-resonant for lambda1=-2, lambda2=1: the
# corner map has no log(x) correction and reduces to the classical x^r.
assert normal_form.resonant_monomials == ()
assert len(derived.expansion.terms) == 1
assert derived.residual_verified is True
```

A resonant monomial *does* produce a `log(x)` term automatically -- it is
never declared or assumed to line up with the normal form's resonances; see
the module docstring's hand-solvable Bernoulli cross-check
(`lambda1=-1`, `lambda2=1`, `p_star = c*xi**2*eta`).

## Collar membership and a genuine unique-cycle proof

```python
from dataclasses import replace
from fractions import Fraction as Q

from omnibias.dynamics.dulac import DulacExpansion
from omnibias.dynamics.membership import certify_collar_membership, verify_collar_membership

# A displacement (a-1)*x + b*x^2 with an exact interior root at x=0.1/20=0.005.
custom_map = DulacExpansion.create(
    ((1, 0, Q(9, 10)), (2, 0, 20)), truncation_order=2, remainder_bound=Q(0)
)
custom_target = replace(target, return_map=custom_map)
custom_derived = replace(derived, expansion=custom_map)

collar = certify_collar_membership(
    custom_target, custom_derived, delta=Q(3, 1000), delta0=Q(7, 1000), grid_points=4
)

assert collar.status == "PROVED_COLLAR"
assert collar.unique_cycle.proved_unique_cycle
lo, hi = collar.unique_cycle.enclosure
assert lo <= 0.005 <= hi
assert verify_collar_membership(collar)
```

`certify_collar_membership` is restricted to `x in [delta, delta0]` with
`delta > 0` strictly: the corner window `x -> 0` itself stays
`corner_window_external: True` on every certificate. A `PROVED_COLLAR`
status earns the narrow `collar_return_membership_proved` flag while
`physical_return_membership_proved` stays `False`.

## The Hilbert-16 obligation ledger

```python
from omnibias.dynamics.hilbert16_ledger import (
    certify_h16_ledger,
    check_ledger,
    default_h16_ledger,
    derived_parent_flags,
    verify_h16_ledger,
)

ledger = default_h16_ledger()
result = check_ledger(ledger)
flags = derived_parent_flags(ledger)
certificate = certify_h16_ledger(ledger)

assert result.status in ("BLOCKED", "CONDITIONAL")
assert not any(flags.values())
assert verify_h16_ledger(certificate)
```

Every gate `G1`-`G6`, every dated DRR case, and the Part-A (22-oval octic)
obligation is one `H16Obligation` with a declared status and
`external_premises`. `derived_parent_flags` re-derives
`full_hilbert16_solved` / `hilbert16_part_a_solved` /
`hilbert16_part_b_quadratic_solved` from the entries themselves --
`payload_earns_parent_claim` guards against a stored certificate forging a
parent claim, and `DISCHARGED_LOCAL_SCOPE` (a genuine narrow win from this
plan) never counts toward any of them. As shipped, every parent flag on
this ledger is `False`.

## Reference

::: omnibias.dynamics.focal
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.dynamics.bautin
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.dynamics.saddle_normal_form
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.dynamics.membership
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.dynamics.hilbert16_ledger
    options:
      show_root_heading: false
      heading_level: 3

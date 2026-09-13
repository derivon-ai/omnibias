# Static sources from the improved cubic vacuum

`su2_wilson_static_family` applies the center-cut theorem to an
actual cubic vacuum from the [Wilson linear source](gauge-wilson-linear-source.md)
or the [gauge-polar source](gauge-wilson-polar-source.md).
It requires a canonical replay of that source, including its nonlinear
correction and positive original-edge curvature.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_wilson_linear_vacuum,
    su2_wilson_static_family,
    replay_su2_wilson_static_family_certificate,
)

source = su2_wilson_linear_vacuum(16, correction_radius=Q(1, 8))
result = su2_wilson_static_family(source["certificate"])
assert result["status"] == "PASS"
assert result["witness"]["linear_static_energy_coefficients"] == ["2", "6"]
assert result["finite_graph_family_confinement"]
assert replay_su2_wilson_static_family_certificate(result["certificate"])
assert not result["continuum_claim"]
```

For every specified finite open three-dimensional cubic box and
every distinct pair of source vertices,
\[
\frac{\kappa\rho}{2}d(s,t)\le E_{s,t}-E_0
\le\frac{3\kappa}{8}d(s,t),\qquad
\rho=\frac12-\frac{16g}{3}-\frac{4r}{3}.
\]
The sector has one static fundamental and one antifundamental source,
no dynamical matter, and zero bare source rest energies.
All energies are in original dimensionless \(aH\) units.
\(E_0\) is the actual neutral vacuum energy.

The proof uses the actual ground-state Dirichlet transform.
A source-separating center flip makes every charged component
conditionally odd on its cut. Conditional Poincare supplies
\(\rho\) per cut, and the distance-ball cuts are edge disjoint.
Fundamental transport along a shortest path supplies the upper bound.
At \(\kappa=16,r=1/8\), these coefficients are exactly 2 and 6.
The polar source passes at \(\kappa=15,r=1/10\), giving coefficients
\(367/180\) and \(45/8\) with the same original-edge cut proof.
The written derivation is in the companion
`ensemble-laws/docs/constructive/wilson-source-confinement.md`.

An inverse-only cubic source returns `INCONCLUSIVE` with no static
coefficients. A strip source is refused because its vertical-chart
curvature cannot supply this original-edge cut premise. A reference
diffusion certificate is also refused. Replay recomputes the entire
upstream certificate and the derived bounds.

The fixed-coupling finite-family conclusion is distinct from an
infinite-volume static-potential limit or an asymptotic string
tension. Those claims, continuum identification and all formal
analytic tiers remain false.

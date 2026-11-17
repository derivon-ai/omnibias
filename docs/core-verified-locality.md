# Exact finite locality bounds and quadratic ball budgets

`omnibias.core.verified.locality` supplies three pure-Python primitives
with exact integer/Fraction inputs. Results contain rational strings,
explicit scope, and a canonically replayable certificate. They do not
verify a differential-operator locality theorem or a nonlinear PDE.
No floating-point input is silently promoted to an exact proof premise.

## Rectangular weighted Schur bounds

For a rectangular matrix A, row weights r and column weights c define
`diag(r) @ A @ diag(c)^(-1)`. The maximum absolute row and column sums
R,C bound its Euclidean operator norm squared by R*C. Returning the
square avoids rounding a square root. Omitted weights are one.

```python
from fractions import Fraction as Q
from omnibias.core.verified.locality import (
    weighted_schur_bound,
    replay_locality_certificate,
)

result = weighted_schur_bound([[1, -2, 3], [0, 4, -1]], [2, 3], [1, 2, 4])
assert Q(result["bounds"]["row_upper"]) == Q(27, 4)
assert Q(result["bounds"]["column_upper"]) == 8
assert Q(result["bounds"]["l2_norm_squared_upper"]) == 54
assert replay_locality_certificate(result["certificate"])
```

This is an l2 Schur bound. It must not be substituted for a row-only
sup-norm estimate or a column-only weighted l1 estimate without checking
the norm required by the consuming argument.

## Symmetric generators and finite resolvents

For a finite symmetric rational matrix K, supply an integer metric d and
a rational spatial weight w>1. The code checks symmetry, positivity off
the metric diagonal, zero diagonal, and every triangle inequality. It
computes

\[
 \lambda=\min_i\left(K_{ii}-\sum_{j\ne i}|K_{ij}|\right),\qquad
 \eta=\max_i\sum_{j\ne i}|K_{ij}|(w^{d_{ij}}-1).
\]

A nonpositive lambda is refused; this does not prove the matrix is
indefinite. For a metric-Lipschitz height h, put D=diag(w^h). The entries
of `D K D^(-1)-K` have a symmetric magnitude majorant with row/column
cap eta. Therefore the weighted finite semigroup has decay margin
lambda-eta. Integrating it gives the finite resolvent bound
1/(lambda-eta), when positive.

For the finite Sylvester generator `X -> K X + X K`, the analogous
Hilbert--Schmidt bounds have denominators `2*(lambda-eta)` for the
two-sided weight `D X D`, or `2*lambda-eta` for a one-sided weight.
Identifying a PDE Hessian equation with this finite matrix model requires
a separate proof; the function never makes that identification.

```python
from fractions import Fraction as Q
from omnibias.core.verified.locality import weighted_symmetric_generator_bounds

result = weighted_symmetric_generator_bounds(
    [[4, -1, 0], [-1, 4, -1], [0, -1, 4]],
    [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    Q(3, 2),
)
assert result["passed"]
assert result["bounds"]["gap_floor"] == "2"
assert result["bounds"]["weighted_commutator_upper"] == "1"
assert result["bounds"]["weighted_resolvent_upper"] == "1"
assert result["bounds"]["two_sided_hessian_resolvent_upper"] == "1/2"
assert not result["mixed_hessian_pde_locality_verified"]

inconclusive = weighted_symmetric_generator_bounds(
    [[2, -1], [-1, 2]], [[0, 1], [1, 0]], 2,
)
assert not inconclusive["passed"]
assert inconclusive["bounds"]["two_sided_hessian_resolvent_upper"] is None
```

All values concern the supplied finite matrix. Uniformity across a
family of matrices must be established separately. Metric validation
costs cubic time in matrix size. A nonpositive weighted margin produces
`INCONCLUSIVE`, retaining any independently positive one-sided margin.

## A conditional quadratic contraction gate

For `T(x)=L(r-B(x,x))`, let defect bound `||r||`, c bound `||L||`,
and b bound the bilinear operator in compatible spaces. On the radius-R
ball about zero, the mapped radius is at most `c*(defect+b*R**2)`
and the Lipschitz constant is at most `2*c*b*R`. Both the self-map
inequality and a strictly subunit Lipschitz constant must pass.

```python
from fractions import Fraction as Q
from omnibias.core.verified.locality import quadratic_ball_budget

result = quadratic_ball_budget(Q(1, 16), 2, Q(1, 4), Q(1, 4))
assert result["passed"]
assert result["bounds"]["mapped_radius_upper"] == "5/32"
assert result["bounds"]["contraction_upper"] == "1/4"
assert not result["nonlinear_existence_verified"]
assert set(result["external_premises"].values()) == {"UNVERIFIED"}
```

The caller must prove completeness, the source and operator bounds in
the same declared spaces, and identification with the target equation.
The gate records these premises as unverified even when its scalar
inequalities pass. It complements the existing `verified.kantorovich`
and `verified.radii_spectral` APIs; it does not replace their validated
finite-residual and tail constructions.

## Replay and scope

`replay_locality_certificate` recomputes every input, bound, claim and
scope field. It rejects a changed result even after its digest is
resealed. A digest establishes integrity, not analytical truth.
PDE locality, nonlinear PDE closure, infinite-volume, continuum and
theorem-prover flags remain false for every result.

# Combining spherical products and directional inverse bounds

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import su2_adjacent_cone_vacuum, replay_su2_adjacent_cone_vacuum_certificate
r=su2_adjacent_cone_vacuum(6, correction_radius=Q(6,25))
assert r["actual_vacuum_verified"]
assert not r["curvature_gap_verified"]
assert r["physical_gap_lower"] is None
assert replay_su2_adjacent_cone_vacuum_certificate(r["certificate"])
assert not r["continuum_claim"]
```

The fixed seven-edge adjacent-square SU(2) model now has an actual
vacuum certificate at \(\kappa=6\), correction radius \(r=6/25\).
A separate actual-measure comparison gives a neutral physical gap
greater than \(4/3\) in original \(aH\) units. All electric spins
are included. This is a fixed-graph result.

## Three independently proved ingredients

The [spherical-product lemma](gauge-theta-bilinear.md) replaces the
generic quadratic constant \(4/3\) by \(8/9\) on this theta graph:
\[
N(T(U,V))\le\frac89N(U)N(V),\qquad
T=C_0^{-1}\Pi_H\Gamma.
\]
Its proof uses nonnegative squared overlap coefficients of normalized
trivalent spherical functions, not positivity of the unknown correction.
The mean output Casimir equals the sum of the input Casimirs.
This cancels the mean signed derivative coefficient before absolute
values are estimated. The theorem includes all representation labels
and arbitrary signed or complex Fourier coefficients.

The [directional dual construction](gauge-adjacent-cone-inverse.md) gives
\[
\|L^{-1}\|_{N\to N}<57/25,\qquad
L=I-2T(S_*,\cdot),\quad S_*=(g/3)(\chi_p+\chi_q),\quad g=4/\kappa^2.
\]
It uses the original seven-edge metric and norm, with the whole
omitted-spin tail.

The independently replayed
[residual enclosure](gauge-adjacent-vacuum.md) gives
\[
N(L^{-1}R_*)<1231/10000,\qquad R_*=T(S_*,S_*).
\]
That enclosure is valid even though the older source's nonlinear
radius test fails. Its exact solve, reference parent, and full
tail are checked again; no failed actual-vacuum flag is copied.

## A complete nonlinear gate at six

Put \(J=57/25\), \(\epsilon=1231/10000\), \(B=8/9\), \(r=6/25\).
The larger simple rational bounds already give
\[
r-\epsilon-JBr^2=\frac{41}{250000}>0,\qquad
2JBr=\frac{608}{625}<1.                              \tag{1}
\]
Therefore
\[
U=L^{-1}R_*+L^{-1}T(U,U)
\]
has a unique solution in the stated radius ball. The complete Fourier
norm reconstructs a \(C^2\) function. Its positive exponential
\(\psi_0=e^{S_*+U}\) satisfies the original Hamiltonian eigenproblem,
and the exact groundstate form identity identifies it as the true
unique vacuum. This is the same analytic reconstruction as the
preceding source theorem, with sharper proved constants.

The original curvature estimate at this radius is
\[
\rho_{\rm lower}=\frac12-\frac{8g}{3}-\frac{4r}{3}
                =-\frac{157}{1350}.
\]
This negative **lower bound** neither proves negative actual
curvature nor implies a vanishing gap. It only fails the previous
positive-curvature sufficient test.

## A spectral conclusion without that curvature test

The [comparison theorem](gauge-adjacent-gap-comparison.md) proves
the sharp coefficient bound
\[
\sum_j|U_j|\le\frac{13}{72}N(U),
\qquad \operatorname{osc}(2U)\le\frac{13r}{18}.
\]
The coefficient bound is sharp; optimality of the resulting
oscillation constant is not claimed.
A tree gauge exposes a product reference while preserving domination
by the original electric form. Comparison with the actual measure gives
\[
\operatorname{gap}_{\rm neutral}(aH)
\ge\frac{3\kappa}{8}\exp\!\left(-\frac{8g}{3}-\frac{13r}{18}\right).
\]
At six, the exponent is \(317/675\). The exact four-step exponential
minorant gives
\[
\boxed{\operatorname{gap}_{\rm neutral}(aH)\ge
\frac94\left(\frac{2383}{2700}\right)^4
=\frac{32247508758721}{23619600000000}>\frac43.}         \tag{2}
\]
The comparison consumer separately records the full-scalar bound,
which is smaller and applies before imposing gauge invariance.

## Replays and remaining obstruction

The new source requires canonical replays of both the earlier
residual enclosure and the directional inverse, including identical
reference parents. It accepts a radius only when that radius contracts.
At six, \(r=1/4\) fails contraction even though some other radius is
feasible. At five, these current nonlinear bounds remain inconclusive.
The preceding v1 source and inverse certificates are unchanged.

The separate parametric Lean development in the ensemble-laws consumer proves reusable finite
algebraic lemmas and the rational radius theorem. It does not
formalize the spherical harmonic, infinite-series, groundstate or
spectral analysis. No analytic source inherits a Lean tier from it.

The improvement concerns a fixed graph. A compatible local source
and inverse across growing interacting volumes and weak-coupling
refinements remain unproved, as do the continuum OS construction,
nontriviality and Clay mass gap. Mathematical novelty beyond this
program has not been established.

# Exact Wilson residual vacuum source

`su2_wilson_residual_vacuum` constructs the actual full-spin SU(2)
vacuum using sharper bounds on a reference and its exact residual.
It applies to every finite open square strip or rectangular
three-dimensional cubic box in its stated family, at fixed coupling.
The constants are independent of box size. There is no representation
cutoff and no discarded interaction tail.

The implementation checks exact rational premises for a written
analytic implication. Certificate replay verifies those premises and
their scope; it does not formalize the analytic implication in Lean.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_wilson_residual_vacuum,
    replay_su2_wilson_residual_vacuum_certificate,
)

result = su2_wilson_residual_vacuum(
    17, family="cubic", correction_radius=Q(1, 12),
)
assert result["status"] == "PASS"
assert result["actual_vacuum_verified"]
assert result["beyond_legacy_source_criterion_verified"]
assert Q(result["physical_gap_lower"]) > Q(29, 10)
assert replay_su2_wilson_residual_vacuum_certificate(result["certificate"])
assert not result["continuum_claim"]
assert not result["theorem_prover_verified"]
```

## Domain and normalization

The families are open \(1\times n\) square strips, \(n\ge1\), and
finite open rectangular three-dimensional cubic boxes with at least
one elementary cell in each direction. All electric and elementary
plaquette weights are one. Gauss law is imposed at every original
vertex. Every original edge points along its positive coordinate axis.
The wrapper does not accept or verify membership of an arbitrary graph.

\[
aH=\frac\kappa2\sum_e C_e+\frac2\kappa\sum_p(2-\chi_{1/2}(U_p)),
\qquad C_e=-\Delta_e,\quad g=\frac4{\kappa^2}.
\]

The Casimir convention is \(j(j+1)\). Gap floors use the same original
dimensionless \(aH\) units. Coupling is the Hamiltonian parameter
\(\kappa\); it is not substituted for a Euclidean simulation's \(\beta\).

## Exact local residual and uniform bounds

Use the complete invariant spin-weighted Fourier space, with norm
\[
\mathcal N_b(f)=\max_e\sum_{\mathbf j\ne0}
 j_e E_{\mathbf j}b^{\operatorname{diam}X_{\mathbf j}}
 \|F_{\mathbf j}\|_1,\qquad b\ge1.
\]
Distances are in the original ambient line graph.
In this same fixed-orientation space,
\(T(f,h)=C_0^{-1}\Pi_0\Gamma(f,h)\) has bilinear bound \(B=4/3\).
Set \(S_*=(g/3)\sum_p\chi_{1/2}(U_p)\).
Its entire centered residual is
\[
T(S_*,S_*)=g^2\left[
 -\frac1{72}\sum_p\chi_1(U_p)
 +\sum_{\{p,q\}\text{ adjacent}}\left(\frac{b_0}{27}-\frac{b_1}{39}\right)
\right],
\]
where \(b_0=\chi_{\rm outer}/2\) and
\(b_1=(\chi_p\chi_q-\chi_{\rm outer}/2)/3\).
Their exact electric energies are \(9/2\) and \(13/2\).
The inverse removes the Haar constant first.

In the canonical orientation, the coefficient nuclear norms of the
fundamental and adjoint square are \(8\) and \(27\). Both normalized
adjacent-pair channels have nuclear norm at most \(16\). These follow
from the vertex-intertwiner singular values, not numerical SVD.
Independent exact tensor regressions cover coplanar and perpendicular
adjacent pairs. Inverting one original group coordinate is not assumed
to preserve this Fourier norm.

Each incident square contributes \(4gb^2\) to the seed bound and
\(3g^2b^2\) to the residual bound. Each adjacent-pair union touching
the anchor contributes at most \(8g^2b^3/3\).
The geometric counts give:

| Family | Squares per edge | Adjacent pairs touching edge | Seed \(v\) | Residual \(D\) |
|---|---:|---:|---|---|
| Strip | 2 | 3 | \(8gb^2\) | \((6b^2+8b^3)g^2\) |
| Cubic box | 4 | 42 | \(16gb^2\) | \((12b^2+112b^3)g^2\) |

For the cubic count, each square has at most 12 edge-adjacent squares.
Four incident squares produce at most \(4\cdot12-\binom42=42\)
distinct pairs. Boundary counts are smaller.

## Actual-vacuum implication and radius gate

The correction solves
\[
u=T(S_*,S_*)+2T(S_*,u)+T(u,u).
\]
Writing \(L=2Bv\), a radius \(r>0\) is accepted if
\[
D+Lr+Br^2\le r,\qquad L+2Br<1.
\]
For \(D>0\), such a radius exists exactly when
\[
L<1,\qquad (1-L)^2-4BD>0.
\]
This is feasibility of this sufficient criterion.
The automatic rational witness is \(r=2D/(1-L)\).
An explicitly supplied radius must independently pass both gates.
No caller-supplied residual bound is accepted.

Banach contraction controls the entire correction, including all spins
and supports. The Fourier reconstruction is \(C^2\) on each finite
graph. Elliptic bootstrap gives a smooth real solution; \(e^{S_*+u}>0\)
solves the original eigenfunction equation. Its ground-state transform
identifies it as the true vacuum. Finite-graph reconstruction constants
may depend on graph size; the contraction and local derivative bounds
used for the gap do not.

At \(b=1\), the new source criterion is equivalent to
\[
3\kappa^2>256+16\sqrt{42}\quad\text{(strip)},\qquad
3\kappa^2>512+32\sqrt{93}\quad\text{(cubic)}.
\]
Its first passing positive integers are 11 and 17. The old conservative
criterion has first passing integers 19 and 27; its APIs and certificates
are unchanged.

## Gap routes and arbitrary finite coordinate marginals

The unweighted curvature route gives
\[
\rho_{\rm strip}=\frac12-\frac{4g}3-\frac{4r}3,\qquad
\rho_{\rm cubic}=\frac12-\frac{16g}3-\frac{4r}3.
\]
A positive \((\kappa/2)\rho\) is a gap floor.
The [conditional Poincare Schur theorem](gauge-marginal-majorant.md)
uses actual single-coordinate Haar comparison with
\[
\Omega_{\rm strip}=(16g+8r)/3,\quad
\Omega_{\rm cubic}=(32g+8r)/3,\quad
\gamma_N=\frac34(1-\Omega/N)^N.
\]
The rational exponential bound requires \(0\le\Omega<N\).
The mixed-Hessian weighted rows are
\[
c_{\rm strip}=\frac{gb}3+\frac{2r}{3b},\qquad
c_{\rm cubic}=\frac{4g(2b+b^2)}6+\frac{2r}3.
\]
If \(\eta=\gamma_N-2c>0\), the preserved comparison margin supplies
gap \((\kappa/2)\eta\) through arbitrary finite nested coordinate
marginals. Spatial exponential covariance control is earned only for
\(b>1\).

Strip coordinates use the horizontal forest gauge, retain both
endpoints, and carry the exact induced prefix-gradient kinetic weights.
Cubic coordinates are original edges. Pullbacks to retained edges
have exactly the sum of their unit electric gradient squares;
require a retained elementary plaquette so the gauge-invariant
physical subspace is nontrivial. The compression uses the actual
vacuum marginal. It is not a claim that compressed time evolution
equals compression of the fine semigroup, or that the marginal is a
new Wilson action with one coupling.

At cubic \(\kappa=17,b=1,r=1/12,N=4\), the curvature floor is
\(1639/612\). The conditional floor is
\(334205080594177/114868566764928>29/10\).
The wrapper reports the larger positive route.

## Scope and refusal

Only exact `int` and `Fraction` inputs are accepted; booleans and floats
are refused. A failed gate returns `INCONCLUSIVE`, with no physical gap
floor. A successful source can remain visible even if both gap routes
fail. At \(\kappa=10\) on strips and \(16\) on cubic boxes, the new
source discriminant is negative; this does not imply a physical
gap vanishes.

The analytic derivation and original-edge tensor proof are in the
companion `ensemble-laws/docs/constructive/strip-source-window.md` and
`strip-reference-residual.md`. Replay all six examples with
`uv run python -m ensemble_laws.wilson_residual` in that checkout.

The linear constraint \(L<1\) alone prevents this source from reaching
\(\kappa\to0\), even if its residual could be made zero without changing
the linear bound. No infinite-volume limit, uniform continuum scaling,
OS reconstruction, static confinement, or Clay mass gap is claimed.
The corresponding runtime flags and formal-verification tiers remain false.

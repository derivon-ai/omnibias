# Two nested actual-vacuum reductions

`su2_shared_strip_two_step` reduces the physical four-plaquette SU(2)
strip to its actual interacting pair of coarse loops, then to its
outer loop. It retains the generated joint interaction and the
electric weights inherited from the first reduction.

All three Hamiltonians use the original dimensionless \(aH\) units:
\(\alpha=\kappa/2\), \(g=4/\kappa^2\). The original graph has thirteen
unit electric edges, four unit plaquettes, and neutral Gauss law at
every vertex. Neither spin truncation nor a new lattice spacing is
introduced by the reductions.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_shared_strip_two_step,
    replay_su2_shared_strip_two_step_certificate,
)

result = su2_shared_strip_two_step(19, correction_radius=Q(1, 9))
assert result["status"] == "PASS"
assert Q(result["intermediate_physical_gap_lower"]) > Q(126701, 10000)
assert Q(result["physical_gap_lower"]) > Q(63386, 10000)
assert result["actual_marginal_derivative_bound_verified"]
assert replay_su2_shared_strip_two_step_certificate(result["certificate"])
assert not result["all_scale_refinement_claim"]
assert not result["continuum_claim"]
assert not result["theorem_prover_verified"]
```

## Actual marginal derivatives

The source is the
[shared-strip actual-vacuum construction](gauge-shared-strip-refinement.md).
Its fixed point constructs \(\psi_0\propto e^{S_*+u}\), with original-edge
correction norm \(\mathcal N(u)\le r\). Only this construction, its first
conditional complement, and its relative-form estimate are used.
The source's old compression gap, full gap, and global logarithm ball
are not premises of the new spectral calculation.

Fixing the eight horizontal links leaves five vertical variables
\((X,A,Z,B,Y)\), with residual left/right endpoint gauge actions. Then
\[
S_1(X,Z,Y)=\tfrac12\log\int e^{2S(X,A,Z,B,Y)}\,dA\,dB.
\]
The integral uses fixed product Haar measure. For retained-coordinate
derivatives,
\[
D S_1=\mathbb E[DS],\qquad
D\mathbb E[f]=\mathbb E[Df]+2\operatorname{Cov}(f,DS).
\]
The covariance is essential. All mixed derivatives used below act
on distinct group factors, whose product connection has no mixed term.

Set \(W=XY^{-1}\). The differential adjoints \(dp_X^*,dp_Y^*\)
depend only on \(X,Y\), so their derivatives in \(Z,A,B\) vanish.
The original correction Hessian has symmetric block row sums at most
\(c=2r/3\). A mixed seed plaquette block has norm at most \(k=g/6\).
If \(\gamma_1\) is the first conditional pair's Poincare floor, these
bounds give
\[
\left|\nabla_Z\left\langle
5dp_X\nabla_XS_1+5dp_Y\nabla_YS_1,v
\right\rangle\right|
\le L_2=5c+\frac{20(k+c)^2}{\gamma_1}
\]
for unit \(v\in T_WG\). Right endpoint transformations leave \(W\)
and this scalar contraction unchanged; fixing \(Y=I\) therefore
introduces no additional frame variance.

## Physical kinetic form and the whole second complement

The intermediate operator has exact kinetic weights \((5,5,1)\).
The final compression has kinetic coefficient \(10\alpha\).
After right endpoint reduction, the full energy dominates \(7/2\)
times the gradient energy of the physical fiber \(R=ZY^{-1}\).

Put
\[
\Omega_{f,1}=(32g+16r)/3,\quad
\Omega_{f,2}=(16g+8r)/3,\quad
\Omega_{m,2}=8(g+r)/3.
\]
For a positive integer \(m\), use the rational lower bound
\(E_-(x)=(1-x/m)^m\le e^{-x}\) whenever \(0\le x<m\). Set
\[
\gamma_i=\tfrac34 E_-(\Omega_{f,i}),\qquad
d_{A,2}=\tfrac{15}{2}\alpha E_-(\Omega_{m,2}),\qquad
d_{H,2}=\tfrac72\alpha\gamma_2,\qquad
\beta_2^2=\frac{2\alpha L_2^2}{5\gamma_2}.
\]
The conditional bound applies to every physical complementary mode.
The cross term satisfies the energy-relative estimate
\(\|Q_2K_1J_2f\|^2\le\beta_2^2\mathfrak q_2(f)\).

## Two Schur gates and exact compatibility

A compression floor \(a\), complementary floor \(h\), and relative
coefficient \(b=\beta^2<h\) give the lower gap
\[
\lambda(a,h,b)=\frac{a+h-\sqrt{(a-h)^2+4ab}}2.
\]
The implementation rounds the square root upward to a dyadic rational.
First it derives
\(z_2\le\lambda(d_{A,2},d_{H,2},\beta_2^2)\).
Then it uses that derived intermediate floor in
\[
z_1\le\lambda\left(
z_2,\frac{11}{5}\alpha\gamma_1,
\frac{4\alpha}{5\gamma_1}(7g/6+4r/3)^2
\right).
\]
No user-supplied or previously asserted coarse gap is accepted.

At \(\kappa=19,r=1/9,m=8\), the resulting floors exceed
\(12.6701\) and \(6.3386\). Simpler exact determinant checks pass
intermediate target \(12\), followed by fine target \(6\).
The purpose of this result is to verify the composed reduction.
The earlier one-step estimate remains a stronger bound on the same
fine graph.

Both coarse vacua are square roots of actual marginal densities.
Their Haar-space embeddings multiply as
\[
J_{01}J_{12}F
=\frac{\psi_0}{\psi_1}\frac{\psi_1}{\psi_2}F
=\frac{\psi_0}{\psi_2}F.
\]
Fubini identifies the iterated marginal with direct outer-loop
integration. This compatibility concerns marginals and compressed
forms; the two Schur tests supply the additional control over the
complete fine spectrum.

## Refusal and scope

The gates for actual marginal derivatives, second conditional estimates,
intermediate gap, and composed gap are reported separately. A failed
source or sufficient inequality returns `INCONCLUSIVE` with the
unsupported gap set to `None`. Canonical replay recomputes every
nested source, derivative bound, spectral step and scope field.

The local derivative argument can succeed when the source's separate
global logarithm ball is inconclusive:

```python
local = su2_shared_strip_two_step(
    19, correction_radius=Q(1, 9), exponent_steps=1,
)
assert local["status"] == "PASS"
source_witness = local["witness"]["source_certificate"]["payload"]["witness"]
assert source_witness["actual_joint_log_ball_verified"] is False

refused = su2_shared_strip_two_step(18, correction_radius=Q(1, 9))
assert refused["status"] == "INCONCLUSIVE"
assert refused["physical_gap_lower"] is None
assert replay_su2_shared_strip_two_step_certificate(refused["certificate"])
```

The conclusion uses a written analytic implication plus exact finite
arithmetic. No Lean tier, volume-uniform induction, continuum theory,
or Yang–Mills mass-gap claim follows from this fixed nested graph.

# Directional dual weights for the complete theta inverse

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import su2_adjacent_cone_inverse, replay_su2_adjacent_cone_inverse_certificate
r=su2_adjacent_cone_inverse(6)
assert Q(r["original_N_inverse_upper"]) < Q(57,25)
assert replay_su2_adjacent_cone_inverse_certificate(r["certificate"])
assert not r["actual_vacuum_verified"]
```

This note sharpens the inverse estimate on the fixed seven-edge adjacent
SU(2) graph. It changes neither the reference \(S_*=(g/3)(\chi_p+\chi_q)\)
nor the original electric norm. It replaces a uniform scalar norm
conversion by exact, separately verified directional weights.
The result is a new estimate in this program; mathematical novelty in
the literature has not been established.

## A finite-column shortcut that fails

Let \(w_{ij}=(j_i/2)E_j(a_j+1)^2(b_j+1)^2\), using doubled-spin
labels \(j=(a_j,b_j,s_j)\). Write
\[
N(x)=\max_i\sum_j w_{ij}|x_j|,\qquad M(x)=\sum_{ij}w_{ij}|x_j|.
\]
Admissibility gives \(2N\le M\le3N\).
It does **not** make \(N\) a single weighted \(\ell^1\) norm.

For example, the normalized basis functions
\(b_{101}/6,b_{011}/6,b_{110}/36\) have anchored vectors
\((1,0,1),(0,1,1),(1,1,0)\). Define a rank-one operator sending
each to \(b_{101}/6\), and all other basis functions to zero.
Every individual column has norm ratio one. Their sum has input
norm two and output norm three. The induced operator norm is
exactly \(3/2\), using the triangle comparison for the upper bound.
Thus the hypothesis that the largest column ratio always equals
the induced original norm is false. This is an operator counterexample,
not a claim about a physical Hamiltonian's gap.

## The directional dual lemma

Suppose nonnegative weights \(\lambda_1,\lambda_2,\lambda_3\) satisfy
\[
\sum_i\lambda_i w_{ij}\ge c_j\quad\hbox{for every input column }j.
\]
Then absolute summation gives
\[
\sum_j c_j|x_j|
\le\sum_i\lambda_i N_i(x)
\le\Big(\sum_i\lambda_i\Big)N(x).                    \tag{1}
\]
For a matrix \(A\), use \(c_j=\sum_o w_{io}|A_{oj}|\) separately
for each output anchor \(i\). The largest resulting weight sum
bounds \(\|A\|_{N\to N}\). Using
\(c_j=\sum_{io}w_{io}|A_{oj}|\) instead bounds \(\|A\|_{N\to M}\).
These are bounds: global dual optimality is not required or asserted.

The entire omitted representation cone also has a finite description.
Every nonnegative triangle triple decomposes as
\[
(a,b,s)=\tfrac{a+b-s}{2}(1,1,0)
       +\tfrac{a+s-b}{2}(1,0,1)
       +\tfrac{b+s-a}{2}(0,1,1).                   \tag{2}
\]
Consequently a linear functional is nonnegative on every such triple
exactly when its three pair sums are nonnegative. The three rays
are themselves admissible doubled-spin labels; scaling puts them
beyond any finite cutoff.

The separate parametric Lean development in the ensemble-laws consumer proves finite weighted
triangle closure, cone duality, dual summation bounds and the
column counterexample parametrically. Passing from finite sums
to an infinite absolutely summable Fourier series remains the
analytic monotone-limit argument in this note.

## Keep the finite inverse before estimating its correction

Use the exact retained inverse from
[adjacent-linearized-inverse.md](gauge-adjacent-resolvent.md):
\[
L=I-K,\qquad R=\operatorname{diag}((PLP)^{-1},I),\qquad
Z=I-RL,\qquad \|Z\|_{M\to M}\le z<1.
\]
Its exact operator identity is
\[
L^{-1}=R+(I-Z)^{-1}ZR.                              \tag{3}
\]
If (1) certifies \(B_N\ge\|R\|_{N\to N}\) and
\(D\ge\|ZR\|_{N\to M}\), then
\[
\boxed{\|L^{-1}\|_{N\to N}\le B_N+\frac{D}{2(1-z)}.} \tag{4}
\]
The factor \(1/2\) is applied only to the output of the correction.
All operators and projections are those already proved bounded by
the complete parent tail. No new spectral assumption enters (3).

Here is how the dual constraints cover every spin.

- On retained inputs, \(R\) is computed exactly; on every omitted
  input it is identity. The latter constraints are the three pair
  sums \(\lambda_a+\lambda_b\ge1\) when the output anchor belongs
  to that pair, and zero otherwise.
- On retained inputs, \(ZR=QKR_P\); all outgoing coefficients are
  summed before taking absolute values.
- On the first omitted shell, \(R=I\), so \(ZR=Z\). The retained
  part is \(R_PPK\), and the omitted part is \(QK\).
- Beyond that shell, \(ZR=K\). The parent full-column bound
  \(M(Ke_j)\le\tau M(e_j)\) is covered by
  \(\lambda_a+\lambda_b\ge2\tau\), and its cyclic versions.

Those last constraints follow from (2). They are not a sampling
of high spins. The implementation checks all finite constraints
over \(\mathbb Q\) and all three analytic tail constraints.

## Deterministic search and exact acceptance

The multiplier search uses two-variable nonnegative families:
\((x,0,z)\), \((0,x,z)\), and \((x,x,z)\) for the three
preconditioner anchors, and \((x,x,z)\) for the mixed remainder.
It enumerates the finite line intersections over \(\mathbb Q\),
with a feasible uniform fallback, and checks every inequality.
These choices are sufficient families, not a theorem of global
optimization. No floating optimizer, LP success flag, or SVD
is accepted as proof.

At \(\kappa=6\), cutoff three, the exact bound is below \(57/25=2.28\),
compared with the parent conversion bound greater than three.
The parent defect and all electric spins are unchanged.
The older inverse certificate remains canonical and valid;
the new certificate has a separate type.

This is an inverse of an explicitly specified reference operator
on one graph. Nonlinear vacuum construction, a physical gap,
volume-uniform estimates, and continuum reconstruction remain
separate obligations.

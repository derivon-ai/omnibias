# Spatial covariance and marginal majorants

`omnibias.geometry.gauge.transfer.marginal_majorant_closure` verifies
an exact rational matrix elimination and a weighted row bound.
The companion [actual strip hierarchy](gauge-strip-marginal-hierarchy.md)
identifies a physical vacuum to which this mathematics applies.
A caller-supplied matrix alone does not identify such a vacuum.

## Covariance on a compact product

Let each compact connected Riemannian factor have Ricci curvature at
least \(\rho>0\), and let \(d\mu\propto e^{2S}dH\), with smooth \(S\).
Assume the covariant Hessian blocks, including diagonal blocks, obey
\(\sup_x\|\operatorname{Hess}_{ij}S(x)\|_{\rm op}\le M_{ij}\), where
\(M\) is symmetric and entrywise nonnegative.
For integrated indices \(R\), set
\[
B_R=\rho I_R-2M_{RR}>0,\qquad K_R=B_R^{-1}\ge0.
\]
At every fixed retained configuration,
\[
|\operatorname{Cov}_{\mu_R}(f,g)|\le a^T K_R b,\qquad
a_i=\|\nabla_i f\|_{L^2(\mu_R)},\quad
b_i=\|\nabla_i g\|_{L^2(\mu_R)}.
\]

Indeed solve \(Lu=g-\mu_Rg\), where
\(L=-\Delta_R-2\nabla_RS\cdot\nabla_R\).
At each finite conditioning compact ellipticity supplies this inverse
without assuming a uniform gap. The weighted Weitzenböck identity gives
\[
(\nabla_\mu^*\nabla+\operatorname{Ric}
-2\operatorname{Hess}S)\,du=dg.
\]
The product connection preserves each cotangent-factor block.
Testing the \(i\)-th block against \((du)_i\), and discarding its
nonnegative connection energy, yields \(B_Rx\le b\) for
\(x_i=\|(du)_i\|_2\). Zero \(x_i\) requires no division.
Inverse positivity gives \(x\le K_Rb\); integration by parts proves
the covariance estimate. The actual finite Fourier vacua are smooth
by elliptic bootstrap from their \(C^2\) fixed points.

This comparison-matrix method has established precedents:
[Menz, *A Brascamp–Lieb type covariance estimate*](https://arxiv.org/abs/1402.5160),
and manifold intertwining is treated by
[Huguet, *Intertwining relations for diffusions in manifolds and applications to functional inequalities*](https://arxiv.org/abs/1912.05376).
The argument above specifies the compact-product normalization used here.

## The same weighted row bound survives

For \(C=R^c\) and
\(S_C=\frac12\log\int_R e^{2S}dH_R\), differentiation under the fixed
Haar integral gives
\[
\operatorname{Hess}_{ij}S_C
=\mathbb E_R\operatorname{Hess}_{ij}S
 +2\operatorname{Cov}_R(D_iS,D_jS).
\]
For diagonal retained blocks the same connection term is subtracted
on both sides. Retained tangent vectors are fixed across the fiber.
Consequently the new majorant is
\[
T_R(M)=M_{CC}+2M_{CR}(\rho I_R-2M_{RR})^{-1}M_{RC}.
\]

If \(M\mathbf1\le m\mathbf1\), \(m<\rho/2\), then
\[
K_RM_{RC}\mathbf1
\le K_R(m\mathbf1-M_{RR}\mathbf1)
=\tfrac12\mathbf1-(\rho/2-m)K_R\mathbf1
\le(m/\rho)\mathbf1.
\]
Here \(K_R\mathbf1\ge\rho^{-1}\mathbf1\).
Thus the retained and eliminated entries share the same row budget:
\[
T_R(M)\mathbf1
\le m\mathbf1-(1-2m/\rho)M_{CR}\mathbf1\le m\mathbf1.
\]
Bounding the two parts independently would lose this closure.

For any metric \(d\) and \(b\ge1\), the same result holds for
\(\sup_i\sum_j b^{d(i,j)}M_{ij}\le m\).
The triangle inequality bounds every weighted Neumann path by the
corresponding path for \(\widehat M_{ij}=b^{d(i,j)}M_{ij}\).
Apply the preceding row estimate to \(\widehat M\).
The covariance kernel also has weighted row at most
\((\rho-2m)^{-1}\). At \(b>1\) these give exponential spatial tails.
Distances remain the original metric restricted to retained indices.

Finally,
\[
\rho I_C-2T_R(M)=\operatorname{Schur}_R(\rho I-2M).
\]
Schur complement associativity makes sequential elimination agree
with direct elimination. This is established matrix structure, also
used in [Dörfler–Bullo, *Kron Reduction of Graphs with Applications to Electrical Networks*](https://arxiv.org/abs/1102.2950).
Together with the row estimate it allows any finite marginal depth
without inflating \(m\).

## Exact API

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    marginal_majorant_closure,
    replay_marginal_majorant_closure_certificate,
)

M = [[Q(1, 20), Q(1, 40), 0],
     [Q(1, 40), Q(1, 20), Q(1, 40)],
     [0, Q(1, 40), Q(1, 20)]]
result = marginal_majorant_closure(
    M, [0, 2], ricci_lower=Q(1, 2),
    distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]], decay_base=2,
)
assert result["output_matrix"] == [["17/320", "1/320"],
                                   ["1/320", "17/320"]]
assert replay_marginal_majorant_closure_certificate(result["certificate"])
```

The API accepts exact integers/Fractions and an optional integer metric.
An insufficient supplied `row_cap` or \(m\ge\rho/2\) is
`INCONCLUSIVE`; malformed or inexact inputs raise.
It recomputes inverse identities, positivity, weighted path domination
and row savings. Replay rejects even resealed alterations.
This certificate verifies finite matrix arithmetic. The analytic
covariance implication and identification of a physical Hessian are
separate obligations; formal and continuum flags remain false.

## Extension using actual conditional Poincare floors

Absolute diagonal Hessian bounds are not necessary for the comparison
method. Suppose instead every actual one-site conditional measure has
a uniform Poincare floor \(\gamma_i>0\), and the mixed covariant
Hessians of \(S\) obey \(c_{ij}=c_{ji}\ge0\), \(c_{ii}=0\).
Set
\[
H_{ii}=\gamma_i,\qquad H_{ij}=-2c_{ij}\quad(i\ne j).
\]
Assume \(H\) is a positive definite M-matrix.
For the Poisson solution used above, \(d_i u\) is an **exact** one-form
on each one-site fiber. Its local weighted one-form energy equals
\(\|L_i u\|_2^2\ge\gamma_i\|d_i u\|_2^2\).
Other-coordinate connection energies are nonnegative.
The same component argument therefore gives covariance kernel \(H^{-1}\).
This use of the scalar conditional gap is on exact forms; it does not
give the same bound on arbitrary one-forms.

After eliminating \(R\), the actual marginal has sufficient bounds
\[
\gamma'_i=\gamma_i-4c_{iR}H_{RR}^{-1}c_{Ri},\qquad
c'_{ij}=c_{ij}+2c_{iR}H_{RR}^{-1}c_{Rj}\quad(i\ne j).
\]
For the first identity condition on all surviving sites other than
\(i\), and apply the covariance estimate on \(\{i\}\cup R\) to
a function of \(i\) only. Its variance is bounded by the corresponding
diagonal inverse entry times its gradient energy; that entry's
reciprocal is the displayed \(\gamma'_i\).
For the second identity differentiate the true log marginal and use
the conditional covariance kernel on \(R\).
The updated comparison matrix is exactly
\[
H'=\operatorname{Schur}_R H.
\]
Thus this broader set of actual analytic premises also propagates
through arbitrary finite marginal depth.

For the exact sufficient weighted gate
\[
\delta=\min_i\left(\gamma_i-
2\sum_{j\ne i}b^{d(i,j)}c_{ij}\right)>0,
\]
write \(\rho=\max_i\gamma_i\),
\(M_{ii}=(\rho-\gamma_i)/2\), and \(M_{ij}=c_{ij}\).
Then \(H=\rho I-2M\) and
\(\|M\|_{\text{weighted row}}\le(\rho-\delta)/2\).
The previous row theorem preserves the same \(\delta\) and spatial
kernel bound \(\delta^{-1}\). The auxiliary \(\rho\) here is an
algebraic reference, not the manifold's Ricci curvature.

```python
from omnibias.geometry.gauge.transfer import (
    conditional_poincare_schur,
    replay_conditional_poincare_schur_certificate,
)

result = conditional_poincare_schur(
    [2, 3, 2],
    [[0, Q(1, 8), 0], [Q(1, 8), 0, Q(1, 8)], [0, Q(1, 8), 0]],
    [0, 2],
    distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    decay_base=2,
)
assert result["output_gaps"] == ["95/48", "95/48"]
assert result["output_mixed_hessian"] == [["0", "1/96"], ["1/96", "0"]]
assert replay_conditional_poincare_schur_certificate(result["certificate"])
```

This wrapper validates only exact finite comparison-matrix arithmetic.
Supplying positive numbers does not establish actual conditional gaps
or mixed derivatives. Its corresponding actual-measure flags remain
false. Applying it beyond the current strong-coupling vacuum source
still requires uniform conditional spectral floors and mixed bounds
for that actual physical measure.

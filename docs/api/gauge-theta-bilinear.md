# Complete-spin theta bilinear estimate

For the original seven-edge SU(2) theta graph and anchored norm `N`,
`T=C0^-1*Pi_H*Gamma` satisfies `N(T(f,h)) <= (8/9)*N(f)*N(h)`.
This is a sharper analytic constant for the same norm and physical electric
weights, rather than a replacement norm or a finite-spin fit.

The adjacent cone-vacuum consumer recomputes the complete nonlinear gate:

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import (
    su2_adjacent_cone_vacuum,
)

result = su2_adjacent_cone_vacuum(6, correction_radius=Fraction(6, 25))
assert result["witness"]["arithmetic"]["quadratic_constant"] == "8/9"
assert result["actual_vacuum_verified"]
assert result["curvature_gap_verified"] is False
```

The theorem alone does not supply a vacuum or gap certificate. The consumer
also needs its independently replayed inverse, residual and contracting
radius. The full bilinear proof and its normalization follow.


12 September 2026. On the neutral seven-edge SU(2) two-plaquette graph,
the original unweighted anchored Fourier norm satisfies
\[
\boxed{N\!\left(C_0^{-1}\Pi_H\Gamma(f,h)\right)
 \le\frac89N(f)N(h).}
\tag{1}
\]
This improves the previously used generic constant \(4/3\). It follows
from the positive product decomposition of trivalent invariant matrix
coefficients and an exact mean-Casimir identity. No truncated spin search,
unknown vacuum, spectral gap, or positivity assumption about a general
Fourier coefficient matrix is used.

The theorem concerns this fixed theta graph and its complete neutral
representation space. Its use in a nonlinear actual-vacuum proof still
requires a separately certified inverse, residual and radius. It supplies
neither a continuum theory nor a mass-scaling statement.

## 1. The space, orientation and normalization

Use the two adjacent square plaquettes with original edge orientations
rightward on the horizontal rows and upward on the verticals. The three
paths between the two trivalent vertices have electric weights
\[
(w_a,w_b,w_s)=(3,3,1).
\]
Spins \(j=(j_a,j_b,j_s)\) are nonnegative half-integers obeying the
triangle conditions and integer total spin. Let \(\eta_j\) be the unit
vector in the one-dimensional diagonal SU(2)-invariant subspace of
\(V_{j_a}\otimes V_{j_b}\otimes V_{j_s}\), using the invariant dualities
required by the fixed orientations. Write
\[
b_j(X,Y,Z)=\langle\eta_j,
 [D_{j_a}(X)\otimes D_{j_b}(Y)\otimes D_{j_s}(Z)]\eta_j\rangle.
\tag{2}
\]
Each path product of independent Haar edges is Haar, and the original
Gauss reduction leaves exactly these invariant matrix coefficients.
One can choose the duality convention so that setting the shared holonomy
to the identity gives
\[
b_j(U,V,I)=\frac{\operatorname{Tr}[P_{j_s}
 (D_{j_a}(U)\otimes D_{j_b}(V))]}{2j_s+1}.
\tag{3}
\]
Thus \(b_j(I,I,I)=1\), and the modes agree with the normalized projector
basis used in [the adjacent inverse](gauge-adjacent-resolvent.md).

Set \(d_j=2j+1\). The original seven-edge electric energy and coefficient
nuclear norm are
\[
E_j=3j_a(j_a+1)+3j_b(j_b+1)+j_s(j_s+1),
\qquad n_j=d_{j_a}^2d_{j_b}^2.
\tag{4}
\]
The second formula follows from the canonical endpoint/bivalent tensor
flattenings; it is an original-edge norm, not the smaller norm after gauge
fixing. The derivation in the adjacent inverse includes arbitrary
admissible spins and trivial legs.

For \(f=\sum_{j\ne0}f_jb_j\), put
\[
N_e(f)=\sum_{j\ne0}j_eE_jn_j|f_j|,
\qquad N(f)=\max_{e\in\{a,b,s\}}N_e(f).
\tag{5}
\]
Every geometric edge along one path gives the same anchor, so (5) is the
original seven-edge unweighted norm. All nonconstant states have
\(E_j\ge3\): nonzero spin at the shared path requires another active
outer path, whereas with \(j_s=0\) both outer paths are active and equal.
The smallest first case has four spin-\(1/2\) electric edges and energy
three; the latter has energy at least \(9/2\).

The identity \(j(j+1)\ge3j/2\) for nonnegative half-integer \(j\) also gives
\[
\sum_p w_pj_p\le\frac23E_j.
\tag{6}
\]
Both (4) and (6) use the original path weights \(3,3,1\).

## 2. Positive products are proved by complete unitary fusion

Multiply two matrix coefficients (2). The product is the matrix
coefficient of the unit vector \(\eta_j\otimes\eta_k\). Regroup the
factors by path. At each path, use the complete multiplicity-free SU(2)
Clebsch–Gordan decomposition
\[
V_{j_p}\otimes V_{k_p}
 =\bigoplus_{\ell_p=|j_p-k_p|}^{j_p+k_p}V_{\ell_p}.
\tag{7}
\]
The full fusion map is unitary. The product vector remains invariant
under the diagonal group. In each output triple \(\ell\), its projection
is therefore either zero or a multiple \(c_{\ell;j,k}\eta_\ell\) of the
unique normalized trivalent invariant vector. Distinct triples are
orthogonal invariant subrepresentations, and group matrices do not mix
them. Consequently
\[
\boxed{b_jb_k=\sum_\ell\mu_{\ell;j,k}b_\ell,
\qquad \mu_{\ell;j,k}=|c_{\ell;j,k}|^2\ge0,
\qquad\sum_\ell\mu_{\ell;j,k}=1.}
\tag{8}
\]
Every allowed fusion output is included. The constant output, when present,
is included in the unit mass. This is a proof of positivity specific to
the two-vertex spherical matrix coefficients. It is not an assumption
that arbitrary coefficient-matrix products have positive Fourier entries.
Input coefficients \(f_j,h_k\) may have arbitrary signs or be complex.

In the standard real CG convention the same coefficient is
\[
\mu_{\ell;j,k}=d_{\ell_a}d_{\ell_b}d_{\ell_s}
\begin{Bmatrix}
j_a&j_b&j_s\\k_a&k_b&k_s\\\ell_a&\ell_b&\ell_s
\end{Bmatrix}^{\!2}.
\tag{9}
\]
The proof of (8) does not require evaluating the 9j formula. The exact
finite oracle uses (9) as an independent arithmetic check of the fusion
normalization and the known fundamental squared-6j multiplication.

## 3. The mean output Casimir is exactly the input sum

The reduced one-leg density matrix of \(\eta_j\) commutes with the
irreducible SU(2) representation: the vector is diagonally invariant.
Schur's lemma and trace one make that reduced density \(I/d_{j_p}\).
In particular the mean of each traceless angular-momentum generator
\(J_{p,r}\) vanishes.

Before fusion the coupled Casimir on path \(p\) is
\[
(J_p+K_p)^2=J_p^2+K_p^2+2\sum_rJ_{p,r}K_{p,r}.
\]
The input vector is the product \(\eta_j\otimes\eta_k\). The mean cross
term factors into two zero one-leg means. Evaluating the same operator
after the unitary fusion (7) therefore gives
\[
\sum_\ell\mu_{\ell;j,k}\,\ell_p(\ell_p+1)
 =j_p(j_p+1)+k_p(k_p+1).
\tag{10}
\]
Multiplying by the physical weights and summing paths,
\[
\sum_\ell\mu_{\ell;j,k}E_\ell=E_j+E_k.
\tag{11}
\]
Equivalently, the electric product rule writes
\[
\Gamma(b_j,b_k)=\sum_\ell\gamma_{\ell;j,k}\mu_{\ell;j,k}b_\ell,
\qquad \gamma_{\ell;j,k}=\frac{E_j+E_k-E_\ell}{2},
\qquad\sum_\ell\mu_{\ell;j,k}\gamma_{\ell;j,k}=0.
\tag{12}
\]
The sign and zero-mean statement include the constant output.

## 4. Mean cancellation improves the absolute generator bound

The complete fusion range obeys \(\ell_p\le j_p+k_p\). Hence
\[
E_\ell\le E_j+E_k+2\sum_pw_pj_pk_p,
\qquad\gamma_{\ell;j,k}\ge-q(j,k),
\quad q(j,k)=\sum_pw_pj_pk_p.
\tag{13}
\]
A real random variable of mean zero has equal mean positive and negative
parts. Applying this elementary identity to the finite probability pack
(8), and using (13), gives
\[
\boxed{\sum_\ell\mu_{\ell;j,k}|\gamma_{\ell;j,k}|
 =2\sum_\ell\mu_{\ell;j,k}(-\gamma_{\ell;j,k})_+
 \le2q(j,k).}
\tag{14}
\]
This replaces the generic absolute three-generator estimate by an exact
mean-cancellation argument. It does not discard a positive contribution or
assume the signs of the eventual nonlinear correction.

Also, dimension fusion implies
\(d_{\ell_p}\le d_{j_p}d_{k_p}\), including trivial spins. Formula (4)
therefore gives
\[
n_\ell\le n_jn_k,\qquad \ell_e\le j_e+k_e.
\tag{15}
\]
These inequalities are used after the unweighted mean identity (14);
there is no claim that the nuclear-reweighted product still has mean-zero
energy defect.

## 5. The full anchored estimate

First take finite coefficient sums and write
\(\alpha_j=n_j|f_j|\), \(\beta_k=n_k|h_k|\). Constants in either input
have zero derivative and may be omitted. In the output of
\(T=C_0^{-1}\Pi_H\Gamma\), the factor \(E_\ell\) in (5) cancels its
inverse. The constant output has zero anchor and can be harmlessly included
in the ensuing upper sums. Equations (12)–(15) yield
\[
N_e(T(f,h))\le
2\sum_{j,k}\alpha_j\beta_k(j_e+k_e)
       \sum_pw_pj_pk_p.
\tag{16}
\]
For the part containing \(j_e\), use
\[
\sum_k k_p\beta_k\le N_p(h)/3\le N(h)/3
\]
and then (6). Its contribution is at most
\[
\frac{2N(h)}3\sum_j\alpha_jj_e\sum_pw_pj_p
\le\frac49N(h)N_e(f).
\tag{17}
\]
The other part is bounded by \(4N(f)N_e(h)/9\). Thus the stronger
anchor-by-anchor estimate is
\[
N_e(T(f,h))\le\frac49\bigl[N(h)N_e(f)+N(f)N_e(h)\bigr].
\tag{18}
\]
Taking the maximum proves (1).

The space defined by (5) is a finite maximum of weighted \(\ell^1\)
norms; finite coefficient sums are dense. Inequality (1) extends \(T\)
continuously to the complete space. For every nonconstant theta
state, \(j_a+j_b+j_s\ge1\). Thus
\[
\sum_{j\ne0}E_jn_j|f_j|\le\sum_eN_e(f)\le3N(f).
\]
The generator and covariant second-derivative bounds on each representation
are controlled by a fixed multiple of \(E_j\), giving absolute uniform
convergence through two derivatives. Passing the finite-sum identity
\(CT(f,h)=\Pi_H\Gamma(f,h)\) to these limits identifies the bounded
extension with the actual electric differential product followed by Haar
centering and the electric inverse. No omitted-spin
remainder is left unbounded, and no cutoff enters \(8/9\).

## 6. Independent checks and use in the vacuum correction

For example \(b_{1/2,0,1/2}=\chi_p/2\). The character square identity and
original electric energies give
\[
T(b_{1/2,0,1/2},b_{1/2,0,1/2})
 =-\frac3{32}b_{1,0,1},
\]
whose ratio in the norm (5) is \(3/16\). For the two different fundamental
plaquettes the corresponding ratio is \(1/12\). These low-spin checks do
not determine the optimal bilinear norm and are not the proof of (1).
The scratch `artifacts/theta_bilinear_audit.py` checks the complete 9j
product packs, unit mass, the three separate mean Casimirs, exact energy
cancellation and the nuclear inequalities on a finite grid and a seeded
selection of additional labels. Its JSON is a regression artifact, not a
standalone all-spin or actual-vacuum certificate.

In a separate preconditioned correction gate one may now replace the
previous \(B=4/3\) by \(B=8/9\), provided all operators use exactly the
norm (5) and this physical theta graph. A bound on the complete reference
inverse and on its action on the full reference residual is still needed.
The nonlinear self-map and contraction must then be checked again. The
improved constant alone earns no actual-vacuum, physical-gap,
volume-uniform, continuum, all-scale or Clay flag.

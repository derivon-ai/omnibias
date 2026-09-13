# SU(2) plaquette reference inverse

The finite rational API bounds an explicit reference linearization on the full
character space. The all-spin Fourier gate can pass after an unpreconditioned
Neumann estimate has failed.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.plaquette_resolvent import (
    su2_plaquette_linearized_inverse,
    replay_su2_plaquette_linearized_inverse_certificate,
)

result = su2_plaquette_linearized_inverse(1, cutoff=4)
assert result["status"] == "PASS"
assert result["fourier_inverse_upper"] == "2408805/7181"
assert result["weighted_hilbert_inverse_upper"] == "1"
assert result["actual_vacuum_verified"] is False
assert replay_su2_plaquette_linearized_inverse_certificate(result["certificate"])

stronger = su2_plaquette_linearized_inverse(Fraction(1, 2), cutoff=32)
assert stronger["full_spin_fourier_reference_inverse_verified"]
```

`kappa` must be an exact positive integer or `Fraction`; `cutoff` must be a
positive integer. Floats and booleans are refused. Insufficient cutoffs return
`INCONCLUSIVE` for the Fourier bound. The separately proved weighted Hilbert
inverse bound remains one. Canonical replay accepts faithful passing and
inconclusive certificates, and rejects modified arithmetic, normalization,
inputs and scope, even when resealed.

The two inverse estimates use different norms. No target-vacuum, physical-gap,
volume-uniform, continuum or formal-verification flag is earned. The complete
analytic implication used by the certificate follows.


12 September 2026. The explicit SU(2) Wilson reference has an invertible
linearization at every finite positive coupling. In a particular weighted
Hilbert norm the inverse bound is exactly one. A separate finite rational
block calculation controls its inverse in the original Fourier nuclear norm,
including every omitted character. Neither statement assumes a target
spectral gap or constructs the target vacuum.

This is one compact neutral plaquette. It supplies a concrete inverse module
for an a posteriori correction argument; a many-plaquette local inverse in
the required volume-uniform norm remains a separate problem.

## 1. Original electric normalization and the exact matrix

Take four unit electric edges, canonical square orientation, Gauss law at
all four vertices, and normalized Haar measure. The physical Hilbert space
is the SU(2) class-function space with orthonormal characters
\(e_n=\chi_{n/2}\), \(n\ge0\). The convention inherited from the Wilson
source is
\[
aH_\kappa=\frac\kappa2 C+\frac2\kappa(2-e_1),\qquad
Ce_n=\lambda_ne_n,\quad \lambda_n=n(n+2),
\quad g=\frac4{\kappa^2}.
\tag{1}
\]
Here \(C=4(-\Delta_{SU(2)})\). In particular the fundamental electric
energy is three, not \(3/4\).

Let \(t=g/3\), \(S_*=t e_1\). On zero-Haar-mean class functions define
\[
\mathcal L=I-2C_0^{-1}\Pi_0\Gamma(S_*,\cdot),\qquad
\Gamma(f,h)=\sum_{\text{four edges},a}(X_af)(X_ah).
\tag{2}
\]
The constant mode is removed before the inverse. The SU(2) character product
and the electric product rule give, with \(e_{-1}=0\),
\[
e_1e_n=e_{n+1}+e_{n-1},\qquad
\Gamma(e_1,e_n)=-n e_{n+1}+(n+2)e_{n-1}.
\tag{3}
\]
Consequently
\[
\mathcal L e_n=e_n+a_ne_{n+1}-b_ne_{n-1},\quad
 a_n=\frac{2tn}{(n+1)(n+3)},\quad
 b_n=\frac{2t(n+2)}{(n-1)(n+1)}\ (n\ge2),\quad b_1=0.
\tag{4}
\]
The vanished lower term at \(n=1\) was a constant, not a missing spin.
All coefficients in (4) are exact rational functions of \(t\).

An independent radial check uses \(x=e_1\in[-2,2]\), for which
\[
Cf=-(4-x^2)f''+3xf',\qquad
\Gamma(f,h)=(4-x^2)f'h'.
\tag{5}
\]
Inserting the Chebyshev characters \(e_n=U_n(x/2)\) reproduces (3)–(4).

## 2. The explicit drift and qualitative invertibility

Define the reference probability measure and its scalar diffusion by
\[
d\nu_t=Z_t^{-1}e^{2t e_1}dH,\qquad
A_t=C-2\Gamma(S_*,\cdot).
\tag{6}
\]
Integration by parts gives
\[
\langle f,A_tf\rangle_{\nu_t}=\int\Gamma(f,f)\,d\nu_t.
\tag{7}
\]
On class functions its radial density is proportional to
\(e^{2tx}\sqrt{4-x^2}\). Formula (5) gives the same weighted divergence
operator directly, with the natural smooth SU(2) domain at the endpoints.

For every finite \(t\), the density is positive and smooth. Its ratio of
maximum to minimum is \(e^{8t}\), so comparison with the Haar class gap
three proves a reference Poincare floor \(3e^{-8t}>0\). This elementary
floor is not an assumed physical gap and is not needed in the sharper
inverse calculation below.

The equation \(\mathcal Lh=f\) is equivalent to
\[
A_th=Cf-\nu_t(Cf),\qquad \int h\,dH=0.
\tag{8}
\]
The scalar subtraction in (8) is necessary: the drift is symmetric in
\(\nu_t\), whereas \(\Pi_0\) uses Haar mean. The solution's normalization
is zero Haar mean, not necessarily zero reference mean. In particular,
\(\mathcal Lh=0\) implies \(A_th\) is constant; integrating against
\(\nu_t\) makes that constant zero and (7) makes \(h\) constant. Its Haar
normalization then gives \(h=0\).

The density in (6) is explicitly constructed. It is not the unknown vacuum
of the target \(C-ge_1\): its quadratic reference residual is nonzero for
\(g>0\). An inverse for (2) is therefore not a target mass-gap certificate.

## 3. A weighted Hilbert inverse bound of one at every coupling

Equip the nonconstant character coefficients with
\[
\|h\|_{\mathscr H_D}^2=\sum_{n\ge1}
 \lambda_n^2(n+1)|h_n|^2,\qquad D_n=\lambda_n\sqrt{n+1}.
\tag{9}
\]
The exact balance identity
\[
\lambda_{n+1}^2(n+2)a_n
 =\lambda_n^2(n+1)b_{n+1}
\]
shows that
\[
D\mathcal L D^{-1}=I+J,\qquad
J_{n+1,n}=\frac{2t}{\sqrt{(n+1)(n+2)}},\qquad
J_{n,n+1}=-J_{n+1,n}.
\tag{10}
\]
The weighted shift defining \(J\) is bounded and compact on \(\ell^2\).
Indeed \(\|J\|_{HS}^2=4t^2\), by the telescoping sum
\(2\sum_{n\ge1}4t^2/[(n+1)(n+2)]=4t^2\). The displayed transpose relation
therefore proves \(J^*=-J\) on the whole Hilbert space, not just on finite
vectors. Thus
\[
\|(I+J)z\|_2^2=\|z\|_2^2+\|Jz\|_2^2\ge\|z\|_2^2.
\]
Its range is closed; its orthogonal complement is the kernel of \(I-J\),
which is zero by the same estimate. Hence it is onto, and
\[
\boxed{\ \|\mathcal L^{-1}\|_{\mathscr H_D\to\mathscr H_D}\le1
\quad\text{for every finite positive }\kappa.\ }
\tag{11}
\]
This is an inverse norm for a preconditioned drift in (9), not a lower
bound of one on the physical Hamiltonian's spectral gap.

## 4. The original Fourier norm is different

In the canonical four-edge orientation, the coefficient matrix for \(e_n\)
has nuclear norm \((n+1)^3\), as proved by the vertex tensors in
[the original-edge Wilson tensor proof](gauge-wilson-residual-source.md). Therefore the
original anchored spin norm on this single plaquette is exactly
\[
\|h\|_{\mathcal N}=\sum_{n\ge1}\omega_n|h_n|,\qquad
\omega_n=\frac n2\lambda_n(n+1)^3
 =\frac{n^2(n+2)(n+1)^3}{2}.
\tag{12}
\]
Every active support is the same four-cycle. Adding any fixed diameter
weight \(b^2\) multiplies both sides of an inverse estimate by the same
factor and does not change its operator norm.

Let \(K=I-\mathcal L\). Its weighted column magnitudes are
\[
A_n=a_n\omega_{n+1}/\omega_n
 =\frac{2t(n+2)^2}{n(n+1)^2},\qquad
B_n=b_n\omega_{n-1}/\omega_n
 =\frac{2tn(n-1)}{(n+1)^3},\quad B_1=0.
\tag{13}
\]
They obey the exact positive difference
\[
(A_n+B_n)-(A_{n+1}+B_{n+1})
=\frac{4t(n^5+6n^4+22n^3+47n^2+47n+16)}
 {n(n+1)^3(n+2)^3}>0.
\tag{14}
\]
Consequently \(\|K\|_{\mathcal N}=A_1=9t/2=3g/2\), and its column
sums tend to zero. The simple unpreconditioned Neumann criterion therefore
fails once \(3g/2\ge1\), although (11) continues to hold.

The operator \(K\) is compact in (12): deleting both rows and columns
beyond \(N\) leaves only one boundary column on each side plus tails whose
column sums are \(O(t/N)\). Also \(\mathcal N\) embeds continuously into
\(\mathscr H_D\), since
\(D_n/\omega_n=2/[n(n+1)^{5/2}]\) is bounded. Equation (11) therefore makes
\(I-K\) injective on \(\mathcal N\). The compact Fredholm alternative
implies a bounded inverse on \(\mathcal N\) at every finite \(t\).
This qualitative assertion supplies neither a numerical inverse bound
nor a coupling-independent bound in (12).

## 5. Exact finite blocks and a complete omitted-spin bound

Retain the nonconstant characters \(1,\ldots,N\), and let
\(R_N=(P_N\mathcal LP_N)^{-1}\). All these finite blocks are invertible:
their LU pivots satisfy
\[
d_1=1,\qquad d_n=1+a_{n-1}b_n/d_{n-1}>0.
\tag{15}
\]
The implementation solves every inverse column over \(\mathbb Q\) and
checks its full tridiagonal residual exactly. Define its weighted column
norms and the block preconditioner by
\[
c_j=\frac1{\omega_j}\sum_{i=1}^N\omega_i|(R_N)_{ij}|,
\qquad \mathcal B_N=R_N\oplus I,\qquad
\|\mathcal B_N\|=\max(1,c_1,\ldots,c_N).
\tag{16}
\]
There are exactly three types of columns in the defect
\(Z_N=I-\mathcal B_N\mathcal L\):

- For retained columns below \(N\), the defect is zero. Column \(N\)
  leaks into the first omitted row with weighted magnitude \(A_N\).
- Column \(N+1\) couples into the retained block through its last inverse
  column and upward into the tail. Its exact weighted column sum is
  \(A_{N+1}+B_{N+1}c_N\).
- For every \(n\ge N+2\), the column sum is \(A_n+B_n\), whose supremum
  is attained at \(n=N+2\) by (14).

Thus the exact all-spin defect norm is
\[
z_N=\max\{A_N,\ A_{N+1}+B_{N+1}c_N,\ A_{N+2}+B_{N+2}\}.
\tag{17}
\]
If \(z_N<1\), the full infinite operator has the quantitative bound
\[
\boxed{\quad
\|\mathcal L^{-1}\|_{\mathcal N\to\mathcal N}
 \le\frac{\max(1,c_1,\ldots,c_N)}{1-z_N}.
\quad}
\tag{18}
\]
This follows from a Neumann inverse for \(\mathcal B_N\mathcal L=I-Z_N\).
It does not replace the omitted tail by a Dirichlet boundary condition.

For every fixed rational positive coupling some finite \(N\) satisfies
this gate. Indeed, \(P_N\mathcal LP_N\oplus I\) converges to \(\mathcal L\)
in the norm (12), and the latter is invertible there. Their inverses are
then uniformly bounded for sufficiently large \(N\), while their
preconditioned defects tend to zero. This is existence of a finite
witness, not a bounded-resource promise for a prescribed cutoff.

## 6. A witness beyond both scalar norm tests

At \(\kappa=1\), \(g=4\), \(t=4/3\), the generic bilinear linear bound is
\(32t=128/3>1\), and even the exact unpreconditioned norm is
\(\|K\|_{\mathcal N}=6>1\). Nevertheless the finite gate passes.
For example \(N=4\) gives
\[
z_4=187174/194355<1,\qquad
\|\mathcal L^{-1}\|_{\mathcal N}\le2408805/7181.
\]
The default \(N=16\) sharpens the bound to
\[
\|\mathcal L^{-1}\|_{\mathcal N}
\le\frac{12149367650561223346581}{402676022206271014505},
\quad
z_{16}=\frac{4282180805264211022}{13646739461224002057}<1.
\tag{19}
\]
The different Hilbert norm still has the bound one from (11).
At insufficient cutoffs the Fourier gate returns `INCONCLUSIVE` while
retaining that independently valid Hilbert statement.

The package API is `su2_plaquette_linearized_inverse` and its canonical
replayer, in `omnibias.geometry.gauge.transfer.plaquette_resolvent`.
Certificates reconstruct every coefficient, LU pivot, weighted column,
full-tail bound and scope. Successful replay of an `INCONCLUSIVE` result
is a faithful refusal replay, not an earned Fourier inverse.

To use this in a target-vacuum correction proof, one still needs a small
enough preconditioned residual and a bounded nonlinear correction term
in the same norm. A strip or cubic-box application additionally needs
locality and constants independent of the number of plaquettes. Neither
requirement is supplied by the one-plaquette inverse alone, and no target
vacuum, physical gap, infinite-volume, continuum or Clay flag is earned here.

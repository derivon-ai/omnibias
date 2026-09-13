# Actual SU(2) vacuum local inequalities

`omnibias.geometry.gauge.transfer.vacuum_local` encloses the local comparison
floors proved below. It validates exact graph incidence and uses the rigorous
interval backend in certificate mode. Numerical zero lower endpoints from
underflow are `INCONCLUSIVE`. Replay recomputes the entire witness and scope.
The enclosure concerns the explicit comparison floor, not the actual
conditional eigenvalue. These analytic implications have not been formalized.

```python
from omnibias.geometry.gauge.transfer.vacuum_local import (
    su2_vacuum_local_bounds, replay_su2_vacuum_local_certificate,
)

result = su2_vacuum_local_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)],
    plaquettes=[(1, 2, 3, 4)], kappa=16, blocks=[(1, 2)],
)
assert result["status"] == "PASS"
assert replay_su2_vacuum_local_certificate(result["certificate"])
assert not result["continuum_claim"]
```

11 September 2026. This independent review proves a volume-independent
conditional Poincare bound for the **actual Hamiltonian vacuum measure**.
It verifies the partial-Bochner argument proposed during this campaign.
It also gives an explicit counterexample to promoting its first-derivative
bound alone into a global gap. No novelty or continuum claim is made.

## Normalization and the logarithmic groundstate equation

Let \(M=SU(2)^{\mathcal E}\) for a finite graph, with normalized product Haar
measure. Use the bi-invariant metric for which
\(-\Delta_e\) has spin-\(j\) eigenvalue \(j(j+1)\). Each factor is the
round three-sphere of radius two. In this convention,

\[
\operatorname{Ric}_e=\tfrac12 g_e,\qquad
\operatorname{diam}(SU(2))=2\pi,\qquad
\lambda_{\rm Haar}=\tfrac34.
\]

Let the character potential be

\[
V(U)=\sum_p v_p\chi_{1/2}(U_p),\qquad
v_p\ge0,\qquad
d_e=\sum_{p:\,e\in p}v_p.
\]

Each supplied cycle uses an edge at most once. This condition ensures the
one-link gradient estimate below; repeated traversals need their own
multiplicity factors. Electric weights are one in this note.

For the Hamiltonian convention

\[
A_\kappa=\frac\kappa2 C+
\frac2\kappa\sum_p v_p(2-\chi_{1/2}(U_p)),\qquad
C=-\sum_e\Delta_e,
\]

multiplication by \(2/\kappa\), followed by an additive constant shift,
gives

\[
H'=C-gV,\qquad g=\frac4{\kappa^2}.
\]

These operations preserve the groundstate. For a finite graph this is an
elliptic self-adjoint operator with bounded smooth real potential on a compact
connected manifold. Its positive heat semigroup has a unique strictly positive
smooth groundstate \(\psi_0\), normalized in Haar \(L^2\).
Gauge invariance of the potential and uniqueness make it gauge invariant.
Write \(S=\log\psi_0\).

The groundstate equation is

\[
C S-|\nabla S|^2-gV=E_0',
\]

so, since Haar mean \(V\) vanishes,

\[
E_0'=-\int|\nabla S|^2\,dU,\qquad
S-\int S\,dU
=C_0^{-1}\!\left(gV+|\nabla S|^2
-\int|\nabla S|^2\,dU\right).
\]

Here \(C_0^{-1}\) acts on Haar-mean-zero functions. This identity fixes the
Riccati sign. It does not by itself establish a contraction in any
volume-independent function or interaction norm.

## The partial-gradient estimate is uniform in volume

For every edge and every finite graph,

\[
\boxed{\quad
\|\nabla_e\log\psi_0\|_\infty\le2g d_e.
\quad}
\]

Only the weighted local incidence \(d_e\) occurs. In a family with
\(\sup_e d_e\le d_*\), the same constant applies in all volumes and at every
link, without presuming that \(\psi_0^2\,dU\) is a Wilson-action Gibbs measure.

To prove it, first observe that
\(\chi_{1/2}(U)=2\cos(r/2)\) in distance \(r\) from the identity on
the radius-two sphere, so

\[
\|\nabla\chi_{1/2}\|_\infty=1.
\]

With the other links fixed, a once-traversed edge enters a cycle through
left and right multiplication, with possible inversion. These are isometries.
The triangle inequality therefore gives
\(\|\nabla_e V\|_\infty\le d_e\).

Set

\[
u_t=e^{-tH'}1,\qquad S_t=\log u_t,\qquad
\partial_tS_t=\Delta S_t+|\nabla S_t|^2+gV,
\quad S_0=0,
\]

where \(\Delta=\sum_j\Delta_j\). The heat kernel with a bounded real
potential preserves strict positivity, so these functions are well defined
and smooth for the calculation.

For \(h_e=|\nabla_eS_t|^2\), the product-manifold Bochner identity gives

\[
\begin{aligned}
(\partial_t-\Delta-2\nabla S_t\cdot\nabla)h_e
={}&-2\sum_j\|\nabla_j\nabla_e S_t\|^2
-2\operatorname{Ric}_e(\nabla_eS_t,\nabla_eS_t)\\
&+2g\langle\nabla_eS_t,\nabla_eV\rangle\\
\le{}&-h_e+2g d_e\sqrt{h_e}.
\end{aligned}
\]

The mixed derivatives from different factors commute. The only Ricci term
comes from the factor indexed by \(e\); the sum of Hessian squares still
includes **every** factor \(j\). Differentiating the nonlinear
\(|\nabla S_t|^2\) term gives exactly the displayed drift term, rather than
an uncontrolled sum over volume.

Let \(b(t)=2g d_e(1-e^{-t/2})\), which solves
\(b'=-b/2+g d_e\), \(b(0)=0\). The scalar maximum principle gives
\(h_e\le b(t)^2\). One can avoid differentiating \(\sqrt{h_e}\) at its zeros
by using the strict positive barrier
\(b_\varepsilon(t)=b(t)+\varepsilon e^t\). At a first contact
\(h_e=b_\varepsilon^2\), the parabolic operator applied to their difference
is nonnegative, whereas the preceding inequality gives

\[
-b_\varepsilon^2+2g d_e b_\varepsilon
-2b_\varepsilon b_\varepsilon'
=2b_\varepsilon(-b_\varepsilon/2+g d_e-b_\varepsilon')<0.
\]

This is a contradiction. Letting \(\varepsilon\downarrow0\) proves

\[
\|\nabla_e S_t\|_\infty
\le2g d_e(1-e^{-t/2}).
\]

On each fixed finite graph, compact elliptic spectral theory and positivity
give
\(e^{tE_0'}u_t\to\langle1,\psi_0\rangle\psi_0\) in \(C^1\), indeed
smoothly after positive time. The coefficient is positive and \(\psi_0\)
has a positive minimum. Therefore
\(\nabla_e S_t\to\nabla_e\log\psi_0\) uniformly on that graph.
The bound just obtained does not depend on its volume or on a uniform rate
for this final convergence. Passing to the limit proves the boxed estimate.

## Conditional Poincare bounds for the true vacuum

Let \(d\mu=\psi_0^2\,dU\). For each fixed exterior configuration,
the conditional one-link density relative to Haar is proportional to
\(e^{2S(U_e,U_{e^c})}\). The gradient bound and diameter give

\[
\operatorname{osc}_{U_e}(2S)\le8\pi g d_e.
\]

If a density \(r\) relative to a probability measure \(dH\) obeys
\(\sup r/\inf r\le e^\omega\), and Haar has Poincare constant
\(\lambda_H>0\), then

\[
\begin{aligned}
\operatorname{Var}_{rH}(f)
&=\inf_c\int|f-c|^2r\,dH\\
&\le(\sup r)\operatorname{Var}_H(f)\\
&\le\frac{\sup r}{\lambda_H}\int|\nabla f|^2\,dH\\
&\le\frac{e^\omega}{\lambda_H}\int|\nabla f|^2r\,dH.
\end{aligned}
\]

No unspecified density normalization enters this ratio. Using the exact
one-link Haar gap \(3/4\) gives the actual conditional bound

\[
\boxed{\quad
\operatorname{Var}_{\mu(\cdot\,|\,U_{e^c})}(f)
\le\frac1{\gamma_e}
\mathbb E_\mu\!\left[|\nabla_e f|^2\,\middle|\,U_{e^c}\right],
\qquad
\gamma_e=\frac34e^{-8\pi g d_e}.
\quad}
\]

For a fixed nonempty block \(B\), change its coordinates one at a time.
Then

\[
\operatorname{osc}_{U_B}(2S)\le8\pi g\sum_{e\in B}d_e.
\]

The product Haar gap is still \(3/4\), as follows by tensorization or by
the sum spectrum of the factor Laplacians. Thus the same proof gives

\[
\gamma_B=\frac34\exp\!\left(-8\pi g\sum_{e\in B}d_e\right)
\]

for the conditional block Dirichlet form
\(\sum_{e\in B}|\nabla_e f|^2\), uniformly in every exterior configuration.
For bounded block size and bounded incidence this is uniform in graph volume.
It is valid at every finite \(g\), although it becomes very small at large
\(g\). It is a local inequality, not a global spectral-gap conclusion.

For unweighted plaquettes in a three-dimensional cubic bulk,
\(d_e=4\), so the one-link exponent is \(128\pi/\kappa^2\).
This explicit scale makes the weakness of the estimate at small \(\kappa\)
visible. A finite rational implementation must use an outward upper bound on
the exponent and a justified lower bound on its negative exponential.
Floating evaluation alone is not a certificate.

## What is still required to make the bound global

For a block cover of overlap multiplicity at most \(m\), the additional
variance-factorization hypothesis is

\[
\operatorname{Var}_\mu(f)
\le C\sum_B\mathbb E_\mu
\operatorname{Var}_{\mu(\cdot\,|\,U_{B^c})}(f),
\]

with \(C<\infty\) independent of volume. If that is proved and
\(\gamma_B\ge\gamma>0\), the conditional estimate now actually established
above yields

\[
\operatorname{Var}_\mu(f)
\le\frac{Cm}{\gamma}\int\sum_e|\nabla_e f|^2\,d\mu,\qquad
\Delta_{A_\kappa}\ge\frac{\kappa\gamma}{2Cm}.
\]

The present calculation removes the need to assume the local conditional
constant. It does not establish a volume-uniform factorization constant \(C\).

The implementation also computes a separate dense finite-graph bound:
the mixed coordinate oscillation of \(2S\) is at most
\(16\pi g\min(d_i,d_j)\). Conditional total-variation influence is
therefore bounded by \(a_{ij}=4\pi g\min(d_i,d_j)\) for \(i\ne j\),
with zero diagonal, using \(\mathrm{TV}\le\tanh(b/4)\le b/4\) for a
log-likelihood oscillation bound \(b\). If its maximum row sum is
\(q<1\), [Wu's Theorem 2.1](https://arxiv.org/pdf/math/0611635) gives
\(C\le(1-q)^{-1}\) for singleton conditionals. The code may then earn
`finite_graph_gap_verified` from \(\kappa\min_i\gamma_i(1-q)/2>0\).
In a homogeneous dense majorant the row sum is
\(4\pi gd_*(|\mathcal E|-1)\), so this bound grows with volume.
`status=PASS` concerns the local floors, independently of this extra gate.

A different sufficient route would be a genuine pointwise Hessian bound
\(\operatorname{Hess}S\le hI\) with \(h<1/4\), uniformly in volume.
For the groundstate diffusion
\(L=\Delta+2\nabla S\cdot\nabla\), the Bochner identity is

\[
\Gamma_2(f)
=\|\operatorname{Hess}f\|^2+
(\operatorname{Ric}-2\operatorname{Hess}S)(\nabla f,\nabla f).
\]

Such a Hessian bound implies
\(\Gamma_2(f)\ge(1/2-2h)\Gamma(f)\). Integrating this inequality for an
eigenfunction of \(-L\) gives
\(\lambda^2\|f\|^2\ge(1/2-2h)\lambda\|f\|^2\). Hence the positive spectrum
of \(-L\) starts at least at \(1/2-2h\), and

\[
\Delta_{A_\kappa}\ge\frac\kappa2(1/2-2h).
\]

That implication is complete, but the required Hessian estimate has not been
derived here. The scalar partial-gradient maximum principle is not a tensor
Hessian estimate.

## An explicit obstruction to using the gradient bound alone

Uniformly small one-link log-density gradients and positive conditional
Poincare constants do not, by themselves, give a global Poincare constant.
The following smooth example proves this even when those gradients are
arbitrarily small.

On \(SU(2)^N\), let

\[
x_e=\tfrac12\operatorname{Tr}U_e,\qquad
M_N=\sum_{e=1}^N x_e,\qquad
S_N=\varepsilon\sqrt{M_N^2+1},\qquad
d\mu_N=Z_N^{-1}e^{2S_N}\,dU,\quad\varepsilon>0.
\]

Each \(x_e\) lies in \([-1,1]\), has symmetric Haar law, and
\(\|\nabla x_e\|\le1/2\). Therefore
\(\|\nabla_eS_N\|_\infty\le\varepsilon/2\), independently of \(N\).
The same diameter comparison as above gives one-link conditional gap at
least \((3/4)e^{-2\pi\varepsilon}\), also independently of \(N\).

Let \(\phi(t)=\mathbb E_H e^{t x_1}\). Since \(x_1\) is nonconstant and
has mean zero, strict Jensen gives \(\phi(t)>1\) for \(t\ne0\). Independence
and \(\sqrt{M_N^2+1}\ge M_N\) imply

\[
Z_N\ge\mathbb E_H e^{2\varepsilon M_N}
=\phi(2\varepsilon)^N.
\]

Choose the odd Lipschitz form-domain function
\(f_N=\max(-1,\min(M_N,1))\). Symmetry gives \(\int f_N\,d\mu_N=0\).
Put \(A_\varepsilon=e^{2\varepsilon\sqrt2}\) and
\(b_N=A_\varepsilon/\phi(2\varepsilon)^N\). On \(|M_N|<1\),
the unnormalized density is at most \(A_\varepsilon\). Consequently

\[
\mu_N(|M_N|<1)\le b_N,\qquad
\operatorname{Var}_{\mu_N}(f_N)\ge1-b_N.
\]

The gradient of the clipped function vanishes almost everywhere outside
this region, so

\[
\int\sum_e|\nabla_e f_N|^2\,d\mu_N
\le\frac N4 b_N.
\]

For \(b_N<1\), the global Poincare gap is therefore at most

\[
\frac{Nb_N}{4(1-b_N)}\longrightarrow0
\quad\text{exponentially as }N\to\infty.
\]

This also forces every single-link variance-factorization constant to diverge
along this sequence, since its conditional gaps have the uniform positive
floor just proved. All functions involved are class functions of the factors;
the test can be kept in the independently conjugation-invariant sector.

The curvature failure is visible directly. At a configuration with every
\(x_e=0\), the gradient of \(M_N\) has squared norm \(N/4\), and

\[
\operatorname{Hess}S_N
=\varepsilon\,dM_N\otimes dM_N
\]

has largest eigenvalue \(\varepsilon N/4\).
The first-derivative bound does not control this collective direction.

This example is **not** the local Wilson Hamiltonian. It is a smooth
nonlocal density, used to show exactly what is lost if the locality of the
original operator is discarded after proving the gradient lemma. It can
itself be realized as the positive groundstate density of a smooth nonlocal
Schrodinger potential: take
\(V_N=\Delta S_N+|\nabla S_N|^2\), so that
\((-\Delta+V_N)e^{S_N}=0\), with nonnegativity verified by the groundstate
Dirichlet-form identity.
It supplies no counterexample to confinement or to a global estimate that
successfully uses the Wilson interaction structure.

## Provenance and the remaining concrete task

The Bochner formula with drift, including
\(\operatorname{Ric}-\operatorname{Hess}\log(d\mu/dU)\), is the classical
framework of [Bakry--Emery, Proposition 3, equations (4a)--(4b)](https://www.numdam.org/item/SPS_1985__19__177_0.pdf).
The specialization to a single SU(2) factor, its constants, and the
conditional-density comparison have been derived explicitly above.

Matrix heat-equation estimates require their own tensor argument.
[Hamilton's matrix Harnack paper](https://intlpress.com/site/pub/files/_fulltext/journals/cag/1993/0001/0001/CAG-1993-0001-0001-a006.pdf)
is relevant precedent, rather than a license to infer an upper Hessian bound
for the stationary solution with the present potential.

The companion [Fourier construction](gauge-vacuum-fourier.md) now exploits
the locality of the Wilson operator to control collective conditional
influence and the log-groundstate Hessian in an explicit strong-coupling
window. It supplies a volume-independent gap and a weighted summable tail
there. Extending that control toward the continuum remains open.
The first-derivative lemma alone does not imply the stronger result;
the Fourier proof supplies the additional interaction-norm estimate.

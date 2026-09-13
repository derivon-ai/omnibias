# Refinement through the actual SU(2) vacuum

This module constructs a finite refinement through the **actual** fine
vacuum and its actual coarse marginal. It bounds every physical
complementary mode and the mixed energy form. The preceding Hamiltonian's
full gap is not an input. All estimates concern the specified seven-edge
graph; no all-scale or continuum conclusion is earned.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_theta_vacuum_refinement,
    replay_su2_theta_vacuum_refinement_certificate,
)

result = su2_theta_vacuum_refinement(19, correction_radius=Q(1, 9))
assert result["status"] == "PASS"
assert result["actual_vacuum_embedding_verified"]
assert Q(result["physical_gap_lower"]) >= Q(1223, 100)
assert replay_su2_theta_vacuum_refinement_certificate(result["certificate"])
assert not result["all_scale_refinement_claim"]
assert not result["continuum_claim"]
```

## The constructed fine theory and its actual marginal

The two unit plaquettes have three-edge outer paths \(X,Y\) and one
shared edge \(Z\), with Gauss's law imposed at every vertex. Set

\[
A_\kappa=\frac{\kappa}{2}\sum_{e=1}^7 C_e+
\frac2\kappa(4-\chi_L-\chi_R),\quad
\alpha=\kappa/2,\quad g=4/\kappa^2,\quad c_*=3/4.
\]

Here \(\chi_L=\operatorname{Tr}(XZ^{-1})\) and
\(\chi_R=\operatorname{Tr}(YZ^{-1})\). The
[invariant Fourier construction](gauge-invariant-vacuum-fourier.md)
gives the actual positive vacuum

\[
\psi_f=\frac{e^S}{\|e^S\|_2},\qquad
S=\frac g3(\chi_L+\chi_R)+U,\qquad \mathcal N(U)\le r,
\]

provided the exact fixed-point inequalities hold:

\[
\frac43(16g+r)^2\le r,\qquad \frac83(16g+r)<1.
\tag{1}
\]

The implementation regenerates the explicit graph certificate and uses
only structural membership and (1). Its source gap and factorization
results are not used as premises.

Let \(p\) be the outer holonomy \(W=XY^{-1}\), and let
\(\nu=p_*(\psi_f^2dH_f)\) be the **true** coarse marginal. Its normalized
positive Haar density defines \(\psi_\nu=\sqrt{d\nu/dH_c}\).
The Haar-space map

\[
(\mathcal J\Phi)(U_f)=
\psi_f(U_f)\frac{\Phi(p(U_f))}{\psi_\nu(p(U_f))}
\tag{2}
\]

is an isometry and satisfies \(\mathcal J\psi_\nu=\psi_f\), directly by
integration against a pushforward. In the vacuum measure spaces it is
simply \(Jf=f\circ p\), with \(J1=1\). Its orthogonal complement consists
of all functions with zero conditional mean given \(W\).

The formula is specified through the analytically constructed vacuum.
The API does not return a numerical evaluator for that infinite Fourier
series or replace it by the trial exponential. In particular \(\psi_\nu\)
need not be the vacuum of an independently prescribed coarse Wilson
Hamiltonian.

## Exact physical geometry

After endpoint gauge reduction the second coordinate is \(R=ZY^{-1}\);
physical functions are invariant under simultaneous conjugation of
\((W,R)\). Their weighted electric form is

\[
\Gamma(h)=3|\nabla_W h|^2+|\nabla_R h|^2+
3|\nabla_W h+\nabla_R h|^2
=6|\nabla_W h+\tfrac12\nabla_R h|^2+
\tfrac52|\nabla_R h|^2.
\tag{3}
\]

The equality follows by differentiating the path products with
bi-invariant generators; each of the three segments contributes its
Casimir. Thus the exact coarse compression is
\(a[f]=6\alpha\int|\nabla f|^2d\nu\).
For conditional-mean-zero \(h\), (3) supplies the entire physical fiber
kinetic coefficient \(5/2\). Treating the redundant endpoint gauge
variables as physical fiber modes would give a different, weaker bound.

## Actual density comparisons

The inherited single-edge oscillation estimate gives

\[
\Omega_f=\frac{16g+8r}{3},\qquad
\operatorname{osc}_R\log\mu(R\mid W)\le\Omega_f.
\tag{4}
\]

It applies because varying \(R\) at fixed outer links varies only the
shared physical edge. Changing one outer link similarly gives
\(\operatorname{osc}\log\nu\le8(g+r)/3\).

A second-order alternative comes from exact Haar integration of the
seed. In quaternion coordinates, its unnormalized marginal is
\(\mathbb E_{S^3}e^{c q_0}\), where \(0\le c\le8g/3\).
The moment series is

\[
\mathbb E_{S^3}e^{c q_0}
=\sum_{k\ge0}\frac{(c^2/4)^k}{k!(k+1)!}
\le e^{c^2/8},
\]

using \((k+1)!\ge2^k\); its minimum is one. Every nonconstant invariant
Fourier support satisfies \(E_{\mathbf j}\sum_ej_e\ge6\).
Summing the seven anchored norms therefore gives
\(\sum_{\mathbf j}\|U_{\mathbf j}\|_1\le7r/6\), hence
\(\operatorname{osc}(2U)\le14r/3\). Positive integration transfers this
logarithmic ratio bound to the marginal. Consequently

\[
\Omega_m=\min\left\{\frac{8(g+r)}3,\frac{8g^2}{9}+\frac{14r}3\right\},
\qquad \operatorname{osc}\log\nu\le\Omega_m.
\tag{5}
\]

Normalization does not change oscillation. If a probability density has
log oscillation at most \(\Omega\), its Poincare constant is at least
the Haar constant times \(e^{-\Omega}\): write \(l\le w\le u\), use
\(\operatorname{Var}_w f\le u\operatorname{Var}_H f\) and
\(\int|\nabla f|^2w\ge l\int|\nabla f|^2dH\), with \(l/u\ge e^{-\Omega}\).

For chosen integer \(m>\max(\Omega_m,\Omega_f)\), the exact rational
bounds \(\ell_m=(1-\Omega_m/m)^m\) and
\(\ell_f=(1-\Omega_f/m)^m\) are positive lower bounds on those
exponentials. Hence

\[
\gamma_R=c_*\ell_f,\quad
d_A=6\alpha c_*\ell_m,\quad
d_H=\tfrac52\alpha\gamma_R
\tag{6}
\]

bound the actual conditional Poincare constant, coarse centered
compression, and all physical complementary modes respectively.
The marginal density itself lies in \([\ell_m,\ell_m^{-1}]\).

## Relative coupling through the actual drift

Ground-state conjugation gives the centered fine Dirichlet form
\(\alpha\int\Gamma\,d\mu\). For a coarse function,

\[
QKJf=-2\alpha\langle b-\mathbb E[b\mid W],\nabla f(W)\rangle.
\tag{7}
\]

The six outer-edge derivatives imply \(|b|\le2(g+r)\).
For a sharper conditional estimate, vary only the shared edge.
The outer-holonomy derivative frames are independent of it.
Six seed mixed Hessian blocks contribute at most \(6g/6=g\);
the entire correction Hessian row contributes at most \(2r/3\).
Therefore for every unit \(v\in T_WG\),

\[
|\nabla_R\langle b,v\rangle|\le g+2r/3,\qquad
\operatorname{Cov}(b\mid W)\preceq
K I,\quad
K=\min\{4(g+r)^2,(g+2r/3)^2/\gamma_R\}.
\]

Apply the conditional Poincare inequality to each scalar contraction;
no factor equal to the dimension of the Lie algebra is needed.
Equations (6)--(7) yield

\[
\|QKJf\|^2\le\beta^2 a[f],\qquad
\beta^2=\frac{2\alpha}{3}K.
\tag{8}
\]

This is an energy-relative estimate, not a bounded mixed operator on
coarse \(L^2\). A smooth positive compact conditional density ensures a
common smooth form core and a conditional projection preserving the form
domain on this fixed graph.

## Gap comparison and exact arithmetic

If \(\beta^2<d_H\), Young's inequality applied to the mixed form proves

\[
\Delta_{\rm physical}\ge
\lambda_*=\frac{d_A+d_H-
\sqrt{(d_A-d_H)^2+4d_A\beta^2}}2>0.
\tag{9}
\]

Indeed for \(\beta^2/d_H<\theta<1\), the form is at least
\((1-\theta)a[f]+(d_H-\beta^2/\theta)\|h\|^2\).
Equating the two positive lower coefficients gives (9).
This controls the whole centered physical Hilbert space.

The square root is rounded **up** to a dyadic rational using integer
square-root arithmetic and an exact squared comparison. Subtraction
therefore returns a lower bound. No transcendental backend or
floating-point eigenvalue participates in certificate production.

At \(\kappa=19,r=1/9,m=8\), (1) holds with slack
\(6791/31668003\) and contraction \(7496/9747\). The returned gap
lower bound exceeds \(1223/100\). All energies are dimensionless \(aH\).
This is a gap of the physical finite graph; it is not asserted for
the larger scalar product-group space, nor expressed in physical GeV.

## Earned and remaining claims

- The fixed-point gate earns the analytic actual-vacuum embedding.
- Valid exponential bounds also earn the conditional estimates.
- The relative coercivity and positive rounded root earn the finite
  physical gap. A failure returns `INCONCLUSIVE` with named constraints.
- Both success and refusal replay by full canonical recomputation,
  including the source certificate. Rehashing an altered claim does
  not make it replay.
- Closure of a prescribed coarse Wilson family, control through every
  scale, infinite volume, continuum identification, and formal prover
  tiers remain unearned.

Conditional projection and spectral comparison are established methods;
see [Legoll--Lelièvre](https://arxiv.org/abs/0906.4865) and
[Nüske--Koltai--Boninsegna--Clementi](https://arxiv.org/abs/1901.01557).
The compact-group constants and physical-space implications used here
are explicitly derived above. Their novelty is not asserted.

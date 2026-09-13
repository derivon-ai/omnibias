# Full physical SU(2) plaquette spectrum through weak coupling

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.weak_plaquette import (
    su2_weak_plaquette_gap,
    replay_su2_weak_plaquette_gap_certificate,
)

result = su2_weak_plaquette_gap(Fraction(1, 100), cutoff=80, bisection_steps=36)
assert result["all_positive_couplings_gap_lower"] == "26/33"
assert Fraction(result["gap_lower"]) > Fraction(39937, 10000)
assert Fraction(result["gap_upper"]) < Fraction(39938, 10000)
assert result["full_omitted_spin_comparison_verified"]
assert replay_su2_weak_plaquette_gap_certificate(result["certificate"])
assert not result["volume_uniform_claim"]
assert not result["continuum_claim"]
assert not result["theorem_prover_verified"]
```

`kappa` must be an exact positive integer or `Fraction`. Omitting `cutoff`
returns the analytic all-coupling comparison. Supplying an integer cutoff
at least one adds finite rational eigenvalue brackets with a full omitted
character comparison. `bisection_steps` is a positive integer. No floating
special-function evaluation feeds the certificate.

For one open square with four unit electric edges, neutral Gauss law at
all four vertices, and the normalization below, the complete physical
Hamiltonian satisfies
\[
\boxed{E_1(\kappa)-E_0(\kappa)\ge\frac{26}{33}
\quad\text{for every }\kappa>0.}
\tag{1}
\]
This elementary uniform bound follows from two independent variational
comparisons. It does not use the earlier strong-coupling Fourier radius,
an assumed true vacuum, or a spin cutoff. Exact finite Jacobi calculations
with a proved comparison for *all* omitted characters give much tighter
enclosures at individual weak couplings.

The physical one-plaquette problem is already known to be exactly
reducible to Mathieu functions; see D'Andrea, Bauer, Grabowska and
Freytsis, [Phys. Rev. D 109, 074501 (2024), Appendix B](https://doi.org/10.1103/PhysRevD.109.074501).
The purpose here is a self-contained, replayable comparison in this
repository's normalization. No new-physics or literature-novelty claim is
made for the solvable one-plaquette model.

## 1. The exact physical Hilbert space and radial operator

Use normalized Haar measure on SU(2), \(x=\operatorname{Tr}U\in[-2,2]\),
and the neutral class-function Hilbert space. Its orthonormal characters
are \(e_n(x)=\chi_{n/2}(U)=U_n(x/2)\), for every integer \(n\ge0\).
The original four edges give
\[
 H_\kappa=\frac\kappa2 C+\frac2\kappa(2-x),\qquad
 C=-(4-x^2)\partial_x^2+3x\partial_x,\qquad
 Ce_n=n(n+2)e_n.
\tag{2}
\]
The energies are the original dimensionless \(aH\) energies. In
particular the fundamental electric eigenvalue is three, not \(3/4\).
The Hilbert space is the entire physical space for this graph; it is not
the full unconstrained \(L^2(\mathrm{SU}(2))\) rotor, whose angular
sectors would be different.

Set \(x=2\cos\theta\) and
\[
 (\mathcal U f)(\theta)=\sqrt{2/\pi}\sin\theta\,
 f(2\cos\theta),\qquad0<\theta<\pi.
\]
This is a unitary onto \(L^2(0,\pi)\), since radial Haar measure is
\((2/\pi)\sin^2\theta\,d\theta\). The characters map to the complete
sine basis \(\sqrt{2/\pi}\sin((n+1)\theta)\). Hence both the operator
and its self-adjoint domain are identified exactly:
\[
 \mathcal U H_\kappa\mathcal U^{-1}
 =-\frac\kappa2\partial_\theta^2
  +\frac4\kappa(1-\cos\theta)-\frac\kappa2,
 \qquad\mathcal D=H^2(0,\pi)\cap H_0^1(0,\pi).
\tag{3}
\]
The magnetic multiplication is bounded for each finite positive
\(\kappa\), so it preserves this domain. The Dirichlet spectrum is
discrete and simple, with the usual positive ground eigenfunction.

## 2. An eikonal solution with an admissible antipodal cusp

Consider
\[
 F(x)=8-4\sqrt{2+x},\qquad\phi_\kappa(x)=e^{-F(x)/\kappa}.
\tag{4}
\]
On \(-2<x\le2\), the derivatives obey
\[
 F'=-\frac2{\sqrt{2+x}},\quad F''=(2+x)^{-3/2},\quad
 (4-x^2)(F')^2=4(2-x).
\tag{5}
\]
Thus the terms of order \(1/\kappa\) in \(H_\kappa\phi/\phi\)
cancel *exactly*, leaving
\[
 W(x):=\frac{H_\kappa\phi_\kappa}{\phi_\kappa}
 =\frac{2+5x}{2\sqrt{2+x}}.
\tag{6}
\]
This is not a smooth global WKB reference: \(W(x)\to-\infty\) as
\(x\downarrow-2\). In particular its residual oscillation is infinite,
so the bounded local-energy comparison cannot
be applied to it.

It is nevertheless a valid Rayleigh trial. In the Dirichlet coordinate,
up to the irrelevant constant \(\sqrt{2/\pi}\), it is
\[
 y_\kappa(\theta)
 =\sin\theta\exp[-8(1-\cos(\theta/2))/\kappa].
\tag{7}
\]
This is smooth on the closed interval, vanishes at both endpoints, and
belongs to the operator domain in (3). It need not be a smooth radial
class function across the SU(2) antipode. Writing \(s=\pi-\theta\),
\[
 y_\kappa(\pi-s)
 =e^{-8/\kappa}\{s+(4/\kappa)s^2+O(s^3)\}.
\tag{8}
\]
Its second derivative remains finite. In original radial coordinates the
Laplacian cusp is of order \(1/s\), whose square is integrable against
the three-dimensional radial Haar factor \(s^2\,ds\). These facts justify
the Rayleigh calculation without pretending the trial is the true vacuum.

For \(z=\sqrt{2+x}\in(0,2]\),
\[
 3-W(x)=\frac{(2-z)(5z+4)}{2z}\ge0.
\tag{9}
\]
The negative singularity of \(W\) is integrable against
\(|y_\kappa|^2d\theta\). Integrating the operator identity (6) and
using the domain statement proves
\[
 E_0(\kappa)\le3.
\tag{10}
\]
The independent constant-character trial has electric energy zero and
Haar mean \(\int x\,dH=0\), giving
\[
 \boxed{E_0(\kappa)\le U_0(\kappa)
 :=\min(3,4/\kappa).}
\tag{11}
\]

## 3. Two independent lower bounds on the first excitation

The magnetic potential in (2) is nonnegative. Electric operator order
and min-max therefore give
\[
 E_1(\kappa)\ge3\kappa/2.
\tag{12}
\]
For a bound useful at weak coupling, the concavity of sine on
\([0,\pi/2]\) gives \(\sin(\theta/2)\ge\theta/\pi\), hence
\[
 1-\cos\theta\ge2\theta^2/\pi^2\qquad(0\le\theta\le\pi).
\tag{13}
\]
The Dirichlet operator (3) thus dominates
\(-\kappa\partial_\theta^2/2+8\theta^2/(\kappa\pi^2)-\kappa/2\)
on this interval. Extension by zero embeds its form domain isometrically
into the half-line Dirichlet oscillator form domain. Min-max then bounds
each interval eigenvalue from below by the corresponding half-line
eigenvalue. The latter are the odd full-line oscillator levels:
\[
 \lambda_n^{\rm half}=(4n+3)\frac2\pi-\frac\kappa2,
 \qquad n=0,1,2,\ldots.
\tag{14}
\]
In particular \(E_0\ge6/\pi-\kappa/2\) and
\(E_1\ge14/\pi-\kappa/2\). No localization assumption about the actual
wavefunction is required for this comparison.

The elementary rational bound \(\pi<22/7\) can itself be proved by
\[
 \frac{22}{7}-\pi
 =\int_0^1\frac{t^4(1-t)^4}{1+t^2}\,dt>0.
\tag{15}
\]
Polynomial division gives the quotient
\(t^6-4t^5+5t^4-4t^2+4\) and remainder \(-4\), verifying the identity.
Consequently the exact rational comparisons used by the certificate are
\[
 \boxed{E_1(\kappa)\ge L_1(\kappa)
 :=\max(3\kappa/2,49/11-\kappa/2),}
 \qquad E_0(\kappa)\ge\max(0,21/11-\kappa/2).
\tag{16}
\]

## 4. A uniform lower bound for every positive coupling

Combine (11) and (16):
\[
 \Delta(\kappa):=E_1-E_0\ge L_1(\kappa)-U_0(\kappa).
\tag{17}
\]
For \(0<\kappa\le4/3\), the right side is at least
\(16/11-\kappa/2\ge26/33\). For \(4/3\le\kappa\le2\), use
\[
 \frac{49}{11}-\frac\kappa2-\frac4\kappa-\frac{26}{33}
 =\frac{(3\kappa-4)(6-\kappa)}{6\kappa}\ge0.
\tag{18}
\]
For \(\kappa\ge2\), the electric and constant-trial comparisons give
\(3\kappa/2-4/\kappa\ge1\). This proves (1). The method works through
the intermediate region as well as both coupling extremes.

## 5. Exact all-spin Jacobi bracketing

Character multiplication is \(xe_n=e_{n+1}+e_{n-1}\), with
\(e_{-1}=0\). Thus the full physical Hamiltonian is the infinite Jacobi
matrix with
\[
 d_n=\frac\kappa2n(n+2)+\frac4\kappa,
 \qquad H_{n,n+1}=H_{n+1,n}=-c,\qquad c=2/\kappa.
\tag{19}
\]
For a finitely supported coefficient vector, its form is exactly
\[
 \langle u,Hu\rangle
 =\frac\kappa2\sum_{n\ge0}n(n+2)|u_n|^2
  +c|u_0|^2+c\sum_{n\ge0}|u_{n+1}-u_n|^2.
\tag{20}
\]
Both the boundary term \(c|u_0|^2\) and the final difference to zero for
a finitely supported vector are essential.

Retain \(0\le n\le N\), \(N\ge1\). Let \(H_N^D\) be the principal
finite matrix, and put
\[
 H_N^N=H_N^D-c|N\rangle\langle N|,
 \qquad t_N=\frac\kappa2(N+1)(N+3).
\tag{21}
\]
Discarding just the nonnegative interface square
\(c|u_{N+1}-u_N|^2\) proves the full form comparison
\[
 H\ge H_N^N\oplus t_N I.
\tag{22}
\]
The omitted block's remaining magnetic differences are nonnegative and
every omitted electric energy is at least \(t_N\). Finite support is a
form core, so (22) holds on the full form domain. This is an explicit
all-spin comparison, not an extrapolation over cutoffs.

If \(\lambda_j^D,\lambda_j^N\) are the ordered eigenvalues of these
finite matrices, min-max and Ritz restriction give, for \(j=0,1\),
\[
 \boxed{\min(\lambda_j^N,t_N)\le E_j\le\lambda_j^D.}
\tag{23}
\]
The minimum is required: if the tail floor is low, an omitted state could
precede that finite eigenvalue. The elementary bounds (11) and (16) may
be intersected with (23). A small cutoff merely gives a wider sound
interval.

For rational \(\kappa\), all matrix entries are rational. Sturm
determinants evaluated with exact integers count eigenvalues below each
rational query; zero determinants are removed in counting sign changes.
Dyadic bisection yields outward rational eigenvalue brackets. No floating
eigenvalue, Mathieu evaluation, or guessed spin tail enters the sealed
result. Replay rebuilds both matrices, both bisections, the omitted floor
and every claim field.


With 36 bisection steps, the following decimal endpoints are rational
outward widenings of the sealed intervals:

| Coupling | Retained highest character index | Actual physical gap enclosure |
|---|---:|---|
| `1/10` | 32 | `[3.9366734357, 3.9366734406]` |
| `1/100` | 80 | `[3.9937419610, 3.9937419824]` |
| `1/1000` | 256 | `[3.9993748111, 3.9993750950]` |

The interval widths include the entire omitted spectrum and rational
bisection error. These are fixed compact-graph results at small coupling.

## 6. Exact solvability and what weak scaling means here

With \(z=\theta/2\), (3) is Mathieu's equation with
\[
 a=8E/\kappa+4-32/\kappa^2,\qquad q=-16/\kappa^2.
\]
Dirichlet conditions at \(z=0,\pi/2\) select the even-index sine
characteristic values \(b_{2n+2}\). The reflection
\(z\mapsto\pi/2-z\) reverses the sign of \(q\) while preserving both
boundaries. Hence the complete spectrum can also be written
\[
 E_n(\kappa)=\frac\kappa8
 \left[b_{2n+2}(16/\kappa^2)-4+32/\kappa^2\right].
\tag{24}
\]
This reproduces the known Mathieu reduction after matching electric and
magnetic conventions; copying a literature coupling symbol without that
match would change the frequency.

Using the characteristic-value asymptotics in
[DLMF 28.8.1](https://dlmf.nist.gov/28.8#E1), or the rigorous treatment of
[Ogilvie and Olde Daalhuis](https://arxiv.org/abs/1507.04984), with
\(s=4n+3\), gives
\[
 E_n=s-\frac{33+s^2}{64}\kappa
       -\frac{s^3+3s}{4096}\kappa^2+O(\kappa^3),
 \qquad
 \Delta=4-\frac58\kappa-\frac{41}{512}\kappa^2+O(\kappa^3).
\tag{25}
\]
The asymptotic notation here is not a finite numerical remainder bound.
The optional certificate instead uses (23) for finite-coupling errors.
An ordinary large-parameter Mathieu routine can select an inaccurate
characteristic value; it is used only as a diagnostic where checked, never
as a certificate backend.

In particular \(E_0\to3\), \(E_1\to7\), and \(\Delta\to4\) as
\(\kappa\to0\). The underlying quadratic oscillator frequency is two;
neutral radial states occupy its odd half-line levels, separated by four.
This matches the four-edge single-face quadratic geometry.

Shrinking a physical lattice spacing on this fixed one-plaquette graph
does not produce a continuum box or the scaling
\(a\Delta_{\rm phys}\to0\). The fixed compact model remains gapped in
dimensionless units even at weak coupling. Extending this result to a
growing interacting graph would require control of its low spatial modes,
shared plaquettes and thermodynamic/continuum limits. Those obligations
are not supplied by the exactly solvable single plaquette.

## 7. Implementation and regression scope

`omnibias.geometry.gauge.transfer.weak_plaquette` exposes
`su2_weak_plaquette_gap(kappa, cutoff=None, bisection_steps=24)` and
`replay_su2_weak_plaquette_gap_certificate`. This page contains its complete written analytic implication.

Independent regressions check radial Chebyshev normalization, the exact
eikonal identity and its antipodal failure, the polynomial uniform-bound
proof, zero-pivot Sturm counts, and the interface-square identity on exact
grid and seeded vectors. Larger Jacobi matrices and moderate-parameter
Mathieu values are floating diagnostics for the rational intervals.
Canonical replay rejects altered boundaries, tail floors, normalizations
and parent flags. All formal tiers and all continuum/volume claims remain
false.

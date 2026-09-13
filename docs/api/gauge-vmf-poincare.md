# All-angular vMF Poincare bounds and an interacting strip reference

13 September 2026. Written analysis with independent algebra and
Friedrichs-domain review. Exact replay certifies the stated scalar budgets
and reference definition. The differential-operator proof below is not
formally verified.

For the von Mises--Fisher measure
\[
 d\mu_h(U)=Z_h^{-1}e^{h q_0}\,dH(U),\qquad
 U=q_0I+iX\cdot\sigma,\qquad q_0^2+|X|^2=1,\quad h>0,
 \tag{1}
\]
for every positive real \(h\), with the standard SU(2) metric whose Haar
Casimir gap is \(3/4\), we prove
\[
 \operatorname{gap}_{\mu_h}
 \ge\frac5{16}+\frac h8\ge\frac h8.
 \tag{2}
\]
This is an all-function bound, including every angular sector. The field
is the coefficient of \(q_0=\operatorname{Tr}(U)/2\), not of the trace.
An arbitrary real four-vector field is reduced to (1) by a rotation of
the unit quaternion sphere; \(h\) is its norm.

The second API applies (2) to every one-site conditional of a specified
**interacting compact reference**, obtaining a Poincare floor
\(1/(12\kappa)\), uniform in strip length. Its original-electric
reference Hamiltonian on the rooted cycle space has gap at least
\(1/24\). The reference is not identified with the Wilson vacuum.

## 1. Full angular decomposition

Work first with the unit-radius sphere \(S^3\). Write
\[
 q=(\cos\theta,\sin\theta\,\omega),\qquad
 0<\theta<\pi,\quad \omega\in S^2.
\]
The radial weight is \(\rho_h(\theta)=e^{h\cos\theta}\sin^2\theta\),
up to a normalization that cancels from quadratic forms. Expansion in
the full \(S^2\) spherical-harmonic basis gives the radial operators
\[
 A_\ell=-\partial_\theta^2-
       (2\cot\theta-h\sin\theta)\partial_\theta
       +\ell(\ell+1)\csc^2\theta,\qquad \ell=0,1,\ldots .
 \tag{3}
\]
Every \(\ell\ge1\) mode is automatically centered on the full sphere.
The \(\ell=0\) sector contains the constant vacuum and its orthogonal
radial excitations. No angular cutoff is made.

For positive smooth \(u\) on the open interval and compactly supported
smooth \(f\), the exact ground-state identity is
\[
 \int\bigl(|f'|^2+V|f|^2\bigr)\rho_h
 =
 \int u^2|(f/u)'|^2\rho_h+
 \int\frac{(-\partial^2-(\rho_h'/\rho_h)\partial+V)u}{u}
             |f|^2\rho_h .
 \tag{4}
\]
It follows by expanding the square and integrating by parts. This
identity extends its lower bounds to the Friedrichs form closure.

Take
\[
 u(\theta)=\sin^{1/2}\theta\,e^{-h\cos\theta/8}.
\]
A direct differentiation gives
\[
 \frac{A_1u}{u}
 =\frac54+\frac54\csc^2\theta+
                       \frac{7h^2}{64}\sin^2\theta
 \ge\frac54+\frac{\sqrt{35}}8h
 \ge\frac54+\frac h2.
 \tag{5}
\]
For an entirely rational check of the last step, put
\(z=h\sin^2\theta\). After multiplication by \(64\sin^2\theta\),
the relevant difference is
\[
 7z^2-32z+80
 =7(z-16/7)^2+304/7>0.
 \tag{6}
\]
Since \(A_\ell\ge A_1\) as forms for every \(\ell\ge1\), equation
(5) controls all angular modes.

The comparison function behaves like \(\theta^{1/2}\) at an endpoint.
It belongs to the weighted form domain but generally not the operator
domain: applying \(A_1\) can produce a logarithmically divergent squared
norm. We do not assert that it is a smooth spherical eigenfunction or an
operator-domain trial vector. Identity (4) is first used only on
compactly supported functions. Smooth spherical functions in each
\(\ell\ge1\) sector have radial amplitudes \(O(\theta^\ell)\) at a
pole; cutting them off in an interval of length \(\varepsilon\) changes
the form by \(O(\varepsilon^{2\ell+1})\). The required closure therefore
includes every smooth spherical mode, and then the full form domain.

## 2. The nonconstant radial sector

The radial bound can also be proved without importing an angular
conclusion from a one-dimensional theorem. With \(D=\partial_\theta\),
\[
 DA_0=(A_1+h\cos\theta)D.
 \tag{7}
\]
Indeed differentiating the drift adds
\(-\partial_\theta(2\cot\theta-h\sin\theta)
  =2\csc^2\theta+h\cos\theta\).

For the derivative partner, use
\[
 v(\theta)=\sin^{1/2}\theta\,e^{-3h\cos\theta/8}.
\]
Its exact ratio is
\[
 \frac{(A_1+h\cos\theta)v}{v}
 =\frac54+\frac54\csc^2\theta+
                          \frac{15h^2}{64}\sin^2\theta
 \ge\frac54+\frac{\sqrt{75}}8h
 \ge\frac54+h.
 \tag{8}
\]
The final difference has the rational numerator
\[
 15z^2-64z+80
 =15(z-32/15)^2+176/15>0.
 \tag{9}
\]

The weighted elliptic generator on the compact sphere has a complete
smooth eigenbasis. Its radial invariant subspace has the same property.
If \(A_0f=\lambda f\) and \(\lambda>0\), then \(g=f'\) is nonzero
and (7) gives
\((A_1+h\cos\theta)g=\lambda g\). A smooth radial function is a smooth
function of \(\cos\theta\) at a pole, so \(g\) vanishes at least
linearly there. It lies in the derivative partner's Friedrichs form
domain. Apply (4), (8) and closure to obtain
\(\lambda\ge5/4+h\).

The only omitted radial eigenfunction is the constant. Existence of the
finite-volume centered spectral decomposition does not assume the
desired field-dependent estimate: smooth positive density on a compact
sphere already gives a finite Poincare constant by bounded-density
comparison with Haar. The supersolutions supply the stronger estimate.

Combining (5) and (8), and using the orthogonality of the complete
angular decomposition, proves a unit-sphere gap at least \(5/4+h/2\).
The radius-two SU(2) metric divides the squared gradient and the
generator by four, proving (2).

For primary context,
[Li--Ma--Zhang, Proposition 1.1 and Theorem 1.2,
*On the spectral gap of Boltzmann measures on the unit sphere*
(2021)](https://www.sciencedirect.com/science/article/pii/S0167715220302662)
distinguish the polar marginal from the whole sphere. Their Proposition
1.1 gives polar gap at least \(\max(n-1,|h|)\) on \(S^{n-1}\);
the displayed full-sphere lower bound in Theorem 1.2 is \(n-2\).
The all-angular linear estimate used here follows from (3)--(9), not
from treating their radial proposition as a full-sphere theorem.
No priority claim about vMF spectral-gap estimates is made.

## 3. The specified interacting compact reference

Let \(n\ge1\), let \(A_{\rm path}\) be the adjacency matrix of the
open path on \(n\) sites, and define
\[
 M=4I-A_{\rm path},\qquad B=M^{-1/2}.
 \tag{10}
\]
For \(U_i=(q_{0,i},X_i)\in SU(2)\), put
\[
 A_i=2-2q_{0,i},\qquad
 F_n=2\sum_iB_{ii}A_i+
                    4\sum_{i<j}B_{ij}X_i\cdot X_j,
 \qquad
 d\nu_{n,\kappa}=Z^{-1}e^{-2F_n/\kappa}\prod_i dH(U_i).
 \tag{11}
\]
Here \(\kappa>0\). For \(n\ge2\) the off-diagonal coefficients
are positive, so this is an interacting compact law.

The inverse-square-root power series has nonnegative coefficients.
The path adjacency has row norm at most two. Therefore
\[
 B_{ii}\ge\frac12,\qquad B_{ij}\ge0,\qquad
 \sum_jB_{ij}\le\frac1{\sqrt2},\qquad 2I\le M\le6I.
 \tag{12}
\]
These follow directly from
\(B=\frac12(I-A_{\rm path}/4)^{-1/2}\), including at path endpoints.
No finite collection of numerical matrix square roots is substituted
for these all-length bounds.

With every other cycle frozen, (11) is exactly the one-site vMF density
with four-vector field
\[
 h_i=\frac8\kappa
     \left(B_{ii},-\sum_{j\ne i}B_{ij}X_j\right),\qquad
 |h_i|\ge\frac{8B_{ii}}\kappa.
 \tag{13}
\]
Thus (2) gives the uniform conditional bound
\[
 \gamma_i\ge\frac{B_{ii}}\kappa.
 \tag{14}
\]
The proof holds for every frozen exterior value, not merely on a
high-probability set.

Use \(S_{\rm ref}=-F_n/\kappa\), so \(\nu\propto e^{2S_{\rm ref}}\).
For \(i\ne j\), the mixed Hessian comes only from the corresponding
bilinear term. Each map \(U_i\mapsto X_i\) has differential operator
norm at most \(1/2\) in the radius-two metric. Hence
\[
 \|\operatorname{Hess}_{ij}S_{\rm ref}\|_{\rm op}
       \le\frac{B_{ij}}\kappa.
 \tag{15}
\]
Apply the conditional Poincare comparison proved below to the actual
conditionals of this reference. Its comparison matrix has
diagonal \(B_{ii}/\kappa\) and off-diagonal \(-2B_{ij}/\kappa\), and
every row margin is at least
\[
 \frac{B_{ii}-2\sum_{j\ne i}B_{ij}}\kappa
 =\frac{3B_{ii}-2\sum_jB_{ij}}\kappa
 \ge\frac{3/2-\sqrt2}\kappa
 >\frac1{12\kappa},
 \tag{16}
\]
where \(\sqrt2<17/12\). Therefore every smooth scalar cycle-space
function satisfies
\[
 \operatorname{Var}_{\nu_{n,\kappa}}f
 \le12\kappa\int\sum_i|\nabla_i f|^2\,d\nu_{n,\kappa}.
 \tag{17}
\]
The same hypotheses and row margins survive arbitrary exterior
conditioning; no separate isolated-block measure is inserted.

### Why the conditional comparison controls variance

Here is the finite compact-product argument used in (16)--(17).
It is related to [Menz's conditional covariance comparison,
Theorems 2.3 and 2.7](https://arxiv.org/abs/1402.5160).
The companion proof with Schur marginalization is at the repository path
    ensemble-laws/docs/constructive/conditional-poincare-schur.md.
No result about arbitrary one-form fields is needed.

For a smooth positive density \(\mu\propto e^{2S}\) on a finite
product of radius-two SU(2) factors, let
\[
 A_i=-\Delta_i-2\nabla_iS\cdot\nabla_i,\qquad A=\sum_i A_i.
\]
Suppose every one-site conditional has scalar gap \(\gamma_i\),
and \(\|\operatorname{Hess}_{ij}S\|\le c_{ij}=c_{ji}\) for \(i\ne j\).
Put \(H_{ii}=\gamma_i,\ H_{ij}=-2c_{ij}\), and suppose its row margins
are at least \(m>0\). It is symmetric with \(H\ge mI\). Its inverse
is entrywise nonnegative: writing \(H=D-2c\), the nonnegative matrix
\(2D^{-1}c\) has row norm less than one, and its Neumann series gives
\(H^{-1}=(I-2D^{-1}c)^{-1}D^{-1}\ge0\).

Solve \(Au=f-\mu f\) with zero mean. Such a smooth solution exists
at each fixed finite size by positive-density Haar comparison and
elliptic regularity. This uses no proposed uniform gap.
Let \(E_i=\|d_i u\|_{L^2(\mu)}\) and
\(a_i=\|d_i f\|_{L^2(\mu)}\).
For each frozen exterior, the conditional gap implies
\[
 \|A_i u\|_2^2\ge\gamma_i\|d_i u\|_2^2.
\]
Indeed \(\|d_i u\|_2^2=\langle u-\mu_i u,A_i u\rangle\), and
conditional Poincare followed by Cauchy--Schwarz proves the claim.
The weighted one-form identity on that factor is
\[
 d_i A_i u=
 (\nabla_{i,\mu}^*\nabla_i+\operatorname{Ric}_i
                       -2\operatorname{Hess}_{ii}S)d_i u.
\]
It applies to the exact conditional differential \(d_i u\);
it is not a claim that the same coercivity holds for arbitrary one-forms.
In the full product Weitzenbock identity for \(dAu=df\), testing
component \(i\) against \(d_i u\) retains additional nonnegative
connection energies from directions \(j\ne i\). Bounding only the
mixed Hessian terms gives
\[
 a_i E_i\ge\gamma_i E_i^2-2\sum_{j\ne i}c_{ij}E_iE_j.
\]
After division when \(E_i>0\), this says \(HE\le a\).
If \(E_i=0\), its row inequality is immediate from \(c_{ij}\ge0\).
The nonnegative inverse gives \(E\le H^{-1}a\), so
\[
 \operatorname{Var}_\mu f
 =\langle df,du\rangle
 \le a^\mathsf T H^{-1}a
 \le m^{-1}\sum_i\|d_i f\|_2^2.
\]
This proves the comparison with the full product connection and
Ricci terms included. It also proves the same bound after conditioning,
because all one-site hypotheses were already uniform in every exterior.

## 4. Original-electric comparison and its scope

On the original open square strip, fix the based gauge using the top
horizontal links and all vertical rungs as a tree. Each bottom chord
is then its elementary plaquette cycle. The original electric form on
the rooted cycle space contains every independent bottom-chord
derivative as a nonnegative summand. In particular,
\[
 \Gamma_{\rm original}(f)\ge\sum_i|\nabla_i f|^2.
 \tag{18}
\]
This is a lower bound only; non-Abelian derivatives on the other
original edges are retained in the original form. No claim is made
that the full metric equals the constant harmonic matrix away from
the identity. The companion geometry analysis is at the repository path
    ensemble-laws/docs/constructive/strip-metric-obstruction.md.

Let \(C=-\Delta_{\rm original}\) denote the reduced original electric
operator, \(v=e^{-F_n/\kappa}\), and define the constructed reference
Hamiltonian
\[
 H_{\rm ref}=\frac\kappa2 C+\frac12 CF_n+
                             \frac1{2\kappa}\Gamma_{\rm original}(F_n).
 \tag{19}
\]
Here the final two terms are multiplication operators. The chain rule
gives \(Cv/v=-CF_n/\kappa-\Gamma(F_n)/\kappa^2\), so \(H_{\rm ref}v=0\).
Its ground-state form is
\[
 \langle vf,H_{\rm ref}vf\rangle
 =\frac\kappa2\int\Gamma_{\rm original}(f)v^2\,dH.
 \tag{20}
\]
Equations (17)--(20) prove a centered reference gap at least
\[
 \frac\kappa2\frac1{12\kappa}=\frac1{24}.
 \tag{21}
\]
This holds on the **full rooted cycle space**, hence on the globally
conjugation-invariant physical subspace containing the reference vacuum.
It is not a bound on unrestricted original-link scalar functions
which retain the eliminated gauge coordinates; gauge-charged modes
there can have energies of order \(\kappa\).

The Wilson magnetic potential is not generally the reference potential
in (19). Its residual is
\[
 W=\frac2\kappa\sum_iA_i-\frac12 CF_n-
                            \frac1{2\kappa}\Gamma_{\rm original}(F_n).
 \tag{22}
\]
No bound on \(W\), inverse of the reference correction map, comparison
of actual conditional kernels, or actual Wilson gap is earned here.
Those are required for a transfer of (21) to the nonlinear target.

## 5. API and certificate boundaries

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.vmf_poincare import (
    replay_su2_vmf_poincare_certificate,
    su2_strip_reference_poincare,
    su2_vmf_poincare_bound,
)

scalar = su2_vmf_poincare_bound(Q(4))
reference = su2_strip_reference_poincare(16, Q(1, 64))
assert scalar["arithmetic"]["su2_linear_poincare_gap_lower"] == "1/2"
assert reference["arithmetic"]["rooted_cycle_reference_electric_gap_lower"] == "1/24"
assert replay_su2_vmf_poincare_certificate(reference["certificate"])
```

The scalar field norm and coupling accept positive integers or Fractions,
and reject floats and booleans. The finite strip API accepts integer
lengths \(n\ge1\), without constructing a dense matrix. The
reference certificate attaches and replays the canonical scalar source
at the minimum field \(4/\kappa\). The parameterized written theorem,
not monotonicity of the unknown exact gap, supplies (14) when the actual
field norm is larger.

Canonical replay regenerates the full body, nested source, measure
definition, metric normalizations, bounds, and honesty flags.
All actual Wilson-vacuum, unrestricted original-link scalar-gap,
continuum, and formal analytic-verification flags remain false.

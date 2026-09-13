# Actual plaquette-path gaps with all boundary flux

For a contiguous path of \(m=1,2,3\) edges in one isolated SU(2) square,
condition the **actual square vacuum** on every complementary-link value.
Impose Gauss law only at the path's internal vertices. The resulting
conditional quantum form has gap

\[
 \boxed{\operatorname{gap}_{\mathrm{path}\mid\mathrm{exterior}}
          \ge \frac{m}{33}\quad(\kappa>0).}                 \tag{1}
\]

All boundary-flux representations are retained. This is a finite-graph
theorem, uniform in the coupling and in the specified complementary path.
It does not identify this measure with the conditional vacuum of a block
embedded in a larger lattice.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.plaquette_path import (
    replay_su2_plaquette_path_conditional_certificate,
    su2_plaquette_path_conditional_gap,
)

result = su2_plaquette_path_conditional_gap(Q(1, 64), block_edges=3)
assert result["status"] == "PASS"
assert result["arithmetic"]["full_rotor_gap_lower"] == "4/33"
assert result["arithmetic"]["conditional_quantum_gap_lower"] == "1/11"
assert result["arithmetic"]["conditional_poincare_lower"] == "128/11"
assert result["all_boundary_flux_sectors_included"]
assert replay_su2_plaquette_path_conditional_certificate(result["certificate"])
assert not result["embedded_ambient_block_claim"]
assert not result["theorem_prover_verified"]
```

## 1. The radial source and the unrestricted rotor

Use normalized Haar measure and the single-link Casimir

\[
 C=-\Delta_{SU(2)}=-\frac14\Delta_{S^3},\qquad C_{1/2}=\frac34.
\]

The four original unit electric edges give the rotor extension

\[
 H_4=2\kappa C+\frac2\kappa(2-\operatorname{Tr}U)
     =-\frac\kappa2\Delta_{S^3}+\frac4\kappa(1-\cos\theta),
 \quad U=\cos\theta+i\sin\theta\,\widehat n\cdot\sigma.       \tag{2}
\]

Here \(H_4\) acts on the full \(L^2(SU(2))\). Its smooth bounded potential
and elliptic kinetic operator on the compact connected group give compact
resolvent and a positivity-improving semigroup. Its unique positive ground
state is invariant under conjugation, by symmetry and uniqueness.
Consequently its vacuum and ground energy are exactly those of the neutral
class-function problem, not a chosen reference density.

The dynamically replayed
[neutral plaquette source](gauge-weak-plaquette.md) proves

\[
 E_0\le3,\qquad E_{1,l=0}-E_0\ge\frac{26}{33}.
                                                               \tag{3}
\]

The first bound uses the admissible eikonal Rayleigh trial with an
antipodal cusp; the source supplies its operator-domain justification.
The second includes all radial characters. Neither statement alone is
an unrestricted-rotor gap. The angular comparison below supplies the
missing sectors.

## 2. Complete angular decomposition and Friedrichs domains

Expand in all spherical harmonics \(Y_{lm}(\widehat n)\) on \(S^2\),
where \(l=0,1,\ldots\) and \(m=-l,\ldots,l\). This gives the complete
orthogonal decomposition of \(L^2(SU(2))\); no angular or Peter–Weyl
cutoff is imposed. Multiplication of the radial function by \(\sin\theta\)
transforms its radial Haar norm to the Lebesgue norm on \((0,\pi)\).
In each angular sector, (2) becomes

\[
 L_l=\frac\kappa2
       \left[-\frac{d^2}{d\theta^2}
              +\frac{l(l+1)}{\sin^2\theta}-1\right]
       +\frac4\kappa(1-\cos\theta).                         \tag{4}
\]

Use the Friedrichs form of (4). For \(l=0\) it is the Dirichlet radial
operator of the source. For \(l\ge1\), the centrifugal term is retained
on its form domain; near an endpoint a smooth angular-sector function
has transformed radial behavior \(O(\theta^{l+1})\), or its corresponding
behavior at \(\pi\). These domains embed into the \(l=0\) form domain.
The form comparisons below first hold on compactly supported smooth
radial functions and extend by Friedrichs closure. They do not suppress
endpoint conditions or angular multiplicities.

Since \(\sin^2\theta\le1\), min–max applied on these domains gives

\[
 L_l\ge L_0+\frac\kappa2 l(l+1),\qquad
 E_{l,0}-E_0\ge\kappa\quad(l\ge1).                       \tag{5}
\]

This estimate is particularly useful at large coupling.

## 3. A complementary weak-coupling angular bound

For \(0\le\theta\le\pi\),

\[
 1-\cos\theta\ge\frac{2\theta^2}{\pi^2},\qquad
 \sin\theta\le\theta.
\]

Thus the angular operator (4), extended by zero from the interval to the
positive half-line in its form domain, dominates

\[
 \mathcal O_l=-a\frac{d^2}{d\theta^2}
        +a\frac{l(l+1)}{\theta^2}+b\theta^2-a,
 \qquad a=\frac\kappa2,\quad b=\frac{8}{\kappa\pi^2}.
                                                               \tag{6}
\]

The positive radial function

\[
 u_l(\theta)=\theta^{l+1}
              \exp\left[-\frac12\sqrt{b/a}\,\theta^2\right]
\]

belongs to the half-line Friedrichs domain. Direct differentiation and
the ground-state form identity give its lowest eigenvalue

\[
 \inf\mathcal O_l=(2l+3)\sqrt{ab}-a
                 =(2l+3)\frac2\pi-\frac\kappa2.            \tag{7}
\]

Interval restriction raises this lower bound. Equations (3) and (7), and
the source's elementary \(\pi<22/7\) inequality, therefore give

\[
 E_{l,0}-E_0\ge\frac{10}{\pi}-3-\frac\kappa2
             \ge\frac2{11}-\frac\kappa2\quad(l\ge1).       \tag{8}
\]

Combining (5) and (8), the nonradial gap is at least

\[
 \max\left\{\kappa,\frac2{11}-\frac\kappa2\right\}
       \ge\frac4{33}.                                      \tag{9}
\]

For \(\kappa\le4/33\) the second branch proves (9); for
\(\kappa\ge4/33\) the first branch proves it. Together with the radial
gap in (3), the complete orthogonal angular decomposition proves

\[
 \boxed{\operatorname{gap}_{L^2(SU(2))}H_4\ge\frac4{33}
        \quad\text{for every }\kappa>0.}                    \tag{10}
\]

The rational certificate also records the stronger pointwise lower bound
\(\min(26/33,\max(\kappa,2/11-\kappa/2))\). No exponentials, spectral
samples, asymptotic expansion or numerical angular truncation enter the
arithmetic gate.

## 4. Exact conditional path geometry and the actual vacuum

Orient the square as \(0\to1\to2\to3\to0\), and take the contiguous
block to be its first \(m\) edges. Every rotated or reversed contiguous
path is equivalent by relabeling and inversion. Fix all complementary
edges and put

\[
 U=U_1\cdots U_m,\qquad B=U_{m+1}\cdots U_4,
 \qquad H=UB.                                             \tag{11}
\]

The block has \(m-1\) internal bivalent vertices. Quotienting only their
gauge actions leaves precisely the holonomy \(U\): every internally
invariant scalar function is an arbitrary function of \(U\). The change
from individual links to \(U\) and \(m-1\) internal gauge variables
preserves product Haar measure. No gauge constraint is imposed at the two
endpoints; in particular functions of \(U\) need not be class functions.

Let \(\psi_0\) be the normalized positive source vacuum. The original
square vacuum is \(\psi_0(H)\). Its conditional density in this quotient
is exactly

\[
 d\mu_B(U)=\psi_0(UB)^2\,dU,\qquad
 \int\psi_0(UB)^2\,dU=1.                                \tag{12}
\]

The normalizer is independent of \(B\) by Haar translation. This is a
disintegration of the actual vacuum; it is not an assumption about the
ground state of a frozen bare magnetic operator.

Each original path-link derivative becomes a left- or right-invariant
derivative of \(U\), conjugated by a product of the other path links.
Bi-invariance of the metric preserves its squared norm. Consequently
the original conditional quantum form is exactly

\[
 q_{m,B}(f)=\frac\kappa2\,m\int|\nabla_U f|^2\,d\mu_B.    \tag{13}
\]

The ground-state transform of (2), followed by \(H=UB\), instead has
the coefficient \(2\kappa\) in front of the same integral. Hence (13)
is \(m/4\) times that unrestricted rotor form. Equation (10) proves
(1), or equivalently

\[
 \boxed{\int|\nabla_U f|^2\,d\mu_B
       \ge\frac{2}{33\kappa}\operatorname{Var}_{\mu_B}f,\qquad
 \int\sum_{e\in\mathrm{path}}|\nabla_e f|^2\,d\mu_B
       \ge\frac{2m}{33\kappa}\operatorname{Var}_{\mu_B}f.} \tag{14}
\]

Smooth functions are dense in the conditional form domain, so the claim
includes every square-integrable internally invariant function of finite
energy, for every frozen complementary configuration. Boundary flux and
all angular sectors remain included throughout.

## 5. Replay and limits of the statement

`kappa` accepts a positive integer or `Fraction`. `block_edges` must be an
integer in \(\{1,2,3\}\); booleans, floats and string inputs are refused.
An arbitrary disconnected edge subset is not a path and is not covered.

The result mirrors its sealed payload and attaches the entire canonical
radial certificate with `cutoff=None`. Replay regenerates that source,
the full angular budgets, the exact path geometry, units and every scope
flag. No caller supplies an actual-vacuum or gap premise. A failed source
replay returns `INCONCLUSIVE` and cannot earn the rotor or path claims.
Resealing a changed source, endpoint constraint, arithmetic value or scope
flag is insufficient to pass canonical replay.

Only the written analytic implications are earned; the arithmetic replay
does not formalize the angular decomposition, Friedrichs comparisons or
vacuum disintegration. No ambient-lattice conditional, volume-uniform,
refinement, continuum, or Yang–Mills mass-gap claim is made. In particular
this theorem does not promote the restricted center-even commutator gap
to all boundary-flux sectors of another graph.

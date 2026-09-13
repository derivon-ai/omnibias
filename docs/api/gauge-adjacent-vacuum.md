# Actual vacuum on the seven-edge adjacent-square graph

The consumer generates and canonically replays a complete Fourier inverse,
applies it to the exact reference residual, encloses every omitted spin,
and checks a nonlinear contraction and original-edge curvature bound.

The primary exact witness is \(\kappa=7\), correction radius \(1/10\):
the actual dimensionless Hamiltonian gap is at least \(73/140\).
This is the fixed graph of two adjacent squares, with seven unit electric
edges. It is not a family of arbitrary volumes or a continuum limit.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_adjacent_preconditioned_vacuum,
    replay_su2_adjacent_preconditioned_vacuum_certificate,
)

result = su2_adjacent_preconditioned_vacuum(
    7, cutoff=3, correction_radius=Q(1, 10),
)
assert result["status"] == "PASS"
assert result["actual_vacuum_verified"]
assert result["physical_gap_lower"] == "73/140"
assert replay_su2_adjacent_preconditioned_vacuum_certificate(result["certificate"])
assert not result["uniform_in_volume_claim"]
assert not result["yang_mills_mass_gap_claim"]

secondary = su2_adjacent_preconditioned_vacuum(
    8, cutoff=3, correction_radius=Q(1, 16),
)
assert secondary["physical_gap_lower"] == "1"

insufficient = su2_adjacent_preconditioned_vacuum(5, cutoff=3)
assert insufficient["replayed_reference_inverse_verified"]
assert not insufficient["actual_vacuum_verified"]
assert insufficient["status"] == "INCONCLUSIVE"
```

Coupling and an optional correction radius accept exact integers or
Fraction values; floating-point numbers and booleans are refused.
Coupling and any supplied radius must be positive. The cutoff is an exact
integer at least two because every source-residual mode must be retained.

With correction_radius=None, the implementation chooses
\(r=2\varepsilon\) only after the exact discriminant
\(1-4JB\varepsilon>0\) passes. Here \(J\) is the actual replayed complete
inverse bound, \(B=4/3\), and \(\varepsilon\) is the source-specific
preconditioned residual bound. A supplied radius must pass its own gates.

The witness stores the explicit graph, source inverse certificate, exact
source coefficients, retained residual solve, complete outgoing
coefficients, norm conversion, and all rational inequalities. Its
actual-vacuum flag follows only after those premises and the nonlinear
contraction pass; its finite-graph gap flag additionally requires positive
original-edge curvature. An unsuccessful sufficient gate returns
INCONCLUSIVE and does not assert nonexistence.

Canonical replay rebuilds the nested inverse rather than accepting its
hash alone. Changes to the source, graph, tail, path weights, norms,
coupling, or scope are rejected unless the entire result is the canonical
certificate for those inputs. Infinite-volume, uniform-in-spacing,
continuum, Yang–Mills parent, and formal-verification flags remain false.

## Full analytic proof

12 September 2026. This construction earns the actual logarithmic vacuum,
with every spin included, on the fixed seven-edge SU(2) graph of two
adjacent elementary squares. At \(\kappa=7\), the correction radius
\(r=1/10\) gives \(\operatorname{gap}(aH)\ge73/140\). At \(\kappa=8\),
\(r=1/16\) gives a gap at least one.

These are finite-graph theorems, not volume-uniform inverse estimates or
continuum limits. The argument is written analysis plus exact rational
replay, not a Lean formalization; mathematical novelty has not been
established. It uses the complete-spin inverse in
[adjacent-linearized-inverse.md](gauge-adjacent-resolvent.md) and the
original invariant Fourier product bound from
[invariant-vacuum-review.md](gauge-invariant-vacuum-fourier.md).

## 1. Fixed graph, metric, and Hamiltonian

Use vertices \(0,1,2\) on the lower row and \(3,4,5\) on the upper row.
The original oriented edges are
\[
 (0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5).
\]
The square boundaries in signed one-based indices are
\((1,6,-3,-5)\) and \((2,7,-4,-6)\). Gauge invariance is imposed at all
six vertices, including boundary vertices. There are no matter fields;
all seven electric weights are one.

The dimensionless Hamiltonian and reference are
\[
 aH=\frac{\kappa}{2}C+\frac2\kappa(4-\chi_p-\chi_q),
 \quad C=-\sum_{e=1}^7\Delta_e,\quad
 g=\frac4{\kappa^2},\quad V=\chi_p+\chi_q,\quad S_*=\frac g3V.
 \tag{1}
\]
Here \(\chi\) is the fundamental SU(2) character. The group metric has
fundamental Casimir \(3/4\) and Ricci tensor \(1/2\) times the metric.
Every square has electric energy three, so \(CS_*=gV\).

The invariant graph has paths of lengths \(3,3,1\). Its admissible doubled
spins \(a,b,s\) have the original electric energy
\[
 E_{abs}=\frac{3a(a+2)+3b(b+2)+s(s+2)}4.                 \tag{2}
\]
A sum of three unit-link Casimirs would describe a different operator.

## 2. The original norm and its triangle improvement

Use normalized theta functions
\[
 b_{abs}=\frac{\operatorname{Tr}[P_s(D_{a/2}\otimes D_{b/2})]}{s+1},
 \qquad b_{abs}(I)=1,
\]
Here \(P_s\) projects onto total spin \(s/2\), with
\(\lvert a-b\rvert\le s\le a+b\) and \(a+b+s\) even.
Their seven-link coefficient nuclear norms are
\((a+1)^2(b+1)^2\). This is an original-edge tensor identity:
normalized trivalent tensors flattened across the isolated outer
representation have singular values \(1/\sqrt{a+1}\) and
\(1/\sqrt{b+1}\), respectively, and the paths' bivalent identity/vector
factors give the remaining dimension powers. It does not assume
nuclear-norm invariance under partial transpose.

For a Haar-centered invariant function \(u=\sum u_{abs}b_{abs}\), set
\[
 \begin{split}
 N_i(u)&=\sum_{abs\ne000}
 \frac{i}{2}E_{abs}(a+1)^2(b+1)^2|u_{abs}|,\quad i=a,b,s,\\
 N(u)&=\max(N_a(u),N_b(u),N_s(u)),\qquad
 M(u)=N_a(u)+N_b(u)+N_s(u).
 \end{split}                                          \tag{3}
\]
These are the distinct original-edge anchored sums: bivalent vertices
force equal spins along each path. The locality weight is one.

Every admissible label obeys \(a\le b+s\), and cyclically.
Multiplying these inequalities by the positive coefficient weights and
summing gives
\[
 \boxed{2N(u)\le M(u)\le3N(u).}                         \tag{4}
\]
An \(M\)-inverse bound \(B_M\) therefore gives the \(N\)-inverse bound
\(J=(3/2)B_M\). An individual error already bounded in \(M\) has an
\(N\)-bound half as large. These are different uses of (4).

## 3. Inverse and exact reference residual

Let \(\Pi_H\) remove the Haar constant and define
\[
 T(u,v)=C_0^{-1}\Pi_H\Gamma(u,v),\quad
 K=2T(S_*,\cdot),\quad L=I-K,\quad R_*=T(S_*,S_*).       \tag{5}
\]
The constant is removed before applying \(C_0^{-1}\).
The parent certificate encloses \(L^{-1}\) on the complete original
invariant Fourier space using an exact retained inverse and an omitted
spin bound. The new source consumer builds and canonically replays that
parent; it accepts no supplied inverse number or honesty flag.

Since \(CV=3V\), the Casimir product identity yields
\[
 R_*=(g/3)^2(3C_0^{-1}-1/2)\Pi_H(V^2)
 =g^2\left(-\frac{b_{202}}{24}-\frac{b_{022}}{24}
            +\frac{b_{110}}{27}-\frac{b_{112}}{39}\right).             \tag{6}
\]
Exact character multiplication checks all four coefficients. Their
original anchored norms are
\[
 N_a(R_*)=N_b(R_*)=\frac{17}{3}g^2,\quad
 N_s(R_*)=\frac{26}{3}g^2,\quad M(R_*)=20g^2.            \tag{7}
\]
Thus \(N(R_*)=(26/3)g^2\); the larger family majorant \(14g^2\)
is not needed for this graph-specific residual.

## 4. Applying the complete inverse to this residual

Let \(P\) retain every nonconstant admissible state with
\(\max(a,b,s)\le n\), put \(Q=I-P\), and set
\[
 D=I-PKP,\qquad {\cal R}=\operatorname{diag}(D^{-1},I_Q).
\]
Require \(n\ge2\), so the entire residual (6) lies in \(P\).
The inverse parent earns
\[
 \|H\|_{M\to M}\le z<1,\quad H=I-{\cal R}L,\qquad
 \|L^{-1}\|_{N\to N}\le J.                             \tag{8}
\]
The defect contains retained-to-omitted columns, the exact first omitted
shell, and an all-spin bound on every farther column.

Solve the rational retained system \(Du_0=R_*\). All components of
\(u_0\) are retained, so
\[
 R_*-Lu_0=QKu_0=:h,\qquad Hu_0=h.
\]
The outgoing polynomial \(h\) is evaluated with all coefficients and
their exact cancellations. Since \({\cal R}R_*=u_0\),
\[
 L^{-1}R_*-u_0=(I-H)^{-1}h.
\]
Consequently
\[
 \boxed{N(L^{-1}R_*-u_0)\le e:=\frac{M(h)}{2(1-z)},\qquad
        N(L^{-1}R_*)\le\varepsilon:=N(u_0)+e.}          \tag{9}
\]
The reverse triangle inequality gives the lower bound
\(\max(0,N(u_0)-e)\) as well.

There is no additional preconditioner norm in (9). The inverse series is
for \(I-H={\cal R}L\), and \({\cal R}\) acts as identity on \(h\).
At cutoff one the source has omitted components and these identities
would be incomplete. The API refuses cutoff below two rather than
discarding those components.

## 5. Nonlinear contraction in the same norm

On this original girth-four invariant graph the complete bilinear bound is
\[
 N(T(u,v))\le B N(u)N(v),\qquad B=4/3.                  \tag{10}
\]
Here is why that constant applies. Each nonconstant invariant support has
no degree-one vertex; it contains a cycle of at least four active edges,
hence has electric energy at least three. The spin inequality
\(\sum_ej_e\le2E/3\), the three generator directions, and each of the
two output anchors give \(2/E_{\min}\) in the coefficient product
estimate. The total is \(4/E_{\min}=4/3\). Unitary fusion, nuclear
pinching, and cancellation of the output electric denominator prove the
finite-polynomial inequality; Banach completion includes every spin.
The norm is precisely the original-link norm (3).

Writing \(S=S_*+U\) transforms the logarithmic vacuum equation into
\[
 LU=R_*+T(U,U),\qquad
 U=\Phi(U):=L^{-1}R_*+L^{-1}T(U,U).                    \tag{11}
\]
No target vacuum or target gap was assumed in obtaining the inverse.
On \(N(U)\le r\), equations (8)–(10) give
\[
 N(\Phi(U))\le\varepsilon+JB r^2,\qquad
 \operatorname{Lip}(\Phi)\le2JB r.
\]
The sufficient gates are therefore
\[
 \boxed{\varepsilon+JB r^2\le r,\qquad 2JB r<1.}        \tag{12}
\]
For positive \(J,B,\varepsilon\), such a radius exists exactly when
\[
 {\cal D}:=1-4JB\varepsilon>0.                         \tag{13}
\]
Equality gives a tangent fixed-point radius with contraction one and
does not pass. If (13) holds, the automatic rational choice
\(r=2\varepsilon\) has slack \(\varepsilon{\cal D}\) and contraction
\(1-{\cal D}\). A supplied radius must pass (12) for that radius;
feasibility of a different radius does not license it.

## 6. Actual-vacuum identification and physical gap

The real invariant Haar-centered coefficient space is complete.
At a finite graph its norm ensures absolute reconstruction through
two derivatives. Banach's theorem gives an actual \(C^2\) correction
satisfying (11), with \(N(U)\le r\). Thus \(S=S_*+U\) satisfies
\[
 CS=gV+\Gamma(S,S)-H[\Gamma(S,S)],\qquad
 (C-gV)e^S=E_0e^S,\quad E_0=-H[\Gamma(S,S)].             \tag{14}
\]
The positive function \(e^S\) is smooth by elliptic regularity. For every
smooth scalar \(f\), including noninvariant \(f\),
\[
 \langle e^Sf,(C-gV-E_0)e^Sf\rangle_H
                  =\int e^{2S}|\nabla f|^2\,dH.        \tag{15}
\]
By form closure this identifies \(e^S\) as the true unique groundstate.
The potential in (1) was not replaced by a manufactured one.

Each seed plaquette contributes at most \(g/6\) to each of its four
original-edge Hessian blocks. At most two squares meet an edge, so
the seed Hessian row is at most \(4g/3\). For the correction the
symmetric block majorant is
\[
 M_{ef}=\sum_{\mathbf j}j_ej_f\|U_{\mathbf j}\|_1,\qquad
 \sum_fM_{ef}
 \le\frac23\sum_{\mathbf j}j_eE_{\mathbf j}\|U_{\mathbf j}\|_1
 \le\frac{2r}{3}.                                     \tag{16}
\]
These are covariant Hessian bounds obtained along product geodesics.
The compact-group connection has not been omitted. A block Schur
estimate bounds the Hessian operator norm by the row bound; there is
no extra componentwise factor of three.

The actual density \(e^{2S}/Z\) consequently has weighted Ricci floor
\[
 \rho=\frac12-\frac{8g}{3}-\frac{4r}{3}.                \tag{17}
\]
If \(\rho>0\), integrated Bochner gives its scalar Poincaré gap at least
\(\rho\). The groundstate identity and normalization (1) imply
\[
 \boxed{\operatorname{gap}(aH)\ge\frac{\kappa}{2}\rho.} \tag{18}
\]
This full-scalar lower bound also holds on the neutral gauge-invariant
physical sector. All seven original unit electric weights remain
present.

## 7. Exact witnesses and failed sufficient gates

At cutoff three and \(\kappa=7\), the exact stored fractions imply
\[
 J<13/5,\qquad\varepsilon<8/125.
\]
Choose \(r=1/10\). Even those larger rounded rational upper bounds give
\[
 r-\frac8{125}-\frac{13}{5}\frac43r^2=\frac1{750}>0,
 \qquad 2\frac{13}{5}\frac43r=\frac{52}{75}<1.
\]
With \(g=4/49\),
\[
 \rho=\frac12-\frac{32}{147}-\frac2{15}=\frac{73}{490},
 \qquad \frac{\kappa\rho}{2}=\frac{73}{140}.
\]
The stored exact inequalities are stronger than these simple witnesses.

At \(\kappa=8\), radius \(1/16\) passes and gives \(\rho=1/4\),
hence gap at least one. Approximate displays of the stored rational
bounds are:

| Coupling | Complete \(N\)-inverse upper | Preconditioned residual \(N\)-upper | Outcome |
|---|---:|---:|---|
| \(5\) | \(5.02801418\) | \(0.27423420\) | Nonlinear criterion fails |
| \(6\) | \(3.22459391\) | \(0.12308593\) | Nonlinear criterion fails |
| \(7\) | \(2.56907028\) | \(0.06391633\) | Radius \(1/10\) passes |
| \(8\) | \(2.24156957\) | \(0.03656932\) | Radius \(1/16\) passes |

At five and six, the reference inverse passes but (13) is negative.
This is failure of these sufficient nonlinear bounds, not
nonexistence of a finite-graph vacuum or gap. A passing inverse by
itself does not earn the nonlinear result.

## 8. Replay, fixed scope, and the next missing estimate

The API builds and replays the actual inverse, derives (6) through exact
multiplication, verifies the retained solve, includes every outgoing
coefficient, and checks (9), (12), and (17). Its certificate preserves
the fixed graph, path lengths, source, norm conversion, nested parent,
and all rational bounds. Rehashing a modified parent, source,
representation tail, norm, or scope does not make it a canonical replay.

The theorem concerns one graph. This finite inverse does not bound
overlapping-block interactions over arbitrary spatial volumes. A next
meaningful estimate is a uniform local inverse and a quadratic product
bound that can be assembled over those graphs, or a better verified
reference with a controlled derivative inverse in the same norm where
the current radius gate fails. Neither an all-scale trajectory nor
continuum reconstruction is assumed or earned here.


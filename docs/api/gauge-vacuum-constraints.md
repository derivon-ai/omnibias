# Exact vacuum constraints

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    solve_vacuum_constraints,
    replay_vacuum_constraint_certificate,
)

boundary = solve_vacuum_constraints("su3", 45, Q(298, 45))
assert boundary["status"] == "FEASIBLE"
assert boundary["witness"]["root_rational"] == "32/75"
assert replay_vacuum_constraint_certificate(boundary["certificate"])

# Resolve a target beyond the first-order exponential approximation.
stronger = solve_vacuum_constraints("su3", 45, Q(71, 10), exponent_steps=32)
assert stronger["target_gap_lower"] == "71/10"

# A negative decision concerns this particular sufficient criterion.
outside = solve_vacuum_constraints("su3", 45, Q(298, 45) + Q(1, 10**60))
assert outside["status"] == "INFEASIBLE_CRITERION"
assert replay_vacuum_constraint_certificate(outside["certificate"])
assert outside["gap_absence_claim"] is False
```

The API eliminates the scalar correction radius **over the real numbers**.
It evaluates polynomial constraints at an exact quadratic-algebraic root,
using rational arithmetic throughout. It is not a general differential
system solver. All physical coefficient estimates are reused from the
[invariant-vacuum proof](gauge-invariant-vacuum-fourier.md), not accepted
as arbitrary caller-supplied proof flags.

`gap_method` is `"factorization"` by default or `"curvature"`.
`exponent_steps=m` is a freely chosen positive integer for the inequality
`exp(-Omega) >= (1-Omega/m)**m`, with the strict constraint `Omega<m`.
It is not the earlier API's canonical `floor(Omega)+1`. Larger explicit
orders may certify a tighter lower floor for the same physical model.

The structural keywords match `invariant_vacuum_fourier_family`:
`weighted_incidence_cap`, `minimum_girth`, `max_cycle_length`,
`max_cycle_diameter`, and `decay_base`. This version requires a strictly
positive incidence cap, as well as positive coupling and target gap.
Inputs must be exact integers or fractions. The family statement applies
only to finite simple graphs satisfying the stated all-vertex Gauss,
electric-weight, full-graph girth and interaction hypotheses. Concrete
graph membership is not inferred from a family call.

A certificate's `radius_root=[a,b,D]` means `a+b*sqrt(D)`.
`target_bound_pair=[x,y]` uses the same `D`. `root_rational` and
`target_bound_rational` expose rational values when available. A bound
pair outside its recorded domain is merely a polynomial value, not an
earned positive gap. `target_gap_lower` is earned only on `FEASIBLE`.

`constraint_decision_verified` and canonical replay can be true for
**either** status. They verify the finite algebraic decision.
`feasible`, `finite_gate_verified`, and
`volume_uniform_target_gap_verified` distinguish a positive witness from
an excluded sufficient criterion. An excluded criterion says nothing
about absence of the actual Hamiltonian gap. It may pass with a different
candidate or exponential approximation. Rehashed changes to inputs,
root signs, physical constants, scope or honesty fields are rejected.

All energies use dimensionless `aH`. Continuum, infinite-volume, gap
absence, Yang--Mills and formal-verification flags remain false.
The written analytic implication below is separate from finite arithmetic
replay. No existing v1 vacuum certificate was changed.

## Written implication

11 September 2026. The local vacuum correction theorem can be reduced to
a finite system of rational polynomial inequalities once the group,
coupling, graph family, candidate and exponential approximation order
are fixed. This note eliminates the remaining scalar radius exactly.
It also proves that multiplying the first Wilson reference by a scalar
cannot improve these sufficient bounds. Neither statement decides the
existence of a Yang–Mills continuum theory.

The analytic premises come from
[the invariant vacuum construction](gauge-invariant-vacuum-fourier.md) and
[its conditional factorization bound](gauge-invariant-vacuum-fourier.md#a-gap-from-the-corrected-vacuum-without-positive-global-curvature).
In particular, the Fourier coefficient norm, gauge-invariant support
energy floor, reconstruction of the actual positive groundstate and
conditional Poincare implication must already have been proved. A CSP
solver cannot accept these as arbitrary caller-supplied honesty flags.

## Fixed analytic inputs and normalization

Use the compact-link Hamiltonian
\[
A_\kappa=\frac\kappa2 C+\frac2\kappa
\sum_pv_p(N-\operatorname{Re}\operatorname{Tr}_F U_p),
\qquad g=4/\kappa^2.
\]
The entire electric graph is simple, its girth is at least \(g_0\ge3\),
electric weights are one, all vertices impose Gauss law, and the simple
cycle weights are nonnegative. The structural hypotheses and locality
weights must be checked as in the linked proof. The family statement
quantifies over every finite graph meeting those hypotheses; it does
not supply a thermodynamic or continuum limit. Energies are in
dimensionless \(aH\) units.

Let \(c_*\) denote the first nonzero one-link Casimir,
\(E_{\min}=g_0c_*\), and let the correction norm be the one in this table.
The notation \(\alpha,\beta,\zeta\) here denotes three error coefficients;
\(\beta\) is not the Wilson action coupling.

| Input | SU(2), spin norm \(\mathcal N_b\) | SU(3), Casimir norm \(\mathcal M_b\) |
|---|---:|---:|
| \(c_*\) | \(3/4\) | \(4/3\) |
| \(\rho_0\) | \(1/2\) | \(3/4\) |
| Bilinear constant \(B\) | \(4/E_{\min}\) | \(2/E_{\min}\) |
| Hessian error coefficient \(\alpha\) | \(2/3\) | \(1\) |
| Influence error coefficient \(\beta\) | \(16/3\) | \(3/2\) |
| Log-density oscillation error coefficient \(\zeta\) | \(8/E_{\min}\) | \(4/E_{\min}\) |

For the explicit first reference \(F_0=C_0^{-1}gV\), write its norm bound
as \(A\), and its independently derived Hessian, conditional influence
and single-edge log-density oscillation bounds as \(h_0,\eta_0,\Omega_0\).
For an explicit graph these valid bounds are
\[
\begin{aligned}
A&=g\max_e\sum_{p:e\in p}a_N(\ell_p)v_pb^{D_p},
&a_2(\ell)&=2^{\ell-1},&a_3(\ell)&=3^\ell,\\
h_0&=\frac g{2c_*}\max_e\sum_{p:e\in p}v_p,\\
\eta_0&=g\max_e\sum_{p:e\in p}
 \frac{2N(\ell_p-1)}{\ell_pc_*}v_p,\\
\Omega_0&=g\max_e\sum_{p:e\in p}
 \frac{4N}{\ell_pc_*}v_p.
\end{aligned}
\tag{1}
\]
Here \(D_p\) is the actual ambient line-graph support diameter. Family
caps replace these sums by their proved upper bounds. For example, a
weighted incidence cap \(d\), cycle length cap \(\ell\), and diameter
cap \(D\) give
\[
A=g a_N(\ell)b^Dd,\quad h_0=gd/(2c_*),\quad
\eta_0=2Ngd(\ell-1)/(\ell c_*),\quad
\Omega_0=4Ngd/(g_0c_*).
\tag{2}
\]
The physical derivative bounds in (1)–(2) need no extra factor of \(b\).
Using the unweighted incidence cap is valid even when \(b>1\) is used
in the stronger Fourier norm.

More generally, a fixed candidate \(S_0\) with independently verified
norm bound \(v\), residual bound \(\varepsilon\), and physical bounds
\(h_0,\eta_0,\Omega_0\) gives the same scalar argument below. For
\(S_0=F_0\), these are \(v=A\) and \(\varepsilon=BA^2\). A finite
character candidate permits exact coefficient calculations; a candidate
with infinitely many coefficients also needs a proved tail.

## A polynomial CSP for a specified lower gap

Fix a positive rational target \(\delta\) and a positive integer \(m\).
The integer is part of the problem specification. For a radius \(r\ge0\)
define
\[
\begin{aligned}
P(r)&=r-\varepsilon-2Bvr-Br^2,\\
L(r)&=1-2B(v+r),\\
E(r)&=1-\eta_0-\beta r,\\
\Omega(r)&=\Omega_0+\zeta r,\\
F_m(r)&=\frac{\kappa c_*}{2}E(r)
                 \left(1-\frac{\Omega(r)}m\right)^m.
\end{aligned}
\tag{3}
\]
The exact sufficient constraints are
\[
\boxed{r\ge0,\quad P(r)\ge0,\quad L(r)>0,\quad E(r)>0,\quad
m-\Omega(r)>0,\quad F_m(r)-\delta\ge0.}
\tag{4}
\]
All coefficients are rational and every condition is polynomial in
\(r\). Positive denominators can be cleared without changing signs.
For the Wilson reference,
\(P(r)=r-B(A+r)^2\). The non-strict self-map inequality is intentional:
contraction must be strict, while a closed ball may map to its boundary.

The first three conditions construct the true vacuum. The influence and
oscillation bounds imply a Poincare constant on every conditional block.
Since \(\Omega\ge0\) and \(m>\Omega\),
\[
e^{-\Omega}\ge(1-\Omega/m)^m>0
\]
follows by applying \(e^{-x}\ge1-x\) to \(x=\Omega/m\) and taking a
positive integer power. Thus (4) proves the lower floor \(\delta\) for
the full scalar gap and the charged energy per graph-distance edge, under
the previously proved source and boundary conventions. It optimizes a
sufficient lower floor; it does not optimize or measure the true mass.

One may instead, or additionally, impose the curvature target
\[
\rho_0-2h_0-2\alpha r\ge2\delta/\kappa.
\tag{5}
\]
For \(\delta>0\) this also imposes positive curvature. The two gap
implications remain distinct.

## Exact elimination of the radius

Put
\[
a=1-2Bv,\qquad D=a^2-4B\varepsilon.
\]
Assume \(B>0\), \(v,\varepsilon\ge0\). A radius satisfying the first
three inequalities in (4) exists if and only if
\[
a>0,\qquad D>0.
\tag{6}
\]
Indeed, \(P(r)=-Br^2+ar-\varepsilon\) is increasing below its vertex
\(r_v=a/(2B)\), while \(L(r)>0\) is precisely \(r<r_v\).
Its maximum is \(D/(4B)\). If \(D=0\), its sole zero is the excluded
vertex. If \(a\le0\), no nonnegative radius has positive \(L\).
When (6) holds, all correction radii are exactly
\[
\boxed{r\in[r_-,r_v),\qquad
r_-:=\frac{a-\sqrt D}{2B}.}
\tag{7}
\]
Nonnegativity follows from \(D\le a^2\). At the lower endpoint,
\(P(r_-)=0\) and \(L(r_-)=\sqrt D>0\).
For the first Wilson reference, (6) reduces to \(A<1/(4B)\), with
\[
r_- = \frac{1-2BA-\sqrt{1-4BA}}{2B}.
\tag{8}
\]

Both \(E\) and \(m-\Omega\) decrease with \(r\). On their positive
domain, differentiating the polynomial in (3) gives
\[
F_m'(r)=-\frac{\kappa c_*}{2}
\left[\beta\left(1-\frac\Omega m\right)^m
 +\zeta E\left(1-\frac\Omega m\right)^{m-1}\right]<0.
\tag{9}
\]
The strict sign uses \(\beta>0\), as in the table. Consequently the
entire CSP (4) is feasible over the reals if and only if (6) holds and
\[
\boxed{E(r_-)>0,\qquad m-\Omega(r_-)>0,\qquad F_m(r_-)\ge\delta.}
\tag{10}
\]
The curvature condition (5), when requested, is tested at this same
endpoint because it also decreases with radius. This is exact
elimination, not a radius grid search.

## Rational arithmetic can decide the algebraic endpoint

For rational inputs every polynomial at \(r_-\) reduces to
\(p+q\sqrt D\), with \(p,q\in\mathbb Q\). Multiplication of pairs uses
\[
(p,q)(u,v)=(pu+qvD,\;pv+qu).
\]
The sign is decided without a floating approximation. When \(D>0\),
zero coefficients and equal signs are immediate. When \(p,q\) have
opposite signs, compare the rational numbers \(p^2\) and \(q^2D\):
\[
\operatorname{sign}(p+q\sqrt D)
=\operatorname{sign}(p)\operatorname{sign}(p^2-q^2D).
\tag{11}
\]
Equality in this comparison gives an exact zero. This also handles a
rational-square \(D\); it is not necessary to assume an irreducible
quadratic extension. The strict signs in (10) must remain strict.

Real feasibility and a rational radius witness are separate statements.
If \(F_m(r_-) > \delta\) and the other strict endpoint tests pass, a
rational radius just above \(r_-\) satisfies (4), by continuity and
density. It can be constructed by rational bisection for \(\sqrt D\)
and exact checking of the proposed radius. The search terminates when
these margins are strict. A resource-limited search that has not yet
found it is incomplete, not an infeasibility result.

If \(F_m(r_-)=\delta\), strict monotonicity (9) forces the only feasible
radius to be \(r_-\). A rational radius exists in this equality case
if and only if \(D\) is a rational square. When \(D\) is not a rational
square, the exact algebraic witness still proves real feasibility, but
a rational-radius exporter must report that distinction. The same issue
arises if a required curvature target is attained exactly at an
irrational endpoint. More generally, all requested non-strict target
constraints must have slack to justify the rational-density argument.
The zero-residual case permits \(r_-=0\) as an exact singleton correction
ball; an API requiring strictly positive radii must handle that policy
separately.

## A stronger rational witness at SU(3), kappa 45

For four-edge cycles, actual girth at least four, incidence cap four and
\(b=1\), SU(3) at \(\kappa=45\) has
\[
A=16/25,\quad B=3/8,\quad h_0=2/675,\quad
\eta_0=2/75,\quad\Omega_0=4/225.
\]
The lower root is rational:
\[
D=1/25,\quad r_-=32/75,\quad L(r_-)=1/5,
\quad 2B(A+r_-)=4/5.
\]
The self-map equality is allowed. The remaining bounds are
\[
\eta=2/3,\qquad\Omega=76/225.
\]
Hence \(m=1\) attains the exact target
\[
\boxed{F_1(r_-)=10(1-76/225)=298/45.}
\tag{12}
\]
For the same actual-vacuum construction, choosing \(m=8\) improves the
certified exponential approximation and gives
\[
\boxed{F_8(r_-)=10(431/450)^8>298/45.}
\tag{13}
\]
Both are finite rational inequalities. Increasing \(m\) changes the
lower approximation, not the vacuum candidate or the theory. Even the
limit \(10e^{-76/225}\) is a sufficient gap floor, not an identification
of the exact spectrum.

For this cubic family, write \(x=\kappa^2\). Positive factorization
with some radius is possible precisely when
\[
\boxed{x^2-1332x-1190700>0,
\quad\text{equivalently }x>666+36\sqrt{1261}.}
\tag{14}
\]
To verify this elimination, the fixed-point prerequisite is \(x>1944\).
The influence endpoint is \(r_\eta=2/3-36/x\), below the correction
vertex on that range. At that endpoint,
\[
B(A+r_\eta)^2-r_\eta
=-\frac{x^2-1332x-1190700}{2x^2}.
\]
Strict negativity is exactly \(r_-<r_\eta\). The positive root in
(14) exceeds 1944, and the oscillation endpoint
\(r_\Omega=4/3-48/x\) exceeds \(r_\eta\). Thus \(m=1\) already
gives a positive floor whenever (14) passes. This explains why 45 is
the first positive integer coupling passing this particular family
criterion; it does not locate a physical phase boundary.

## Scalar rescaling of the first reference cannot help

Consider \(S_0=tF_0\), \(t\ge0\), using only
\[
v=tA,\quad \varepsilon=|1-t|A+Bt^2A^2.
\]
The correction inequalities become
\[
|1-t|A+B(tA+r)^2\le r,\qquad 2B(tA+r)<1.
\tag{15}
\]
Suppose all seed constants are nonnegative and
\[
h_0\le\alpha A,\qquad\eta_0\le\beta A,
\qquad\Omega_0\le\zeta A.
\tag{16}
\]
These conditions hold for (1)–(2). The physical bounds for the scaled
candidate are \(th_0+\alpha r\), \(t\eta_0+\beta r\), and
\(t\Omega_0+\zeta r\).

For \(0\le t\le1\), set \(z=tA+r\). Equation (15) is
\(A+Bz^2\le z\), so \(z\ge A\). At \(t=1\), choose
\(r_1=z-A\ge0\). The self-map and contraction conditions are exactly
preserved. For each pair \((p_0,c)\) from (16),
\[
(tp_0+cr)-(p_0+cr_1)=(1-t)(cA-p_0)\ge0.
\]
Thus every physical upper bound improves or stays equal.

For \(t\ge1\), keep the same radius \(r_1=r\) and replace \(t\) by
one. Both terms on the left of (15) decrease, contraction improves,
and each nonnegative seed bound decreases. Therefore every passing
scaled candidate has a feasible \(t=1\) candidate with no larger
Hessian, influence or oscillation bounds. Its curvature floor and its
exact-exponential factorization floor are no smaller. For an explicit
fixed integer \(m\), the same statement holds for the polynomial floor
\(F_m\), since the improved oscillation remains below that same \(m\).

A generic rule that recomputes \(m=\lfloor\Omega\rfloor+1\) is not
globally monotone: its rational lower estimate jumps upward when
\(\Omega\) crosses an integer. For example the estimates at
\(\Omega=99/100\) and \(101/100\) are \(1/100\) and
\((99/200)^2\), respectively. This is an approximation artifact.
Keep \(m\) fixed when comparing target floors, or compare the exact
exponential. For the current simple-cycle constants an additional
observation avoids this issue: \(\Omega_0\le\eta_0\) and
\(\zeta\le\beta\), since cycle lengths and girth are at least three.
Thus any passing factorization gate has \(\Omega\le\eta<1\), so its
canonical order is one. Larger explicit orders such as (13) still help.

## What could improve the candidate rather than merely rescale it

A next candidate can include actual connected character terms:
\[
S_1=F_0+T(F_0,F_0).
\]
Its exact equation residual is
\[
-2T(F_0,T(F_0,F_0))-T(T(F_0,F_0),T(F_0,F_0)).
\]
These terms can be computed in the existing rational representation
algebra on a fixed finite graph, then
bounded locally with all omitted coefficients accounted for. Direct
coefficient cancellation, sharper support energies, or a verified
preconditioner for \(I-2T(S_1,\cdot)\) could improve the certificate.
None is earned by introducing a free scalar alone.

In particular, the generic estimates
\(v=A+BA^2\) and
\(\varepsilon=2B^2A^3+B^3A^4\) for this second candidate still give
\[
v+\varepsilon-Bv^2=A.
\]
Putting \(z=v+r\) reduces its self-map condition to the same
\(A+Bz^2\le z\). Without more precise residual or physical derivative
information, this iteration does not enlarge the norm feasibility
window. Genuine gains require a newly proved estimate.

An exact negative CSP result is therefore
`INFEASIBLE_CRITERION`: no radius in this specified sufficient
criterion achieves the target. It is neither absence of a physical
mass gap nor a disproof of confinement. The continuum existence,
cutoff scaling and quantum-field-theory axioms remain separate parent
obligations.

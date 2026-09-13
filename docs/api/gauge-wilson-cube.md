# SU(2) attached cube with earned curvature feedback

12 September 2026. A dependent cube face can be controlled by its other
faces, retaining a quantified old-curvature term. For one cube glued to
an old periodic cubic vacuum, the existing actual moment bound turns
that term into a coupling-independent vacuum-energy increment at most
\(45\). This is a finite graph family with arbitrary old torus volume,
not a gap theorem or a uniform renormalization induction.

## Exact family and operator

Start with the [SU(2) periodic cubic Hamiltonian](gauge-wilson-large-field.md)
on any torus of integer side \(L\ge3\), at \(\kappa>0\). Select
any elementary old face as a cube's bottom. Introduce four **entirely new
vertices**, four vertical links and four top links, and add exactly four
side-face potentials and one top-face potential. All new links have
electric coefficient \(\kappa/2\); all five new Wilson actions have
magnetic coefficient \(2/\kappa\). Original Haar coordinates, unit
electric weights, fundamental Casimir \(3/4\), and neutral Gauss law
are retained.

The resulting graph is a torus with a fresh cube attached along one face.
It is not a larger ordinary periodic cubic lattice or an embedded cube
removed from an arbitrary exterior. In particular new vertices are not
silently identified with existing neighbors and no extra potentials are
added. Energy is dimensionless \(aH\).

## Noncommutative cube identity

Label cyclic bottom and top links \(B_i,T_i\), and vertical links
\(V_i\), with indices modulo four. Define the side holonomies and
bottom transports

\[
P_i=B_iV_{i+1}T_i^{-1}V_i^{-1},\qquad
C_i=B_0\cdots B_{i-1},\quad C_0=I.
\]

Solving each side identity for \(T_i\) and telescoping the vertical
links gives the exact **ordered** identity

\[
V_0(T_0T_1T_2T_3)V_0^{-1}
=\left[\prod_{i=0}^3 C_iP_i^{-1}C_i^{-1}\right]
 B_0B_1B_2B_3.
\]

For SU(2), \(A(U)=2-\operatorname{Tr}U\) satisfies
\(\|U-I\|_F^2=2A(U)\). Unitary invariance, the telescoping triangle
inequality for products, and Cauchy--Schwarz now prove

\[
\boxed{A_{\rm top}\le5\left(A_{\rm base}+
                         \sum_{i=0}^3A_{P_i}\right).}
\]

The transports and base action are essential. Neither an unordered
product of the sides nor a side-only action bound gives this identity.

## Weighted physical attachment

The preceding inequality gives the form comparison

\[
H_{\rm cube}\le H_{\rm old}+\frac\kappa2\sum_{\rm new}C_e
        +\frac{12}\kappa\sum_{\rm sides}A_p
        +\frac{10}\kappa A_{\rm base}.
\]

The four side faces have fresh edges in cyclic order. Apply the
[normalized attachment isometries](gauge-wilson-attachment.md) to the
comparison Hamiltonian with side magnetic weight \(w=6\). Their exact
per-stage energy is

\[
e_{t,w}=\frac{3\kappa t}2m(4t)+\frac{4w}\kappa(1-m(4t))
 \le\frac{3\kappa t}2+\frac{3w}{2\kappa t}.
\]

At \(\kappa t=2\), this is at most \(15/2\). The composed physical
isometry \(J\) preserves every old multiplication observable. Hence,
for **any** smooth old gauge-invariant multiplication potential with the
stated old electric terms,

\[
\boxed{J^*H_{\rm cube}J\le
       H_{\rm old}+30I+(10/\kappa)A_{\rm base}.}
\]

This is an operator-form comparison, not invariance of the image of
\(J\) or an identity for the true new vacuum. In an old actual vacuum
with an earned mean bound \(\mathbb E A_{\rm base}\le M\), it gives
\(0\le E_{\rm cube}-E_{\rm old}\le30+10M/\kappa\).

For the old periodic family, the canonical
[actual Wilson moment certificate](gauge-wilson-large-field.md) proves
\(M=\min(3\kappa/2,2)\). The source is included and replayed;
the bound is not a user-supplied arbitrary moment. Thus

\[
\delta=\min\left(30+\frac{10M}\kappa,
 4\min(3,4/\kappa)+8/\kappa,\ 20/\kappa\right)\le45.
\]

The latter two alternatives are the ordinary sequential closed-face and
global Haar fallbacks. All bounds concern the same actual graph and
operator.

## Actual subtraction, localization, and the noniterable premise

There are five added actions and their maximum added-face incidence is
two. The general attachment moment and cutoff proof therefore applies
with \(n=5,d=2\) and the improved \(\delta\), writing
\(q(f)=\langle f,(H_{\rm cube}-E_{\rm cube})f\rangle\):

\[
\begin{split}
\mathbb E_{\rm cube}\sum_{\rm added}A_p
 &\le\min(20,\kappa\delta/2),\\
\|\chi_g\psi_{\rm cube}\|^2
 &\ge\max(0,1-\kappa\delta/s),\\
q(f)&\ge(s/\kappa-\delta)\|\chi_bf\|^2
                -\frac{1936\kappa}{49s}\|f\|^2.
\end{split}
\]

The report also gives the refined vacuum localization cost using
\(m_*:=\min(\kappa\delta/2,10)\) and
\(\mathbb E\Gamma S\le2m_*(4-m_*/5)\).

After attachment the actual graph is no longer the old periodic family,
so its per-face mean theorem cannot simply be reapplied. In fact, from
an old mean \(\le C\kappa\), the above increment and positivity only
give a new-face upper budget \(C_{\rm next}=15+5C\). It has no
nonnegative invariant constant. This is a failure of that upper-bound
induction, not evidence that actual moments diverge. Old moments are
preserved by \(J\) on trial states, not by the change of true vacuum.
Stable feedback on genuine cubic blocks remains unproved.

## Replay

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_wilson_attached_cube,
    replay_su2_wilson_attached_cube_certificate,
)

t = Q(1, 64)
result = su2_wilson_attached_cube(t**5, action_threshold=t**4)
a = result["witness"]["arithmetic"]
assert result["status"] == "PASS"
assert a["vacuum_increment_upper"] == "45"
assert a["vacuum_subtracted_bad_support_floor"] == "19"
assert Q(a["good_localized_vacuum_norm_squared_lower"]) == Q(19, 64)
assert Q(a["universal_ims_error_upper"]) < 1
assert replay_su2_wilson_attached_cube_certificate(result["certificate"])
assert not result["continuum_claim"]
assert not result["yang_mills_mass_gap_claim"]
```

The canonical replay verifies its embedded sources, geometry, rational
comparison, moment and cutoff calculations, and all scope fields. It
earns neither Lean verification of the analytic proof nor a physical
spectral gap. Coupling independence of this finite attachment cost does
not identify a physical spacing trajectory or reconstruct a QFT.

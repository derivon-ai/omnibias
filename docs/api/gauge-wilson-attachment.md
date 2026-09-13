# SU(2) Wilson attachment and actual exterior subtraction

12 September 2026. This page proves a coupling-independent vacuum-energy
increment for an ordered family of Wilson faces. Each inexpensive stage
must introduce a fresh link. It supplies an actual exterior subtraction,
not a spectral gap, an invariant trial subspace, or a continuum theorem.
The variational, conditional-normalization and IMS methods are standard;
no literature-priority claim is made.

## Operator and scope

On any finite simple link graph take

\[
H_{\rm old}=\frac\kappa2\sum_{e\in E_{\rm old}}C_e+V_{\rm old},
\qquad
H_{\rm new}=H_{\rm old}+\frac\kappa2\sum_{e\in E_{\rm new}}C_e
 +\frac2\kappa\sum_{p\in P_{\rm added}}A_p,
\quad A_p=2-\operatorname{Tr}U_p.
\]

Here \(\kappa>0\), \(C_{1/2}=3/4\), and each face is a simple
four-edge cycle. The old potential is any smooth real gauge-invariant
**multiplication** function of the old links. The old electric weights are
one; nonlocal old kinetic terms are not covered. The graph contains the
listed old edges, none of the introduced edges, and may contain arbitrary
other old edges. Only the listed new Wilson potentials are added.
The Hilbert space satisfies neutral Gauss law at every vertex. Positivity
and uniqueness of the compact scalar vacuum make its energy equal the
physical vacuum energy, as in the
[actual Wilson moment proof](gauge-wilson-large-field.md).
Energies use the [Kogut--Susskind Hamiltonian convention](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.11.395)
in dimensionless \(aH\) units.

## Exact normalized attachment

First add one face with at least one fresh edge. Write old links as \(y\)
and new links as \(x\), and define

\[
\phi_t(x\mid y)=Z_t^{-1/2}\exp(t\operatorname{Tr}U_p),\qquad
(J_t\eta)(x,y)=\phi_t(x\mid y)\eta(y).
\]

Integrating any fresh Haar link makes the entire loop Haar, even when the
fresh edges are not contiguous. Consequently \(\int\phi_t^2dx=1\)
for every old configuration. Thus \(J_t\) is a physical isometry and
\(\int\phi_t\nabla_y\phi_tdx=0\). For every old quadratic-form
state, the latter identity cancels all old-state cross derivatives.
It does **not** remove the derivatives of the trial with respect to old
links. All four original edge derivative costs must be included.

Let \(m(c)\) be the mean of \(q_0\) on \(SU(2)\) weighted by
\(e^{cq_0}\), with \(\operatorname{Tr}U=2q_0\). Haar integration
by parts gives \(c\mathbb E(1-q_0^2)=3m(c)\). The
[one-link differential comparison](gauge-wilson-large-field.md) proves
\(0\le m(c)\le1\) and \(1-m(c)\le3/(2c)\).
The exact compression, as quadratic forms, is therefore

\[
J_t^*H_{\rm new}J_t=H_{\rm old}+e_tI,\qquad
e_t=\frac{3\kappa t}2m(4t)+\frac4\kappa(1-m(4t)).
\]

Taking \(t=1/\kappa\) gives \(e_t\le3\). A constant fresh-link
trial separately gives \(4/\kappa\). The new electric and magnetic
terms are nonnegative. Hence the **actual** vacuum energies obey

\[
\boxed{0\le E_{\rm new}-E_{\rm old}\le\min(3,4/\kappa).}
\]

Exact compression is weaker than subspace invariance and makes no
assertion about the excited spectrum.

## Ordered stages and the closed-face obstruction

Conditional isometries compose: a later face may reuse an edge introduced
earlier, provided it introduces at least one edge at its own stage.
Thus \(n\) such stages cost at most \(n\min(3,4/\kappa)\).
If a stage has no fresh edge, the normalization argument does not apply.
The safe bound is then \(0\le2A_p/\kappa\le8/\kappa\).
Let \(f\) count fresh stages and \(r=n-f\) the others. A second trial,
constant on all initially absent edges, has mean character zero on every
face touching an initially new edge. If \(h\) is that face count, set

\[
\delta=\min\left(
 f\min(3,4/\kappa)+8r/\kappa,
 (4h+8(n-h))/\kappa\right).
\]

Then \(0\le E_{\rm new}-E_{\rm old}\le\delta\) still holds.
`PASS` means \(r=0\); `INCONCLUSIVE` means the inexpensive complete
attachment route failed, although these fallback bounds remain valid.
This graph gate does not certify that the supplied faces are all faces
of a cubic block. A cube's final face is a simple example of a dependent
stage. Full open cubic blocks cannot be built entirely by fresh stages.

## Actual vacuum moments and localized energy

Put \(S=\sum_pA_p\) and let \(d\) be the maximum number of added
faces meeting one edge, calculated from the actual pattern. Do not replace
it by the cubic value on an arbitrary graph. Positivity gives

\[
\mathbb E_{\rm new}S\le M:=\min(4n,\kappa\delta/2).
\]

For \(0<s\le4n\), take a Lipschitz angle \(\theta(S)\), zero
below \(s/2\), equal to \(\pi/2\) above \(s\), and linear
between them; set \(\chi_g=\cos\theta\), \(\chi_b=\sin\theta\).
Using \(\pi<22/7\),

\[
\begin{split}
\mathbb P(S\ge s)&\le\min(1,M/s),\\
\|\chi_g\psi_{\rm new}\|^2&\ge\max(0,1-2M/s),\\
\mathrm{IMS}&\le\frac{968\kappa d}{49s},\\
\langle f,(H_{\rm new}-E_{\rm new})f\rangle
 &\ge(s/\kappa-\delta)\|f\|^2
 \quad\text{if }\operatorname{supp}f\subset\{S\ge s/2\}.
\end{split}
\]

The last line subtracts the actual full vacuum energy: the old energy
cancels through \(\delta\), with no ambient-volume estimate. For
general states the partition bound is
\(q(f)\ge(s/\kappa-\delta)\|\chi_bf\|^2-\mathrm{IMS}\|f\|^2\).
This controls the bad component only and is not a gap on all excitations.

For the refined vacuum cost use
\(\Gamma S\le d(4S-\sum_p A_p^2)\) and Jensen. With
\(m_*:=\min(\kappa\delta/2,2n)\),

\[
\mathbb E\Gamma S\le d m_*(4-m_*/n),\qquad
C_{\rm vac}\le\frac\kappa2\left(\frac{22}{7s}\right)^2
             d m_*(4-m_*/n).
\]

The normalized good vacuum has energy above \(E_{\rm new}\) at most
\(C_{\rm vac}/(1-2M/s)\) when this denominator is positive. These
are unconditional statements about the true new vacuum. No uniform
probability or Poincare constant under exterior conditioning follows.

## Exact replay

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_wilson_attachment,
    replay_su2_wilson_attachment_certificate,
)

t = Q(1, 64)
result = su2_wilson_attachment(
    t**5, [[0, 1], [1, 2], [2, 3]], [[0, 1, 2, 3]],
    action_threshold=t**4,
)
a = result["witness"]["arithmetic"]
assert result["status"] == "PASS"
assert a["vacuum_increment_upper"] == "3"
assert a["vacuum_subtracted_bad_support_floor"] == "61"
assert Q(a["universal_ims_error_upper"]) < 1
assert replay_su2_wilson_attachment_certificate(result["certificate"])
assert not result["spectral_gap_claim"]
assert not result["continuum_claim"]
```

The certificate replays canonical graph gates, all rational arithmetic,
the complete witness and scope. Its transcendental backend is `not_used`:
the analytic comparison above supplies the universal bound, not a sampled
transcendental evaluation. No Lean flag is earned by this arithmetic
certificate. The separate ensemble-laws formal bundle checks supporting
rational lemmas only; it does not formalize the Haar or variational proof.

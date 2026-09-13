# Actual finite-strip normalized-kernel tails

For an isolated open strip of \(n\ge2\) square plaquettes, the actual SU(2)
vacuum has an explicit tail bound for its joint kernel normalized by its own
two block marginals. For every nonempty proper partition \(I,J\) of the cycles,

\[
\int_{F_0\ge R\kappa}\frac{\rho_\kappa^2}
 {\rho_{\kappa,I}\rho_{\kappa,J}}\,dH^{\otimes n}
\le \exp\left(1600000n^2+13120n-\frac{17R}{330}\right),
\qquad 0<\kappa\le\frac1{64},\quad R>0.
\tag{1}
\]

Here \(\rho_\kappa\) is the squared normalized positive ground state, all
representations are included, and

\[
F_0=\sum_{i=1}^n8(1-\cos(\theta_i/2)),\qquad
\operatorname{Tr}U_i=2\cos\theta_i,\quad 0\le\theta_i\le\pi.
\]

The constants grow with \(n\). Equation (1) is uniform in the coupling at
each fixed \(n\); it is not uniform in volume or in frozen exterior links.
Its integral is the **squared** Hilbert--Schmidt tail. It does not alone
give a maximal-correlation bound or a numerical coupling threshold for
comparison with a Gaussian.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.strip_kernel_tail import (
    su2_strip_normalized_kernel_tail,
    replay_su2_strip_kernel_tail_certificate,
)

row = su2_strip_normalized_kernel_tail(
    Q(1, 2**48), 4, 500_000_000,
    retained_cycles=[1, 2], target=Q(1, 1024),
)
assert row["status"] == row["tail_target_status"] == "PASS"
assert not row["tail_region_has_zero_haar_measure"]
assert row["arithmetic"]["squared_hs_tail_exponent"] == "-3468160/33"
assert row["factored_dyadic_squared_hs_tail_upper"]["negative_integer_exponent"] == 105095
assert replay_su2_strip_kernel_tail_certificate(row["certificate"])
assert not row["uniform_in_volume_claim"]
assert not row["maximal_correlation_upper_verified"]
```

Cycle IDs are \(1,\ldots,n\). The default partition retains
\(1,\ldots,\lfloor n/2\rfloor\). Rational inputs must be `int` or
`Fraction`; booleans and floats are rejected. A target is tested with a
factored outward dyadic upper bound; no very small floating exponential or
very large power of two is constructed. Analytic `status` and
`tail_target_status` have different meanings. A positive coupling outside
the proved range returns `INCONCLUSIVE`. If \(R\kappa\ge8n\), the region
has Haar measure zero, including equality, and its exact effective bound is
zero. This observation does not promote an out-of-range coupling.

## 1. Actual original-link geometry and the energy source

Let \(B_i,T_i\) be bottom and top links directed right, and let
\(V_0,\ldots,V_n\) be the upward rungs. The coherently oriented elementary
face is

\[
P_i=B_iV_iT_i^{-1}V_{i-1}^{-1},\qquad i=1,\ldots,n.
\]

There are \(3n+1\) original links, \(2n+2\) vertices and \(n\) cycles. Set
all top links and rungs to the identity using gauge transformations away
from a fixed top root. This is a spanning tree, and each remaining bottom
chord is exactly its elementary plaquette \(U_i\). Tree reduction preserves
product normalized Haar. The rooted reduced space is the full
\(L^2(SU(2)^n)\), with residual simultaneous conjugation imposed for physical
observables. We retain this full space when forming the kernel.

The operator is the Friedrichs realization of

\[
H_\kappa=\frac\kappa2\sum_{e=1}^{3n+1}C_e+
\frac2\kappa S,\qquad S=\sum_i A(U_i),\quad A(U)=2-\operatorname{Tr}U,
\quad C=-\Delta,\quad C_{1/2}=\frac34.
\tag{2}
\]

The original scalar operator is positivity improving on a connected compact
product, so its ground state is unique, positive, smooth and invariant under
every vertex gauge action. The rooted reduced operator is also smooth and
uniformly elliptic: its bottom-link rows already supply each independent
one-link gradient. Its unique positive ground state identifies the original
physical vacuum. There is no assumed vacuum or gap input.

For each cycle take the same normalized central one-plaquette trial with
energy at most three, supplied by the canonical
[weak-plaquette source](gauge-weak-plaquette.md). The product of these trials
is normalized in the tree coordinates. Its diagonal original electric
energies are the four-link energies of the individual factors. Distinct
central face factors interact in the electric form only on a shared rung.
In tree gauge these terms are inner products of independent central scores
in the same Lie-algebra frame. Each score has Haar mean zero. Thus every
cross expectation vanishes. Equivalently one may use the actual central
one-plaquette ground states, whose energies obey the same canonical cap.
The potential is additive. Consequently, for **every** \(n\),

\[
0\le E_\kappa\le3n,\qquad
\int S\rho_\kappa\,dH^{\otimes n}\le\frac{3n\kappa}{2}.
\tag{3}
\]

The certificate rebuilds the one-plaquette source and uses its sealed energy
upper bound. Equation (3)'s product construction is part of the written
proof. A caller cannot supply an arbitrary energy or spectral gap.

For general noncentral reduced functions the tree-edge derivative rows
also contain transported conjugations of several cycle variables. We do
**not** bound the full nonlinear reduced co-metric by the tridiagonal
harmonic matrix. The following estimates concern additive central phases
and are proved on the original links.

## 2. Global phase inequalities and barriers

Set \(f(\theta)=8(1-\cos(\theta/2))\). One-link radial calculus gives

\[
\Gamma f=A,\qquad Cf=\sec(\theta/2)-\frac52\cos(\theta/2)\ge-\frac32.
\]

Each face has four original derivative rows. Only adjacent faces share a
link, and the mixed inner product is bounded by
\(\sqrt{A(U_i)A(U_{i+1})}\). Cauchy's inequality and the path degree at most
two therefore give

\[
2S\le\Gamma F_0\le6S,\qquad CF_0=4\sum_i Cf_i,
\qquad\Gamma S\le6S,\qquad\Delta S=6n-3S.
\tag{4}
\]

Use the smooth approximation

\[
f_\epsilon=8\left(\sqrt{1+\epsilon^2}
 -\sqrt{\cos^2(\theta/2)+\epsilon^2}\right),\qquad\epsilon=\frac1{64}.
\]

Rationalizing the difference of square roots and differentiating gives

\[
\frac{32}{33}f\le f_\epsilon\le f\le2A,
\qquad\Gamma f_\epsilon\le A,\qquad Cf_\epsilon\le\frac3{2\epsilon}=96.
\tag{5}
\]

The same original-link argument gives \(\Gamma F_\epsilon\le6S\).
For \(a=4/5,b=3/2\), direct application of (2) gives

\[
\frac{(H_\kappa-E_\kappa)e^{-aF_\epsilon/\kappa}}
 {e^{-aF_\epsilon/\kappa}}
\ge\frac{2S}{25\kappa}-\frac{783n}{5},
\qquad
\frac{(H_\kappa-E_\kappa)e^{-bF_0/\kappa}}
 {e^{-bF_0/\kappa}}
\le\frac{9n}{2}-\frac{S}{4\kappa}.
\tag{6}
\]

These have the required opposite signs outside \(S\le2048n\kappa\),
where \(2S/\kappa-E_\kappa>0\). The lower barrier has a favorable cusp at a
face antipode. The plaquette map is a submersion to a three-dimensional
group. In its normal distance \(r\), the lower barrier is a positive
constant times \(e^{cr/\kappa+O(r^3/\kappa)}\); the singular \(1/r\)
Laplacian term contributes negatively to its Schrödinger residual. This
term is locally square integrable and the flux across a removed tube is
\(O(r^2)\). The weak subsolution inequality therefore extends through the
cusp; finite intersections are handled by the same weak chain rule. The
upper barrier is smooth. The weak maximum principle on the complement of
the core does not require choosing a regular value of \(S\).

## 3. Explicit core estimate on the original product

Work on \(SU(2)^{3n+1}\), of dimension \(d=9n+3\le12n\), with nonnegative
Ricci curvature. Write \(h=\log\psi_\kappa\), \(q=|\nabla h|^2\). Then

\[
\Delta h=v-q,\qquad v=\frac{4S}{\kappa^2}-\frac{2E_\kappa}{\kappa}.
\]

Put \(B=8192n\), \(\zeta=1-S/(B\kappa)\), \(\eta=(\zeta_+)^2\), and
\(Q=\eta q\). At a positive maximum of \(Q\), Bochner's identity yields

\[
\frac2d(Q-\eta v)^2\le
 2\eta^{3/2}\sqrt Q\,|\nabla v|
 +2Q^{3/2}\frac{|\nabla\eta|}{\sqrt\eta}
 +Q\left(-\Delta\eta+\frac{2\Gamma\eta}{\eta}\right).
\tag{7}
\]

On the cutoff support (3)–(4) give

\[
|v|\le\frac{32774n}{\kappa},\quad
|\nabla v|\le4\sqrt{49152n}\,\kappa^{-3/2},\quad
\frac{|\nabla\eta|}{\sqrt\eta}\le2\sqrt{6/(8192n)}\,\kappa^{-1/2},
\]
\[
-\Delta\eta+\frac{2\Gamma\eta}{\eta}
=\frac{2\zeta(6n-3S)}{B\kappa}+
 \frac{6\Gamma S}{B^2\kappa^2}
\le\frac{12n+36}{8192n\kappa}.
\tag{8}
\]

If \(A=\kappa Q\ge2^{18}n^2\), then \(A\ge2(32774n)\). Equations
(7)–(8) imply

\[
\frac{\sqrt A}{24n}\le
\frac{8\sqrt{49152n}}A+4\sqrt{\frac6{8192n}}
 +\frac{12n+36}{8192n\sqrt A}.
\tag{9}
\]

The left side is at least \(64/3\). The three right terms are less than
\(1/100,1/8,1/100\), respectively: use \(49152<222^2\), \(n\ge1\), and
the assumed lower bound on \(A\). This contradicts \(64/3>29/200\).
On the core, \(\zeta\ge3/4\), whence

\[
|\nabla\log\psi_\kappa|\le\frac{2048n}{3\sqrt\kappa}.
\tag{10}
\]

Gauge fixing preserves the value of the vacuum. In the top-and-rung tree
gauge, follow simultaneous radial geodesics on the bottom **original**
links to the identity. No prefix products occur: each bottom chord equals
one face. The path remains in the core, and its length is

\[
\left(\sum_i(2\theta_i)^2\right)^{1/2}
\le\pi\sqrt S\le64\pi\sqrt{n\kappa}.
\]

Combining this with (10), \(\pi<22/7\), and \(n^{3/2}\le n^2\), gives

\[
\left|\log\frac{\psi_\kappa(U)}{\psi_\kappa(I)}\right|
\le200000n^2\quad\text{on the core}.
\tag{11}
\]

The comparison principle applied to (6) now yields, everywhere,

\[
L e^{-3F_0/(2\kappa)}\le
 h_0:=\frac{\psi_\kappa}{\psi_\kappa(I)}
\le U e^{-4F_\epsilon/(5\kappa)},\qquad
L=e^{-200000n^2},\quad U=e^{200000n^2+3277n}.
\tag{12}
\]

For the upper core prefactor use \(F_\epsilon\le2S\le4096n\kappa\)
and \(4\cdot4096/5<3277\). If the core fills the product, its direct
estimate already proves (12).

## 4. Own marginal denominators and the tail

Let \(H_I=\int h_0^2\,dH_J\) and \(H_J=\int h_0^2\,dH_I\).
The normalization of the actual ground state cancels **exactly**:

\[
\frac{\rho^2}{\rho_I\rho_J}=\frac{h_0^4}{H_IH_J}.
\]

Set \(Z_c(\kappa)=\int_{SU(2)}e^{-cf/\kappa}\,dH\). From (12),

\[
H_IH_J\ge L^4 e^{-3F_0/\kappa}Z_3(\kappa)^n,
\quad
\frac{\rho^2}{\rho_I\rho_J}
\le (U/L)^4 Z_3(\kappa)^{-n}
 e^{-17F_0/(165\kappa)}.
\tag{13}
\]

This uses the actual block marginals, with no assumption that they are
Haar. The factor is \(Z_3^n\), independently of the sizes of the two blocks.
With radial Haar \(2\sin^2\theta\,d\theta/\pi\), integrating
\(0\le\theta\le\sqrt\kappa\le1\), using \(f\le\theta^2\),
\(\sin\theta\ge5\theta/6\), and \(e^{-3}>1/27\), gives

\[
Z_3(\kappa)\ge\frac{175}{32076}\kappa^{3/2}
>\frac{\kappa^{3/2}}{200}.
\]

Conversely \(f\ge4\theta^2/\pi^2\). Gaussian integration and
\(\sqrt\pi<2\) yield, for \(c=17/330<1\),

\[
Z_c(\kappa)\le\frac{\pi^{5/2}}{16c^{3/2}}\kappa^{3/2}
\le\frac{121}{98c^2}\kappa^{3/2}<466\kappa^{3/2}.
\tag{14}
\]

Split the exponential in (13) into two factors with exponent \(c\). On
\(F_0\ge R\kappa\) the first is at most \(e^{-cR}\); integrate the second
using (14). The factors \(\kappa^{3n/2}\) cancel exactly. Thus

\[
\int_{F_0\ge R\kappa}|k|^2\le
93200^n\exp\left(1600000n^2+13108n-\frac{17R}{330}\right).
\]

Since \(e>8/3\) and \((8/3)^{12}>93200\), this proves (1). Coarsening each
block to the list of its class angles contracts this restricted squared
kernel integral, by conditional Jensen under the product of its own
marginals. The tail event is measurable under that coarsening. Therefore
(1) also bounds the corresponding radial kernel tail.

## 5. Replay and the limits of the result

The replay rebuilds the canonical source, all original links and oriented
plaquette words, the tree, the partition, every rational gate, the region,
the factored exponential and all scope flags. Resealing altered constants,
geometry, source fields or claims does not make them canonical.

The proof above is a written analytic theorem; the exact replay does not
formalize Bochner, the maximum principle, or the spectral identification.
All Lean, continuum, ambient-exterior and volume-uniform flags remain false.
The two-plaquette specialization agrees with the geometry of the
[theta tail theorem](gauge-theta-kernel-tail.md), which has substantially
better constants. This more general source does not replace that payload.
The missing quantitative local Gaussian comparison is separate from (1).

# Actual weak-theta normalized-kernel tails

`su2_theta_normalized_kernel_tail` proves a uniform rescaled
Hilbert–Schmidt tail estimate for the **actual** vacuum of two adjacent
SU(2) square plaquettes. The graph has seven original unit-weight links.
For every \(0<\kappa\le1/64\) and \(R>0\), the squared tail is at most

\[
 26\,000\,000\exp(81640-2R/9).                         \tag{1}
\]

The large constant is explicit. This theorem controls rare configurations
in a normalized joint kernel; it does not prove a Gaussian approximation,
a maximal-correlation upper bound, or a growing-graph result.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.theta_kernel_tail import (
    su2_theta_normalized_kernel_tail,
    replay_su2_theta_kernel_tail_certificate,
)

result = su2_theta_normalized_kernel_tail(
    Fraction(1, 2**40), 400000, target=Fraction(1, 1024),
)
assert result["status"] == "PASS"
assert result["tail_target_status"] == "PASS"
assert not result["tail_region_has_zero_haar_measure"]
assert result["arithmetic"]["dyadic_upper_negative_integer_exponent"] == 7248
assert replay_su2_theta_kernel_tail_certificate(result["certificate"])
assert not result["maximal_correlation_upper_verified"]
```

The certificate and every displayed summary are canonical and replayable.
Positive couplings outside the proved interval return `INCONCLUSIVE`.
An optional target has a separate decision from the analytic theorem.
Inputs are exact integers or fractions; floats and booleans are refused.

## 1. Defined actual kernel and source

Use normalized Haar measure on \(G=SU(2)\), with fundamental Casimir
\(3/4\), and write \(A(U)=2-\operatorname{Tr}U\). The exact reduced
operator of the seven-link graph is

\[
 H_\theta=\frac\kappa2 C_\theta+\frac2\kappa S,
 \quad C_\theta=3(C_x+C_y)+C_d,\quad S=A(x)+A(y),             \tag{2}
\]

where \(C_d\) generates simultaneous left multiplication. Its unique
positive smooth ground state \(\Phi\) is invariant under simultaneous
conjugation and lifts to the actual physical original-link vacuum.
The [canonical theta source](gauge-theta-weak-blocks.md) proves these facts
and \(0\le E_0\le6\) by an explicit product trial. The implementation
replays that source and consumes only its ground-energy upper bound and
vacuum identification. Its spectral-gap fields are not premises.

Let \(\rho=\Phi^2\), normalized in product Haar, and let \(\rho_x,\rho_y\)
be its **own** marginals. The normalized kernel is

\[
 k_\kappa(x,y)=\frac{\rho(x,y)}{\sqrt{\rho_x(x)\rho_y(y)}}.
                                                               \tag{3}
\]

For \(U=\cos\theta+i\sin\theta\,\omega\cdot\sigma\),
\(0\le\theta\le\pi\), define

\[
 f(U)=8(1-\cos(\theta/2)),\qquad F_0=f(x)+f(y).
\]

The left side bounded by (1) is exactly
\(\int_{F_0\ge R\kappa}|k_\kappa|^2\,dx\,dy\), the **square** of a
Hilbert–Schmidt norm. If \(R\kappa\ge16\), this set has zero Haar measure
because \(F_0\le16\), with equality only at \((-I,-I)\). The API then
also returns the sharper exact effective upper bound zero.

## 2. Explicit local logarithmic control on original links

Work on the full original product \(SU(2)^7\), of dimension 21, with its
product bi-invariant metric and nonnegative Ricci curvature. Let \(h\) be
the logarithm of the lifted actual ground state, and put

\[
 q=|\nabla h|^2,\qquad
 v=4S/\kappa^2-2E_0/\kappa,\qquad \Delta h=v-q.
                                                               \tag{4}
\]

All derivatives here are original-link derivatives. Direct differentiation
of the two words gives

\[
 \Gamma S\le5S,\qquad \Delta S=12-3S.                       \tag{5}
\]

For example each nonshared link contributes one individual gradient square,
while the shared link differentiates both traces; Cauchy–Schwarz gives the
constant five. This does not require Ricci positivity of a quotient metric.

Set \(B=1024\), \(\zeta=1-S/(B\kappa)\),
\(\eta=\zeta_+^2\), and \(F=\eta q\). A positive maximum of \(F\)
lies in \(\zeta>0\), where the cutoff is smooth. Bochner's formula and
\(\|\nabla^2h\|^2\ge(\Delta h)^2/21\), followed by
\(\nabla q=-q\nabla\eta/\eta\) at that maximum, imply

\[
 \frac2{21}(F-\eta v)^2
 \le2\eta^{3/2}\sqrt F\,|\nabla v|
 +2F^{3/2}\frac{|\nabla\eta|}{\sqrt\eta}
 +F\left[-\Delta\eta+\frac{2\Gamma\eta}{\eta}\right].     \tag{6}
\]

On this support, (4)–(5) and \(E_0\le6\) give

\[
 |v|\le4108/\kappa,\quad
 |\nabla v|\le4\sqrt{5120}\,\kappa^{-3/2},\quad
 \frac{|\nabla\eta|}{\sqrt\eta}
 \le2\sqrt{5/1024}\,\kappa^{-1/2}.
\]

The final bracket in (6) is exactly

\[
 \frac{2\zeta(12-3S)}{B\kappa}
 +\frac{6\Gamma S}{B^2\kappa^2}
 \le\frac{54}{B\kappa}\le\frac{64}{B\kappa}.               \tag{7}
\]

If \(a=\kappa F\ge16384\), then \(|a-\kappa\eta v|\ge a/2\).
After dividing (6) by \(a^{3/2}/\kappa^2\),

\[
 \frac{\sqrt a}{42}
 \le\frac{8\sqrt{5120}}a+4\sqrt{5/1024}
       +\frac1{16\sqrt a}
 \le\frac9{256}+\frac9{32}+\frac1{2048}
 =\frac{649}{2048}<\frac{64}{21},                           \tag{8}
\]

contradicting \(\sqrt a/42\ge64/21\). Thus \(\kappa F<16384\).
On the core \(S\le256\kappa\), \(\eta\ge9/16\), and

\[
 |\nabla\log\Phi|\le\frac{512}{3\sqrt\kappa}.              \tag{9}
\]

To integrate (9), gauge-fix the five tree links and vary the two remaining
chords along shortest group geodesics to the identity. Gauge invariance
preserves vacuum values. The path is in the original-link configuration
space, stays in the core, and has length at most \(\pi\sqrt S\): each
link distance \(2\theta\) is bounded by \(\pi\sqrt{A(U)}\).
Consequently

\[
 |\log\Phi(x,y)-\log\Phi(I,I)|
 \le8192\pi/3<10000                                        \tag{10}
\]

throughout the core. This is an explicit local estimate, not an assumed
Harnack constant or a global oscillation bound of order \(1/\kappa\).

## 3. Global supersolution and subsolution

Put \(\epsilon=1/16\) and

\[
 f_\epsilon(U)=8\left(\sqrt{1+\epsilon^2}
 -\sqrt{\cos^2(\theta/2)+\epsilon^2}\right),\quad
 F_\epsilon=f_\epsilon(x)+f_\epsilon(y).
\]

Direct one-rotor differentiation gives

\[
 \Gamma_C f=A,\quad Cf=\sec(\theta/2)-\tfrac52\cos(\theta/2)\ge-3/2,
 \quad\Gamma_C f_\epsilon\le A,\quad Cf_\epsilon\le24.
                                                               \tag{11}
\]

Moreover \((8/9)f\le f_\epsilon\le f\) and \(f\le2A\).
Indeed the ratio is
\((1+c)/(\sqrt{1+\epsilon^2}+\sqrt{c^2+\epsilon^2})\),
where \(c=\cos(\theta/2)\); bounding the denominator by
\(1+c+2\epsilon\) gives \(8/9\).
The original metric obeys \(3C_\Sigma\le C_\theta\le5C_\Sigma\).
For additive \(F\), \(C_\theta F=4(C_xF+C_yF)\).

Define \(w_+=e^{-4F_\epsilon/(5\kappa)}\) and
\(w_-=e^{-6F_0/(5\kappa)}\). The exact exponential identity is

\[
 \frac{Hw}{w}
 =-\frac{C_\theta F}{2}
   +\frac{2S-\Gamma_\theta F/2}{\kappa},\quad w=e^{-F/\kappa}.
\]

Using (11) gives

\[
 \frac{(H-E_0)w_+}{w_+}\ge\frac{2S}{5\kappa}-\frac{414}{5},
 \qquad
 \frac{(H-E_0)w_-}{w_-}\le\frac{36}{5}-\frac{4S}{25\kappa}.
                                                               \tag{12}
\]

These are respectively positive and negative outside the core.
There \(2S/\kappa-E_0>0\), so the maximum principle compares with the
actual eigenfunction from the core boundary. The upper barrier is smooth
on the whole group. At an antipode the lower barrier has a conical minimum:
it is locally a smooth nonzero factor times \(e^{c r/\kappa}\).
Its Laplacian has an integrable positive \(1/r\) term, with the favorable
sign in \(-\Delta w_-\). The singularity has codimension three, lies in
the local \(H^2\) domain, and integration on a sphere of radius \(r\)
has boundary flux tending to zero as \(r^2\). There is no adverse point
mass. Thus its subsolution inequality holds weakly and the same form
comparison applies. No spin truncation or omitted angular sector is used.

Normalize temporarily by \(h_0=\Phi/\Phi(I,I)\), which need not have
unit norm. Equations (10)–(12), and \(F_\epsilon\le F_0\le2S\), give
on the entire reduced group

\[
 L e^{-6F_0/(5\kappa)}\le h_0\le
 U e^{-4F_\epsilon/(5\kappa)},\qquad
 L=e^{-10000},\quad U=e^{10410}.                            \tag{13}
\]

On the core, \((4/5)F_\epsilon/\kappa\le2048/5<410\), which
explains the extra upper constant. Outside it, use the comparison just
proved. If the core fills the group, the core estimates alone suffice.

## 4. Actual marginal denominators and the tail integral

Let \(H_x(x)=\int h_0(x,y)^2dy\) and similarly \(H_y\). Normalizing
\(h_0\) cancels exactly from the kernel:

\[
 |k_\kappa|^2=\frac{h_0^4}{H_xH_y}.
\]

For \(Z_b(\kappa)=\int_G e^{-12f/(5\kappa)}dU\), (13) gives
\(H_x\ge L^2Z_b e^{-12f(x)/(5\kappa)}\). The same applies to \(H_y\).
Therefore, using \(F_\epsilon\ge8F_0/9\),

\[
 |k_\kappa|^2
 \le (U/L)^4 Z_b^{-2}\exp[-4F_0/(9\kappa)].                \tag{14}
\]

Both actual marginal denominators have been bounded; a small unweighted
vacuum error alone would not justify (14).

The exact radial Haar density is \((2/\pi)\sin^2\theta\,d\theta\).
For \(0<\kappa\le1\), restrict to \(0\le\theta\le\sqrt\kappa\).
Here \(f\le\theta^2\), \(\sin\theta\ge5\theta/6\), and
\(e^{-12/5}\ge1/27\), since \(e<3\). Hence

\[
 Z_b\ge\frac{175}{32076}\kappa^{3/2}>
 \frac1{200}\kappa^{3/2}.                                 \tag{15}
\]

Conversely \(f\ge4\theta^2/\pi^2\) and \(\sin\theta\le\theta\).
For any \(a>0\), extension of the radial Gaussian integral gives

\[
 \int_G e^{-af/\kappa}dU
 \le\frac{\pi^{5/2}}{16a^{3/2}}\kappa^{3/2}.
\]

At \(a=2/9\), use \(\sqrt\pi\le2\), \(\pi\le22/7\), and
\(a^{-3/2}\le a^{-2}\), obtaining \(9801\kappa^{3/2}/392\).
On \(F_0\ge R\kappa\), split the exponential in (14) into two halves.
One contributes \(e^{-2R/9}\), while the two remaining Haar integrals
contribute at most \((9801/392)^2\kappa^3\). Equations (14)–(15) yield

\[
 \int_{F_0\ge R\kappa}|k_\kappa|^2
 \le40000(9801/392)^2e^{81640-2R/9}
 <26\,000\,000e^{81640-2R/9},                              \tag{16}
\]

which proves (1) uniformly in the stated coupling interval.

## 5. Radial coarse-graining, scope and finite target gates

The same tail bound holds for the radial joint kernel relative to the
radial Haar marginals. To see this without assuming product conditionals,
let \(\nu=\mu_x\otimes\mu_y\) and \(r=d\mu/d\nu\). After applying
the coordinatewise radial map, the new density ratio is the conditional
expectation of \(r\) under \(\nu\). Jensen's inequality contracts its
squared integral on any radially measurable set, including this tail.
That integral is exactly the normalized-kernel squared norm in (16).

For \(E=81640-2R/9\le0\), the API uses
\(e^E\le2^{-\lfloor-E\rfloor}\). For \(E>0\), it uses
\(e^E\le2^{2\lceil E\rceil}\). These follow from \(2<e<4\).
The prefactor and signed integer exponent are stored separately, so no
underflow or expanded huge denominator occurs. Exact rational bit-length
comparisons decide whether this outward upper bound is below a supplied
target. Failure of that conservative target gate does not invalidate (16).

The central region still requires a quantitative normalized-kernel
comparison to the Gaussian reference. Small unweighted wavefunction error
alone cannot replace it: tiny correlated events can dominate maximal
correlation. Accordingly this certificate earns the actual tail theorem,
not \(\delta\le1/2\), and retains false flags for Gaussian comparison,
ambient/growing-graph scope, continuum limits, and formal analytic proof.

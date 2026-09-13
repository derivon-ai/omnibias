# An explicit actual theta correlation interval

For the actual seven-link, two-plaquette SU(2) vacuum, the following
explicit interval suffices:

\[
0<\kappa\le2^{-17022271}
\quad\Longrightarrow\quad
\delta_{\rm radial}(\kappa)\le\frac{143}{480}<\frac3{10},
\qquad
\delta_{\rm full\ cycle}(\kappa)\le\frac{95}{224}<\frac12.
\tag{1}
\]

This is an extremely conservative mathematical threshold. Its size comes
from the existing core-comparison constants, rather than a numerical
transition measurement. The statement retains every mode of the actual
vacuum on this fixed graph. It is not an ambient-lattice or continuum
mass-gap theorem.

The [actual kernel-tail theorem](gauge-theta-kernel-tail.md) supplies
the tail estimates used below. A smooth quasimode and an explicit density
comparison make the compact-region error quantitative. No unspecified
elliptic-regularity constant is used. The general harmonic-approximation
context is [Simon, 1983](https://www.numdam.org/item/AIHPA_1983__38_3_295_0.pdf);
the quantitative constants and own-marginal estimates used below are
derived here and in the linked source proofs.

```python
from omnibias.geometry.gauge.transfer.theta_kernel_threshold import (
    su2_theta_kernel_contraction,
    replay_su2_theta_kernel_contraction_certificate,
)

row = su2_theta_kernel_contraction()
assert row["status"] == "PASS"
assert row["actual_radial_maximal_correlation_upper"] == "143/480"
assert row["explicit_coupling_threshold_verified_in_written_analysis"]
assert replay_su2_theta_kernel_contraction_certificate(row["certificate"])
assert not row["continuum_claim"]

# Asking for a larger interval does not inherit the proven bound.
outside = su2_theta_kernel_contraction(dyadic_exponent=17022270)
assert outside["status"] == "INCONCLUSIVE"
assert outside["actual_radial_maximal_correlation_upper"] is None
```

The sole input is a positive integer exponent `N`. The returned domain is
the entire interval `0 < kappa <= 2**(-N)`, and the proof succeeds for
`N >= 17022271`. The implementation stores this exponent symbolically; it
does not construct a rational denominator with millions of digits, or
round the upper endpoint to floating-point zero. Booleans and floats are
refused.

## 1. A positive smooth trial with an actual spectral comparison

Write the two group elements as unit quaternions \((s,X),(t,Y)\), with
\(S=4-2s-2t\). Define

\[
a=\frac1{\sqrt3}+\frac1{\sqrt5},\quad
b=2\left(\frac1{\sqrt5}-\frac1{\sqrt3}\right),\qquad
F=aS+bX\cdot Y,\qquad v_\kappa=Z^{-1/2}e^{-F/\kappa}.
\tag{2}
\]

This trial is globally smooth and strictly positive. Since
\(|X\cdot Y|\le S/2\), it obeys
\(F\ge(2/\sqrt5)S\); its only zero is the identity pair. The
[exact quaternion quasimode proof](gauge-theta-quasimode.md)
establishes

\[
\lambda=\frac32(\sqrt3+\sqrt5)<6,\qquad
\|(H_\kappa-\lambda)v_\kappa\|_2\le83\kappa.
\tag{3}
\]

This is a residual for the actual Hamiltonian, including the shared-link
kinetic term. The canonical weak-theta comparison gives the absolute
excited-energy floor \(E_1\ge32/5\) for \(\kappa\le1/64\). Thus the
excited component of \(v_\kappa\) has norm at most \((5/2)83\kappa\).
Both it and the normalized actual vacuum \(\psi_\kappa\) are positive,
so their overlap is nonnegative. The spectral decomposition gives

\[
\|\psi_\kappa-v_\kappa\|_2\le332\kappa,
\qquad
\|\psi_\kappa^2-v_\kappa^2\|_1\le664\kappa.
\tag{4}
\]

The eigenvalue floor identifies the correct state; a small residual alone
would not do so.

## 2. Quantitative comparison with the Gaussian density

Use the fixed rescaled Lebesgue coordinates of the kernel-limit proof:
\(x=\exp(i\sqrt\kappa u\cdot\sigma/2)\), and similarly for \(y,v\).
Let \(r^2=|u|^2+|v|^2\), \(M=\left(\begin{smallmatrix}4&1\\1&4\end{smallmatrix}\right)\),
and \(B_0=M^{-1/2}\otimes I_3\). Remove the common Haar scaling constant
from the unnormalized trial density and put

\[
g_\kappa=j(\sqrt\kappa u)j(\sqrt\kappa v)e^{-2F/\kappa},
\qquad g_0=e^{-z^TB_0z},\quad z=(u,v),
\quad j(w)=\left(\frac{\sin(|w|/2)}{|w|/2}\right)^2.
\tag{5}
\]

Extend \(g_\kappa\) by zero outside the product of balls of radius
\(2\pi/\sqrt\kappa\). On that domain, elementary Taylor remainders give

\[
\left|\frac{2F}{\kappa}-z^TB_0z\right|
\le\frac{a+|b|}{96}\kappa r^4
\le\frac{\kappa r^4}{64},\qquad
0\le1-j(\sqrt\kappa u)j(\sqrt\kappa v)\le\frac{\kappa r^2}{12}.
\tag{6}
\]

Indeed \(|2(1-\cos\theta)-\theta^2|\le\theta^4/12\), and
\(|\sin\alpha\sin\beta-\alpha\beta|
\le\alpha\beta(\alpha^2+\beta^2)/6\).
Use \(a\le11/10\), \(|b|\le1/3\) and \(\alpha\beta\le(\alpha^2+\beta^2)/2\).
These bounds hold throughout the coordinate domain, not just near zero.

Both exponents in (5) are at least \(r^2/8\): for the trial use
\(S\ge\kappa r^2/\pi^2\), \(F\ge(2/\sqrt5)S\),
\(\sqrt5<9/4\) and \(\pi^2<10\). Outside the coordinate domain,
\(\kappa r^2\ge4\pi^2>36\). The exponential mean-value inequality
therefore yields the whole-space estimate

\[
|g_\kappa-g_0|
\le\kappa\left(\frac{r^2}{9}+\frac{r^4}{64}\right)e^{-r^2/8}.
\tag{7}
\]

The Gaussian integral is \((8\pi)^3<32768\), with normalized second
and fourth moments \(24\) and \(768\). Hence
\(\|g_\kappa-g_0\|_1\le2^{19}\kappa\). Since
\(\int g_0=\pi^3 15^{3/4}>1\), normalization costs at most a factor
two. Let \(p_\kappa\) be the scaled actual density and \(p_0\) the
normalized Gaussian density. Combining this estimate with (4) proves

\[
\tau:=\|p_\kappa-p_0\|_1
\le(664+2^{20})\kappa<2^{21}\kappa.
\tag{8}
\]

## 3. A compact density-to-kernel estimate

This lemma applies to any two normalized joint densities \(p,q\).
Suppose on a product region \(D_X\times D_Y\) both joints are at most
\(B\), and all four own marginals are at least \(\ell>0\). If
\(\tau=\|p-q\|_1\) on the full product space, then

\[
\left\|\frac p{\sqrt{p_Xp_Y}}-
             \frac q{\sqrt{q_Xq_Y}}\right\|_{L^2(D_X\times D_Y)}
\le\frac{3\sqrt{B\tau}}\ell.
\tag{9}
\]

To prove this, split the difference into

\[
\frac{p-q}{\sqrt{p_Xp_Y}}+
\frac q{\sqrt{p_Y}}\left(\frac1{\sqrt{p_X}}-\frac1{\sqrt{q_X}}\right)+
\frac q{\sqrt{q_X}}\left(\frac1{\sqrt{p_Y}}-\frac1{\sqrt{q_Y}}\right).
\]

The first squared norm is at most \(B\tau/\ell^2\), since
\(|p-q|^2\le B|p-q|\). For the second use \(q^2\le Bq\) and
\((\sqrt a-\sqrt b)^2\le|a-b|\). Its squared integrand is at most
\((B/\ell^2)(q/q_X)|p_X-q_X|\). Integration over \(D_Y\) costs at
most one; marginalization contracts \(L^1\), so this norm has the
same bound. The third term is symmetric. The triangle inequality proves
(9). No lower bound on a trial marginal substitutes for an actual one.

For centered kernels a further rank-one difference has norm at most
\(2\sqrt\tau\). This follows by marginal \(L^1\) contraction and the
same square-root inequality. If the two uncentered squared-kernel tails
outside the region are \(T_p,T_q\), the complete centered error is thus

\[
\|T_p^{\rm centered}-T_q^{\rm centered}\|_{\rm op}
\le\frac{3\sqrt{B\tau}}\ell+2\sqrt\tau+
\sqrt{T_p}+\sqrt{T_q}.
\tag{10}
\]

## 4. Explicit own-marginal floors

Fix \(L=2048\) and \(\kappa\le2^{-20}=4/L^2\). Use the product of
two radius-\(L\) balls for (9). The existing
[actual barrier theorem](gauge-theta-kernel-tail.md)
writes \(\psi=m h\), where

\[
e^{-6F_0/(5\kappa)}\le h\le e^{20410}e^{-32F_0/(45\kappa)}.
\]

The one-link bound \(\int e^{-2f/(9\kappa)}dH\le25\kappa^{3/2}\)
also bounds the faster exponential here. Normalization therefore gives
\(m^2\ge[625e^{40820}\kappa^3]^{-1}\).
For \(|u|\le L\), integrate the lower density over \(|v|\le1\).
Both Haar Jacobian shapes exceed \(1/2\), because their class angles
are at most one, and \(F_0\le\kappa r^2/4\). The unit three-ball has
volume greater than four. Consequently

\[
p_{X,\kappa}(u)\ge
\frac{e^{-40820-(3/5)(L^2+1)}}{625(16\pi^2)^2}
\ge e^{-40845-L^2}=:\ell.
\tag{11}
\]

Here \(625(16\pi^2)^2<2^{24}<e^{24}\). The same holds for the other
actual marginal. The Gaussian marginal variance is
\(\sigma^2=(\sqrt5+\sqrt3)/4\in(19/20,1)\), so each Gaussian
marginal is at least \(e^{-5-L^2}\), and hence at least \(\ell\).
The actual global upper envelope from the barrier theorem gives
\(p_\kappa\le2e^{40820}\le e^{40822}=:B\); also \(p_0<1<B\).

## 5. Uniform normalized tails and the dyadic interval

Outside the product region, \(|z|\ge L\). The actual squared-kernel
tail is bounded by

\[
T_p\le26\,000\,000\exp(81640-L^2/45).
\tag{12}
\]

This uses \(F_0/\kappa\ge |z|^2/\pi^2\ge |z|^2/10\) in the
existing all-mode tail theorem. For the Gaussian, its squared kernel
has prefactor below one and precision
\(2B_0-I/(2\sigma^2)\ge I/3\): use
\(2/\sqrt5-10/19>1/3\). Thus
\(T_q\le(6\pi)^3e^{-L^2/6}\), which is also bounded by (12).
At \(L=2048\), the exponent in (12) is \(-520504/45\).
Since \(e>2\) and \(26\,000\,000<2^{25}\), both squared tails
are less than \(2^{-12}\). Their two norm contributions total at most
\(1/32\).

Set \(A=61256+L^2=4255560\). Equations (10)–(11), with \(e<4\), give

\[
\|T_\kappa-T_0\|_{\rm op}
\le2^{2A+3}\sqrt{2^{21}\kappa}+\frac1{32}.
\tag{13}
\]

If \(\kappa\le2^{-N}\) and
\(N\ge4A+21+10=17022271\), the first term is at most \(1/4\).
The centered error is therefore at most \(9/32\).
The Gaussian full-cycle correlation is \(4-\sqrt{15}<1/7\), and its
radial correlation is \(31-8\sqrt{15}<1/60\). Angular projection
contracts the operator norm, so (13) proves (1).

The executable certificate stores the integer exponent \(N\) and the
universal domain \(0<\kappa\le2^{-N}\). It never constructs a rational
with millions of digits or rounds this threshold to floating-point zero.
Arithmetic replay and scalar Lean checks have their stated scopes; the
operator, integral, spectral and density implications above remain written
analysis.

## 6. Remaining mathematical target

This discharges the explicit-threshold sub-obligation on the isolated
theta graph. Better core and marginal estimates could improve the very
poor numerical threshold. The central parent obstacle remains control of
actual conditional laws on growing, higher-dimensional lattices, with a
compatible scale limit, nontrivial QFT reconstruction and a physical gap.

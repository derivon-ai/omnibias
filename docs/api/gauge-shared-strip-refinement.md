# Actual shared-strip vacuum refinement

`su2_shared_strip_refinement` treats the physical thirteen-edge open
four-plaquette SU(2) strip. Two coarse two-plaquette rectangles share a
single electric edge. The actual joint marginal defines the coarse
vacuum and embedding. Every vertex imposes Gauss law, and every spin
sector is included.

The Hamiltonian normalization is
\[
aH=\frac{\kappa}{2}\sum_{e=1}^{13}C_e+
\frac2\kappa\left(8-\sum_{p=1}^4\chi_p\right),\qquad
g=4/\kappa^2.
\]
The additive constant does not affect the gap.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_shared_strip_refinement,
    replay_su2_shared_strip_refinement_certificate,
)

result = su2_shared_strip_refinement(19, correction_radius=Q(1, 9))
assert result["status"] == "PASS"
assert Q(result["physical_gap_lower"]) > Q(13, 2)
assert result["actual_joint_log_ball_verified"]
assert replay_su2_shared_strip_refinement_certificate(result["certificate"])
assert not result["all_scale_refinement_claim"]
assert not result["continuum_claim"]
assert not result["theorem_prover_verified"]
```

## Actual marginal and conditional gap

A spanning-tree gauge leaves four group coordinates
\((W_L,R_L,R_R,W_R)\) modulo simultaneous conjugation.
The coarse map retains \((W_L,W_R)\) and integrates the two internal
coordinates. With actual fine vacuum \(\psi\), define
\[
\rho=\int\psi^2\,dR_LdR_R,\quad \phi=\sqrt{\rho},\quad
J\Phi=\psi\,\Phi(W_L,W_R)/\phi.
\]
This is an isometry with \(J\phi=\psi\). It does not substitute an
independent product or a prescribed Wilson vacuum.

The coarse electric metric has path weights \((5,5,1)\):
\[
\Gamma_A=5|p_L|^2+5|p_R|^2+|p_L+p_R|^2.
\]
Keeping both left and right rotation frames before minimizing the fine
metric gives
\(\Gamma\ge(11/5)(|\nabla_{R_L}|^2+|\nabla_{R_R}|^2)\).

The source's structural and fixed-point gates construct
\(\psi\propto e^{S_*+u}\), with
\(\mathcal N(u)\le r\), when
\[
\frac43(16g+r)^2\le r,\qquad \frac83(16g+r)<1.
\]
Its earlier gap conclusion is not an input here. Put \(\alpha=\kappa/2\),
\[
\Omega_f=(32g+16r)/3,\quad
\Omega_m=\min\{16(g+r)/3,16g^2/9+26r/3\},\quad
h=7g/6+4r/3.
\]
With \(\gamma_f=(3/4)e^{-\Omega_f}\), the written analytic implication is
\[
d_A=\tfrac92\alpha e^{-\Omega_m},\qquad
d_H=\tfrac{11}{5}\alpha\gamma_f,\qquad
\beta^2=\frac{4\alpha h^2}{5\gamma_f}.
\]
Here \(d_H\) controls the entire conditional complement and
\(\|Q\mathcal LJf\|^2\le\beta^2\mathfrak q_A(f)\).
The sufficient gap test is
\((d_A-z)(d_H-z)\ge\beta^2d_A\).

The implementation uses exact rational exponential lower bounds
\((1-\Omega/m)^m\) and an upward dyadic square root. With default
`exponent_steps=8` and `sqrt_bits=64`, the lower gap at \(\kappa=19\)
is greater than \(6.6477\), in dimensionless \(aH\) units.

## Joint density and logarithm

After tree restriction, the source supplies
\[
K_0=16g/3+13r/3,\qquad K_2=104g/3+24r
\]
as bounds on \(\|2S\|_{\mathfrak A_0}\) and
\(\|2S\|_{\mathfrak A_2}\). For
\(E_+=(1-K_0/m)^{-m}\), \(K_0<m\), the actual normalized coarse density
satisfies
\[
\|\rho-1\|_{\mathfrak A_0}\le B_0=E_+-1,\qquad
\|\rho-1\|_{\mathfrak A_2}\le B_2=K_2(1+K_0)E_+.
\]
If \(B_0<1\), then
\(\|\log\rho\|_{\mathfrak A_2}\le B_2/(1-B_0)^2\).
The smallness gate is in \(\mathfrak A_0\); no small
\(\mathfrak A_2\) ball is asserted.

The broad bound on the density's own connected part is
\[
\|\rho-\rho_L\rho_R\|_{\mathfrak A_2}
\le\max\{B_2,B_2^2/4\}.
\]
Its key is `actual_joint_to_own_marginals_A2_upper`. It is a density
bound, not a narrow connected logarithm or potential enclosure.

## Scope and refusal

The physical gap and joint logarithm have separate gates. A failed
logarithm ball does not invalidate a separately earned finite gap.
Failed source or arithmetic premises return `INCONCLUSIVE` and
`None` for the corresponding unsupported bounds. Exact input
parameters refuse floats and booleans.

Canonical replay recomputes the graph, source, arithmetic and flags;
rehashing a promoted or altered certificate does not make it valid.
The analytic implication is written mathematics, not Lean verification.
The constants concern one finite graph. They do not establish a
volume-uniform refinement, a continuum limit, or Yang–Mills mass gap.

# Wilson linear inverse and improved vacuum source

The exact fundamental fusion calculation improves the linear part of
the [Wilson residual source](gauge-wilson-residual-source.md) in the
same complete spin-weighted Fourier norm:
\[
K=2C_0^{-1}\Pi_H\Gamma(S_*,\cdot),\qquad
\|K\|\le L=\frac{28}{3}qgb^2.
\]
Here \(S_*=(g/3)\sum_p\chi_{1/2}(U_p)\), \(g=4/\kappa^2\), and
\(q=2\) for open square strips or \(4\) for open three-dimensional
cubic boxes. Edges retain their canonical positive-coordinate orientation.
All electric and elementary plaquette weights are one.
The previous generic linear bound was \(32qgb^2/3\).

The public call separately records a bounded Fourier inverse, a
nonlinear correction, and its physical gap. An inverse alone does
not earn either of the latter.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_wilson_linear_vacuum,
    replay_su2_wilson_linear_vacuum_certificate,
)

inverse_only = su2_wilson_linear_vacuum(13)
a = inverse_only["witness"]["arithmetic"]
assert a["linear_upper"] == "448/507"
assert a["generic_linear_upper"] == "512/507"
assert a["fourier_linear_inverse_upper"] == "507/59"
assert inverse_only["fourier_linear_inverse_verified"]
assert not inverse_only["actual_vacuum_verified"]
assert inverse_only["status"] == "INCONCLUSIVE"

vacuum = su2_wilson_linear_vacuum(16, correction_radius=Q(1, 8))
assert vacuum["status"] == "PASS"
assert vacuum["physical_gap_lower"] == "2"
assert vacuum["beyond_previous_source_criterion_verified"]
assert replay_su2_wilson_linear_vacuum_certificate(vacuum["certificate"])
assert not vacuum["continuum_claim"]
```

## Why the linear coefficient is smaller

On a differentiated edge, spin \(1/2\) from the reference fuses with
input spin \(k\). The operator
\(\Omega=\sum_a T_a^{1/2}\otimes T_a^k\) is scalar on the two
allowed output spins:
\[
\Omega_{k+1/2}=k/2,\qquad \Omega_{k-1/2}=-(k+1)/2.
\]
At the norm's anchor, maximize the weighted product jointly:
\[
\max_\ell \ell|\Omega_\ell|=\frac{k(k+1/2)}2.
\]
The zero-spin case contributes zero. Elsewhere,
\(\|\Omega\|\le3k/2\).
Product Clebsch--Gordan pinching is contractive in nuclear norm,
and \(\Omega\) commutes with its fusion projectors. No coordinate
inversion or partial-transpose isometry is assumed.

For a plaquette \(p\), input spins \(\mathbf k\), and anchor \(i\),
these bounds give
\[
G_i\le\frac32 k_i\sum_{e\in p}k_e
 +\frac34\mathbf1_{i\in p}\sum_{e\in p\setminus\{i\}}k_e.
\]
The second term charges only the other three edges.
Every nonconstant invariant mode has energy at least three,
\(\sum_e k_e\le2E/3\), and at most \(q\) squares meet an edge.
The two anchored sums are therefore bounded by \(qN_i\) and
\(3qN/4\). Multiplying by the canonical square coefficient norm
eight and the seed factor \(2g/3\) proves \(L=28qgb^2/3\).
Intersecting supports supply the factor \(b^2\); completion
extends the estimate to all spins and support sizes.

When \(L<1\), the Neumann inverse on that very same Fourier space obeys
\[
\|(I-K)^{-1}\|\le\frac1{1-L}.
\]
Thus cubic \(\kappa=13,b=1\) earns inverse bound \(507/59\),
while its old generic linear bound is \(512/507>1\).
This theorem has constants independent of finite box size.

## The nonlinear gate remains necessary

Keep the exact residual bounds
\[
D_{\rm strip}=(6b^2+8b^3)g^2,\qquad
D_{\rm cubic}=(12b^2+112b^3)g^2,\qquad B=4/3.
\]
The wrapper requires \(D+Lr+Br^2\le r\) and \(L+2Br<1\)
to construct the actual vacuum. These are also the self-map and
contraction conditions for the inverse-preconditioned equation
\(u=(I-K)^{-1}[R_*+T(u,u)]\).
Some positive radius exists exactly when \(L<1\) and
\((1-L)^2>4BD\). Failure is inconclusive about the physical theory.

At cubic \(\kappa=16,b=1,r=1/8\),
\[
L=7/12,\quad D=31/1024,\quad
r-D-Lr-Br^2=1/1024,\quad L+2Br=11/12.
\]
The actual-vacuum reconstruction from the source theorem then applies.
Its curvature is at least \(1/4\), giving
\(\operatorname{gap}(aH)\ge(16/2)(1/4)=2\).
This includes every electric spin on every specified finite open box.
At \(\kappa=17,b=17/16,r=1/10\), the weighted source and its
actual conditional marginal hierarchy also pass.

## Contract and limits

Inputs follow the original source: exact positive `kappa`,
`family="strip"|"cubic"` (default cubic here), optional positive
`correction_radius`, `decay_base>=1` and integer `exponent_steps>=1`.
The automatic radius, curvature and conditional gap routes are unchanged.
No supplied linear bound or residual can replace the proved constants.

Existing `su2_wilson_residual_vacuum` v1 certificates retain their
original payloads and constants. The sharper wrapper has certificate
type `su2_wilson_linear_vacuum_v1` and its own canonical replay.
Six prior certificate digests are regression-locked.

The companion proof is
`ensemble-laws/docs/constructive/wilson-linear-bound.md`.
This reduces a sufficient norm overestimate; it does not remove
the remaining weak-coupling obstruction. The linear condition still
requires \(3\kappa^2>448b^2\) on cubic boxes. Continuum,
infinite-volume and formal analytic verification flags remain false.

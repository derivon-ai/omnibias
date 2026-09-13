# Reference diffusion inverse

`su2_reference_linearized_inverse` bounds an auxiliary reference
diffusion on centered \(L^2(\nu)\), where
\[
S_*=\frac g3\sum_p\chi_{1/2}(U_p),\quad
\nu=Z^{-1}e^{2S_*}\,dH,\quad
A=C-2\Gamma(S_*,\cdot),\quad g=4/\kappa^2.
\]
This \(\nu\) is explicitly specified. It is not asserted to be the
unknown quantum vacuum density.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_reference_linearized_inverse,
    replay_su2_reference_linearized_inverse_certificate,
)

cubic = su2_reference_linearized_inverse(7, decay_base=Q(9, 8))
a = cubic["witness"]["arithmetic"]
assert cubic["status"] == "PASS"
assert a["reference_poincare_lower"] == "19/294"
assert a["reference_inverse_l2_upper"] == "294/19"
assert a["weighted_covariance_margin_lower"] == "5/588"
assert cubic["spatial_reference_covariance_verified"]
assert not cubic["actual_vacuum_verified"]
assert not cubic["fourier_nuclear_inverse_verified"]
assert replay_su2_reference_linearized_inverse_certificate(cubic["certificate"])

strip = su2_reference_linearized_inverse(2, family="strip")
assert strip["witness"]["arithmetic"]["reference_inverse_l2_upper"] == "108"
```

## Centering and operator domain

On each finite compact link product, \(A\) is the nonnegative
Friedrichs generator associated to
\[
\langle u,Au\rangle_\nu=\int\sum_e|\nabla_eu|^2\,d\nu.
\]
It has constants as its kernel. For \(\nu\)-centered \(h\), a
Poincare lower bound \(\gamma>0\) gives the unique
\(\nu\)-centered inverse \(u=A_\nu^{-1}h\) with
\[
\|u\|_{L^2(\nu)}\le\gamma^{-1}\|h\|_{L^2(\nu)},\qquad
\mathcal E_\nu(u,u)\le\gamma^{-1}\|h\|_{L^2(\nu)}^2.
\]
The norm statement uses this centered representative.
Subtracting its Haar mean is a different normalization.

Let \(\Pi_H,\Pi_\nu\) denote their respective centering maps.
For smooth Haar-centered inputs the quotient identities are
\[
(\Pi_H A)^{-1}=\Pi_H A_\nu^{-1}\Pi_\nu,\qquad
(I-2C_0^{-1}\Pi_H\Gamma(S_*,\cdot))^{-1}
=\Pi_H A_\nu^{-1}\Pi_\nu C.
\]
The projections cannot be interchanged. The second identity has
the Casimir \(C\) on the input; it is not a bounded inverse
estimate in the original Fourier nuclear norm.

## Uniform finite-family estimates

The families, orientations and unit electric/plaquette weights match
the [Wilson source](gauge-wilson-residual-source.md).
The original-edge reference Hessian row is at most \(2qg/3\),
where \(q=2\) for strips and \(4\) for cubic boxes.
With Ricci \(1/2\), the curvature lower bound is
\[
\rho_*=\frac12-\frac{4qg}{3}.
\]
If positive, it supplies \(\gamma=\rho_*\) on the full original
product space and its gauge-invariant subspace. This yields
\(\gamma=19/294\) at cubic \(\kappa=7\).

For strips there is a second route at every finite \(g\).
Fix all top and vertical links as a spanning tree. The remaining
bottom-link plaquette coordinates give a product reference density
proportional to \(\prod_i e^{(2g/3)\chi(U_i)}\).
Their physical electric form contains the sum of their unit gradient
squares. Haar comparison and tensorization give
\[
\gamma_{\rm strip}\ge\frac34e^{-8g/3}.
\]
The wrapper uses the rational lower bound
\((3/4)(1-\Omega/N)^N\), with \(\Omega=8g/3\) and
`N=exponent_steps`, requiring \(0\le\Omega<N\).
For any finite \(g\), a sufficiently large integer \(N\) meets that
domain. An insufficient chosen \(N\) does not imply noninvertibility.
This strip route applies to gauge-invariant functions in the tree chart.

The optional original-edge spatial covariance estimate uses
\[
m_b=\frac{qg}{6}(1+b)^2,\qquad \rho_b=\frac12-2m_b.
\]
When \(\rho_b>0\), the compact covariance majorant has weighted
row sum at most \(1/\rho_b\). A spatial exponential flag additionally
requires \(b>1\). This estimate uses original-edge coordinates,
independently of the alternative strip product chart.

## What remains unproved by this API

These inverse estimates control \(L^2(\nu)\) forcing. They do not
bound the nonlinear correction in the complete Fourier norm, or
show that \(H^2\) is a multiplication algebra with constants
independent of a growing number of links.
No quantum vacuum construction or target Hamiltonian gap follows
from this certificate alone.

`actual_vacuum_verified`, `fourier_nuclear_inverse_verified`,
`nonlinear_correction_verified` and all continuum/formal tiers
remain false. The companion analytic proof is
`ensemble-laws/docs/constructive/reference-resolvent.md`.

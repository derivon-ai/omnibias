# Uniform large-kappa expansion of the limiting exponential family

This note concerns the **limiting family only**. It does not extend an actual
epsilon-dependent passage estimate to an unbounded kappa interval, and it
does not prove an additional cycle count or full graphic cyclicity.

The subsequent [actual joint theorem](HILBERT16-JOINT-MATCHING.md)
establishes a uniform physical-slope error on the corner
exp(-kappa)->0, epsilon*exp(kappa)->0. It uses this limiting expansion
as one ingredient and proves the expanding-domain comparison separately.
Its joined cycle bounds exclude limiting multiplier resonance.

Fix a compact real coefficient set with

\[
 \Delta=4L-\lambda^2\ge\chi>0,\qquad L\le L_{\max},
 \quad \lambda=\lambda_1.
\]

Thus `L>=chi/4>0`, the two quadratics

\[
 B_\sigma(x)=x^2-\sigma\lambda x+L
\]

are positive for every real `x`, and
`S_sigma(x)=integral_0^x s/B_sigma(s) ds` increases from zero to infinity
on the positive half-line. Define `x_sigma(kappa)>0` by `S_sigma=kappa`,
and `R(kappa)=B_-(x_-)/B_+(x_+)`.

## Explicit constants and expansions

Set

\[
 K_0={\pi\lambda\over\sqrt\Delta},\qquad
 d_0=\sqrt L\exp\!\left[-{\lambda\over\sqrt\Delta}
                    \arctan{\lambda\over\sqrt\Delta}\right],
 \qquad d_\sigma=d_0e^{-\sigma K_0/2}.
 \tag{1}
\]

All these quantities and their reciprocals have uniform finite bounds on
the stated compact set. With `z=e^-kappa`,

\[
 \boxed{
 x_\sigma={d_\sigma\over z}+\sigma\lambda
       -{L\over2d_\sigma}z
       +{\sigma\lambda L\over3d_\sigma^2}z^2+O(z^3).}
 \tag{2}
\]

In particular the requested leading expansion is
`x_sigma=d_sigma exp(kappa)+sigma lambda+O(exp(-kappa))`.
The constant term cancels in the evaluated quadratic:

\[
 B_\sigma(x_\sigma)
 ={d_\sigma^2\over z^2}+{\sigma\lambda d_\sigma\over z}
          +{\sigma\lambda L\over6d_\sigma}z+O(z^2).
 \tag{3}
\]

Therefore

\[
 \boxed{
 R(\kappa)=e^{2K_0}\left[
 1-\lambda\left({1\over d_-}+{1\over d_+}\right)e^{-\kappa}
 +\lambda^2\left({1\over d_+^2}+{1\over d_+d_-}\right)e^{-2\kappa}
 +O(e^{-3\kappa})\right].}
 \tag{4}
\]

In particular

\[
 R_\infty=\exp{2\pi\lambda\over\sqrt\Delta},\qquad
 R-R_\infty=-R_\infty\lambda(d_-^{-1}+d_+^{-1})e^{-\kappa}
                      +O(e^{-2\kappa}).
 \tag{5}
\]

Every remainder is uniform on the fixed compact coefficient set. The same
order holds after any fixed finite number of kappa derivatives. When
`lambda=0`, `x_sigma=sqrt(L)*sqrt(exp(2kappa)-1)` and `R=1` exactly.

For nonzero lambda the sign is also consistent with the exact identity

\[
 {d\over d\kappa}\log R
        =\lambda(1/x_-+1/x_+).
\]

Thus `R` increases to its limit from below when `lambda>0`, and decreases
to its limit from above when `lambda<0`.

## Derivation and uniform remainder proof

The exact elementary primitive is

\[
 S_\sigma(x)={1\over2}\log{B_\sigma(x)\over L}
                     +{\sigma\lambda\over2}I_\sigma(x),
\]
\[
 I_\sigma(x)={2\over\sqrt\Delta}
 \left[\arctan{2x-\sigma\lambda\over\sqrt\Delta}
                   +\sigma\arctan{\lambda\over\sqrt\Delta}\right].
 \tag{6}
\]

Consequently

\[
 I_\sigma(\infty)={\pi+2\sigma\arctan(\lambda/\sqrt\Delta)
                          \over\sqrt\Delta},\qquad
 c_\sigma=-\tfrac12\log L
                     +\tfrac12\sigma\lambda I_\sigma(\infty),
\]

and `d_sigma=exp(-c_sigma)` is exactly (1).

For a rigorous uniform inversion, write `q=1/x`. There is an analytic
function `A_sigma(q)` near zero such that

\[
 S_\sigma(1/q)=-\log q+c_\sigma+A_\sigma(q),\qquad
 A_\sigma(0)=0,
\]
\[
 A_\sigma'(q)={-\sigma\lambda+Lq\over1-\sigma\lambda q+Lq^2}.
 \tag{7}
\]

Choose a common complex disk `|q|<=r0` on which
`|lambda|r0+Lmax*r0^2<=1/2`. The denominator in (7) is then nonzero,
and `A_sigma'` has a uniform bound `M_A`. Shrink `r0` further so that
`M_A*r0<=1/16`. The exact inversion equation is

\[
 z=d_\sigma q\,e^{-A_\sigma(q)}=:\Phi_\sigma(q).
 \tag{8}
\]

On this common disk, `|Phi_sigma'(q)/d_sigma-1|<1/2` and
`|Phi_sigma(q)-d_sigma q|<d_sigma|q|/8`. Since `d_sigma>=d_min>0`,
the equation has a unique analytic inverse for
`|z|<d_min*r0/2`, uniformly in the coefficient set. For example,
Rouche's theorem compares `Phi_sigma(q)-z` to `d_sigma q-z` on
`|q|=r0`, and the derivative bound gives local uniqueness throughout.

The regularized quantity `z*x_sigma(z)=z/q_sigma(z)` is analytic at zero
and uniformly bounded by `d_max exp(1/16)`. Cauchy estimates on smaller
common disks give uniform Taylor remainders to any finite order.
The positive real inverse agrees with the unique `x_sigma(kappa)` by
monotonicity of `S_sigma`.

The coefficients can be read particularly simply from the exact equation

\[
 {dx_\sigma\over d\kappa}
 ={B_\sigma(x_\sigma)\over x_\sigma}
 =x_\sigma-\sigma\lambda+L/x_\sigma.
 \tag{9}
\]

Substituting the analytic Laurent series obtained from (8) determines in
order the constant, linear, and quadratic terms in (2). Substitution into
`B_sigma` gives (3), including its zero constant term; division gives (4).
The normalized denominator `z^2 B_+(x_+(z))` is uniformly separated from
zero on a smaller disk, so this division preserves uniform analytic
remainder bounds. Since `d/dkappa=-z*d/dz`, any fixed number of kappa
derivatives preserves the displayed remainder orders.

## What actual matching is still missing

The [actual exponential transition](HILBERT16-EXPONENTIAL-PASSAGE.md)
converges in `C^1(kappa)` on
each **fixed compact** positive kappa interval. That theorem alone does
not control any diagonal limit with `kappa=kappa(epsilon)->infinity`.
Its radial endpoint constants and derivative bounds can deteriorate as
the endpoints grow.

For the actual canonical family,

\[
 B_{\epsilon,\sigma}(x)
 =L-\sigma\lambda x-x^2\zeta(\sigma\epsilon x,\epsilon)
 =B_\sigma(x)-\sigma\beta_0\epsilon x^3+\cdots.
\]

Since `x_sigma` is of size `exp(kappa)`, its first omitted term has
relative size of order `epsilon exp(kappa)`. This identifies a necessary
smallness scale for that particular coefficient expansion; it is not an
established uniform passage error estimate.

An actual matched result must bound the escape profiles, differentiated
height kernels, physical tails, and endpoint-coordinate factors on
expanding radial domains. To use the correction in (5), it must control
the actual slope error to smaller order than `exp(-kappa)`, not merely
prove convergence to zero. Near `kappa` of order `log(1/epsilon)`, the
physical escape coordinate `V=epsilon*x_sigma` ceases to shrink, so the
full entry-exit coefficients and physical connectors re-enter the
matching. None of those uniform diagonal estimates is proved by this
limiting-family calculation.

## Reusing the existing rigorous register

The reciprocal coordinate in (7) makes the limiting tail regular at zero.
No new distribution or generic asymptotic engine is required to represent
it. The existing pieces can be composed as follows:

- `omnibias.core.verified.taylor_model.TaylorModel.reciprocal` encloses
  the reciprocal quadratic with a geometric analytic remainder once its
  nonzero-denominator and relative-variation conditions hold.
- `omnibias.core.verified.interval.Interval` propagates coefficient
  boxes through the rational tail kernel and its explicit derivatives.
- `omnibias.core.verified.quadrature.trapezoid_integral` encloses a
  tail integral using interval node values and a bound on the second
  derivative over the entire integration interval.
- `omnibias.dynamics.continuation.certify_segment` can certify unique
  finite-dimensional implicit branches over a whole parameter interval,
  provided the residual and Jacobian callbacks enclose those quantities.
- The existing `ValidatedSeries` and `geometric_tail_bound` propagate
  infinite analytic tails after a valid geometric majorant is supplied.
  They can carry a remainder bound derived from the common complex disk
  in the proof above.

The executable composition in
`benchmarks/hilbert16_compactified_tail.py` treats a compact real
coefficient box and a reciprocal-coordinate interval containing zero.
It encloses the limiting rational kernel, its first two derivatives,
and a definite tail integral. The explicit rational derivative formulas
are used; a value-only Taylor remainder is not differentiated.
Deterministic and seeded sample checks supplement the interval arithmetic.
The benchmark also replays the finite expansion identities above.

```bash
python benchmarks/hilbert16_compactified_tail.py --output artifacts/hilbert16/compactified_tail.json
```

This composes existing components for a specific analytic obligation.
The resulting limiting-tail enclosures do not establish the actual
expanding-domain passage estimate. That estimate must retain both small
parameters and the physical endpoint factors, as described above.

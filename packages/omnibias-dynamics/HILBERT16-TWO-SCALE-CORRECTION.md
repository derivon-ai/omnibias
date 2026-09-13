# Two-scale singular correction at negative-lambda multiplier resonance

This note reuses the exact canonical field, the two-height-level actual
matching proof, and the physical event formula already established in the
repository. It does not introduce a new package or infer a derivative of an
error term from a value estimate. Its new conclusion is a value expansion
of the actual physical section derivative. The sharper
[regular-anchor theorem](HILBERT16-REGULAR-ANCHOR.md) now discharges
the regular comparison obligation identified below; its proof is separate
from the singular expansion established here.
This is a written analytic argument, not formal verification or a claim
of novelty. It does not settle full graphic cyclicity or Hilbert XVI.

## 1. Existing mechanics inspected

The relevant existing pieces are:

- [actual joint matching](HILBERT16-JOINT-MATCHING.md), especially the
  actual two-level passage and the exact signed-coordinate formulas in
  sections 6 and 8;
- [limiting compactification](HILBERT16-MATCHING-LIMIT.md), the uniform limiting-family compactification;
- [regular jets](HILBERT16-REGULAR-JETS.md) and [endpoint passage](HILBERT16-ENDPOINT-PASSAGE.md), which give
  the actual fixed-splitting regular derivative and the connected matching
  domain;
- [invariant conics](HILBERT16-INVARIANT-CONICS.md), which already supplies a nearby exact
  invariant-parabola reference and its Darboux cofactor;
- [compensation](HILBERT16-COMPENSATION.md), whose exact first-order polynomial explains
  the first section-coordinate cancellation discussed in section 5 below.

The regular comparison uses an actual asymptotic estimate inside these
existing regular-passage mechanics. The companion proves it using the
invariant parabola and a matched-endpoint multiplier functional.

## 2. Statement for the actual singular derivative

Use the fixed weighted chart `r=-1` and the established canonical family.
Fix the section radius `rho>0` first and a compact coefficient set with

\[
 L=-\lambda_0>0,\quad\lambda=\lambda_1<0,\quad
 \Delta=4L-\lambda^2\ge\chi>0.
\]

For the resonance statement below the negative lambda is bounded away from
zero automatically. More generally this can be imposed on the coefficient
compact. Put

\[
 \omega=e^{-\kappa},\qquad u=\epsilon/\omega,
 \qquad\epsilon=u\omega,\qquad
 H=\epsilon^3 e^{-\kappa/\epsilon}.
\]

Define

\[
 d_\sigma=\sqrt L\exp\!\left[
 -{\lambda\over\sqrt\Delta}\arctan{\lambda\over\sqrt\Delta}
 -{\sigma\pi\lambda\over2\sqrt\Delta}\right],
 \qquad\sigma\in\{-1,+1\},\qquad \beta_0=1/3.
\]

Then the actual physical normal-forward derivative has the uniform expansion

\[
 \boxed{
 \log D_\epsilon'(t_i(\kappa))
 = {2\pi\lambda\over\sqrt\Delta}
 -\lambda(d_-^{-1}+d_+^{-1})\omega
 +3\beta_0(d_-+d_+)u
 +O\!\left(\omega^2+u^2+\epsilon\log(1/\epsilon)\right).}
 \tag{1}
\]

The constants are uniform for the fixed coefficient and physical-section
compacts, with sufficiently small independent upper cutoffs on omega and u.
The error is `o(omega+u)` at arbitrary relative rates. Thus both displayed
corrections are positive when lambda is negative. In particular,

\[
 \log D_\epsilon'-{2\pi\lambda\over\sqrt\Delta}>0
 \tag{2}
\]

throughout a sufficiently small two-parameter corner. This conclusion concerns
the singular derivative relative to its limiting value; it is not yet a
comparison with the actual regular derivative.

## 3. Uniform value expansion without differentiating a remainder

The exact canonical coefficient has the stronger identity

\[
 \zeta(V,\epsilon)=-1+\beta_0V+\epsilon V Z(V,\epsilon),
 \tag{3}
\]

where Z and the required finite derivatives are uniformly bounded on a fixed
compact chart. Indeed `zeta(0,epsilon)=-1` for every epsilon, while
`zeta(V,0)=-1+V/3`. Joint analyticity and division first by epsilon and then
by V establish (3). No limiting coefficient is substituted in the exact
height-dependent fast term.

Use the exact joint variables

\[
 V=\sigma uX,\qquad h=\epsilon u^2Y,
\]
\[
 \epsilon Y_X={XY\over b_\sigma(X)+\widehat k_\sigma(X,Y)Y},
 \qquad Y(0)=\omega^2e^{-\kappa/\epsilon}.
\]

On the fixed normalized interval used in the actual matching proof,

\[
 Q_\sigma(X)=X^2-\sigma\lambda\omega X+L\omega^2,
\]
\[
 b_\sigma(X)=Q_\sigma(X)-\sigma\beta_0uX^3
                                     +O(\epsilon uX^3),
 \tag{4}
\]

and both Q and b are bounded below by a common positive multiple of
`omega^2+X^2`. Expanding the reciprocal with its exact algebraic remainder
gives, uniformly for `a<=X<=K`, with a fixed positive a,

\[
 S_{\epsilon,\sigma}(X)
 :=\int_0^X{s\over b_\sigma(s)}\,ds
 =S_\sigma(X/\omega)
  +\sigma\beta_0u\int_0^X{s^4\over Q_\sigma(s)^2}\,ds
  +O(u^2+\epsilon u).
 \tag{5}
\]

For example, the quadratic reciprocal error is bounded by
`C*u^2*integral_0^X s^7/(omega^2+s^2)^3 ds<=C*u^2`; the coefficient
remainder contributes `C*epsilon*u`. Split the remaining integral at
`s=omega`. On `[0,omega]` its difference from the integral of 1 is
`O(omega)`. On `[omega,K]`, the integrand differs from 1 by at most
`C*omega/s`. Hence

\[
 \int_0^X{s^4\over Q_\sigma(s)^2}\,ds
        =X+O\!\left(\omega[1+\log(1/\omega)]\right).
 \tag{6}
\]

The existing limiting primitive supplies

\[
 S_\sigma(X/\omega)
 =\kappa+\log X+c_\sigma-\sigma\lambda\omega/X+O(\omega^2),
 \qquad c_\sigma=-\log d_\sigma.
\]

Combining these estimates,

\[
 S_{\epsilon,\sigma}(X)-\kappa
 =\log X+c_\sigma-\sigma\lambda\omega/X
                       +\sigma\beta_0uX+O(\mathcal E),
 \tag{7}
\]

where throughout the note one may use

\[
 \mathcal E=\omega^2+u^2+\epsilon\log(1/\epsilon).
\]

Let `X_1` be the actual first level `Y=omega^2`, not a limiting escape
coordinate. The exact logarithmic identity in the joint proof gives

\[
 0\le S_{\epsilon,\sigma}(X_1)-\kappa\le C\epsilon.
\]

The previously proved localization `X_1=d_sigma+O(omega+u)` and (7) then
give, by Taylor expansion on a fixed interval separated from zero,

\[
 \boxed{X_1=d_\sigma+\sigma\lambda\omega
                          -\sigma\beta_0d_\sigma^2u+O(\mathcal E).}
 \tag{8}
\]

The actual compensated exponent through this level differs from
`log[b_sigma(X_1)/(L*omega^2)]` by
`O(omega^2+exp(-c/epsilon)+epsilon*u)`, as already proved using the
exact khat and its partial X derivative. Substituting (8) into (4) gives

\[
 \log b_\sigma(X_1)
 =2\log d_\sigma
       +{\sigma\lambda\over d_\sigma}\omega
       -3\sigma\beta_0d_\sigma u+O(\mathcal E).
 \tag{9}
\]

The factor 3 in the u term has two contributions: the change in the escape
location contributes `-2*sigma*beta0*d_sigma*u`, and the explicit cubic
coefficient in b contributes the remaining `-sigma*beta0*d_sigma*u`.

After `Y=omega^2`, the actual second-height argument gives a transition
width `O(epsilon*kappa)` before `Y=1`, followed by a compensated exponent
of size `O(epsilon*log(1/epsilon))`. The physical tail contributes at most

\[
 C[\epsilon\log(1/u)+\epsilon^2/u+\epsilon].
\]

Here `epsilon^2/u=u*omega^2<=omega^2` for u below 1. The logarithms are
included in E. The exact moving physical section factors contribute
`O(epsilon)` in their logarithms. These are estimates for the actual
height-dependent field all the way to the actual physical events.

Consequently the complete physical exponents satisfy

\[
 \Psi_\sigma=2\kappa+2\log d_\sigma-\log L
       +{\sigma\lambda\over d_\sigma}\omega
       -3\sigma\beta_0d_\sigma u+O(\mathcal E).
\]

The exact physical variation ratio cancels the common central prefactor
and subtracts the incoming sign `+` exponent from the outgoing sign `-`.
This proves (1). All changes of scale are frozen at the reference center
height when this physical derivative is evaluated.

Finally,

\[
 {\mathcal E\over\omega+u}
 \le \omega+u+u\log(1/u)+\omega\log(1/\omega)
 \longrightarrow0.
 \tag{10}
\]

This proves the uniform little-o assertion without assuming that either
small scale dominates the other. No kappa derivative of (1), (8), or the
error term has been asserted.

As an algebraic consistency check, the positive-base normal-forward map
has limiting form `M(B)=a*B/(1-b*B)`, with
`a=d_-/d_+`, `b=beta0*(a+1)`. Its squared-label derivative is
`a^2/(1-b*B)^3`. The u coefficient in its logarithm at `B=d_+*u` is
`3*b*d_+=3*beta0*(d_-+d_+)`, exactly the independently derived coefficient
above. This check is not used as a nonuniform application of that theorem
at a zero fiber.

## 4. What this says, and does not say, at resonance

Let `c(C)=exp(-4*pi/sqrt(C-1))`. At negative-lambda limiting resonance

\[
 (C+3)\lambda^2=16L,
\]

the leading constant in (1) equals `log c(C)`. Define the exact regular
deviation, at the same fixed splitting parameter, by

\[
 E_{\rm reg}(t)=\log H_{\rm reg}'(t)-\log c(C).
\]

The actual comparison has the rigorously separated form

\[
 \boxed{
 \log D_\epsilon'(t_i)-\log H_{\rm reg}'(t_i)
 = A_*\omega+B_*u-E_{\rm reg}(t_i)+O(\mathcal E),}
 \tag{11}
\]

where `A_*=-lambda*(d_-^-1+d_+^-1)>0` and
`B_*=3*beta0*(d_-+d_+)>0`.

The existing regular theorem only bounds

\[
 |E_{\rm reg}|\le30\rho+19/R+2K_w(\rho+\zeta),
 \qquad R=\rho/\nu.
\]

For fixed rho and zeta this is not `o(omega+u)`, and its sign is unknown.
It therefore cannot establish the sign of (11). This is a limitation of
the proved estimate, not a counterexample in the actual quadratic family.
Even taking rho very small first does not fix this: omega+u subsequently
tends to zero. Letting rho shrink with the two scales would require a new
uniform-rho audit of constants and domains.

Thus it is now valid to conclude that the actual singular slope is above
its resonant limiting value. It is still invalid to conclude that it is
above the actual regular slope merely from the existing O(rho) bound.

## 5. Existing regular cancellations and a precise remaining bridge

The exact signed-coordinate ratio has, for bounded input and output labels,
the small-rho expansion at nu=0

\[
 \log\left|{Z_-'(t_i)\over Z_+'(t_o)}\right|
          =6\rho+(t_o-t_i)\rho^2+O(\rho^3).
 \tag{12}
\]

This alone is not the actual regular multiplier. The existing compensated
first-order polynomial for p=0, r=-1, alpha1=3 is

\[
 U=-x^3/2-3x^2/2-Cx/2-C.
\]

With the notation `b,B,h,q` of the compensation note, the exact formal
first-order variation of the regular logarithmic variational integrand
along `w=nu*U` is

\[
 I_1(x)={B\over q}+{-2xh+4xU+b\over q^2}
                     =-3+O(1/x).
 \tag{13}
\]

Its symmetric integral has leading `-6R`. Under `R=rho/nu`, that formal
term is `-6rho`, canceling (12)'s `+6rho`. This verifies that the leading
section effect can cancel; it does not justify resumming a fixed-rho
actual expansion by interchanging nu->0 and R->infinity. Higher formal
terms may also contribute powers of rho.

The repository already contains an exact alternative to a formal truncation:
the invariant-parabola graph and its Darboux identity. In the present
canonical sector its physical parameters obey

\[
 \mu_2=3\nu^2+O(\nu^3),\quad \mu_3=-\nu,
 \quad\mu_1=-2\nu^3+O(\nu^4).
\]

The exact parabola with the same mu2,mu3 has

\[
 \mu_{1,\rm par}=-2\nu^3+O(\nu^4),\qquad
 A_{\rm par}=1+3\nu+(6-2C)\nu^2+O(\nu^3).
\]

Thus the quadratic-height coefficient discrepancy is only `O(nu^4)`.
Comparing the actual admitted passage in that exact Darboux coordinate is
a concrete way to seek an estimate such as

\[
 |E_{\rm reg}(t)|
 \le C_\rho[\nu\log(1/\nu)+|t|+|H_{\rm reg}(t)|].
 \tag{14}
\]

Equation (14) was the sufficient estimate sought after the singular
calculation. It is now proved by the
[regular-anchor companion](HILBERT16-REGULAR-ANCHOR.md), with log(2/nu)
in place of log(1/nu). That argument holds the splitting fixed during
the section derivative and controls the exact reference tail and endpoint
factors as well as the actual matched-parameter variation.

The already proved actual joint height and event bounds imply
`t_i,t_o=O_rho(u^2+epsilon)`: at the physical events
`h=V^2/2+O(u^2+epsilon)`, while
`(v-1)^2=V^2+O(nu)`. At a cycle, `H_reg(t_i)=t_o`. Applying the now-proved
regular estimate gives `E_reg(t_i)=o(omega+u)` at every cycle in the
joint corner. Equation (11) consequently gives a strict negative
displacement derivative `(H_reg-D_epsilon)'` at those resonant zeros.
The [exact-resonance synthesis](HILBERT16-EXACT-RESONANCE.md) supplies
the initial, compact-band and positive-base comparisons and their
connected coverage. It proves the scoped at-most-one conclusion on that
slice; varying detuning remains the next obligation below.

## 6. The near-resonant leading model has two scales

Even with the sharp regular estimate, a count uniform through a
varying nonzero resonance detuning requires another argument. At the level
of the displayed slope model, with `epsilon=omega*u`, write

\[
 G(\kappa)=\Gamma+a e^{-\kappa}+b\epsilon e^\kappa,
 \qquad a,b>0.
\]

The positive correction has minimum `2*sqrt(a*b*epsilon)`, attained at
`omega=sqrt(b*epsilon/a)`. If
`Gamma<-2*sqrt(a*b*epsilon)`, the model has two distinct slope zeros,
given in omega by

\[
 \omega_\pm={-\Gamma\pm\sqrt{\Gamma^2-4ab\epsilon}\over2a}.
\]

When `Gamma->0-` and `epsilon/Gamma^2->0`, both roots lie in the joint
corner: one has `omega` comparable to `|Gamma|`, the other has `u`
comparable to `|Gamma|`. Thus a two-cycle bound through every varying
detuning is not an automatic consequence of a positive correction on
the exact resonance locus. A potential bound on three displacement zeros
requires control of derivatives or an appropriate zero-count operator
for the actual remainder. The subsequent
[actual kappa-jet proof](HILBERT16-SINGULAR-KAPPA-JETS.md) supplies that
derivative control, and the [varying-detuning synthesis](HILBERT16-VARYING-DETUNING.md)
gives the scoped uniform three-zero bound. This value-expansion note
does not assert actual realization of the two model slope roots.

## Finite verification

```bash
python benchmarks/hilbert16_joint_matching.py --output artifacts/hilbert16/joint_matching.json
```

The shared benchmark checks the finite expansion and model algebra along
with the earlier scaling and interval multiplier-gap examples. It does
not certify the analytic remainder constant, its physical smallness
cutoffs, the sharp regular analytic estimate, or a resonant cycle count.

# A uniform displacement bound through varying multiplier detuning

This synthesis uses the actual singular kappa-derivative estimate, the
fixed-field regular section jets, and the existing physical sections and
connected admission proof. It concerns a selected canonical quadratic
itinerary. It is a written analytic argument with independent agent reviews, not a formally
verified physical cycle bound, a novelty claim, or full Hilbert XVI.

## 1. Uniform parameter and itinerary scope

Fix the established r=-1 canonical slow family on a coefficient compact:

    C in [Cmin,Cmax] with Cmin>1,
    L=-lambda0>=Lmin>0,
    lambda=lambda1<=-lambda_min<0,
    Delta=4L-lambda^2>=chi>0,
    A=1+nu*Abar, with Abar bounded,
    epsilon=nu*k_scale, with k_scale uniformly positive.

All slow coefficients are bounded. The canonical embedding, including
mu3=-nu, mu2=3*nu^2+O(nu^3) and mu1=-2*nu^3+O(nu^4), is retained.
Those errors are uniform analytic coefficient errors, not free changes
to the physical family. Define

    Gamma=2*pi*lambda/sqrt(Delta)+4*pi/sqrt(C-1).

The negative-lambda resonance Gamma=0 is exactly
`(C+3)*lambda^2=16L`. There are common positive choices of a sufficiently
small section radius rho, signed input/output boxes, Hmax, Gamma0 and
epsilon0 such that the following implication holds. Every field of this
compact with `|Gamma|<=Gamma0` and `0<epsilon<epsilon0` has at most
three distinct cycles admitted by the selected connector itinerary and
`0<H<=Hmax`.

The detuning can depend arbitrarily on epsilon within that fixed interval.
In particular, the statement does not require separation of Gamma from
zero, a rate bound on Gamma/sqrt(epsilon), or hyperbolicity of the counted
cycles. It asserts an upper bound, not existence of three cycles. Every
positive height and every nearby cycle are not asserted to be admitted.

Each kappa derivative below fixes every physical coefficient. Set

    H(kappa)=epsilon^3*exp(-kappa/epsilon),
    omega=exp(-kappa), u=epsilon*exp(kappa).

The actual signed input t_i is strictly increasing with kappa. The two
actual maps are the normal-forward singular map D and the regular map
Hreg of this same fixed field. The physical closing map is D inverse.
The connected admission interval and the actual positive-label overlap
are those in [the joined proof](HILBERT16-JOINED-POSITIVE-BASE.md).

## 2. Strict curvature on the complete joint interval

For each radial sign define

    d_sigma=sqrt(L)*exp[-lambda/sqrt(Delta)*atan(lambda/sqrt(Delta))
                              -sigma*pi*lambda/(2*sqrt(Delta))],
    a_star=-lambda*(1/dminus+1/dplus), b_star=dminus+dplus.

Both coefficients have common strictly positive lower bounds. The
[actual singular derivative estimate](HILBERT16-SINGULAR-KAPPA-JETS.md)
gives, for j=0,1,2 and independent sufficiently small omega,u cutoffs,

    |partial_kappa^j [log D'(ti(kappa))
            -2*pi*lambda/sqrt(Delta)-a_star*omega-b_star*u]|
        <=Ks*(omega^2+u^2+epsilon*log(1/epsilon)),

    |ti_kappa|+|ti_kappakappa|<=Kl*u^2.

These are actual variational estimates with moving physical events; the
derivatives are not inferred from a value remainder. The exact common
central sensitivity is cancelled before estimating the two radial sides.

On the admitted regular interval the
[fixed-field regular theorem](HILBERT16-REGULAR-KAPPA-JETS.md) gives
uniform bounds L1,L2 for the first two t derivatives of log Hreg'. Thus

    |partial_kappa^2 log Hreg'(ti(kappa))|
        <=L2*ti_kappa^2+L1*|ti_kappakappa|<=Kr*u^2.

The regular output need not equal the singular output here. No matching
parameter is varied, and no small regular value error is assumed away
from cycles. Define the actual logarithmic multiplier difference

    G(kappa)=log D'(ti(kappa))-log Hreg'(ti(kappa)).

For a common positive lower bound c_star for both coefficients and a common K,

    G_kappakappa >= c_star*(omega+u)
                   -K*(omega^2+u^2+epsilon*log(1/epsilon)).

The last error divided by omega+u tends uniformly to zero as independent
omega,u cutoffs shrink, because epsilon=omega*u. For example it is at
most

    omega+u+u*log(1/u)+omega*log(1/omega).

Choose the two cutoffs so that the error is at most
`(c_star/2)*(omega+u)`. Then G is strictly convex throughout the actual
joint admission interval. Constants and cutoffs are independent of the
subsequently chosen detuning and of its rate of approach to zero.

## 3. Uniform negative crossings outside the joint interval

Only the sign of the displacement derivative at its zeros is needed in
the outer regions. Preserve the existing initial-height and thin-height
estimate. It gives a strict negative displacement derivative up to a
fixed positive kappa_star, uniformly over this coefficient compact.

On any fixed band `[kappa_star/2,kappa_b+1]`, the actual singular slope
converges uniformly to its limiting slope R(kappa). Negative lambda gives

    log R(kappa)-2*pi*lambda/sqrt(Delta)>0.

The chosen band and parameter compact supply a common gap delta_band>0.
The signed singular endpoints tend to zero on that fixed band. At every
cycle, the [sharp regular estimate](HILBERT16-REGULAR-ANCHOR.md) therefore
gives

    G(kappa)>=Gamma+delta_band-o(1).

After choosing `Gamma0<=delta_band/4` and then epsilon sufficiently small,
G>0 at every cycle in that band. This is a value comparison at actual
cycle points, with the same fixed-field section derivatives.

For the remaining fixed positive-base interval, write B=sqrt(ti) and
use the actual compact convergence to

    D0(t)=a^2*t/(1-b*sqrt(t))^2,
    a=exp(pi*lambda/sqrt(Delta)), b=(a+1)/3.

Choose Bmax sufficiently small after rho so that b*Bmax<=1/2 and the
quadratic regular error is dominated by the linear singular correction.
As in the [exact-resonance proof](HILBERT16-EXACT-RESONANCE.md), at cycles
on a fixed interval `[Bmin,Bmax]` one obtains

    G>=Gamma+3*bmin*B-Krho*B^2-Krho,Bmin*nu*log(2/nu),

where bmin is a positive lower bound for b. First require
`Krho*Bmax<=bmin`. Then choose `Gamma0<=bmin*Bmin/4` and reduce nu so
that the last error is at most `bmin*Bmin/4`. This gives G>0 at every
cycle in the positive-base interval. Constants multiplying the vanishing
nu error may depend on the subsequently fixed Bmin; no estimate uniform
as Bmin tends to zero is asserted.

## 4. Noncircular choices and coverage

Make choices in this order, initially using the entire fixed coefficient
compact, rather than a compact shrinking with epsilon:

1. Choose rho for the regular endpoint theorem, the signed-coordinate
   jet bounds and the sharp invariant-parabola comparison.
2. Choose Bmax and small signed boxes for the positive-base comparison,
   with output margin for the limiting images. Choose Hmax for the
   existing initial/grazing theorem and these boxes.
3. Choose the initial kappa_star. Choose fixed kappa_b>kappa_star+1
   sufficiently large for the strict-convexity omega cutoff. Choose
   fixed u0 sufficiently small for its u cutoff and the positive anchors.
4. Choose Bmin below half the smallest incoming base anchor on
   `u in [u0/2,u0]`. The anchor is strictly positive by the
   [actual endpoint theorem](HILBERT16-JOINT-ANCHOR.md).
5. Choose the auxiliary paper-section height h_paper for the
   [ordinary connectors](HILBERT16-SINGULAR-TRANSPORT.md). It is smaller
   than the common positive outer-section height for the already fixed
   rho and sufficiently small for the entry-exit theorem on the fixed
   positive base intervals. The strict pole and sign margins in the
   chosen boxes permit this choice without changing rho or Bmin.
6. Choose Gamma0 below both outer-region gap requirements above.
7. Finally choose epsilon0 to meet all finite actual-passage convergence,
   event, overlap and error cutoffs, including
   `log(u0/epsilon)>kappa_b+1`.

The joint interval has boundaries kappa_b and log(u0/epsilon), and it
overlaps the fixed band and the positive-base interval. The existing
anchor and strict center-height monotonicity show that every remaining
admitted height below

    Hcut=epsilon^3*(epsilon/u0)^(1/epsilon)

has positive input between Bmin squared and Bmax squared. Together with
the initial and fixed bands this covers every admitted `0<H<=Hmax`.
The fixed-field regular admission can truncate or miss an overlap; its
intersection with the singular admission remains an interval. The
singular overlap is used for coverage and map identification, rather than
assuming that its anchor itself is a cycle or is regularly admitted.

## 5. One global zero-count argument

Write the displacement in the increasing kappa coordinate as

    f(kappa)=Hreg(ti(kappa))-D(ti(kappa)).

On the joint interval the exact identity is

    f_kappa=p(kappa)*(1-exp(G(kappa))),
    p(kappa)=ti_kappa*Hreg'(ti(kappa))>0.

Thus G=0 if and only if f_kappa=0 there. Strict convexity gives at most
two G zeros and the [checked convexity/Rolle implication](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Resonance.lean) gives at most three f
zeros on that interval, once these actual derivative hypotheses hold.
The following argument joins the outer regions without adding counts.

Suppose four distinct displacement zeros are admitted. Restrict to the
closed interval I between the first and last of these four. It is wholly
admitted by connectedness. Let J be its intersection with the closed
joint interval. If J is empty, all zeros in I have negative derivative,
which already excludes two of them by the downward-zero lemma.

Otherwise, strict convexity makes

    K={kappa in J : G(kappa)<=0}

a compact interval, possibly empty or a singleton. If K is nondegenerate,
strict convexity gives G<0 in its interior, so f is strictly increasing
on K and has at most one zero there. A singleton also contains at most
one zero. Each of the at most two interval components of I outside K
has strictly negative derivative at every zero: inside J because G>0,
and outside J by the outer comparisons. The downward-zero lemma excludes
two zeros in either component. Consequently I contains at most three
zeros, a contradiction. If K is empty the same lemma applies to all of I.

This treats nonhyperbolic zeros at G=0 and possible interval endpoints;
it does not assume in advance that the whole zero set is finite. The
negative-at-zeros implication is checked in
[Hilbert16Rolle.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean).
The increasing-core count and the implication from a nonpositive convex
gap to an increasing core are also checked in
[Hilbert16Resonance.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Resonance.lean).
Selecting the endpoints of K by compactness and convexity remains the
written topological step above. The actual analytic hypotheses are
supplied by the written estimates, not by the finite arithmetic certificates.

## 6. Remaining obligations

This uniform bound retains strict Delta, positive L, bounded negative
lambda, C>1, fixed small physical sections and the selected itinerary.
Coalescing roots, shrinking L, escaping endpoints, other charts, full
graphic coverage and all-degree coverage remain separate open
obligations. The curve/surface classification work is preserved in
[its own target](HILBERT16-ALGEBRAIC-TARGET.md).

No neural surrogate replaces the actual vector field in this argument.
The new joint jets and exact realization/replay mechanism check the
fixed-epsilon derivative algebra as described in
[the verification manifest](HILBERT16-FORMAL.md). A fitted model using
confluent representations or continuation would need sound error
enclosures before it could support an actual passage estimate. The
uniform analytic estimates and selected-itinerary physical bound have not been
formalized in Lean, and no computed sharp cutoff is claimed.

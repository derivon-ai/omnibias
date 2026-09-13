# One cycle at most on the exact limiting resonance

This is a written analytic synthesis for the selected small-label canonical
quadratic itinerary, using the actual sharp regular multiplier and actual
two-scale singular correction. It assumes the existing strict positive
Delta margins and fixes the limiting resonance exactly. It does not prove
a bound uniform through varying negative resonance detuning, full graphic
cyclicity, or Hilbert XVI. The physical analytic hypotheses are not formally
verified. A separately checked abstract zero-count implication does not
remove those hypotheses.

## 1. Statement and parameter choices

Use the r=-1 canonical slow embedding, with C in a compact subset of
(1,infinity), L=-lambda0>=c0>0, bounded negative lambda=lambda1,
Delta=4L-lambda^2>=chi>0, and bounded Abar in A=1+nu*Abar. Impose

    (C+3)*lambda^2=16L, lambda<0.

Write a=exp(pi*lambda/sqrt(Delta)), b=(a+1)/3, and
c(C)=exp(-4*pi/sqrt(C-1)). The resonance means a^2=c(C).
On this compact lambda is bounded negatively away from zero.

Fix rho sufficiently small for the regular endpoint theorem, its strict
contraction Hreg'<=theta<1, and the sharp regular multiplier estimate.
There are then sufficiently small input/output signed-label boxes about
zero, a fixed Hmax>0, and epsilon0>0 such that every field in this compact
with 0<epsilon<epsilon0 has at most one cycle in the selected connector
itinerary admitted by 0<H<=Hmax. If such a cycle exists it is hyperbolic
and attracting in physical time.

The canonical embedding is essential: mu3=-nu, mu2=3*nu^2+O(nu^3),
and mu1=-2*nu^3+O(nu^4), with uniform analytic errors. The
[sharp regular estimate](HILBERT16-REGULAR-ANCHOR.md) and
[two-scale correction](HILBERT16-TWO-SCALE-CORRECTION.md) use this same field.

The boxes and Hmax may be reduced after rho. They retain the fixed
positive-height connector hypotheses at V=+-1 and the fine regular
matching tube. The connected admission proof and exact positive-label
joining anchor are the same ones used in the nonresonant positive-base
join. There is no assertion that every height or every nearby cycle
belongs to this itinerary.

Set F(t)=Hreg(t)-D(t) in the actual signed section label. At a zero,
ti=t and to=Hreg(t)=D(t). Derivatives hold the physical field fixed.
It will suffice to prove F'(t)<0 at every zero in the connected admitted
interval. Positivity of both map derivatives then makes the physical
return multiplier Hreg'/D' strictly between zero and one.

## 2. The initial and compact exponential regions

The existing thin-height and grazing estimates already give F'<0 on

    H>=epsilon^3*exp(-kappa_star/epsilon)

within the admitted height range, for a fixed kappa_star>0 determined
after the regular contraction and coefficient compact are fixed.

On any fixed positive band [kappa_a,kappa_b+1], with
kappa_a=kappa_star/2, the actual exponential theorem gives
D'(ti(kappa))->R(kappa) uniformly, and ti,to->0 uniformly. Here

    R(kappa)>R_infinity=a^2=c(C)

for every finite kappa, because the limiting multiplier is strictly
decreasing for negative lambda and has limit a^2. Compactness supplies
a positive absolute gap on the fixed band.

At every cycle in this band, the sharp regular theorem gives

    |log Hreg'(ti)-log c(C)|
       <=K_rho*(nu*log(2/nu)+|ti|+|to|)->0.

Consequently D'>Hreg' and F'<0 at all such zeros for sufficiently
small nu. Only a value convergence statement is used here.

## 3. The actual joint corner

Use omega=exp(-kappa), u=epsilon/omega and epsilon=omega*u. The proved
actual singular expansion is

    log D'=log c(C)+Astar*omega+Bstar*u+O(E),
    Astar=-lambda*(1/dminus+1/dplus)>0,
    Bstar=dminus+dplus>0,
    E=omega^2+u^2+epsilon*log(1/epsilon).

Both coefficients have uniform positive lower bounds on the compact.
The actual physical endpoint estimate gives |ti|+|to|<=K_rho*(u^2+epsilon).
At a cycle the sharp regular estimate therefore yields

    |log Hreg'(ti)-log c(C)|
       <=K_rho*(u^2+epsilon*log(2/epsilon)).

Since nu and epsilon are uniformly comparable, all constants remain
uniform. Combining these bounds gives at every cycle in the corner

    log(D'/Hreg')>=c1*(omega+u)-K_rho*E.

The proved E/(omega+u)->0 is uniform at arbitrary relative rates.
Choose independent omega and u cutoffs sufficiently small to obtain
log(D'/Hreg')>0. Thus F'<0 at every cycle in this corner. This is a
statement at zeros of F, not an assertion about arbitrary unmatched
regular outputs elsewhere in the admission interval.

Choose kappa_b>kappa_star+1 sufficiently large for the small-omega
cutoff, then a fixed u0 sufficiently small for this corner and for the
positive-label anchor. The previous compact band overlaps this corner.

## 4. The fixed positive-base interval

For B=sqrt(t) in a fixed [Bmin,Bmax], Bmin>0 and 1-bB>=ell>0,
the existing actual singular theorem gives

    D0(t)=a^2*t/(1-bB)^2,
    log D0'(t)=log c(C)-3*log(1-bB),
    D=D0+O_C2,rho,Bmin(nu*log(1/nu)).

Shrink Bmax first so that bBmax<=1/2 uniformly. Then at any cycle in
this interval,

    |ti|+|to|<=K0*B^2+K1*nu*log(2/nu),

where K0 can be chosen independently of Bmin: the limiting output
D0(t)<=4*amax^2*B^2. The coefficient K1 and the final nu cutoff may
depend on the fixed Bmin. The sharp regular estimate, the lower bound
on the positive D0', and actual C1 convergence imply

    log(D'/Hreg')
       >=3*bmin*B-K2_rho*B^2-K3_rho,Bmin*nu*log(2/nu).

Choose Bmax small enough that K2_rho*Bmax<=bmin. This is permissible
after rho and the sharp regular constant are fixed, and before Bmin
or the final small-parameter cutoff are selected. Then choose nu small
enough for the remaining error to be at most bmin*Bmin. The right side
is at least bmin*B>0. Thus F'<0 at every positive-base cycle.

For precise choice order, choose the small signed boxes with this Bmax
bound after rho, then Hmax for the initial estimate, then kappa_star,
kappa_a,kappa_b and a smaller u0 whose limiting positive anchors lie
strictly inside the boxes. Choose Bmin less than half the smallest
incoming anchor on u in [u0/2,u0]. Finally choose nu to meet every
finite cutoff. The exact positive-label anchor supplies overlap and
the center-height monotonicity proves coverage of every admitted
0<H<=Hmax by the initial, compact, joint and positive-base regions.
There is no circular requirement of a nu cutoff uniform as Bmin->0.

## 5. Downward crossings and the count

Let f be C1 on an interval. If f'(x)<0 at every zero, then it has at
most one zero. Here is an elementary proof without assuming finiteness
of all zeros. Suppose a<b are two zeros. Negative derivatives supply
p,q with a<p<q<b, f(p)<0 and f(q)>0. The zero set in [p,q] is nonempty
by the intermediate value theorem, closed and compact. Let c be its
smallest point. It satisfies p<c<q, and f<0 on [p,c), since a sign
change would produce an earlier zero. But f'(c)<0 forces f>0 just to
the left of c, a contradiction. The same proof only needs continuity
on the closed subinterval and differentiability at its zeros.

Apply this lemma to the connected admitted displacement interval from
the positive-base join. The preceding four regions cover it, and F'<0
at every zero. Therefore there is at most one zero and at most one
physical cycle. Its positive multiplier Hreg'/D'<1 proves the stated
hyperbolic attraction.

This settles the exact limiting-resonance slice only within the stated
compact canonical sector and small-label itinerary. A uniform bound
through varying negative detuning still requires additional control:
the leading slope model Gamma+Astar*omega+Bstar*epsilon/omega can
have two zeros near the balanced scale. No derivative bound for the
actual joint remainder, broad graphic coverage, coalescing-root result,
or full-Hilbert conclusion is inferred here.

The abstract downward-zero implication is checked in
[Hilbert16Rolle.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean).
The physical estimates above remain written analytic proofs. Attraction
is pointwise for each field; a uniform multiplier gap as epsilon tends
to zero and existence of a cycle are not asserted.

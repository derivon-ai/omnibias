# Discriminant and pole boundary reductions for the restricted itinerary

This is a written analytic argument for the same restricted normal/regular
itinerary as [the interior criterion](HILBERT16-INTERIOR-CYCLICITY.md).
It supplies no full-graphic
capture theorem and does not cover zero or unbounded fiber labels. No
uniform interior C2 remainder is used at discriminant zero.

## 1. Exact central chart

Write the exact normal-forward family as

    Vdot = epsilon [epsilon^2 lambda0 + epsilon lambda1 V
                    + V^2 zeta(V,epsilon;theta)] + h g(V,h,epsilon;theta),
    hdot = -V h,

where zeta(V,0;theta)=-1+beta V, beta>0, and g(V,h,0;theta)=-1.
The coefficients are jointly analytic on the common neighborhoods already
established in the singular-transport companion. In particular
zeta(0,epsilon;theta)=-1 exactly, by the coefficient normalization.

Set

    s=lambda1/2, lambda0=-s^2-delta^2, delta>=0,
    u=V/epsilon, W=h^epsilon/epsilon, tau=epsilon^2 t.

At fixed epsilon>0 these are ordinary changes of variables for h>0.
The exact equations are

    du/dtau = -[(u-s)^2+delta^2] + E_epsilon(u;theta)
              + epsilon^-3 (epsilon W)^(1/epsilon)
                g(epsilon u,(epsilon W)^(1/epsilon),epsilon;theta),
    d(log W)/dtau = -u,

where E=u^2[zeta(epsilon u,epsilon;theta)+1]. On any fixed |u|<=K,
|E|<=C_K epsilon. For bounded W the final term tends to zero faster than
any power of epsilon, uniformly in the compact passive parameters.

The leading polynomial alone does NOT determine the actual finite-epsilon
fold: at u=s its perturbation contains epsilon*beta*s^3+O(epsilon^2).
Thus comparing delta^2 only with zero, while ignoring delta^2~epsilon,
would be invalid. The proof below absorbs this entire perturbation.

## 2. Uniform outer half-maps up to u=+/-K

Fix compact positive input/output labels

    B in [Bmin,Bmax], U in [Umin,Umax],
    1-beta B >= omega > 0,

and a fixed compact coefficient set, allowing delta=0. Choose K large
enough that K>sup|s|+1 and both polynomials

    Q_plus(v)=lambda0 v^2+lambda1 v-1,
    Q_minus(v)=lambda0 v^2-lambda1 v-1

stay below -1/2 for 0<=v<=1/K. Enlarge K if the common local normal-form
sections require it. Restrict the constructions in the proofs of HK
Lemmas 3.1 and 4.3 to these outer half-transitions: they never enter the
slow bottleneck at u=s. This is a local extraction of those proofs, not
an assertion of the paper's standing global hypotheses at delta=0.
The construction here involves Q_plus/minus near zero and the slow
coefficient along fixed positive/negative base intervals, not negativity
of P on all of R.

The leading positive heights in the W coordinate at u=+K and u=-K are

    W_plus^0(B) = K exp(I_plus) (1-beta B)/B,
    W_minus^0(U) = K exp(I_minus) (1+beta U)/U,
    I_plus/minus = integral_0^(1/K) (Q_plus/minus(v)+1)/(v Q_plus/minus(v)) dv.

The integrands have removable singularities at zero. They and the
coefficient derivatives are uniformly bounded. The ordinary finite-nu
connectors from the physical sections only change the labels by o(1).
Consequently there are fixed 0<m<M<infinity such that all candidate
passages with these endpoints have

    m <= W(+K), W(-K) <= M

for sufficiently small epsilon, uniformly as delta tends to zero.

The parameter-family justification uses the explicit additional-parameter
allowance in De Maesschalck--Schecter, Remark 1, and the same finite-order
normal forms as the singular-transport companion. It does not apply HK's
full entry-exit Theorem 2.4 at the discriminant boundary.

Primary sources:

- HK, Sections 3--4, especially Lemma 4.3 and Remark 4.4:
  https://arxiv.org/html/2510.02770v1
- De Maesschalck--Schecter, Remark 1, p.5:
  https://schecter.math.ncsu.edu/entry-exit.pdf

## 3. Actual exclusion near a nonzero discriminant point

**Claim.** Fix either s in [s0,S] or s in [-S,-s0], s0>0, and the
compact fiber/coefficient sets above. There exist delta_* and epsilon_*
strictly positive such that no candidate normal-forward passage connects
these input and output sets when

    0 <= delta <= delta_*, 0 < epsilon <= epsilon_*.

There is no required relation between delta and epsilon. The conclusion
includes possible shifts or splitting of the actual slow equilibria.

**Proof.** Restrict to a putative passage in the specified small-height
normal itinerary. On that fixed small-height neighborhood, g<0 uniformly
for epsilon sufficiently small. At u=+K and u=-K, the leading polynomial
is at most -(K-S)^2, the E term is uniformly small, and the h term is
negative. Thus both boundaries have strictly negative u derivative.
The passage enters the central strip at +K and exits at -K, crosses each
boundary in that direction only, and cannot make an excursion outside
the strip and reenter. Here inward means entry through +K in forward
time and entry through -K in backward time. At u=0 the exact u derivative is
lambda0 plus the negative h term; it is strictly negative. The trajectory
therefore crosses u=0 only from positive to negative. W decreases before
that crossing and increases afterwards. Its two endpoint bounds imply
W<=M throughout the central passage. This also justifies the uniform
smallness estimate for the flat term; there is no bootstrap assuming an
interior entry-exit remainder.

Put L=log(M/m). Consider s>=s0. On -K<=u<=0, the leading polynomial
is at most -s0^2. For small epsilon the exact u derivative is at most
-s0^2/2, because the h term has negative sign. Thus the time in that
half is at most 2K/s0^2, and its total positive contribution to log W
is at most 2K^2/s0^2.

Choose eta>0 satisfying

    eta < s0/2, eta < (K-S)/2,
    s0/(3 eta) > L + 2K^2/s0^2.

Take delta<=eta and then epsilon so small that the sum of the absolute
E and flat-term bounds is at most eta^2. On [s-eta,s+eta],

    |du/dtau| <= 3 eta^2, u>=s0/2.

Any passage from +K to -K must traverse this interval. Between its last
visit to s+eta preceding its first visit to s-eta it stays in that
interval. This traversal takes at least 2/(3 eta). Its contribution to
log W is therefore at most -s0/(3 eta). Other time with u>0 only adds
contraction. Combining both halves gives

    log[W(-K)/W(+K)] <= -s0/(3 eta)+2K^2/s0^2 < -L,

contradicting the endpoint bounds. This argument allows nonmonotone u
inside the bottleneck; monotonicity was used only on the opposite half,
where it follows from a strict differential inequality.

For s<=-s0 the positive-u half has uniformly bounded duration and hence
bounded contraction, while the negative bottleneck contributes at least
s0/(3 eta) to log W. This contradicts the upper bound L. QED.

**Scope.** This is an exclusion for compact positive endpoint ranges.
It does not exclude passages whose incoming label tends to zero, whose
outgoing label tends to zero/infinity, or whose incoming base approaches
1/beta. It does not prove a global exclusion of cycles at the discriminant.

## 4. A pole cannot occur with bounded output on a compact interior parameter set

Now retain a compact lambda set with discriminant <=-chi<0 and beta in a
compact positive interval. Allow B near the leading singular-map pole,
but keep B,U in compact positive intervals and 1-beta B>=omega>0.

Construct the incoming and outgoing HALF-maps separately to V=0.
HK Proposition 4.6 gives their asymptotics before the final closing IFT is
applied. Comparing their positive heights gives the actual matching
equation

    log[((1-beta B) U)/(B(1+beta U))] - log a + error_epsilon(B,U)=0,
    a=exp(pi lambda1/sqrt(-4lambda0-lambda1^2)),

where error tends uniformly to zero on these compact B,U sets (rho is
fixed first). This use of half-maps does not assume that the limiting
full entry map exists at B: equality of the two half-map heights is the
matching condition itself.

After possibly changing the sign of error, the equation is exactly

    1/B-beta = a exp(error) (1/U+beta).

Let amin=min a>0, betamax=max beta, and

    d0=min(1/4,1/(4 betamax Umax)), e0=-log(1-d0).

If |error|<=e0, then with b=beta(a+1),

    1-bB = B a [exp(error)/U + beta(exp(error)-1)]
          >= Bmin amin/(2 Umax) > 0.

Indeed exp(error)>=1-d0, so the bracket is at least
(1-d0)/Umax-betamax*d0 >=1/(2Umax).

Thus actual cycles with bounded positive input/output labels acquire a
strict pole margin automatically. This uses only a C0 half-map remainder
on compact sets; it does not assume uniform C2 convergence at the pole.
For fixed parameters the reduced input interval has 1-bB>=ell with
ell=Bmin*amin/(2Umax), and its leading outputs satisfy
amin*Bmin <= M(B) <= amax*Bmax/ell. Thus those leading outputs belong
to a new common compact interval; they need not belong to the original
observed output interval. Uniform C2 convergence on this reduced domain
can now be obtained from the interior theorem legitimately. Intersect
this input interval with the connected regular matching domain proved
in that companion. The intersection remains connected. For sufficiently
small rho and then epsilon, the same regular-curvature gate consequently
gives **at most two cycles on this single reduced domain**, subject to
the companion's other itinerary and regular-capture premises. No count
is added over overlapping parameter charts. The pole margin is a derived
consequence of compact positive actual input/output ranges, rather than
an additional hypothesis on these cycles.

## 5. The remaining origin and fiber layers

The discriminant exclusion leaves the corner lambda1->0,lambda0->0.
There the bottleneck lies at u=0, so the lower bound |u|>=s0/2 fails and
the large opposite-sign logarithmic contributions can cancel. One must
not infer exclusion there from a tending to zero/infinity: if
lambda1/delta stays bounded, a can tend to any finite positive value.

The natural secondary weighted parameter chart is

    lambda1=kappa*l1, lambda0=kappa^2*l0, u=kappa*v,
    tau_hat=kappa*tau, kappa>=0.

Then the polynomial part becomes l0+l1*v-v^2, while the exact correction
is O(epsilon*kappa) on bounded v, and the flat term is multiplied by
kappa^-2. The exact logarithmic height equation becomes
d(log W)/d(tau_hat)=-v. Thus even this chart requires an appropriate
height normalization and common domains; it is not automatically an
application of the interior theorem. Matching the outer labels can
introduce epsilon*log kappa through factors kappa^(2epsilon).

At a positive nonzero discriminant point, the surviving label layer
suggested by the limiting formula is B~a^-1 when a->infinity. At a
negative point, U~a when a->0 for B away from 1/beta. These are suggestions
for the next boundary coordinates, not uniform finite-epsilon formulas.
The proof above establishes only that fixed compact positive endpoints
are absent there; rates in those shrinking layers remain to be proved.

The pole layer with unbounded U can be charted by q=1/U. Its leading
matching relation is 1/B=beta(a+1)+a q, regular at q=0. Actual geometry,
return-map derivative control, and capture in that reciprocal-output
chart have not been established here.

## 6. Finite replay

Run from the repository root:

```sh
uv run --no-sync python benchmarks/hilbert16_boundary_reductions.py \
  --output artifacts/hilbert16/boundary_reductions.json
uv run --no-sync python -O benchmarks/hilbert16_boundary_reductions.py \
  --output artifacts/hilbert16/boundary_reductions_optimized.json
```

The replay checks four symbolic identities and rational examples of the
strict bottleneck and pole inequalities, including inconclusive and
invalid-input cases. Its constants are declared example bounds. It does
not estimate actual singular-map errors, produce an analytic cutoff, or
formally prove the normal-form and ODE arguments above.

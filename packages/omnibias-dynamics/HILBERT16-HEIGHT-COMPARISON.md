# A uniform slope bound for nonnegative lambda1, without a height cutoff

This is a direct written comparison for the exact normal quadratic family.
It does not assert that all center heights reach the physical sections.
A precise rectangular-domain capture statement is proved in section 6.
No novelty or formal analytic verification is claimed.

## 1. Precise hypotheses and conclusion

Use r=-1, the exact canonical embedding, epsilon=nu*k, C in a fixed compact
subset of (1,infinity), and A=1+nu*Abar with |Abar|<=A0. Fix compact bounds

    |lambda0|<=L0, 0<=lambda1<=L1.

The discriminant has no sign restriction. Write the exact normal field

    Vdot=F(V,h)=f(V)+h*g(V,h), hdot=-V*h,
    f(V)=epsilon^3*lambda0+epsilon^2*lambda1*V
         +epsilon*V^2*zeta(V,epsilon), zeta(0,epsilon)=-1.

Fix rho>0 and a common compact normal-coordinate domain. Coefficient
estimates are taken on an enlarged rectangle containing it, so the
segments from 0 to V used below remain inside the analytic chart. Every admitted
passage starts at V=0,h=H>0, reaches its selected incoming outer physical
section on the V>0 branch in backward normal time, and its selected outgoing
outer physical section on the V<0 branch in forward normal time. Both
branches remain in that common compact domain. Their endpoint heights obey

    0<hmin<=h_i,h_o<=hmax<infinity.

Require the center to cross in the specified normal-forward direction:

    F(0,H)<0.

This is automatic for lambda0<=0 and H>0; it is retained explicitly for
positive lambda0. Equality would be an equilibrium, not a passage.
There is no lower bound on H/epsilon^3. Assume the endpoints belong to fixed
compact portions of the large section branches. Explicit sufficient
transversality requirements on their height ranges are

    1+sigma*rho-rho^2*h <= -d_* < 0,
    1-sigma*rho*h has sign -sigma and magnitude >=v_* >0,

for sigma=-1 on the incoming side and sigma=+1 on the outgoing side.
These hold for a small fixed signed-label neighborhood of the large
sections already used in the rapid-passage proof. All cutoff choices below
depend only on these fixed compact domains and coefficient bounds.

At every such passage, the actual section map satisfies

    D_epsilon'(t_i(H)) >= ((1-e_epsilon)/(1+e_epsilon))
                          *exp[-2*epsilon*log(hmax/hmin)],
    e_epsilon=O(epsilon)->0.

Both section-label derivatives in H are strictly negative. The bound is
uniform for every admitted H>0, including H/epsilon^3 tending to zero and
lambda0=0.

On any connected interval of admitted center heights, the signed incoming
labels form one interval. If the captured regular map on the connected
intersection has derivative 0<Hreg'<=theta<1, there is at most one cycle
there for sufficiently small epsilon. Every such cycle is hyperbolic and
attracting in physical time. Without connected-domain capture, the slope
and stability conclusions still hold at each admitted cycle, but this
argument does not add a global count over distinct domain components.

## 2. Two exact coefficient signs

The exact normal-coordinate expansion from the
[grazing companion](HILBERT16-GRAZING-PASSAGE.md) gives

    g(V,h,nu)=-1+nu*(V-1)+O(nu^2)

in C1 on the fixed compact domain. Therefore

    g<0, g_V=nu+O(nu^2)>0

uniformly for sufficiently small positive nu. The derivative statement
uses joint analyticity and its C1 Taylor remainder; it is not obtained by
differentiating an uncontrolled pointwise O term.

Likewise zeta(V,0)=-1+beta0*V, beta0=1/3>0. Define exactly

    B(V,epsilon)=2*integral_0^1 zeta_V(q*V,epsilon) dq
                  +zeta_V(V,epsilon).

It tends uniformly to 3*beta0>0 on the fixed compact V range. Thus B>0 for
small epsilon, and for V!=0

    f_V(V)/V = -2*epsilon + epsilon^2*lambda1/V
                          +epsilon*V*B(V,epsilon).

Adding the transverse coefficient gives

    F_V(V,h)/V = -2*epsilon + epsilon^2*lambda1/V
                    +epsilon*V*B(V,epsilon)+h*g_V(V,h)/V.

Every correction to -2*epsilon has positive sign for V>0 and negative
sign for V<0. Consequently the exact pointwise envelope is

    F_V(V,h)/V >= -2*epsilon  (V>0),
    F_V(V,h)/V <= -2*epsilon  (V<0).

The inequalities hold at arbitrary unequal positive/negative V values.
No comparison of their magnitudes or of the corresponding trajectories
is required.

## 3. Variation initialized at the turning point

On each sign branch set T=V^2/2 and use h as independent variable. Height
increases away from the center on both branches, and exactly

    T_h=-F(V,h)/h, V=+sqrt(2T) or -sqrt(2T).

At a fixed H>0 put F0=F(0,H). The crossing hypothesis says

    F0=epsilon^3*lambda0+H*g(0,H)<0.

Moreover F_h(0,h)=g(0,h)+h*g_h(0,h)<0 uniformly on the compact chart for
small epsilon. Hence F(0,h)<0 for all h>=H in that chart. Neither outward
branch can recross V=0: at that boundary the original normal velocity is
strictly negative, while backward normal velocity is strictly positive.
Height consequently increases strictly away from the center on both
branches until a boundary of the admitted chart is reached.

Thus the original flow crosses V=0 transversely. Its analytic local
expansion in normal time q is

    V(q,H)=F0*q+O_H(q^2),
    h(q,H)=H-H*F0*q^2/2+O_H(q^3).

The subscript H permits constants depending on this positive reference
height. Analytic factorization of h-H=q^2*c(q,H), c(0,H)>0, gives smooth
inverse branches in sqrt(h-H). Hence, on either branch,

    T(h;H)=(-F0/H)*(h-H)+O_H((h-H)^(3/2)),
    Xi(h;H):=partial_H T(h;H)=F0/H+O_H(sqrt(h-H)).

The derivative here holds h fixed, and the physical coefficients and
epsilon are fixed. The expansion is differentiated on a neighborhood of
the positive reference H using the analytic square-root factorization.
It is not necessary that this local neighborhood be uniform as H->0.

For h>H the exact variational equation is

    Xi_h = -[F_V(V,h)/(h*V)]*Xi.

Its coefficient is locally O_H((h-H)^(-1/2))+O_H(1), so it is integrable at
the center. Passing the exact initial limit into the scalar variation
formula therefore gives

    Xi(h;H)=(F0/H)*exp(integral_H^h -F_V(V(r),r)/(r*V(r)) dr).

In particular Xi<0. No uniformly small Taylor remainder at H=0 is assumed.

## 4. Uniform comparison at the two different endpoint heights

Let Q0(H,epsilon)=-F0/H>0, possibly unbounded as H->0. Applying the exact
envelope from section 2 separately to the two branch variations gives

    |Xi_i(h_i)| <= Q0*(h_i/H)^(2*epsilon),
    |Xi_o(h_o)| >= Q0*(h_o/H)^(2*epsilon).

The same Q0 appears on both sides and cancels exactly. Therefore

    |Xi_o|/|Xi_i| >= (h_o/h_i)^(2*epsilon)
                  >= exp[-2*epsilon*log(hmax/hmin)].

This step does not extend either trajectory to the other trajectory's
endpoint height. It requires no common-height bridge or control beyond
either selected physical endpoint. The possible divergence of Q0, or of
the individual variation integrals as H->0, does not affect this ratio.

## 5. Exact physical section factors

In normal coordinates the physical section equation is

    E_sigma(V,h)=V-1+sigma*rho*h+nu*rho^2*h^2
                 +C*nu^2*sigma*rho*h^2=0.

Thus V=V_sigma(h) with (V_sigma)_h=-(E_sigma)_h. Put

    B_sigma(h)=V_sigma(h)^2/2,
    t_sigma(h)=(sigma*rho*h-1)^2-2*h.

Differentiating T(h_sigma(H);H)=B_sigma(h_sigma(H)) gives exactly

    dt_sigma/dH = A_sigma*Xi_sigma,
    A_sigma = -t_sigma'(h_sigma)
               /[T_h+V_sigma*(E_sigma)_h].

At nu=0, T_h=1, V_sigma=1-sigma*rho*h, (E_sigma)_h=sigma*rho, and

    t_sigma'=-2*(1+sigma*rho*V_sigma).

Hence A_sigma=2. On the admitted compact endpoint height ranges, the
denominators are bounded away from zero by d_*. Analyticity gives

    2*(1-e_epsilon)<=A_i,A_o<=2*(1+e_epsilon),
    e_epsilon=O(epsilon)->0,

uniformly without requiring convergence of the actual endpoints as H->0.
Indeed the comparison is a uniform coefficient estimate as a function of
the endpoint height h, which already lies in a fixed compact set.

Both label derivatives are negative since Xi<0 and A_sigma>0. Their ratio,
combined with section 4, proves the stated lower bound on D_epsilon'.

## 6. A precise interval-capture statement

For the following strengthening choose a rectangular common chart

    |V|<=Vmax, 0<h<=hmax,

and fixed connected compact endpoint-height intervals I_i,I_o inside the
outer-section height ranges of section 1. Let hmin be their common positive
lower bound and restrict centers to 0<H<=Hmax<hmin. The rectangle and the
height intervals must have the strict coordinate/transversality margins
already assumed. A center is admitted if F(0,H)<0 and each sign branch
reaches its selected outer section at a height in I_sigma while remaining
in this rectangle. The following proves that this admitted set is an
interval; it does not prove that it is nonempty or equals (0,Hmax].

Take two admitted H1<H2. On either sign branch, comparison gives

    T(h;H1)>T(h;H2)

at every common height after H2. Indeed at h=H2 the first trajectory has
positive T whereas the second starts at zero, and uniqueness for the
scalar equation at T>0 prevents their graphs from meeting. Positivity
persists because T_h at T=0 is -F(0,h)/h>0 for h>=Hj.

First establish that the corresponding endpoint heights obey h2<h1.
Suppose instead h1<=h2. At h1 the trajectory from H2 has smaller T than
the one reaching its section from H1. For the incoming V>0 branch its
section residual E_sigma is then negative; for the outgoing V<0 branch
it is positive. These are the respective already-passed sides of the
outer section. To reach its proposed later outer crossing at h2, the
second trajectory would have to make an opposite-direction zero first.
That is impossible: at every section zero in the connected height interval
I_sigma the derivative along increasing height is

    dE_sigma/dh = [T_h+V*(E_sigma)_h]/V.

Its numerator is uniformly negative on this outer interval. The derivative
is therefore negative for the incoming branch and positive for the
outgoing branch. All zeros there are transverse in this fixed direction.
Equality h1=h2 is also impossible by strict ordering. Thus h2<h1.

Now let H1<H<H2. The inequality F_h(0,h)<0 also gives F(0,H)<0,
so this intermediate center has the required crossing. From h=H onward its T graph starts below T(h;H1) and
stays below it. It cannot return to T=0, and the upper comparison keeps it
inside the rectangular chart until h1. Hence it continues at least that
far. From h=H2 onward it is also strictly above T(h;H2), up to h2.

At h2 its section residual is on the before-crossing side; at h1 it is on
the after-crossing side. Continuity supplies an intersection between h2
and h1. The fixed crossing direction makes it unique within that outer
height interval. It lies in I_sigma, which is an interval. This reasoning
applies on both sign branches, proving admission of every intermediate H.
The use of Hmax<hmin ensures both comparisons begin before either selected
endpoint. No bounding trajectory is extended beyond its own endpoint.

If a different, nonrectangular notion of bounded itinerary is used, this
capture argument must be checked for that domain or connectedness must be
retained as a hypothesis; it cannot be inferred just from Xi<0.

## 7. Cycle count and stability scope

Use the [regular capture and derivative bounds](HILBERT16-REGULAR-JETS.md)
and the [exact signed-coordinate transport](HILBERT16-RAPID-PASSAGE.md).
The strict negativity of dt_i/dH maps the connected admitted set to one
interval of section labels. Its intersection with the captured regular
domain is connected. The lower bound on D' tends uniformly to 1, so
D'>theta for small epsilon and Hreg-D is strictly decreasing there.
At a cycle the physical return map D^-1 composed with Hreg has multiplier
Hreg'/D' strictly between zero and one. This yields the stated count and
stability, under the specified domain assumptions.

The slope theorem itself remains valid pointwise for every admitted
bounded passage even when the domain-capture hypotheses in section 6 are
not imposed. It then proves attracting stability at each captured cycle
but does not count across unproved domain components.

No condition H>=gamma*epsilon^3 was used. The proof requires lambda1>=0,
F(0,H)<0, the fixed compact itinerary and endpoint domains, the exact
canonical embedding, and A-1=O(nu). The crossing condition is automatic
for lambda0<=0; positive lambda0 is allowed when it holds. Escape or a
different itinerary can still intervene at extremely small H. The proof
does not cover a changing sign of lambda1 or an unbounded section chart.

## Finite checks and reproduction

The [thin-passage benchmark](../../benchmarks/hilbert16_thin_passage.py)
replays 17 exact identities, including the compensated variation, drift
envelope, moving physical endpoints, radial escape inequality, and the
cancellation of the earlier symmetric-window term. Rational examples
check the sufficient escape constants, scale exponents, and a declared
slope margin.

```sh
uv run --no-sync python benchmarks/hilbert16_thin_passage.py \
  --output artifacts/hilbert16/thin_passage.json
uv run --no-sync python -O benchmarks/hilbert16_thin_passage.py \
  --output artifacts/hilbert16/thin_passage_optimized.json
```

The algebra replay does not verify the analytic continuation or uniform
limit arguments. Actual physical cutoff constants are not computed as
interval certificates, and no formal analytic verification flag is earned.

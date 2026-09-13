# Negative Schwarzian for the actual compensated regular passage

This note proves a uniform Schwarzian perturbation estimate for the actual
quadratic unfolding in the common sector of the [endpoint passage proof](HILBERT16-ENDPOINT-PASSAGE.md).
It uses the fixed-alpha passage established there, not a separate choice of
alpha for each initial condition. The result concerns the regular passage
in the normalized height coordinates specified below. It does not supply a
singular closing map or a bound on zeros of the complete return displacement.
The analytic argument is not formally verified; the companion script checks
its finite algebra and rational sufficient conditions.

## Statement and notation

Fix 1<C_-<=C_+ and L>=1, with C in [C_-,C_+] and |p|,|r|,|m|<=L. The
physical field is

    xdot=(1+alpha)*x-y+x^2+epsilon*(p+r)*x*y+epsilon^2*m*y^2,
    ydot=C*x+x^2+x*y+epsilon*r*y^2.

Let q=(x^2+2*x+C)/2, y0=(x^2-C)/2, w=y-y0, j=1+|x|, R>=2,
S=R+1, epsilon>=0, zeta>0, and delta=epsilon+zeta/R. Use the constants
Q=C_+/2, kappa, M_e, d0, t1, n0, p0, Y, K_w, gamma and eta_p from the
endpoint proof. In particular

    gamma=4*C_- /[(C_++3)*(C_++1)],
    p0=1+2*L+2*L*Y, n0=z0+t0,
    K_w=4*d0/kappa^2+2*t1/kappa+4*n0*p0/kappa^2.

The fixed compensated parameter is alpha_*=alpha_R(epsilon,0,0). The
endpoint theorem gives a genuine flow map G_R from x=-R to x=R on the
whole interval [-zeta,zeta], in coordinates

    z_in=w(-R)/q(-R), z_out=w(R)/q(R),

provided delta*S<=eta_p. Every trajectory of this same field satisfies

    |w(x)|<=M_e*delta*j^3, |alpha_*|<=M_e*delta,
    P=xdot>=q/2, kappa*j^2<=q<=Q*j^2.

Let G_R^0 denote the base map with epsilon=alpha=0 near z=0. We prove

    sup[|z|<=zeta] |S(G_R)(z)-S(G_R^0)(0)| <= K_S*delta*S,       (1)

where S(g)=g'''/g'-(3/2)*(g''/g')^2 and K_S is explicit below. This
compares the actual Schwarzian throughout its input interval to the base
Schwarzian at zero; it does not require an already established uniform
C^3 convergence theorem.

Define the fully rational constants

    n1=2+t1, k_p=2*L*(1+Y),
    K3=48*L*p0/kappa^2 +24*L*n1/kappa^2
         +48*t1*p0^2/kappa^3 +96*k_p*(p0+1)/kappa^3
         +456*d0/kappa^4 +96*n0*p0*L/kappa^3
         +96*n0*p0^3/kappa^4,
    A_c=Q^2/kappa,
    K_S=A_c^2*[(8/5)*K3+48*K_w*kappa^(-3)].

Also set

    n_D=ceil(1/(3*kappa)), r_D=3^(-n_D),
    b_gamma=3*gamma/(1+3*gamma),
    sigma=(5/9)*kappa*r_D^2*b_gamma/Q^2,
    eta_s=min(eta_p,sigma/(2*K_S)).                            (2)

All these constants are positive, and rational whenever C_-,C_+,L are
rational. Then for every R>=2 and delta*S<=eta_s,

    S(G_R)(z)<=-sigma/2<0 for all |z|<=zeta.                   (3)

The derivative and positivity conclusions of the endpoint theorem remain
valid, including G_R(0)=0 and 0<G_R'<=1/(1+gamma/2). For the joint choice
R=rho/epsilon, it suffices to fix rho,zeta>0 with
rho+zeta<=(2/3)*eta_s and then take 0<epsilon<=rho/2. Thus (3) holds
on a common nonzero entrance interval independent of epsilon in this sector.

## Any fixed parameter and its connected admitted domain

The zero anchor is only one source of admitted passages. More generally,
fix any single alpha_bar and an interval I of normalized entrance heights
such that the same physical field has trajectories from x=-R to x=R for
every z in I, all lying in the displayed endpoint amplitude tube with the
same delta and constants. Then estimates (1), (3), and (15) hold uniformly
on I under their stated sector restrictions. The estimate 0<G_R'<=theta
also holds there. In every derivative calculation below, alpha_* may be
replaced by this arbitrary fixed alpha_bar: the proof uses only the tube
bounds and the initial first variation A(-R)=q(-R). It does not use a zero
trajectory of the perturbed field or the condition G_R(0)=0. The latter
condition, and existence on the entire prescribed input interval
[-zeta,zeta], are specific conclusions for alpha_R(epsilon,0,0).

The endpoint theorem supplies a connected domain for an arbitrary fixed
alpha_bar in its matching range. Write
A_R(s,t)=alpha_R(epsilon,s,t) on the closed endpoint square. Its proved
strict signs are partial_s A_R>0 and partial_t A_R<0. Consequently

    I_alpha={s in [-zeta,zeta]:
                A_R(s,zeta)<=alpha_bar<=A_R(s,-zeta)}.          (16)

Both boundary functions of s are strictly increasing. The first
inequality defines a left interval and the second a right interval, so
I_alpha is an interval, possibly empty or a singleton. For every s in
I_alpha there is exactly one t in [-zeta,zeta] with
A_R(s,t)=alpha_bar, by continuity and strict decrease in t. These matched
trajectories form the actual passage G_R(s)=t of the same fixed field.
The endpoint theorem's extension to a neighborhood of the closed square
and partial_t A_R<0 provide smooth local graphs, which agree by uniqueness.
The derivatives therefore exist through the endpoints of I_alpha as well.
At a singleton this is a local statement; no interval of positive length
is asserted. On every nondegenerate I_alpha, (1), (3), and (15) apply
with the same constants.

In particular, the interval between any two admitted entrance heights is
also admitted, which is the connectedness needed for Rolle's theorem.
This statement does not require the intermediate passages to satisfy a
stronger coarse-neighborhood condition used to select particular cycles;
the common fine endpoint tube is sufficient for the derivative bounds.

## Exact scalar-flow identity

Put F(x,w)=N/P, the x-time slope at the fixed alpha_*. From the previously
verified field transformation,

    N=2*x*w+Slin+T,
    Slin=-alpha_*x^2+epsilon*f4+epsilon^2*m*b1,
    T=(-epsilon*p*x^2-C*epsilon*r)*w+epsilon*r*w^2
         -epsilon^2*m*x*(2*y0*w+w^2),
    P=q+d,
    d=-w+alpha_*x+epsilon*(p+r)*x*(y0+w)
         +epsilon^2*m*(y0+w)^2.

For each finite R, the endpoint theorem and P>0 give a smooth flow on a
neighborhood of the closed input interval. This only establishes existence
of derivatives at finite R; the following estimates supply their uniform
control. Differentiate with respect to z_in, holding alpha_* fixed. Write
A=partial_z w, B=partial_z^2 w, C3=partial_z^3 w. Scalar variational theory
gives

    A'=F_w*A,
    B'=F_ww*A^2+F_w*B,
    C3'=F_www*A^3+3*F_ww*A*B+F_w*C3.

Their initial values are A(-R)=q(-R)>0, B(-R)=C3(-R)=0. Consequently

    d/dx [C3/A-(3/2)*(B/A)^2]=F_www*A^2.

The output normalization is affine and does not change the Schwarzian.
Therefore the exact formula is

    S(G_R)(z)=integral[-R,R] F_www(x,w(x,z))*A(x,z)^2 dx.        (4)

No derivatives of the matching parameter appear, precisely because the
endpoint theorem supplied a passage of a single fixed vector field.

## A weighted bound for the third height derivative

Both N and P are quadratic in w. Their exact quotient derivative is

    F_www=-3*N_ww*P_w/P^2 -3*N_w*P_ww/P^2
            +6*N_w*P_w^2/P^3 +6*N*P_w*P_ww/P^3
            -6*N*P_w^3/P^4.                                (5)

Throughout the endpoint tube, the previously proved estimates and
0<=epsilon<=delta, delta*j<=delta*S<=1 give

    |N|<=n0*delta*j^4,
    |N_w-2*x|<=t1*delta*j^2, |N_w|<=n1*j,
    |N_ww|=|2*epsilon*r-2*epsilon^2*m*x|<=4*L*delta,
    |P_w|<=p0,
    |P_w+1|<=k_p*delta*j,
    |P_ww|=2*epsilon^2*|m|<=2*L*delta^2,
    |d|<=d0*delta*j^3, q/2<=P<=3*q/2.

At the base point epsilon=alpha=w=0, F_www=12*x/q^3. Split the third
term of (5) as

    6*N_w*P_w^2/P^3-12*x/q^3
      =6*(N_w-2*x)*P_w^2/P^3
        +12*x*(P_w^2-1)/P^3
        +12*x*(P^(-3)-q^(-3)).

The inverse-cube difference satisfies

    |P^(-3)-q^(-3)|
      =|d|*(P^2+P*q+q^2)/(P^3*q^3)<=38*|d|/q^4,

because P<=3*q/2 and P>=q/2. Substitution into the five terms of (5),
with the third term split into three, yields respectively the seven
constants in K3. Terms containing delta^2*j or delta^3*j^2 are bounded
using delta*j<=1. Hence

    |F_www-12*x/q^3|<=K3*delta*j^(-4).                       (6)

Every power of j is explicit; K3 contains no hidden R dependence.

## The input normalization cancels the apparent tail growth

Use D and H from the endpoint proof:

    D'/D=2/q, D increasing, H=q^2/D, H'/H=2*x/q.

At the base zero trajectory, define

    A0(x)=q(-R)*H(x)/H(-R)
         =[D(-R)/D(x)]*q(x)^2/q(-R).

Monotonicity of D and x>=-R give the useful bound

    0<A0(x)<=A_c*j(x)^4/S^2.                                (7)

There is no factor 1/min(D) in this estimate. The actual first variation
obeys A=A0*E, where

    E(x)=exp(integral[-R,x] [F_w-2*s/q(s)] ds).

The earlier bound |F_w-2*x/q|<=K_w*delta gives
|log E|<=2*K_w*delta*R. The restriction eta_p<=1/(4*K_w) ensures

    E^2<=4, |E^2-1|<=8*K_w*delta*R.                          (8)

Indeed |log E^2|<=4*K_w*delta*R<=1; on [-1,1] one has
|exp(u)-1|<=2*|u|, using exp(1)<3. These elementary exponential
bounds justify (8) without numerical evaluation of a transcendental gate.

Subtract the base version of (4). Equations (6)-(8) and
|12*x/q^3|<=12*kappa^(-3)*j^(-5) give

    |S(G_R)(z)-S(G_R^0)(0)|
      <=4*K3*delta*A_c^2/S^4 * integral[-R,R] j^4
        +8*K_w*delta*R*12*kappa^(-3)*A_c^2/S^4
                                      * integral[-R,R] j^3.

The exact elementary integrals are

    integral[-R,R] j^4=(2/5)*(S^5-1),
    integral[-R,R] j^3=(S^4-1)/2.

Their substitution proves (1). The denominator S^4 supplied by the
squared normalized first variation is what controls the increasing cutoff.
A pointwise closeness argument without (7) would not establish this result.

## A rational negative base margin for every R>=2

At the zero trajectory, the base formula (4) becomes

    S(G_R^0)(0)=12*D(-R)^2/q(-R)^2
                          * integral[-R,R] x*q(x)/D(x)^2 dx.

Pair positive and negative x. Define

    T(x)=q(x)*D(-x)^2/[q(-x)*D(x)^2].

Since (log(q/D^2))'=(x-3)/q,

    log T(x)=-integral[0,x] (5*s^2+3*C)/[q(s)*q(-s)] ds<0

for x>0. Thus the paired integrand is negative everywhere. For x>=1,
restriction of the last integral to [0,1] gives log T(x)<=-3*gamma,
using q(s)<=(C_++3)/2, q(-s)<=(C_++1)/2 and numerator>=3*C_-.
Therefore

    1-T(x)>=1-exp(-3*gamma)>=b_gamma.                        (9)

The second inequality follows from exp(3*gamma)>=1+3*gamma.

For x in [R/2,R], x>=1. The negative-tail D ratio has a rational lower
bound stronger than a global min(D) bound:

    log[D(-x)/D(-R)]
       =integral[-R,-x] 2/q(s) ds
       <=(2/kappa)*[1/(1+x)-1/(1+R)]
       <=(2/kappa)*R/[(R+1)*(R+2)]<=1/(3*kappa).

The final step is equivalent to (R-1)*(R-2)>=0. By exp(1)<3 and the
definition of n_D,

    D(-R)/D(-x)>=exp[-1/(3*kappa)]>=3^(-n_D)=r_D.             (10)

Now restrict the negative paired integral to [R/2,R]. Equations (9)-(10),
q(-x)>=kappa*x^2, q(-R)^2<=Q^2*S^4 and S<=3*R/2 give

    -S(G_R^0)(0)
      >=12*kappa*r_D^2*b_gamma/(Q^2*S^4)
                                  * integral[R/2,R] x^3 dx
      >=[12*(15/64)*(16/81)]*kappa*r_D^2*b_gamma/Q^2
       =sigma>0.                                           (11)

This proves the base margin for the whole ray R>=2. Combining (1), (2)
and (11) proves (3).

## Exact checks and scope

The [benchmark](../../benchmarks/hilbert16_regular_jets.py) checks (5), the
physical N_ww and P_ww formulas, the scalar-flow identity (4), the paired
base identity, the elementary tail algebra, and all rational sufficient
conditions for two parameter boxes. No Python assertion is used.

For C in [2,3], L=1, it gives

    sigma=5/344373768,
    K_S=13516945306830007691017715712/5,
    eta_s=1/491694900216129519270650495377007830887591936.

In this box the existing endpoint sector already satisfies the new
Schwarzian condition; no additional reduction is needed. Thus
S(G_R)<=-5/688747536 and 0<G_R'<=6/7 throughout the stated common sector.
For C in [5/4,7/4], L=2, a further reduction to eta_s approximately
4.463030309894e-58 suffices. These sectors are nonempty but intentionally
very conservative; the result is an analytic uniformity statement.

The conclusion uses z=w/q at both sections. A nonlinear change of
transverse coordinates can alter the Schwarzian sign, so this result must
be transported through the actual chart changes before use in a full
return map. The singular closing transitions and any complete displacement
zero bound require their own arguments. No claim of novelty or a solution of full
Hilbert 16 follows from this bounded regular-passage theorem.

## Uniform first through fourth section derivatives

The same argument supplies the additional finite-order bounds needed to
compose with actual entry and exit coordinate changes. They hold throughout
the original endpoint sector delta*S<=eta_p; Schwarzian negativity itself
uses the smaller eta_s when required. All derivatives here are with respect
to the normalized entrance coordinate, with alpha_* fixed. No parameter
jet or coordinate-change estimate is included implicitly.

From the preceding tube estimates and delta*j<=1,

    |N|<=n0*j^3, |N_w|<=n1*j, |N_ww|<=4*L/j,
    |P_w|<=p0, |P_ww|<=2*L/j^2, |P|>=kappa*j^2/2.

For k>=0 write F_k=partial_w^k F. Differentiating P*F=N gives exactly

    P*F_k=N_k-k*P_w*F_(k-1)-binom(k,2)*P_ww*F_(k-2),

with N_k=0 for k>=3 and terms with negative indices omitted. Therefore

    |F_k|<=C_k*j^(1-2*k),                                (12)

where the explicit positive constants can be chosen as

    C_0=2*n0/kappa,
    C_1=(2/kappa)*(n1+p0*C_0),
    C_2=(2/kappa)*(4*L+2*p0*C_1+2*L*C_0),
    C_k=(2/kappa)*[k*p0*C_(k-1)+k*(k-1)*L*C_(k-2)]
                                                for k>=3.

This proves the bound for every fixed finite order by induction. Only
k<=4 is used below. The coarse F_1 bound in (12) is used for this algebraic
recurrence, not to estimate the first variation: doing the latter would
lose the useful normalized estimate (7)-(8).

Let A=partial_z w, V_n=(partial_z^n w)/A for n>=2, and V_1=1. The
Faà di Bruno/Bell recurrence from scalar variational equations yields

    V_2'=F_2*A,
    V_3'=3*F_2*A*V_2+F_3*A^2,
    V_4'=F_2*A*(4*V_3+3*V_2^2)+6*F_3*A^2*V_2+F_4*A^3.

All V_n for n>=2 start at zero at x=-R because the initial normalized
coordinate is affine. Since A<=2*A_c*j^4/S^2, for k>=2

    integral[-R,R] |F_k|*A^(k-1)
       <=C_k*(2*A_c)^(k-1)/S^(2*k-2)
                               * integral[-R,R] j^(2*k-3)
       <=T_k:=C_k*(2*A_c)^(k-1)/(k-1).                     (13)

In particular, uniformly in x,z,R and admitted parameters,

    |V_2|<=T_2,
    |V_3|<=3*T_2^2+T_3,
    |V_4|<=15*T_2^3+10*T_2*T_3+T_4.                       (14)

For example the last estimate follows by substituting the preceding two
bounds into the V_4 equation and integrating the absolute values; it uses
T_2*(4*(3*T_2^2+T_3)+3*T_2^2)+6*T_3*T_2+T_4.

At the output, partial_z^n G_R=G_R'*V_n. Thus with
theta=1/(1+gamma/2), the promised common-box bounds are

    0<G_R'<=theta,
    |G_R''|<=theta*T_2,
    |G_R'''|<=theta*(3*T_2^2+T_3),
    |G_R''''|<=theta*(15*T_2^3+10*T_2*T_3+T_4).            (15)

These constants are independent of the growing cutoff R. The spatial
powers in (13), including S^(-2*k+2) from the normalized first variation,
are the essential cancellation. Bound (15) concerns actual solutions, not
formal Taylor coefficients at fixed R. Its composition with a changing
phase-chart or entry coordinate still requires that coordinate map's own
uniform derivatives and a verified overlap domain.

## A fixed coarse tube captures every admitted regular passage

This is an a priori extension of the endpoint theorem. It removes the need
to assume that an actual regular trajectory already has the shrinking
weighted amplitude bound used by the contraction construction. It does not
prove that every nearby limit cycle traverses the specified regular and
singular charts.

Use the physical family and all constants of the
[endpoint passage proof](HILBERT16-ENDPOINT-PASSAGE.md).
Let R>=2, S=R+1, epsilon>=0, j=1+|x| and q=(x²+2x+C)/2. Fix a coarse
radius eta>=0. Suppose a C1 trajectory graph w on [-R,R] satisfies

    |w(x)|<=eta*j(x)², |alpha|<=eta,
    w(-R)=q(-R)z_-, w(R)=q(R)z_+, |z_-|,|z_+|<=zeta.

The radius eta is independent of epsilon and R. Define

    d_c=2+2*L*Y+L*Y²,
    T_c=b0+2*L+2*L*Q,
    E_c=2*T_c+6*d_c/kappa,
    F_c=2*d_c/kappa,
    G_c=F1+F_c*(F0+F1),
    eta_c=min(1,kappa/(2*d_c),1/(2*K*E_c),1/G_c).

Here G_c is a scalar constant, not a passage map. If

    eta+epsilon*S<=eta_c,

then the trajectory necessarily satisfies

    max(sup |w|/j³,|alpha|)<=M_e*(epsilon+zeta/R),
    xdot>=q/2>0.

If the endpoint existence sector
(epsilon+zeta/R)*S<=eta_e also holds, it is exactly the unique trajectory
and matching alpha constructed by that theorem.

Proof. Put h=eta+epsilon*S, rho0=epsilon*S, and
n=max(sup |w|/j³,|alpha|). Since eta,rho0<=1, |y0+w|<=Y*j². The exact
quantities d,T,Z,E from the endpoint equation satisfy

    |d|<=d_c*h*j²,
    |T|<=T_c*rho0*n*j⁴,
    |Z|<=[3*n+epsilon*(F0+rho0*F1)]*j⁴.

For d, bound the w and alpha*x terms by 2*eta*j², the epsilon*x*y term
by 2*L*Y*rho0*j², and the epsilon²*y² term by L*Y²*rho0²*j².
Thus |d|<=q/2 and P=q+d>=q/2. This inequality holds everywhere in the
coarse tube, so a physical trajectory traversing it has increasing x and
can be represented as a graph.

For T, the four terms in its exact definition contribute respectively
b0*rho0*n, L*eta*rho0*n, 2*L*Q*rho0²*n, and L*eta*rho0²*n after
factoring j⁴. Their sum is at most T_c*rho0*n. For Z use
|2*x*w-alpha*x²|<=3*n*j⁴ and the stated forcing bounds.

The exact quotient E=(q*T-d*Z)/P consequently satisfies

    ||E||_4<=E_c*h*n+F_c*h*epsilon*(F0+rho0*F1).

Every solution of the graph equation has the projected-inverse
representation, including the same boundary particular solution used in
the endpoint proof. Its norm therefore obeys

    n<=K_b*zeta/R
         +K*epsilon*(F0+rho0*F1)*(1+F_c*h)
         +K*E_c*h*n.

The definition of eta_c gives K*E_c*h<=1/2. Also

    (F0+rho0*F1)*(1+F_c*h)
      <=F0+h*[F1+F_c*(F0+F1)]<=F0+1.

Absorbing the last norm term yields

    n<=2*K_b*zeta/R+2*K*epsilon*(F0+1)
       <=M_e*(epsilon+zeta/R).

This uses the existing M_e>=2*[K_b+K*(F0+1)]. No division by epsilon,
zeta or h is needed, so the zero case is included. The endpoint uniqueness
claim follows once its additional sector restriction is satisfied.

This lemma upgrades a fixed relative-height neighborhood to the fine
weighted tube for a prescribed regular passage. Singular transition
coverage and the all-degree Hilbert conclusion remain separate obligations.


When the finer sector delta*S<=eta_p also holds, each captured trajectory
belongs to the fixed-alpha matching domain (16), so the first through
fourth section derivative bounds (15) apply. Under delta*S<=eta_s,
the Schwarzian bound (3) applies as well. Thus it suffices to locate a
physical regular passage in the fixed coarse tube with the prescribed
endpoints; one need not assume its shrinking weighted bound in advance.

## Reproducing the finite checks

The benchmark checks finite symbolic identities and rational constants for
both regular-jet control and coarse-tube capture. All acceptance conditions
use exact fractions and explicit exceptions, which remain active under
Python optimization. The written analytic implications are not formally
verified by these finite checks.

```sh
uv run --no-sync python benchmarks/hilbert16_regular_jets.py --output artifacts/hilbert16/regular_jets.json
uv run --no-sync python -O benchmarks/hilbert16_regular_jets.py --output artifacts/hilbert16/regular_jets_optimized.json
```

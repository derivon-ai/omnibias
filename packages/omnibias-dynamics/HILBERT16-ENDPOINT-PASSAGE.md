# Common nonzero sections for the nonlinear compensated passage

This extends [the uniform passage proof](HILBERT16-UNIFORM-PASSAGE.md) to prescribed normalized
endpoint heights. All constants are explicit and deliberately conservative.
It is a passage/matching theorem, not a return-displacement zero bound or a
solution of Hilbert 16.

## Common-section theorem

Fix 1<C_-<=C_+<infinity and L>=1. Let C be in this compact interval and
|p|,|r|,|m|<=L. Use the actual quadratic field

    xdot=(1+alpha)*x-y+x^2+epsilon*(p+r)*x*y+epsilon^2*m*y^2,
    ydot=C*x+x^2+x*y+epsilon*r*y^2,

and q=(x^2+2*x+C)/2, y0=(x^2-C)/2, w=y-y0, j=1+|x|.
The normalized section coordinate is z=w/q. Fix R>=2, epsilon>=0 and
zeta>=0, prescribe |z_-|,|z_+|<=zeta, and put

    delta=epsilon+zeta/R, S=R+1.

With the explicit constants M_e,eta_e below, if

    delta*S<=eta_e,                                      (D)

there is a unique pair (w,alpha) in the amplitude tube

    ||w||_3:=sup[-R,R] |w|/j^3<=M_e*delta,
    |alpha|<=M_e*delta,
    w(-R)=q(-R)*z_-, w(R)=q(R)*z_+,                     (B)

that solves the trajectory equation. The solution is C^1, the physical
x velocity obeys xdot>=q/2>0, and its flight time is at most
4*pi/sqrt(C_--1). Uniqueness is within the stated tube.

For delta=0 the pair is exactly w=alpha=0. No division by delta is used.
A sufficient version of (D) is

    epsilon*(R+1)+zeta <= (2/3)*eta_e,

because S/R<=3/2. In particular, fix positive rho and a section height
radius zeta with rho+zeta<=(2/3)*eta_e. Then R=rho/epsilon satisfies
(D) for every 0<epsilon<=rho/2. This gives a nonzero, epsilon-independent
box of entrance and exit heights on a common nonlinear passage sector.

## Explicit boundary solution and signs

Use

    D(x)=exp(4*(atan((x+1)/sqrt(C-1))-pi/2)/sqrt(C-1)),
    H=q^2/D, W=D/q^3,
    J_R=integral[-R,R] W*x^2,
    J_x=integral[-R,x] W*s^2 ds.

For the two prescribed endpoints define

    B_-=D(-R)*z_-/q(-R), B_+=D(R)*z_+/q(R),
    DeltaB=B_+-B_-,
    alpha_b=-DeltaB/J_R,
    w_b=H*[B_-+DeltaB*J_x/J_R].                         (1)

The plus sign in w_b is essential. Since 0<=J_x/J_R<=1, w_b/H is a
convex interpolation between B_- and B_+. Direct differentiation gives

    (w_b/H)'=DeltaB*W*x^2/J_R=-alpha_b*W*x^2,
    L_C w_b=-alpha_b*x^2,
    w_b(+-R)=q(+-R)*z_+-.                               (2)

There is no endpoint sign or orientation convention hidden in (1).

Set, as in the zero-endpoint proof,

    c_*=exp(-4*pi/sqrt(C_--1)), Q=C_+/2,
    kappa=min(1,C_--1)/16,
    H0=Q^2/c_*, W0=kappa^(-3), J0=8/[9*(C_++8)^3],
    G0=81*Q^2*W0,
    A4=2*W0/J0, B4=G0*(1+A4/3), K=max(A4,B4).

The previously proved elementary bounds are

    kappa*j^2<=q<=Q*j^2, c_*<=D<=1,
    H<=H0*j^4, W<=W0*j^-6, J_R>=J0.

Since j(+-R)=S,

    |B_+-|<=zeta/(kappa*S^2),
    |w_b(x)|<=H0*zeta*j^4/(kappa*S^2),
    ||w_b||_3<=H0*zeta/(kappa*S),
    |alpha_b|<=2*zeta/(kappa*J0*S^2).                   (3)

For the contraction one can remove the small c_* from the boundary bound.
For x<=0 use the exact interpolation as

    w_b=H*(1-J_x/J_R)*B_- + H*(J_x/J_R)*B_+.

The first term uses D(-R)/D(x)<=1. In the second term use
J_x/D(x)=integral[-R,x] [D(s)/D(x)]*s^2/q(s)^3,
whose integrand is bounded by W0*j(s)^-4. For x>=0 apply the same
argument with the two endpoints reversed and the complementary tail,
using D(s)/D(x)<=81. The result is

    |w_b(x)|<=81*Q^2*zeta/(kappa*S^2) *
                 [j^4+W0*j/(3*J0)].

Thus, with the fully rational constant

    K_b=max(81*Q^2/kappa*(1+W0/(3*J0)),2/(kappa*J0)),

the product norm of (w_b,alpha_b) is at most K_b*zeta/S, hence at most
K_b*zeta/R. The sharper
O(zeta/R^2) bound for alpha_b in (3) is retained and does not come from
the weaker product norm.

## Explicit contraction constants

Define

    F0=L*(Q^2+2*Q), F1=L*Q^2,
    U0=L*(6+3*C_+), A0=L*(C_++9),
    M_e=max(1,2*(K_b+K*(F0+1)),2*U0,2*A0), Y=Q+1,
    d0=2*M_e+2*L*Y+L*Y^2,
    z0=3*M_e+F0+F1,
    b0=L*(1+C_+),
    t0=b0*M_e+L*M_e+2*L*Q+L,
    e0=2*t0+2*d0*z0/kappa,
    d1=2+2*L+2*L*Y,
    t1=b0+4*L+2*L*Q,
    e1=2*t1+2*d1*z0/kappa+6*d0/kappa+2*e0*d1/kappa,

    eta_e=min(1,1/M_e,kappa/(2*d0),
              1/(F1+e0),1/(2*K*e1)).                  (4)

All quantities are strictly positive. On the tube (B), epsilon<=delta and
delta*S<=eta_e imply exactly the same estimates as the zero-endpoint
proof, with epsilon replaced by delta in upper bounds. In particular, set

    b2=-x^2*y0, b3=y0^2-x^2*y0, b1=-x*y0^2,
    f4=p*b2+r*b3,
    Slin=-alpha*x^2+epsilon*f4+epsilon^2*m*b1,
    T=(-epsilon*p*x^2-C*epsilon*r)*w+epsilon*r*w^2
          -epsilon^2*m*x*(2*y0*w+w^2),
    d=-w+alpha*x+epsilon*(p+r)*x*(y0+w)
          +epsilon^2*m*(y0+w)^2,
    E=(q*T-d*(Slin+2*x*w))/(q+d).

Then throughout the amplitude tube,

    P=q+d>=q/2,
    |E|<=e0*delta^2*j^5,
    |E-Etilde|<=e1*delta*n*j^5,
    n=max(||w-wtilde||_3,|alpha-alphatilde|).             (5)

These estimates use only amplitude bounds, not endpoint values. The exact
equation is L_C w=epsilon*f4+epsilon^2*m*b1+E-alpha*x^2.

Let (v(f),a(f)) be the zero-endpoint projected inverse defined in the
preceding note. On the closed affine space of functions with the prescribed
endpoints, map

    (w,alpha) -> (w_b+v(f), alpha_b+a(f)),
    f=epsilon*f4+epsilon^2*m*b1+E(w,alpha).

The linear inverse estimates and (3),(5) bound the image norm by

    K_b*zeta/R+K*epsilon*F0+K*delta^2*S*(F1+e0)
       <=delta*[K_b+K*F0+K*eta_e*(F1+e0)]
       <=delta*[K_b+K*(F0+1)] <= M_e*delta/2.

Thus the image lies in half the amplitude tube, including when the
endpoint data lie on the boundary of their closed square. Its Lipschitz constant is
at most K*delta*S*e1<=1/2. Banach's theorem closes the existence and
uniqueness proof. The differential equation, positivity, and flight time
follow exactly as in the zero-endpoint theorem. For delta=0 the tube is a
singleton and the assertion follows directly.

## Dependence on endpoint data

The construction is uniform over the full box |z_-|,|z_+|<=zeta: use the
same delta and the same tube for every pair of endpoint values. At each
fixed R it gives an analytic matching relation alpha=alpha_R(epsilon,z_-,z_+)
on a neighborhood of the CLOSED endpoint square. To fix the Banach space,
write w=w_b+u with u(-R)=u(R)=0 and use the fixed-point equation for
(u,alpha). At its solution, the image lies in half the amplitude tube.
Consequently the bound for |d| improves from d0 to
M_e+2*L*Y+L*Y^2<d0, so P>q/2 for delta>0; the derivative norm is
at most 1/2<1. These strict margins persist in some parameter and endpoint
neighborhood of each solution, including solutions over the boundary of
the square. The rational map is analytic there, and I minus its derivative
is invertible by the convergent Neumann series. The analytic implicit-function
theorem therefore supplies the asserted neighborhoods. For delta=0, P=q>0
and the nonlinear derivative at the base point is zero, so the same local
conclusion holds directly. Uniqueness identifies overlapping neighborhoods.
Compactness supplies a C^1 neighborhood of the whole closed square.
This justifies the continuation argument in the fixed-alpha corollary below;
no endpoint differentiability is inferred merely from interior estimates.

## Uniform nondegeneracy and opposite endpoint derivatives

The following variational estimate applies to the actual nonlinear
trajectory above, after a further explicit sector reduction.
Write f=N/P for its x-time slope, where N=2*x*w+Slin+T and P=q+d. Set

    p0=1+2*L+2*L*Y, n0=z0+t0,
    K_w=4*d0/kappa^2+2*t1/kappa+4*n0*p0/kappa^2,
    E_alpha=4*L*Y^2*kappa^(-4),
    eta_t=min(eta_e,kappa/(2*M_e),1/(4*K_w),
              J0/(36*E_alpha)).                        (6)

Assume delta*S<=eta_t. Direct differentiation gives

    |f_w-2*x/q|<=K_w*delta,
    f_alpha=-[x^2*(q+w)+epsilon*r*x*(y0+w)^2]/P^2.       (7)

For the first bound, |N|<=n0*delta*j^4, |T_w|<=t1*delta*j^2,
and |P_w|<=p0 throughout the amplitude tube. Differentiate the quotient:

    f_w-2*x/q=-2*x*d/(q*P)+T_w/P-N*P_w/P^2.

Since q>=kappa*j^2 and P>=q/2, these three terms are bounded by
4*d0*delta/kappa^2, 2*t1*delta/kappa, and
4*n0*p0*delta/kappa^2, respectively. This proves the displayed K_w
bound without a hidden dependence on R. For the second identity,
P_alpha=x and the physical y velocity is
x*(q+w)+epsilon*r*(y0+w)^2; quotient differentiation gives (7).

For the first inequality, write
f_w-2*x/q=2*x*(1/P-1/q)+T_w/P-N*P_w/P^2.
The respective bounds are 4*d0*delta/kappa^2,
2*t1*delta/kappa and 4*n0*p0*delta/kappa^2.

Let V(R,x)=exp(integral[x,R] f_w) denote the scalar variational multiplier.
Relative to the base multiplier H(R)/H(x), write

    V(R,x)=H(R)/H(x)*E_R(x).

Equation (6) implies |log E_R(x)|<=2*K_w*delta*R<=1/2; therefore
1/2<=E_R(x)<=2. No endpoint limits or formal Taylor series are used.
Also |w|<=q/2 and q/2<=P<=3q/2. In the integral

    I_R=integral[-R,R] E_R(x)*f_alpha(x)/H(x) dx,

the first term of f_alpha contributes at most -J_R/9. The absolute
integral of its r-dependent second term is at most 2*epsilon*E_alpha,
because

    |epsilon*r*x*(y0+w)^2/(P^2*H)|
       <=4*epsilon*L*Y^2*kappa^(-4)*j^-3,
    integral[-R,R] j^-3<=1.

Consequently

    I_R<=-J0/9+2*epsilon*E_alpha<=-J0/18<0,
    partial_alpha w(R)=H(R)*I_R.                       (8)

This proves genuine shooting nondegeneracy. It is uniform in the endpoint
box and in the admitted parameter sector. The derivative formulas for the
matching function are

    partial_(z_+) alpha = [D(R)/q(R)]/I_R < 0,
    partial_(z_-) alpha = -E_R(-R)*[D(-R)/q(-R)]/I_R > 0.

Thus

    |partial_(z_+) alpha| <= 18/(kappa*J0*S^2),
    |partial_(z_-) alpha| <= 36/(kappa*J0*S^2).          (9)

These powers are for the NORMALIZED coordinates z_+-=w(+-R)/q(+-R).
Using unnormalized endpoint w heights would introduce another q factor.
For two-sided nondegeneracy set M_alpha=8*W0+2*E_alpha. The same estimates
give |I_R|<=M_alpha, so

    -partial_(z_+) alpha >= c_* /(Q*M_alpha*S^2),
     partial_(z_-) alpha >= c_* /(2*Q*M_alpha*S^2).

Indeed the main absolute integral is at most 12*J_R<=8*W0, and the
remaining integral is at most 2*epsilon*E_alpha<=2*E_alpha. Hence the
R^2-rescaled endpoint derivatives have uniform, opposite, nonzero signs.

## Logarithmic matching relation on the common endpoint box

Intersect the sector (6) with the zero-endpoint sector eta0 from the
preceding note. Its zero-endpoint solution lies in the present larger tube
because M_e includes 2U0,2A0 and exceeds its remaining amplitude constant.
Uniqueness therefore identifies the two zero-endpoint constructions.
Integrating (9) along a straight segment inside the common endpoint box gives

    |alpha_R(epsilon,z_-,z_+)-alpha_R(epsilon,0,0)|
       <=54*zeta/(kappa*J0*S^2).

Combining this with the proved nonlinear zero-endpoint asymptotic yields

    alpha_R(epsilon,z_-,z_+)
       =epsilon*[-(C+6)*p-3*r]
         +8*(C+3)*epsilon^2*[p*(2*p+r)-m]*log R
         +O(epsilon/R+epsilon^2+zeta/R^2).              (10)

All constants in (10) are uniform on the declared common sector and
parameter/endpoint boxes. For R=rho/epsilon and fixed positive rho and zeta,
the endpoint variation is O(epsilon^2*zeta/rho^2). This is an actual
nonlinear matching relation on common nonzero sections, with the logarithmic
term retained. It does not assert a limit-cycle count. Higher section jets,
singular-chart matching, and zeros of the complete return displacement
remain open obligations.

## A genuine fixed-alpha passage map on the full input interval

One can go beyond a different alpha for each endpoint pair. Define

    gamma=4*C_- /[(C_++3)*(C_++1)],
    theta=1/(1+gamma/2)<1,
    eta_p=min(eta_t,eta0,gamma/(4*K_w)).                 (11)

Assume delta*S<=eta_p and zeta>0. Fix once and for all
alpha_*=alpha_R(epsilon,0,0). Then the actual quadratic field at alpha_*
has a section-to-section passage map G_R on the ENTIRE common interval
[-zeta,zeta], with

    G_R(0)=0, 0<G_R'(z)<=theta,
    |G_R(z)|<=theta*|z|.

To prove this, the base normalized variational factor equals

    g0=q(R)*D(-R)/[q(-R)*D(R)],
    log g0=-integral[0,R] (3*x^2+C)/[q(x)*q(-x)] dx.

On [0,1], the numerator is at least C_- and the denominator is at most
(C_++3)*(C_++1)/4, so log g0<=-gamma. Along any of the admitted
nonlinear trajectories the actual normalized derivative is
g0*E_R(-R), and hence is at most

    exp(-gamma+2*K_w*delta*R)<=exp(-gamma/2)<=theta.

Now continue the level set alpha_R(epsilon,z_-,z_+)=alpha_* from (0,0).
Its slope is the positive normalized flow derivative above. The estimate
|z_+|<=theta*|z_-| prevents this graph from reaching the top or bottom of
the common endpoint square before reaching either full input endpoint.
The matching function is continuous on the square, and its z_+ derivative
never vanishes; compactness and the implicit-function theorem therefore
continue the level graph across [-zeta,zeta]. Its uniqueness follows from
the strict decrease of alpha in z_+. Every point of this graph represents
an actual trajectory at the SAME alpha_*, by the endpoint existence theorem.
This proves the claimed fixed-alpha passage map and its uniform contraction.

For C in [2,3], gamma=1/3 and theta=6/7 are valid. The additional rational
sector constraint is eta<=1/(12*K_w). This result treats only the regular
passage; it supplies neither the singular closing map nor a zero bound for
their composition.

## Reproducing the finite checks

The companion [benchmark](../../benchmarks/hilbert16_endpoint_passage.py)
checks the boundary projection signs, the exact shooting source, the paired
base variational identity, and every displayed rational smallness condition
on two compact parameter boxes. Its final eta_p includes the zero-endpoint
sector eta0. These are finite symbolic and rational checks, not a formal
verification of the analytic passage theorem or a cycle-count claim.

```sh
uv run --no-sync python benchmarks/hilbert16_endpoint_passage.py --output artifacts/hilbert16/endpoint_passage.json
uv run --no-sync python -O benchmarks/hilbert16_endpoint_passage.py --output artifacts/hilbert16/endpoint_passage_optimized.json
```

# A uniform nonlinear compensated passage in a small joint-scaling sector

This note proves a local passage theorem for the actual five-parameter
quadratic field. It does not prove a return-map zero bound, quadratic
cyclicity, or Hilbert 16. The argument is analytic; the accompanying script
checks its finite algebra but does not formally verify the analytic theorem.

## Statement with an explicit common domain

Fix 1<C_-<=C_+<infinity and L>=1. Let C be in [C_-,C_+] and let
|p|,|r|,|m|<=L. Consider

    xdot=(1+alpha)*x-y+x^2+epsilon*(p+r)*x*y+epsilon^2*m*y^2,
    ydot=C*x+x^2+x*y+epsilon*r*y^2.

Set j(x)=1+|x|, y0=(x^2-C)/2, w=y-y0, q=(x^2+2*x+C)/2.
The constants M and eta0 below depend only on C_-,C_+,L and are strictly
positive and explicit.

**Theorem 1.** For every R>=2 and epsilon>0 satisfying

    epsilon*(R+1)<=eta0,

there exists a unique pair (w,alpha) in the closed ball

    w(-R)=w(R)=0,
    sup[-R,R] |w(x)|/j(x)^3 <= M*epsilon,
    |alpha|<=M*epsilon                                      (B)

that solves the trajectory equation. The function w is C^1. Throughout the
passage,

    xdot>=q/2>0,  xdot>=(C_--1)/4,
    passage_time <= 4*pi/sqrt(C_--1).

In particular, for any fixed 0<rho<=eta0/2 and 0<epsilon<=rho/2, the
joint choice R=rho/epsilon satisfies the theorem. This is a common
nonlinear passage domain, obtained without substituting into fixed-R jets.
Uniqueness is within (B), not among all possible distant trajectories.

The resulting shooting parameter also retains the logarithmic term:

**Theorem 2.** Write

    alpha1=-(C+6)*p-3*r, beta=p*(2*p+r)-m.

There is K_* depending only on C_-,C_+,L such that throughout the same
sector,

    |alpha-epsilon*alpha1-8*(C+3)*epsilon^2*beta*log R|
       <=K_* (epsilon/R+epsilon^2).                       (A)

Thus at fixed rho as above,

    alpha=epsilon*alpha1+8*(C+3)*epsilon^2*beta*log(1/epsilon)
             +O_rho(epsilon^2).

The log rho term is included in the final O_rho bound. Unlike the earlier
fixed-section Taylor result, (A) concerns actual nonlinear solutions on the
common sector in Theorem 1. The companion
[endpoint passage proof](HILBERT16-ENDPOINT-PASSAGE.md) extends this result
to varying section coordinates and proves the first section derivative bound.

## Explicit constants for Theorem 1

All arithmetic in this list is real arithmetic; these are conservative
analytic bounds, not numerically optimized interval results. Define

    c_*=exp(-4*pi/sqrt(C_--1)), Q=C_+/2,
    kappa=min(1,C_--1)/16,
    H0=Q^2/c_*, W0=kappa^(-3), J0=8/[9*(C_++8)^3],
    G0=81*Q^2*W0,
    A4=2*W0/J0,
    B4=G0*(1+A4/3), K=max(A4,B4),
    F0=L*(Q^2+2*Q), F1=L*Q^2,
    U0=L*(6+3*C_+), A0=L*(C_++9),
    M=max(1,2*K*(F0+1),2*U0,2*A0), Y=Q+1,
    d0=2*M+2*L*Y+L*Y^2,
    z0=3*M+F0+F1,
    b0=L*(1+C_+),
    t0=b0*M+L*M+2*L*Q+L,
    e0=2*t0+2*d0*z0/kappa,
    d1=2+2*L+2*L*Y,
    t1=b0+4*L+2*L*Q,
    e1=2*t1+2*d1*z0/kappa+6*d0/kappa+2*e0*d1/kappa,

    eta0=min(1,1/M,kappa/(2*d0),
             (F0+1)/(F1+e0),1/(2*K*e1)).                 (C)

These constants can make eta0 extremely small. Their purpose is to close
a nonempty uniform sector. Improving that sector is a separate problem.

## Projected linear inverse

Let

    D(x)=exp(4*(atan((x+1)/sqrt(C-1))-pi/2)/sqrt(C-1)),
    H=q^2/D, W=D/q^3, L_C=q*d/dx-2*x.

Then D'/D=2/q, H'=2*x*H/q. Uniformly in the stated C range,

    kappa*j^2<=q<=Q*j^2, c_*<=D<=1,
    H<=H0*j^4, W<=W0*j^-6.

For the lower q bound use j<=|x+1|+2, hence j^2<=2*(x+1)^2+8.
The definitions of kappa and q give the claimed bound directly. The upper
bound follows from C_+>=1. For R>=2,

    J_R=integral[-R,R] W*x^2 >= J0,

by restricting the integral to [1,2], where x^2>=1,
q<=(C_++8)/2, and
D(x)>=exp(-4/(x+1))>=exp(-2)>1/9. The latter inequalities follow
from atan u<=u for u>=0 and e<3.

The two tail kernels admit a substantially better bound than multiplying
the separate H and W bounds. D is increasing. If x<=0 and s<=x,
D(s)/D(x)<=1. If x>=0 and s>=x, then
D(s)/D(x)<=1/D(x)<=exp(4/(x+1))<=exp(4)<81. Consequently, on
either of the endpoint tails used below,

    H(x)*W(s)<=G0*j(x)^4*j(s)^(-6).

This removes the small c_* from the contraction constants. H0 is retained
only as an auxiliary global bound for the later polynomial comparison.

For continuous forcing f define

    a(f)=integral[-R,R] W*f / J_R,
    v(f)(x)=H(x)*integral[-R,x] W*(f-a(f)*x^2).

The integral over [-R,R] vanishes, so for x>=0 one may instead use minus
the integral over [x,R]. These formulas give the unique solution of

    L_C v=f-a*x^2, v(-R)=v(R)=0.

Let ||f||_k=sup |f|/j^k. For degree-four growth,

    |a(f)|<=A4*||f||_4,
    ||v(f)||_3<=B4*||f||_4.                              (L4)

Indeed integral_R j^-2<=2 and the one-sided tails of j^-2 and j^-4
are at most j^-1 and (3*j^3)^-1. For degree-five growth, with S=R+1,

    |a(f)|<=A4*log(S)*||f||_5,
    |v(f)(x)|<=G0*||f||_5 *
         [j^4*log(S/j)+(A4/3)*j*log S],                  (L5a)
    ||v(f)||_3<=B4*S*||f||_5.                           (L5b)

Here integral[-R,R] j^-1=2 log S, and
j*log(S/j)<=S/e<=S, log S<=S. The projected tail representation is
essential: estimating an uncancelled growing homogeneous solution would
lose uniformity.

## Exact nonlinear equation and uniform bounds

Put

    b2=-x^2*y0, b3=y0^2-x^2*y0, b1=-x*y0^2,
    f4=p*b2+r*b3,
    Slin=-alpha*x^2+epsilon*f4+epsilon^2*m*b1,
    T=(-epsilon*p*x^2-C*epsilon*r)*w+epsilon*r*w^2
          -epsilon^2*m*x*(2*y0*w+w^2),
    d=-w+alpha*x+epsilon*(p+r)*x*(y0+w)
          +epsilon^2*m*(y0+w)^2,
    Z=Slin+2*x*w,
    E=(q*T-d*Z)/(q+d).

The physical x velocity is P=q+d and the w velocity is 2*x*w+Slin+T.
Consequently, wherever P!=0, the trajectory equation is EXACTLY

    L_C w=epsilon*f4+epsilon^2*m*b1+E-alpha*x^2.          (N)

No w derivative occurs in E. Also

    ||f4||_4<=F0, |m*b1|<=F1*j^5.

For any input in (B), let eta=epsilon*S. The constraints eta<=1 and
M*eta<=1 imply |y0+w|<=Y*j^2. Then

    |d|<=d0*epsilon*j^3,
    |Z|<=z0*epsilon*j^4,
    |T|<=t0*epsilon^2*j^5.

For T, the four respective bounds are b0*M, L*M, 2*L*Q and L,
after factoring epsilon^2*j^5; use M*epsilon*j<=1 for the last three.
Since d0*eta/kappa<=1/2, one has P>=q/2 throughout the entire input
ball. Hence

    |E|<=e0*epsilon^2*j^5.                              (E0)

For two ball inputs let

    n=max(||w-wtilde||_3, |alpha-alphatilde|).

Elementary difference bounds are

    |d-dtilde|<=d1*n*j^3,
    |Z-Ztilde|<=3*n*j^4,
    |T-Ttilde|<=t1*epsilon*n*j^5.

Using

    E-Etilde = [q*(T-Ttilde)-(d-dtilde)*Z
                  -dtilde*(Z-Ztilde)-Etilde*(d-dtilde)]/(q+d),

and (E0), gives

    |E-Etilde|<=e1*epsilon*n*j^5.                        (E1)

The last term uses epsilon*j<=eta<=1. This explains every constant
in (C); no assertion that an interval gate passed is used.

## Closed contraction and passage proof

On the complete space C([-R,R]) with both zero endpoint conditions,
times R for alpha, use norm max(||w||_3,|alpha|). Map (w,alpha) to

    (v(f),a(f)), f=epsilon*f4+epsilon^2*m*b1+E(w,alpha).

Equations (L4)-(L5b), (E0) give a norm bound

    K*epsilon*[F0+eta*(F1+e0)]
       <=K*epsilon*(2*F0+1)<M*epsilon.

The map preserves (B). Equation (E1) and (L5b) give Lipschitz constant

    K*epsilon*S*e1<=1/2.

Banach's fixed-point theorem gives exactly one fixed point in (B).
Its integral formula is C^1, and (N) and P>=q/2 show that it is a
trajectory of the actual field. The physical time satisfies

    integral[-R,R] dx/P <= 2*integral_R dx/q
       =4*pi/sqrt(C-1)<=4*pi/sqrt(C_--1).

This proves Theorem 1, including positivity and a common flight-time bound.

## Retaining the actual logarithmic renormalization

All constants in the estimates below depend only on C_-,C_+,L; this section
does not claim that K_* is numerically sharp. They can be obtained from the
displayed bounds and the fixed finite polynomials below. Write S=R+1 and

    U=p*(x^3+3*x^2+3*C/2)
        +r*(x^3/2+3*x^2/2+C*x/2+C),
    h=alpha1*x+(p+r)*x*y0, B=-p*x^2-C*r,
    F2(U)=(U-h)*U'+B*U.

The exact identity L_C U=f4-alpha1*x^2 holds. By the first-variation
antiderivative or by taking its explicit polynomial endpoint expression,

    a(f4)-alpha1=O(1/R),
    v(f4)(x)-U(x)=O(j^4/R).                             (P)

For example, the alpha particular polynomial is

    P_alpha=-(x^4+8*x^3+2*(C+12)*x^2+C*(C+12))/(16*(C+3)),
    L_C P_alpha=-x^2.

Combining U, (a(f4)-alpha1)*P_alpha and the homogeneous term that makes
the incoming value zero proves (P); each coefficient of the latter two
degree-four growth terms is O(1/R). There is no logarithmic alpha term in
this first-order calculation.

Apply the SHARP profile (L5a) to the nonlinear fixed point, with
f5=epsilon^2*m*b1+E and ||f5||_5<=epsilon^2*(F1+e0). Together with (P),
this proves

    w-epsilon*U
       =O(epsilon*j^4/R
            +epsilon^2*[j^4*log(S/j)+j*log S]),           (W)
    alpha-epsilon*alpha1
       =O(epsilon/R+epsilon^2*log S).                   (Q)

In particular the potentially growing alpha mode has already been
cancelled by the projection; it is NOT estimated as a naked
epsilon^2*log(S)*P_alpha.

The exact rational expression E has the following pointwise refinements
on the ball, by the same difference calculation used for (E1):

    |E(w,alpha)-E(wtilde,alphatilde)|
       <=K1*epsilon*[j^2*|w-wtilde|+j^3*|alpha-alphatilde|],
    E(epsilon*U,epsilon*alpha1)
       =epsilon^2*F2(U)+O(epsilon^3*j^6).                (R)

For an explicit K1, put d_w=1+2*L+2*L*Y and take the maximum of

    2*t1+2*d_w*z0/kappa+4*d0/kappa+2*e0*d_w/kappa,
    2*(z0+d0+e0)/kappa.

These follow from |delta d|<=d_w*|delta w|+j*|delta alpha|,
|delta Z|<=2*j*|delta w|+j^2*|delta alpha|, and
|delta T|<=t1*epsilon*j^2*|delta w|. Thus the refined Lipschitz bound
does not assume an unproved derivative estimate for the unknown trajectory.

The base pair in (R) obeys the amplitude bounds of the tube by
M>=2U0,2A0; it need not obey the zero endpoint conditions. The pointwise
estimates for E use only those amplitude bounds. For the second
identity expand the displayed finite polynomials for Slin,T,d: the
order-two coefficient is F2(U), and the remaining numerator terms,
after division by P>=q/2, are bounded by epsilon^3*j^6 when
epsilon*j<=eta0. The epsilon^2*m term starts contributing to E at
order three; its order-two contribution was already isolated as m*b1
in (N).

Multiplying (R) by W<=W0*j^-6 and integrating, then using (W)-(Q), gives

    integral[-R,R] W*(E-epsilon^2*F2(U))=O(epsilon^2).    (I)

The terms are respectively bounded by constants times

    epsilon^2/R * integral[-R,R] 1 = O(epsilon^2),
    epsilon^3 * integral[-R,R] log(S/j) = O(epsilon^3*S),
    epsilon^3*log S * integral[-R,R] j^-3,
    epsilon*(epsilon/R+epsilon^2*log S)*integral[-R,R] j^-3,
    epsilon^3 * integral[-R,R] 1.

Here integral[-R,R]log(S/j)<=2*S,
integral[-R,R]j^-3<=1, epsilon*S<=eta0, and
epsilon*log S<=epsilon*S. Every bound in (I) is uniform in R and epsilon.

Finally, exact polynomial algebra gives [x^5]F2(U)=p*(2*p+r)/4. With
c=exp(-4*pi/sqrt(C-1)), the two tails yield

    integral[-R,R] W*(F2(U)+m*b1)
       =2*(1-c)*beta*log R+O(1),
    J_R=(1-c)/(4*(C+3))+O(R^-3).

Both remainders are uniform on the stated compact C and bounded direction
sets. Project (N), use (I), and divide by J_R>=J0. The linear part is
epsilon*alpha1+O(epsilon/R). This proves (A). In particular, (A) is
not inferred from a nonuniform Taylor expansion: the uniform nonlinear
fixed point and the spatially resolved bound (W) supply its remainder.

## Remaining scope

The construction permits all bounded p,r,m, including a nonzero effective
logarithmic direction. It does not require the exact invariant-parabola
surface. It selects alpha to join two fixed zero-height sections and proves
one passage in a small tube. The
[endpoint passage proof](HILBERT16-ENDPOINT-PASSAGE.md) now supplies a common
nonzero endpoint box, the first section derivative, and a uniformly
contracting regular passage for one fixed alpha. Higher uniform section
jets, matching into the singular charts, their endpoint behavior, and a
uniform bound on zeros of the full return displacement remain unproved. The
parameter sector is deliberately small and the constants are conservative.

## Reproducing the finite checks

The companion [benchmark](../../benchmarks/hilbert16_uniform_passage.py)
checks the exact field transformation, source and remainder identities,
and rational constant inequalities on two compact parameter boxes. These
finite checks support the written argument; they do not machine-check its
infinite analytic conclusions. The constants and every acceptance gate use
`fractions.Fraction`; Python optimization does not remove the checks.

```sh
uv run --no-sync python benchmarks/hilbert16_uniform_passage.py --output artifacts/hilbert16/uniform_passage.json
uv run --no-sync python -O benchmarks/hilbert16_uniform_passage.py --output artifacts/hilbert16/uniform_passage_optimized.json
```

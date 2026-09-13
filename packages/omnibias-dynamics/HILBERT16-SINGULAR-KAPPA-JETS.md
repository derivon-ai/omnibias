# Actual second kappa derivatives in the joint singular corner

This note extends the existing actual two-scale singular multiplier estimate
using real inverse-height variational equations. It keeps the physical field
fixed during every kappa derivative. It does not differentiate a value
remainder, infer a uniform estimate from numerical samples, or claim formal
verification of the analytic estimates. The proof uses the exact analytic
normal family, actual height dependence and physical sections already used
in [joint matching](HILBERT16-JOINT-MATCHING.md) and the
[two-scale correction](HILBERT16-TWO-SCALE-CORRECTION.md).

## 1. Statement and the derivative being estimated

Fix the canonical `r=-1` family, a fixed sufficiently small section size
rho, and a coefficient compact with `L=-lambda0>0`, bounded lambda,
`4L-lambda^2>=chi>0`. The physical parameters, including epsilon,
nu, A, C, L and lambda, are held fixed. Write

    omega=exp(-kappa), u=epsilon/omega, epsilon=omega*u,
    H=epsilon^3*exp(-kappa/epsilon).

Let d_sigma and beta0=1/3 be the constants in the two-scale note. Put

    E=omega^2+u^2+epsilon*log(1/epsilon),
    Astar=-lambda*(1/dminus+1/dplus),
    Bstar=3*beta0*(dminus+dplus).

The result established below is, for j=0,1,2,

    |partial_kappa^j [log D_epsilon'(ti(kappa))
        -2*pi*lambda/sqrt(4L-lambda^2)-Astar*omega-Bstar*u]|
       <=C_rho*E.                                             (T)

The independent upper cutoffs on omega,u and the constant are uniform over
the fixed compacts. The actual signed endpoint labels also satisfy

    |partial_kappa ti|+|partial_kappa^2 ti|
      +|partial_kappa to|+|partial_kappa^2 to|<=C_rho*u^2.       (L)

The sign restriction lambda<0 is useful for subsequent cycle comparisons,
but is not needed for these derivative estimates under strict Delta.

In normalized formulas the derivative is the curved scale generator

    E_kappa=u*partial_u-omega*partial_omega,

with epsilon treated as a fixed dependent product. Its second action is

    E_kappa^2 R=omega^2*R_omegaomega-2*omega*u*R_omegau
               +u^2*R_uu+omega*R_omega+u*R_u.

In particular `E_kappa epsilon=E_kappa^2 epsilon=0`. A straight directional
Hessian would omit the last two terms and is not the derivative used here.
Below `O_C2k(F)` means that each of the actual kappa derivatives of orders
zero, one and two is bounded by a common constant times F at that point.

## 2. Exact equations and coefficient estimates

Use the exact radial scaling from the existing proof:

    V=sigma*u*X, h=epsilon*u^2*Y,
    epsilon*Y_X=X*Y/(b+khat*Y),
    Y(0)=omega^2*exp(-kappa/epsilon),

    b=X^2-sigma*lambda*omega*X+L*omega^2
          -sigma*beta0*u*X^3+O(epsilon*u*X^3),
    khat=k(sigma*u*X,epsilon*u^2*Y,epsilon).

The last error is an actual analytic coefficient error, with bounded finite
derivatives. It follows from the exact identity
`zeta(V,epsilon)=-1+beta0*V+epsilon*V*Z(V,epsilon)`.

On the fixed normalized interval `[0,K]`, the existing proof gives

    b>=c*(omega^2+X^2), 1/2<=khat<=2,
    |b_X|<=C*(omega+X), 0<=epsilon*Y_X<=2*X.

The exact normal coefficient has `partial_V^a(k-1)=O(epsilon)` and
`partial_V^a partial_h^b k=O(epsilon^2)` for b>=1 and the finite spatial
orders used below. Epsilon and all passive parameters are fixed in these
derivatives. Epsilon derivatives do not have these orders. Only finitely
many, at most four, spatial coefficient derivatives are needed below.

These bounds follow from the actual off-line coordinate change in
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md), rather than from
its restriction to h=0. With w=1-V, its analytic inverse gives

    v=w-nu*w^2+nu^2*(2*w^3-C*w*h)+O(nu^3),
    g=-1+nu*(V-1)+nu^2*g2+O(nu^3),
    g2=-2*C*V^2+3*C*V-C*h-C
                         +3*V^2-Abar*V-6*V+Abar+3.

Here g is the coefficient of h in the exact V equation, k=-g, and
A=1+nu*Abar. Substitution in the exact field cancels the unspecified
canonical slow coefficients from this two-jet. The canonical conversion
has `k_scale=3-9*nu+O(nu^2)` and hence

    nu=epsilon/3+epsilon^2/3+O(epsilon^3),
    k=1+(epsilon/3)*(1-V)+O(epsilon^2),
    k_h=C*epsilon^2/9+O(epsilon^3).

Joint analyticity on a slightly enlarged fixed spatial and parameter
compact permits Taylor division by epsilon and bounds the required
spatial derivatives of the remainders. In particular, the entire
coefficient of epsilon is height independent. This proves the stated
mixed spatial bounds; it does not infer them by differentiating an
uncontrolled value remainder.

Write Q for the displayed quadratic part of b. For j<=2,

    |E_kappa^j b|<=C*(omega+X)^2

on `[0,K]`. On every fixed `[a,K]` separated from zero, the stronger bounds

    |E_kappa b|+|E_kappa^2 b|
      +|partial_X E_kappa b|+|partial_X E_kappa^2 b|
      <=C*(omega+u)                                       (2.1)

hold. All estimates also hold on a fixed slightly enlarged compact so that
the derivatives at the stated cutoffs are ordinary local derivatives.

There are useful exact scaling identities. At fixed physical epsilon,
`b(X,kappa)=omega^2*B_epsilon(X/omega)`, whence

    E_kappa b=X*b_X-2*b,
    E_kappa khat=X*khat_X+2*Y*khat_Y.

Consequently the phase A defined below satisfies exactly

    A_kappa=X^2/b-1,
    A_kappakappa=-X^2*(X*b_X-2*b)/b^2.                  (2.1a)

The lower integration-boundary term vanishes because `X^2/b` tends to
zero at X=0 for each positive omega. On `[a,K]`, these identities give
`A_kappa,A_kappakappa=O(omega+u)` directly, without differentiating any
phase remainder.

The phase expansion itself can be differentiated independently of the orbit:

    A(X,kappa):=integral_0^X s/b(s) ds-kappa
      =log X-log d_sigma-sigma*lambda*omega/X
                       +sigma*beta0*u*X+O_C2k(E),          (2.2)

uniformly for fixed `a<=X<=K`, with the needed first two X derivatives.
Here is a justification of the derivative remainder. In the reciprocal
expansion used in the existing value proof, applying E_kappa at most twice
preserves the integrable bounds

    C*u^2*s^7/(omega^2+s^2)^3, C*epsilon*u,

because the logarithmic scale derivatives of Q and b are bounded relative
to Q and b. Moreover, for j=0,1,2, the error in replacing
`integral_0^X s^4/Q(s)^2 ds` by X is bounded by
`C*omega*(1+log(1/omega))` after those derivatives: for j>=1 the differentiated
integrand is bounded by `C*omega/(omega+s)`, and the same bound applies to
the undifferentiated difference. The limiting quadratic primitive has an
analytic expansion in omega on `[a,K]`, so its remainder is `O_C2k(omega^2)`.
These facts prove (2.2) by differentiation under integrable majorants.
They also give `A_kappa,A_kappakappa=O(omega+u)` and `A_X>=c>0` there.

## 3. The early exponentially small segment, including derivatives

Choose a fixed a below the common first-height event bracket such that
`A(a,kappa)<=-2c_a<0` on the enlarged parameter rectangle. Define

    z=epsilon*log(Y/omega^2).

At fixed X it solves the exact equation

    z_X=F(X,z,kappa)=X/(b+khat*Y), z(0)=-kappa,
    Y=omega^2*exp(z/epsilon), h=epsilon^3*exp(z/epsilon).    (3.1)

Notice that h at fixed z is exactly independent of kappa. The phase upper
bound gives `z(X)<=-2c_a` on `[0,a]`; hence
`Y/b<=C*exp(-2c_a/epsilon)` on that whole interval. Positivity gives

    F_z=-X*Y*(k+h*k_h)/(epsilon*(b+kY)^2)<=0.              (3.2)

For sufficiently small epsilon the last sign follows from k>=1/2 and the
uniform bound on h*k_h. The first kappa variation Z1 satisfies

    Z1_X=F_z*Z1+F_kappa, Z1(0)=-1.

The coefficient bounds imply `|F_kappa|<=C*X/b` and
`integral_0^a X/b dX<=C*(1+log(1/epsilon))`; here
`kappa<=log(1/epsilon)` since u<=1. The nonpositive homogeneous coefficient
therefore bounds Z1 by a logarithmic polynomial, without exponential
Gronwall growth.

The second variation has the same nonpositive homogeneous coefficient:

    Z2_X=F_z*Z2+F_zz*Z1^2+2*F_zkappa*Z1+F_kappakappa,
    Z2(0)=0.

Its source is bounded by `C*X/b` times a finite polynomial in
`epsilon^-1` and `log(1/epsilon)`; every inverse-epsilon term from a z
derivative also carries the exponentially small factor Y/b. Thus Z2 is
bounded by such a polynomial as well. Consequently the first two kappa
derivatives of Y are bounded by Y times a fixed polynomial in those same
quantities. This is sufficient: no sharp estimate for Z1 or Z2 at the
origin is required.

In the exact phase loss and compensated exponent errors,

    Q_a=integral_0^a X*kY/[b*(b+kY)] dX,

    P_a=integral_0^a [-b_X*kY/(b*(b+kY))+Y*k_X/(b+kY)] dX,

all kappa derivatives up to two have the same exponentially small factor
times integrable rational coefficients and polynomial losses. For example
`integral |b_X|/b=O(1+log(1/epsilon))`; the lower bound
`omega>=epsilon` controls any additional finite inverse powers of omega.
After reducing c_a in the exponent, this proves

    Q_a,P_a=O_C2k(exp(-c_a/epsilon)).                     (3.3)

Therefore the moving initial value for the inverse-height equation is

    z_a=A(a,kappa)-Q_a,
    partial_kappa z_a,partial_kappa^2 z_a=O(omega+u)      (3.4)

where the last line denotes the first and second derivatives, not z_a
itself. The value z_a stays in a fixed compact negative interval.

## 4. Inverse logarithmic height through both small levels

Starting from `X(z_a)=a`, invert (3.1):

    X_z=(b(X)+k(sigma*u*X,epsilon^3*exp(z/epsilon),epsilon)
                       *omega^2*exp(z/epsilon))/X.        (4.1)

We use this equation through

    z=0: Y=omega^2, X=X1,
    z=z2:=2*epsilon*kappa: Y=1, X=X2.

The existing value proof puts X in a common interval bounded away from
zero, and z in a bounded interval; also `Y<=1` on this part. At fixed z,

    Y_kappa=-2Y, Y_kappakappa=4Y, h_kappa=0.

The right side Ftilde of (4.1) has bounded X derivatives through order two,
and its first two explicit kappa derivatives, including the mixed X
derivative, are bounded by `C*(omega+u+Y)`. The crucial integrable identity
is

    integral_za^z2 Y dz<=epsilon.

Ordinary first and second parameter variational equations therefore give

    |X_kappa|+|X_kappakappa|<=C*(omega+u)                  (4.2)

at fixed z throughout this interval. For completeness the moving initial
condition contributes `X_kappa(z_a)=-Ftilde(z_a)*z_a,kappa`.
Twice differentiating `X(z_a(kappa),kappa)=a` gives terms involving
`z_a,kappa`, `z_a,kappakappa`, bounded X derivatives of Ftilde, and
`Ftilde_z(z_a)`. The potentially large `Y/epsilon` part of the last term
is exponentially small at z_a. Thus the same bound holds for the second
initial datum, and standard bounded-interval Gronwall estimates apply.

Explicitly the second initial identity is

    X_kappakappa+2*X_kappaz*z_a,kappa
       +X_zz*(z_a,kappa)^2+X_z*z_a,kappakappa=0,

evaluated at z_a, where
`X_kappaz=Ftilde_X*X_kappa+Ftilde_kappa` and
`X_zz=Ftilde_X*Ftilde+Ftilde_z`.

The exact first-level phase loss now has a sharper derivative estimate:

    A(X1,kappa)=Q_a+integral_za^0 kY/b dz
      =O_C2k(epsilon*omega^2+exp(-c_a/epsilon)).           (4.3)

Indeed X is separated from zero, (4.2) controls its first two variations,
and `integral_za^0 Y dz<=epsilon*omega^2`. The lower moving-endpoint terms
are exponentially small. Combining (4.3) with (2.2), its X derivative
lower bound, and the first two implicit derivative equations proves

    X1=d_sigma+sigma*lambda*omega
                  -sigma*beta0*d_sigma^2*u+O_C2k(E).     (4.4)

This is an estimate of the actual moving level and its derivatives, not
a differentiated localization bound.

For the exponent, the inverse logarithmic height equation cancels its
denominator exactly:

    Psi_z=(b_X+Y*k_X)/X.                                (4.5)

Thus, before z=0, its difference from the derivative of log b is

    Y/X*(k_X-k*b_X/b).

Equations (3.3), (4.2) and the integral of Y show

    Psi(X1)=log[b(X1)/(L*omega^2)]
           +O_C2k(epsilon*omega^2+exp(-c_a/epsilon)).     (4.6)

On `[0,z2]`, the integrand in (4.5) and its first two total kappa
derivatives at fixed z are bounded. Its z derivative at z2 is bounded
as well: the apparent `Y/epsilon` is multiplied by `k_X=O(epsilon*u)`.
More explicitly, `X_z=O(1)` when Y=1 and
`Y_z*k_X=(1/epsilon)*O(epsilon*u)=O(u)` there.
Since `z2_kappa=2*epsilon` and `z2_kappakappa=0`, Leibniz differentiation
gives

    Psi(X2)-Psi(X1)=O_C2k(epsilon*(1+kappa)).             (4.7)

At the second height event, the total derivatives of X2 are still
`O(omega+u)`. In its second derivative the term `X_zz*(2epsilon)^2`
is harmless: `X_zz=O(1+1/epsilon)` at Y=1, giving only O(epsilon).

## 5. Inverse linear height after Y=1

Set `T=epsilon*Y`, so h=u^2*T. The actual inverse equation is

    X_T=k(sigma*u*X,u^2*T,epsilon)/X
                            +epsilon*b(X)/(X*T).         (5.1)

It starts at the fixed T=epsilon with X=X2. Continue until X=K, at
T=T_K. The existing height bounds give common constants
`0<c<=T_K<=C`; X stays bounded away from zero. On this interval
`X_T>=c>0`. The first two explicit kappa derivatives of the right side
are bounded by

    C*[epsilon*u+epsilon*(omega+u)/T],

as are the mixed X,kappa derivatives needed by the two variational
equations. The pure first and second X derivatives instead have the
bound `C*(1+epsilon/T)`, whose integral is uniformly bounded. The
constant term includes the nonvanishing X derivatives of the leading
1/X term. Here the physical h varies at fixed T, but the extra terms are
`2*u^2*T*k_h=O(epsilon^2*u^2*T)`, and are included in the bound.
First and second variational equations and
`integral_epsilon^C epsilon/T dT=O(epsilon*log(1/epsilon))` give

    X_kappa,X_kappakappa=O(omega+u),
    T_K,kappa,T_K,kappakappa=O(omega+u).                  (5.2)

The second line follows from the transverse event X=K; its evaluation
occurs at T_K bounded away from zero. Again the notation denotes only
the derivatives of T_K.

The exact exponent integrand in this coordinate is

    Psi_T=epsilon*b_X/(X*T)+k_X/X.                      (5.3)

Its first two total kappa derivatives obey the integrable bound
`C*(epsilon/T+epsilon*u)`. The moving upper endpoint has the bounds in
(5.2). Therefore

    Psi(K)-Psi(X2)=O_C2k(epsilon*log(1/epsilon)).          (5.4)

At the physical moving radial cut s0=K*u, the height is
`h0=u^2*T_K`. Equations (5.2) yield the useful total derivative bounds

    |h0|+|h0_kappa|+|h0_kappakappa|<=C*u^2.              (5.5)

## 6. Physical tails and actual section-coordinate derivatives

On each radial side put s=|V| and retain the fixed-field equation

    h_s=R(s,h)=s*h/(q(sigma*s)+k(sigma*s,h)*h).

The established tube has `c*s^2<=h<=C*s^2` from s0=K*u to a fixed
neighborhood of the selected physical event. The denominator is comparable
to h there, even where q itself need not remain positive. The exact
analytic field gives

    |q|<=C*(epsilon*s^2+epsilon^2*s+epsilon^3),
    |q_s|<=C*(epsilon*s+epsilon^2), |q_ss|<=C*epsilon,

and the k derivative bounds of section 2. Direct quotient differentiation
then yields

    |R_h|<=C*(epsilon/s+epsilon^2/s^2+epsilon^3/s^3
                                                   +epsilon^2*s),
    |R_hh|<=C*(epsilon/s^3+epsilon^2/s^4+epsilon^3/s^5
                                           +epsilon^2/s+epsilon^2*s).
                                                               (6.1)

At a fixed physical s let p=h_kappa and p2=h_kappakappa. Since the
physical field is fixed, there are no explicit parameter source terms:

    p_s=R_h*p, p2_s=R_h*p2+R_hh*p^2.                    (6.2)

The moving lower datum must be converted before using these equations.
Since `s0_kappa=s0_kappakappa=s0`, the exact identities are

    p(s0)=h0_kappa-R(s0,h0)*s0,
    p2(s0)=h0_kappakappa-2*R_h*p(s0)*s0
               -(R_s+R_h*R)*s0^2-R*s0.                (6.3)

Here R_s is partial at fixed h. The bounds `R=O(s0)`, `R_s=O(1)` and
(5.5) imply p(s0),p2(s0)=O(u^2). Integrating (6.1) gives

    integral_s0^S |R_h| ds<=C,
    u^4*integral_s0^S |R_hh| ds<=C*u^2,                (6.4)

uniformly for fixed S. In particular the leading terms on the second
line are `epsilon*u^2`, `epsilon^2*u`, and `epsilon^3`; each is at most
`C*u^2` because epsilon/u=omega<=1. Equations (6.2)–(6.4) prove

    |p|+|p2|<=C*u^2                                    (6.5)

on the full tails.

The actual section is E_tau(V,h)=0 with the exact E_tau in the joint note.
Its total s derivative `E_s+E_h*R` has a fixed nonzero margin near the
selected event. Explicitly, at epsilon=0 its restrictions to h=s^2/2 are
`s-1-rho*s^2/2` incoming and `-s-1+rho*s^2/2` outgoing. The selected
outer roots are `(1+sqrt(1-2*rho))/rho` and
`(1+sqrt(1+2*rho))/rho`, with radial derivatives
`-sqrt(1-2*rho)` and `sqrt(1+2*rho)`. For fixed rho<1/2 the relative
tail bounds preserve these margins on small fixed event neighborhoods.
The additional incoming inner root is outside the selected outer branch;
no first-hit assertion on the entire physical line is used.
Differentiating this actual event equation once and twice,
using (6.5) and bounded event derivatives, gives

    s_event,kappa,s_event,kappakappa=O_rho(u^2),
    h_event,kappa,h_event,kappakappa=O_rho(u^2).

The label is the exact `T_tau(h)=(tau*rho*h-1)^2-2h`. Its derivatives are
bounded on these fixed event compacts, which proves (L). This argument
does not differentiate the earlier value estimate `t=O(u^2+epsilon)`.

For the compensated exponent, write

    Jtail=integral_s0^s_event A(s,h(s)) ds,
    A=(q_s+h*k_s)/(q+k*h).

The required quotient bounds are

    |A|<=C*(epsilon/s+epsilon^2/s^2+epsilon),
    |A_h|<=C*(epsilon/s^3+epsilon^2/s^4
                                     +epsilon^2/s^2+epsilon^2),
    |A_hh|<=C*(epsilon/s^5+epsilon^2/s^6
                       +epsilon^2/s^4+epsilon^2/s^2+epsilon^2),
    |A_s|<=C*(epsilon/s^2+epsilon^2/s^3+epsilon).       (6.6)

The first and second differentiated interior integrals are respectively
`integral A_h*p` and `integral(A_h*p2+A_hh*p^2)`. By (6.5)–(6.6) both
are O(epsilon). For instance the most singular terms in the second are
`u^4*epsilon/u^4=epsilon` and
`u^4*epsilon^2/u^5=epsilon*omega`.

Leibniz endpoint terms have the same bound. For a temporarily fixed upper
endpoint b, the complete second derivative is exactly

    integral_s0^b (A_hh*p^2+A_h*p2) ds
      -A0*s0-(A_s,0+A_h,0*R0)*s0^2-2*A_h,0*p0*s0.

At the lower event their
possible types are `A*s0`, `A_s*s0^2`, `A_h*p*s0`, and `A*s0` from the
second endpoint derivative; (6.6) bounds each by O(epsilon). Terms with
R from total s derivatives obey the same estimates. At the outer event
all coefficient derivatives are O(epsilon), while event variations are
O(u^2). The undifferentiated integral has the known logarithmic bound.
Consequently

    Jtail=O_C2k(epsilon*log(1/epsilon)).                  (6.7)

The two differentiated orders in fact have the stronger O(epsilon) bound.

## 7. Exact physical prefactors and completion of the expansion

Use the exact identity already proved at the physical events:

    D_epsilon'=(Kout/Kin)
       *[hout/(qout+kout*hout)]/[hin/(qin+kin*hin)]
       *exp(Psiout-Psiin),

    Ktau=Ttau'(h)/(1+E_tau,h*Rcal).

At epsilon=0 on the exact outer event, `Ktau=-2` and `h/(q+kh)=1`
identically throughout the compact event family. Event transversality and
analyticity therefore show that `log(-Ktau/2)` and `log[h/(q+kh)]`
are epsilon times smooth uniformly bounded
functions of the event coordinates. This assertion includes their first
two event-coordinate derivatives. Combining it with (L) gives
`O_C2k(epsilon)` for the physical logarithmic prefactor error. An
`O(epsilon)` value estimate has not been differentiated in this step;
the exact vanishing at epsilon=0 is factored in the smooth event chart.

Equations (4.4) and (4.6), with the coefficient estimates in section 2,
give

    log b(X1)=2*log d_sigma+(sigma*lambda/d_sigma)*omega
                    -3*sigma*beta0*d_sigma*u+O_C2k(E).

Adding (4.7), (5.4) and (6.7), the complete physical exponent is

    Psi_sigma=2*kappa+2*log d_sigma-log L
        +(sigma*lambda/d_sigma)*omega
        -3*sigma*beta0*d_sigma*u+O_C2k(E).

The common central factor in the exact physical variation identity cancels
before approximation or differentiation. Subtracting the incoming sign +
from the outgoing sign -, and including the physical prefactors, proves
(T). Every derivative throughout is at fixed physical epsilon and field.

## 8. Consequence and limits of this proof

For a compact with negative lambda bounded away from zero, Astar and
Bstar are uniformly positive. Since both omega and u satisfy
`partial_kappa^2 omega=omega`, `partial_kappa^2 u=u`, (T) supplies

    partial_kappa^2 log D_epsilon'
      =Astar*omega+Bstar*u+O(E)>0

on a sufficiently small common corner. A varying constant limiting gap
Gamma has zero kappa derivatives. If the actual regular logarithmic
derivative has uniformly bounded first two derivatives in its own signed
input, (L) supplies its second kappa derivative as O_rho(u^2), via the
ordinary chain rule. This is the concrete curvature comparison available
for a subsequent varying-detuning zero-count argument.

This note alone does not count displacement zeros over all heights. Such
a count still needs the single connected admission interval, overlap with
the compact-kappa and positive-base regions, and the correct derivative
or zero-count operator. Nor does it give a formal analytic verification,
unrestricted graphic coverage or a full Hilbert XVI result.

The [fixed-field regular composition](HILBERT16-REGULAR-KAPPA-JETS.md)
uses (L) directly. The [varying-detuning synthesis](HILBERT16-VARYING-DETUNING.md)
combines its curvature estimate with the actual overlap and connected
admission argument. See [formal scope](HILBERT16-FORMAL.md) for the
precise statements checked in Lean.

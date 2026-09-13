# Fixed-field regular jets in the logarithmic center height

This note reuses the already proved actual G_R section jets and the exact
signed coordinate changes. It does not differentiate the sharp anchor's
O term. It proves the regular composition estimate conditional only on the
specified actual singular input-label jets; the first label derivative
bound already follows from the existing exact endpoint identity and joint
value theorem. This is a written analytic estimate; its hypotheses are not formally
verified in Lean.

## 1. What is fixed

Fix one physical field, including nu, C, all mu coefficients and alpha=A-1.
The canonical epsilon=nu*k_scale is therefore fixed as well. Only the
center height varies:

    H_center(kappa)=epsilon^3 exp(-kappa/epsilon),
    omega=exp(-kappa), u=epsilon exp(kappa).

Let t_i(kappa) be the incoming signed label of its actual normal-forward
singular trajectory. Its outgoing signed label is t_o^sing(kappa).
Let H_reg be the regular signed passage of THIS SAME FIXED FIELD.
The regular output is

    t_o^reg(kappa)=H_reg(t_i(kappa)).

It is generally different from t_o^sing(kappa). They are equal at cycles.
Define

    E_reg(kappa)=log H_reg'(t_i(kappa))-log c(C),
    c(C)=exp(-4*pi/sqrt(C-1)).

This is the function relevant to a fixed-field displacement-zero argument.
Its constant value offset is immaterial to its second derivative.

## 2. Existing uniform section jets and a positive lower multiplier

Choose a compact C interval [Cmin,Cmax] inside (1,infinity), the common
regular direction box, and a sufficiently small fixed rho. For explicit
coordinate bounds it is harmless to impose

    0<rho<=1/128, zeta=8rho,
    0<nu<=rho^2/(1+Cmax), R=rho/nu>=4,

as well as the existing endpoint sector and relative-variation bound.
On any fixed-alpha admitted input interval, [the regular-jet theorem](HILBERT16-REGULAR-JETS.md), equations (12)–(15),
already proves

    0<G_R'<=M1,
    |G_R''|<=M2,
    |G_R'''|<=M3,

with M1=theta, M2=theta*T2, M3=theta*(3*T2^2+T3), independent of R.
These are actual solution derivatives, not Taylor coefficients at one R.
The same proof supplies the relative first variation E between 1/2 and 2.
From the exact endpoint formula

    G_R'=[q(R)/q(-R)] [D_reg(-R)/D_reg(R)] E,

one also gets

    G_R'>=gmin:=cmin/2>0,
    cmin=exp(-4*pi/sqrt(Cmin-1)).                         (1)

Indeed q(R)/q(-R)>=1 and D_reg(-R)/D_reg(R)>=c(C)>=cmin.
Thus passing to logarithmic derivatives loses no unbounded factor.

## 3. Exact C3 signed-coordinate transport

Write

    Delta_tau=rho^2+2*tau*nu*rho+C*nu^2,
    A_tau=rho^2/Delta_tau, B_tau=C*nu^2/Delta_tau,
    Z_tau(t)=2/[1+tau*rho+sqrt(1+2*tau*rho+rho^2*t)]-1,
    K_tau(z)=1+(4/rho^2)*[(1+z)^(-2)-(1+tau*rho)*(1+z)^(-1)].

Then the actual input map is phi(t)=A_- Z_-(t)+B_- and the actual output
inverse map is psi(z)=K_+((z-B_+)/A_+). Exactly,

    H_reg=psi composed with G_R composed with phi.       (2)

Here 1/2<=A_tau<=2. For |t|<=1, put
q=sqrt(1-2rho+rho^2*t), J=1-rho+q. Direct differentiation gives

    Z_-'=-rho^2/(q*J^2),
    Z_-''=rho^4*[1/(2q^3 J^2)+1/(q^2 J^3)],
    Z_-'''=-rho^6*[3/(4q^5 J^2)+3/(2q^4 J^3)+3/(2q^3 J^4)].

The domain gives q>=1/2,J>=1 and q*J^2<=6. Consequently

    rho^2/12<=|phi'|<=4rho^2,
    |phi''|<=16rho^4, |phi'''|<=128rho^6.               (3)

For |G_R|<=zeta the inverse affine output coordinate zbar obeys
|zbar|<=16rho+rho^2<1/4. At |zbar|<=1/4 the explicit derivatives of K_+
give

    rho^(-2)<=|K_+'|<=20rho^(-2),
    |K_+''|<=80rho^(-2), |K_+'''|<=512rho^(-2).

For the last bound use
K_+'''=(4/rho^2)*[-24/(1+zbar)^5+6*(1+rho)/(1+zbar)^4]; the bracket is
negative, so its absolute value is at most 24/(1+zbar)^5.
Including the affine scale factors gives

    1/(2rho^2)<=|psi'|<=40/rho^2,
    |psi''|<=320/rho^2, |psi'''|<=4096/rho^2.            (4)

Both phi' and psi' are negative. Equations (1), (3), (4) imply the positive
uniform lower bound

    H_reg'>=m0:=cmin/48.                               (5)

Notice that no bound |t_o^reg|<=1 is needed for (3)–(5): the full admitted
NORMALIZED output box |G_R|<=zeta already keeps zbar in its valid branch.
This prevents an unnecessary restriction of the connected capture domain.

The actual chain rules are

    H_reg''=psi''*(G_R'*phi')^2
                +psi'*(G_R''*(phi')^2+G_R'*phi''),

    H_reg'''=psi'''*(G_R'*phi')^3
       +3psi''*(G_R'*phi')*(G_R''*(phi')^2+G_R'*phi'')
       +psi'*(G_R'''*(phi')^3+3G_R''*phi'*phi''+G_R'*phi''').

For example the following explicit conservative constants suffice:

    B2=rho^2*[5120*M1^2+640*M2+640*M1],
    B3=rho^4*[262144*M1^3+61440*M1*(M2+M1)
                                  +2560*M3+7680*M2+5120*M1].

Thus |H_reg''|<=B2 and |H_reg'''|<=B3. Put

    L1=B2/m0,
    L2=B3/m0+(B2/m0)^2.

The fixed-field function ell(t)=log H_reg'(t) satisfies

    |ell_t|<=L1, |ell_tt|<=L2.                          (6)

These estimates involve no matching-alpha derivative and no unproved
high-order ODE inference. They follow from the existing third section jet.

## 4. The exact kappa composition estimate

On any kappa interval whose actual t_i lies in the admitted regular domain,

    E_reg,kappa=ell_t(t_i)*t_i,kappa,

    E_reg,kappakappa=ell_tt(t_i)*(t_i,kappa)^2
                           +ell_t(t_i)*t_i,kappakappa.  (7)

Consequently

    |E_reg,kappa|<=L1*|t_i,kappa|,
    |E_reg,kappakappa|
       <=L2*|t_i,kappa|^2+L1*|t_i,kappakappa|.          (8)

This is the complete regular-side theorem. In particular, if the actual
singular label satisfies, on the same coefficient compact and domain,

    |t_i,kappa|+|t_i,kappakappa|
         <=K_lab*[u^2+epsilon*log(2/epsilon)]           (J)

and the right-hand scale is bounded by 1, then

    |E_reg,kappa|+|E_reg,kappakappa|
         <=K_rho*[u^2+epsilon*log(2/epsilon)].          (9)

The stronger label estimate O(u^2) is now supplied independently by
[the actual singular variational proof](HILBERT16-SINGULAR-KAPPA-JETS.md).
It gives the stronger O_rho(u^2) in (9).
No outgoing singular label derivative is needed for the FIXED-field result.

## 5. The first actual input-label derivative is already available

This step uses an exact identity, not differentiation of joint asymptotics.
From the existing physical moving-event formula, with
mathcal D=q+k*h and D0=q(0)+k(0,H_center)*H_center,

    dt_i/dH_center=mathcal K_i*(D0/H_center)
                         *(h_i/mathcal D_i)*exp(Psi_i),
    dH_center/dkappa=-H_center/epsilon.

Therefore exactly

    t_i,kappa=-mathcal K_i*(D0/epsilon)
                           *(h_i/mathcal D_i)*exp(Psi_i). (10)

The joint theorem supplies mathcal K_i=-2+O(epsilon), bounded positive
h_i/mathcal D_i, D0=O(epsilon^3), and Psi_i=2kappa+O(1). It follows
immediately and uniformly that

    0<t_i,kappa<=K*epsilon^2*exp(2kappa)=K*u^2.         (11)

The lower positivity follows from the same exact factors. Thus the only
additional singular ingredient needed for (9) is a bound on the actual
second input-label derivative. A value-only formula for t_i cannot supply
that ingredient. It must be obtained from its second variational equation,
including the moving physical event, or another proved derivative estimate.
The existing compact-kappa theorem gives such derivatives only on fixed
bands; its constants cannot be silently used on this growing joint band.

## 6. Zeroth order and cycle anchoring

The existing sharp anchor gives

    |E_reg|<=K_rho*[epsilon*log(2/epsilon)
                              +|t_i|+|t_o^reg|],      (12)

when these signed labels lie in its box (nu and epsilon are uniformly
comparable). The actual singular theorem gives
|t_i|+|t_o^sing|<=K_rho*(u^2+epsilon), but away from a cycle it does not
identify t_o^reg with t_o^sing. Therefore (12) alone does not yield the
requested small zeroth-order error on an arbitrary unanchored fixed-field
interval. This is a limitation of that inference, not a demonstrated
counterexample in the actual canonical family.

At cycles the labels agree, so the j=0 bound already follows. There is also
a useful interval extension. If kappa_a is an actual cycle in the interval,
then for kappa>=kappa_a,

    |t_o^reg(kappa)|
      <=|t_o^sing(kappa_a)|
          +sup|H_reg'|*integral[kappa_a,kappa] |t_i,s| ds
      <=K_rho*(u(kappa)^2+epsilon),

using (11) and integral exp(2s) ds. Shrink u0 and epsilon so these outputs
remain in [-1,1]. Equation (12) now supplies the j=0 estimate throughout
the portion AFTER this anchor. Combined with (J), all j<=2 obey the desired
bound there. For an arbitrary zero-count contradiction one can start at
its leftmost presumed cycle; no globally selected matching parameter is
needed. For convexity of a log-slope difference, only (9) is required and
no zeroth-order anchoring is necessary.

## 7. Why matched-alpha jets are a different object

Let alpha_hat(kappa) be the BVP matching parameter selected so that the
regular passage from t_i(kappa) ends at t_o^sing(kappa). Then

    M_hat(kappa)=log H_(alpha_hat(kappa))'(t_i(kappa)).

Although M_hat equals log H_reg' at a cycle where alpha_hat=alpha_fixed,
its derivative generally differs:

    M_hat,kappa=partial_t log H_alpha' *t_i,kappa
                  +partial_alpha log H_alpha' *alpha_hat,kappa.

There is a corresponding second derivative containing alpha_hat,kappakappa
and mixed/second alpha terms. Equality of the values at a cycle does not
remove those terms. The matched-endpoint multiplier functional is excellent
for the value comparison proved in [the regular anchor](HILBERT16-REGULAR-ANCHOR.md), but differentiating
it along the singular endpoints is not a substitute for (7).

The bound (9) avoids this issue entirely: alpha is held fixed from the
beginning, and only the actual incoming section label is composed with its
already established regular jets. The captured regular input domain is an
interval; since (11) is strictly positive, its preimage in the connected
singular kappa band is also an interval. All derivatives extend locally
through admitted endpoints by the existing strict BVP/section margins.

## Formal scope

The existing Lean parabola algebra remains unchanged and its proven finite
identities are preserved. Neither this kappa composition estimate nor the
input-label hypothesis (J) is claimed as Lean verified here. The local
algebraic chain rules can be checked separately; a proof of uniform
singular label jets is a distinct analytic input, not a missing generic
regular-jet implementation.

The previously separate label hypothesis (J) is now discharged by
[the singular kappa-jet theorem](HILBERT16-SINGULAR-KAPPA-JETS.md),
under its strict discriminant and fixed canonical compact assumptions.
The conditional composition proof above keeps that dependency explicit.

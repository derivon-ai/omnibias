# A second-order logarithmic obstruction to quadratic compensation

This is a written analytic calculation with exact symbolic checks. It is not a
novelty claim, a formal proof, a uniform joint-parameter passage theorem, a return
map theorem, or a solution of Hilbert 16. The result extends the [quadratic unfolding analysis](HILBERT16-UNFOLDING.md).

## Precise result

Fix C>1, p,r,m1 in R. Use the quadratic family

    xdot = A*x-y+x^2+(mu2+mu3)*x*y+mu1*y^2,
    ydot = C*x+x^2+x*y+mu3*y^2,

with mu2=epsilon*p, mu3=epsilon*r, mu1=epsilon^2*m1, and

    A=1+epsilon*alpha1+epsilon^2*alpha2,
    alpha1=-(C+6)*p-3*r.

At the fixed incoming section x=-R impose w=0, where

    y0=(x^2-C)/2, w=y-y0, q=(x^2+2*x+C)/2.

Write G_R(epsilon)=w(R;epsilon)/q(R). For every fixed finite R>0 this is
analytic in epsilon on a neighborhood of zero; the neighborhood can depend on
R. Set c=exp(-4*pi/sqrt(C-1)). Its Taylor coefficients, with NO factorial in
the notation g2_R, satisfy

    G_R(epsilon)=epsilon*g1_R+epsilon^2*g2_R+O_R(epsilon^3),
    g1_R=(2*p+r)*(1+c)*R+O(1),
    g2_R=(1-c)*(p*(2*p+r)-m1)*R^2*log(R)+O(R^2).             (T)

Thus the second derivative is 2*g2_R. The O constants can be chosen uniformly
for C in a compact subset of (1,infinity) and p,r,m1,alpha2 in a bounded set.
Fixed alpha2 contributes alpha2*partial_alpha G_R, and

    R^-2*partial_alpha G_R -> -(1-c)/(8*(C+3)).

It cannot cancel a nonzero R^2*log(R) term.

There is also a result for the ACTUAL compensating surface at each finite R.
The implicit-function theorem gives a unique local analytic alpha_R(epsilon)
with G_R=0 and alpha_R(0)=0, using A=1+alpha_R(epsilon). Write

    alpha_R(epsilon)=a1_R*epsilon+a2_R*epsilon^2+O_R(epsilon^3).

Then

    a1_R=-(C+6)*p-3*r+O(1/R),
    a2_R=8*(C+3)*(p*(2*p+r)-m1)*log(R)+O(1).               (IFT)

For generic directions this compensating surface has a logarithmically growing
second Taylor coefficient. This sharpens the warning that its finite-R
implicit-function neighborhood need not be uniform.

## Proof of the coefficient asymptotics

Put a=sqrt(C-1), theta=atan((x+1)/a),

    D=exp(4*(theta-pi/2)/a), H=q^2/D, L=q*d/dx-2*x.

Then D'/D=2/q, H'=2*x*H/q, D(+infinity)=1, D(-infinity)=c.
For mu1=0 and alpha2=0 the exact x-time equation is

    w'=[2*x*w+epsilon*(b+B*w+r*w^2)]
       /[q-w+epsilon*(h+(p+r)*x*w)],
    b=-alpha1*x^2-(p+r)*x^2*y0+r*y0^2,
    B=-p*x^2-C*r,
    h=alpha1*x+(p+r)*x*y0.

Let w=epsilon*u_R+epsilon^2*v_R+O_R(epsilon^3). Direct multiplication,
including the denominator variation, gives

    L u_R=b,
    L v_R=(u_R-h)*u_R'+B*u_R + m1*b1 + alpha2*b_alpha,
    b1=-x*(x^2-C)^2/4, b_alpha=-x^2,
    u_R(-R)=v_R(-R)=0.                                    (1)

The additional mu1 and alpha2 contributions stated here follow from expanding
the full family to order two. Their denominator terms multiply the zero base
w derivative, so do not contribute at this order.

An exact polynomial particular solution of the first equation is

    U=p*(x^3+3*x^2+3*C/2)
      +r*(x^3/2+3*x^2/2+C*x/2+C).

Therefore

    u_R=U+K_R*H, K_R=-D(-R)*U(-R)/q(-R)^2=O(1/R).         (2)

The exact endpoint formula u_R(R)/q(R), with U's leading coefficient
p+r/2, gives the g1_R assertion in (T).

For a polynomial trial function U, put F2(U)=(U-h)*U'+B*U. It has degree
at most five, with

    [x^5]F2(U)=p*(2*p+r)/4.

Consequently, with beta=p*(2*p+r)-m1,

    D*(F2(U)+m1*b1)/q^3 = 2*beta/x+O(x^-2), x -> +infinity,
    D*(F2(U)+m1*b1)/q^3 = 2*c*beta/x+O(x^-2), x -> -infinity.

The negative tail has the opposite orientation in the symmetric integral:

    integral[-R,R] D*(F2(U)+m1*b1)/q^3
       =2*(1-c)*beta*log(R)+O(1).                         (3)

It is essential to check the actual initial-condition correction (2). Exactly,

    F2(U+K*H)-F2(U)
      =K*[H*U'+(U-h)*H'+B*H]+K^2*H*H'.                 (4)

Uniformly on the stated compact C and bounded direction sets, q is comparable
to (1+|x|)^2, D is bounded above and bounded strictly away from zero,
H=O((1+|x|)^4), H'=O((1+|x|)^3), U,h=O((1+|x|)^3),
U'=O((1+|x|)^2), B=O((1+|x|)^2). Multiplying (4) by D/q^3 and integrating
from -R to R gives

    O(|K_R|*R+K_R^2*R^2)=O(1).

Thus the boundary correction changes no logarithmic coefficient in (3).
Variation of constants in (1) gives

    g2_R=q(R)/D(R) * integral[-R,R]
          D*[F2(u_R)+m1*b1+alpha2*b_alpha]/q^3.

Here q(R)/D(R)=R^2/2+O(R), and the alpha integral converges. This proves (T).
All tail and polynomial bounds used in the proof are uniform on the stated
compact parameter sets. Nothing asserts uniform convergence of the full
Taylor series as R grows.

## Proof of the actual finite-R compensation formula

At the base trajectory the denominator is q>0 on the compact interval [-R,R].
Analytic parameter dependence of the ODE therefore supplies a local analytic
G_R. Moreover

    partial_alpha G_R=q(R)/D(R)*integral[-R,R] -D*x^2/q^3 < 0

for every R>0. The ordinary implicit-function theorem applies.

Set alpha0=-(C+6)*p-3*r. Since the compensated directional first derivative
is O(R) and partial_alpha G_R is a negative constant times R^2+O(R), one
obtains a1_R-alpha0=O(1/R).

This O(1/R) alteration of the first direction does not change (3). To see
this without interchanging any limits, use the exact particular solution

    P_alpha=-(x^4+8*x^3+2*(C+12)*x^2+C*(C+12))/(16*(C+3)),
    L P_alpha=-x^2.

The corrected first variation differs from U by a function E_R with

    E_R=O((1+|x|)^4/R), E_R'=O((1+|x|)^3/R), |x|<=R.

This includes both the changed particular solution and the changed homogeneous
initial-condition correction. Also h changes by O(|x|/R). The difference of
the weighted second-forcing integrals is again O(1): its linear E_R terms
integrate as O(R/R), and its quadratic terms as O(R^2/R^2). The changed h
terms are smaller. Hence the quadratic coefficient computed with a1_R still
has the leading term in (T).

At order epsilon^2 the compensation equation is

    0=tilde_g2_R+a2_R*partial_alpha G_R.

Using partial_alpha G_R=-(1-c)*R^2/(8*(C+3))+O(R) proves (IFT), including
its sign. The assertion concerns Taylor coefficients of a family of local
analytic surfaces, not their common domain of existence.

## An exact invariant-parabola subfamily

For mu1=0, mu2=t, mu3=-2*t, A=1-C*t, set D0=1-t+C*t^2 and

    F_t=(t-1)*x^2+2*C*t*x+2*D0*y+C.

Exact polynomial differentiation gives

    vector_field(F_t)=(2*x-2*t*y)*F_t.

Thus F_t=0 is invariant. For C>1, D0>0 for all real t, so this is a smooth
global graph in y, and is a nondegenerate parabola for t!=1. Near t=0 it
deforms y=(x^2-C)/2. Its parameter tangent matches first-order compensation
exactly: -(C+6)*t-3*(-2*t)=-C*t. This direction has beta=0 when m1=0.
The identity alone does not prove that the entire graph is one oriented
section-to-section trajectory at every cutoff.

This special family cannot be extended to a generic compensated direction
merely by choosing a different conic. For an analytic normalized conic

    F=a*x^2+b*x*y+d*y^2+e*x+y+f,
    vector_field(F)=(ell*x+m*y+n)*F,

with base (a,b,d,e,f,ell,m,n)=(-1/2,0,0,0,C/2,2,0,0), the constant, y,
y^2 and x^3 equations imply n=0, e=-f*m, m=mu3-b,
ell=2+b/a. Its x*y^2 coefficient then requires

    b*(b+mu2+mu3)-d*b/a=0.

Along a compensated tangent (mu2',mu3')=(p,r), the first-order conic
equations force b'=-(2*p+r), d'=0. Therefore the second-order coefficient
of this necessary identity is p*(2*p+r)=0. Generic directions with a
nonzero logarithmic obstruction admit no such analytic invariant-conic
repair. The tangent p=0 remains a separate exceptional direction; its lack
of this obstruction does not by itself establish an invariant conic or a
uniform passage.

## Additional exact consistency identities

The symbolic script also checks F2(U)=-p*(2*p+r)*b1+L V+k*b_alpha, where V
is a displayed cubic polynomial in the script and

    k=-(2*p+r)*[(25*C+108)*p+(11*C+54)*r].

Thus the quadratic forcing has the exceptional source direction with precisely
the effective coefficient mu1-epsilon^2*p*(2*p+r), modulo a polynomial
coordinate correction and an ordinary alpha source. Boundary terms must still
be treated as in (4).

In the already derived nu=0 family chart, d=m1-p*r and

    u'=d+p*u-z, z'=-u*z.

The effective logarithmic expression m1-p*(2*p+r) is d-2*p^2. At its zero
set with p!=0, the equilibrium (u,z)=(0,2*p^2) has linear eigenvalues
{2*p,-p}, a 1:2 saddle resonance. This is an algebraic consistency observation,
not a proof that the computed regular passage has been matched to a Dulac map.

## Reproduction and remaining limits

Run `python benchmarks/hilbert16_compensation.py --numeric` from the workspace. The script writes `artifacts/hilbert16/compensation.json`; all exact checks remain active under `python -O`. All transformation, first and
second variation, polynomial decomposition, and invariant-conic identities
are checked by exact SymPy algebra. The optional 40-digit quadrature for
C=2,p=1,r=m1=0 produces g2_R/(R^2*log R) = 3.3067966, 2.6302911,
2.4052599, 2.3067644 at R=20,80,320,1280, approaching the proved limit
1.9999930. This numerical check is not an interval certificate.

For a prospective R=rho/epsilon regime, the Taylor calculation suggests

    alpha+epsilon*[(C+6)*p+3*r]
       +8*(C+3)*epsilon^2*[m1-p*(2*p+r)]*log(R)

as the logarithmic compensation combination. Substituting R=rho/epsilon
into (T) alone is not justified without a uniform nonlinear remainder and
a uniform passage domain. Those two requirements are now supplied on an
explicit small sector by the separate
[uniform nonlinear argument](HILBERT16-UNIFORM-PASSAGE.md), extended to
[common nonzero sections](HILBERT16-ENDPOINT-PASSAGE.md). Matching to the
singular charts and a uniform return-displacement zero bound remain
unproved. Full Hilbert 16 remains unresolved.


The [polynomial obstruction companion](HILBERT16-POLYNOMIAL-OBSTRUCTION.md)
extends the finite polynomial decomposition to every forcing degree. An
independent 65-digit calculation imposes the actual finite-section first-order
compensation before integrating the second-order source. For C=2, p=1, r=m1=0,
R=16 it gives a1=-4.309346732754569 and a2=13.410363488441785.
Direct shooting of the original vector field at epsilon=0.000025 gives
the second Taylor coefficient 13.410363663979297. This agreement checks
the calculation; it is not a rigorous numerical enclosure. Neither diagnostic
justifies a uniform joint limit.

# A sharp actual regular multiplier near the signed zero labels

This is a written analytic proof using the existing uniform endpoint BVP,
its transverse parameter derivative, and the exact invariant-parabola
surface. It is not a formal Lean proof, a numerical cutoff certificate,
a novelty claim, or a full-graphic result. All constants below are uniform
on the stated coefficient compacts after a sufficiently small fixed rho is
chosen.

## 1. Statement

Use the actual physical family

    xdot=A*x-y+x^2+(mu2+mu3)*x*y+mu1*y^2,
    ydot=C*x+x^2+x*y+mu3*y^2,

with the existing canonical slow embedding at r=-1. Its analytic expansions
on compact slow-parameter sets are

    mu3=-nu,
    mu2=3*nu^2+O(nu^3),
    mu1=-2*nu^3+O(nu^4),
    A=1+nu*Abar, |Abar|<=A0.

C lies in a compact subset of (1,infinity). Fix a sufficiently small
rho>0, use the exact sections x=+-R with R=rho/nu, and the actual signed
section label

    t=(x/(nu*y)-1)^2-2/(nu^2*y).

Fix one physical field. If its actual regular passage is captured by the
existing fine endpoint tube, and its two signed labels ti,to have magnitude
at most one, then

    |log H_reg'(ti)+4*pi/sqrt(C-1)|
        <=K_rho*[nu*log(2/nu)+|ti|+|to|].                 (T)

The derivative holds all physical parameters, including A, fixed. The
constant is uniform over the compact canonical coefficients and captured
bounded compensator. For sufficiently small nu, the logarithm is defined
because the regular passage and both signed coordinate maps are strictly
monotone with the established positive multiplier.

In particular, on a sequence of captured actual passages with ti,to->0,
the regular logarithmic multiplier tends to log c(C), c(C)=exp(-4*pi/sqrt(C-1)),
with rho fixed. The older O(rho) estimate is not the limiting obstruction.
The bound (T) also suffices at a cycle when the singular estimate supplies
|ti|+|to|=O(u^2+epsilon). It does not assert any derivative in the slow
parameters or in kappa.

## 2. A multiplier functional with both endpoints prescribed

Set q=(x^2+2*x+C)/2, y0=(x^2-C)/2, w=y-y0 and j=1+|x|. Write the
regular scalar equation as

    w'=f(x,w;alpha,mu1)=N/P, alpha=A-1,
    P=xdot, N=ydot-x*xdot.

Hold nu,C,mu2,mu3 fixed. The [common-section theorem](HILBERT16-ENDPOINT-PASSAGE.md) supplies the unique
matching alpha for every normalized endpoint pair z_-,z_+ in its small
closed square and for every intervening mu1 considered below. Its strict
margins and ordinary finite-R analytic ODE/implicit-function theory give
smooth dependence in a neighborhood of these data. No uniform analyticity
radius in R is being assumed.

Define M(mu1,z_-,z_+) to be the logarithm of the regular derivative G'
with alpha held fixed during the section derivative, evaluated along that
matched BVP. Exactly,

    M=log(q(-R)/q(R))+integral[-R,R] f_w(x,w(x)) dx.       (1)

Changing the arguments of M changes the matched alpha and the reference
trajectory; it does not change what G' means.

Take zeta=8rho and the usual sufficiently small endpoint/capture sector.
Along all these BVPs delta=nu+zeta/R=9nu. The existing derivative bounds,
or direct differentiation of the rational scalar field, give

    |f|<=K*j, |f_w|<=K/j, |f_ww|<=K/j^3,
    P>=kappa*j^2, |P_w|<=K,
    |f_alpha|<=K, |f_walpha|<=K/j^2,
    |f_mu1|<=K*j^3, |f_wmu1|<=K*j.                     (2)

The first second-derivative bound is the k=2 bound in the regular-jet proof.
For the parameter bounds use the exact identities

    f_alpha=-x*(x+f)/P,
    f_mu1=-y^2*(x+f)/P.                                (3)

They follow from P_alpha=x,N_alpha=-x^2 and
P_mu1=y^2,N_mu1=-x*y^2. Their w derivatives give the displayed powers
using y=O(j^2). Thus (2) does not assume small parameter derivatives of
an unknown orbit.

Let mathcal H be a positive homogeneous solution of v'=f_w v normalized
by mathcal H(0)=H0(0), where H0=q^2/D_reg. The existing relative-variation
bound gives

    k1*j^4<=mathcal H(x)<=k2*j^4.

The endpoint transversality proof, with this harmless positive normalization,
gives a common constant b>0 such that

    I_alpha=integral[-R,R] f_alpha/mathcal H dx <=-b.     (4)

These constants are independent of R and nu in the fixed small sector.

### Endpoint derivatives

For a derivative with respect to either normalized endpoint, let v be the
trajectory derivative and a be the matching-alpha derivative. Then

    v'=f_w*v+f_alpha*a,
    v(-R),v(R) equal 0 or the corresponding q(+-R).

The two endpoint values of v/mathcal H are O(R^-2). Integrating the equation
and using (4) gives |a|<=K/R^2. The integral of
|f_alpha|/mathcal H is bounded, so the same exact variation formula gives

    |v(x)|<=K*j^4/R^2.

Differentiate (1), apply (2), and use integral j dx=O(R^2):

    |partial_(z_-) M|+|partial_(z_+) M|<=K.              (5)

This is the parameter-combined derivative of the matched multiplier,
not a bound for an uncompensated alpha perturbation.

### The mu1 derivative at fixed endpoints

Now the variation equation has source f_mu1 and zero endpoints:

    v'=f_w*v+f_alpha*a+f_mu1, v(-R)=v(R)=0.

Since |f_mu1|/mathcal H<=K/j, (4) gives |a|<=K*log S with S=R+1.
For x<=0 integrate from -R to x; for x>=0 integrate backward from R.
The two zero endpoints then give the spatially resolved estimate

    |v(x)|<=K*[j^4*log(S/j)+j*log S].                   (6)

This is the same two-tail cancellation as the projected Green estimate;
no naked log(S)*j^4 mode is kept. Differentiating (1) gives

    partial_mu1 M=integral[f_ww*v+f_walpha*a+f_wmu1] dx.

Here integral j*log(S/j) dx<=S^2/2 and
integral j^-2 dx<=2. Therefore

    |partial_mu1 M|<=K*R^2.                            (7)

A value-only Taylor-model remainder has not been differentiated anywhere.
All derivatives arise from the actual finite-R variational equations.

### Transfer to the actual signed labels

Write Z_tau,nu(t) for the exact affine signed-coordinate maps from the
[joint proof](HILBERT16-JOINT-MATCHING.md). For |t|<=1, fixed small rho and small nu,

    |Z_tau,nu'|<=K*rho^2,
    |Z_tau,nu''/Z_tau,nu'|<=K*rho^2.

For the endpoint-prescribed multiplier define

    M_signed(mu1,ti,to)
      =M(mu1,Z_-,nu(ti),Z_+,nu(to))
         +log|Z_-,nu'(ti)|-log|Z_+,nu'(to)|.

Equations (5)–(7) imply

    |partial_ti M_signed|+|partial_to M_signed|<=K*rho^2,
    |partial_mu1 M_signed|<=K*R^2.                      (8)

The sign of each Z' is negative, so the actual derivative ratio is positive.
The affine coordinate maps depend on nu,C, not on alpha or mu1.

## 3. The same-(mu2,mu3) invariant-parabola reference

Use the exact Branch-II parabola in [invariant conics](HILBERT16-INVARIANT-CONICS.md), with
p=mu2,r=mu3 fixed. Denote its affine parameter by k_p (distinct from the
canonical epsilon/nu scale). Then

    k_p+5*k_p^2=2p+r,
    m=8*k_p^2+2*k_p-2p,
    F=a*(x+k_p*y)^2-f0*m*x+y+f0,
    vector_field(F)=[2*(1+k_p)*x+m*y]*F.

The coefficients a,f0,A_p,mu1_p are the exact rational formulas from that
note. The canonical expansions imply

    k_p=-nu+nu^2+O(nu^3), m=-2nu+4nu^2+O(nu^3),
    a=-1/2+O(nu), f0=C/2+O(nu),
    mu1_p=-2nu^3+O(nu^4),
    A_p=1+3nu+(6-2C)*nu^2+O(nu^3).                     (9)

Consequently mu1-mu1_p=O(nu^4), uniformly over the canonical compact.
Only the first displayed orders are needed; no expansion is substituted
for the actual field or trajectory in the following variation formula.

Put s=x+k_p*y and J=1+k_p*f0*m. On the invariant parabola, exactly,

    x=(s+k_p*a*s^2+k_p*f0)/J,
    y=(f0*m*s-a*s^2-f0)/J.                              (10)

For small fixed rho, x_s=(1+2*k_p*a*s)/J is strictly positive throughout
the arc between x=-R and x=R; its endpoints s_-,s_+ satisfy
s_-<0<s_+ and |s_+-| comparable to R. Its on-parabola speed U(s)=sdot
is a cubic and its cofactor K(s) is a quadratic. Write their coefficients
as U_j and K_j. Exact polynomial substitution gives

    U_0=C/2+O(nu), U_1=1+O(nu), U_2=1/2+O(nu),
    K_0=O(nu), K_1=2+O(nu), K_2=2*U_3,

and, with E=C*m^2+12*k_p^2+16*k_p+2*m+4,

    U_3=-(1+3*k_p)*(p-3*k_p^2)
                  *(6*k_p^2+8*k_p+m+2)/E.              (11)

Since p=3nu^2+O(nu^3) and k_p^2=nu^2+O(nu^3),

    U_3=O(nu^3).                                       (12)

This is the decisive cancellation: a generic invariant parabola with
coefficients merely O(nu) would not give this order. The proof here
uses the canonical slow-sector embedding.

For |s|<=K_rho/nu the cubic term has size O(nu^3*|s|^3), while the other
speed coefficients differ from those of q(s) by O(nu). Thus

    U(s)=q(s)+O(nu*(1+|s|)^2+nu^3*|s|^3),
    U(s)>=kappa1*(1+|s|)^2>0.                           (13)

The reference is therefore an actual oriented complete regular arc,
not just an algebraic invariant set. Its displacement w from the base
parabola obeys |w|<=K*nu*j^3 and |alpha_p|<=K*nu, directly from (10).
By reducing rho, it lies in the coarse capture neighborhood and hence
is the unique fine BVP with its actual endpoints. All interpolations of
mu1 and signed endpoints used in (8) stay inside the common BVP domain.

## 4. Exact reference multiplier and endpoint cancellation

Darboux variation along F=0 gives an exact formula for the signed reference
multiplier:

    H_p'=[(t_y/F_y)_+/(t_y/F_y)_-]
                  *exp integral[s_-,s_+] K(s)/U(s) ds. (14)

At a fixed x section, delta F=F_y delta y and delta t=t_y delta y.
The moving arrival time creates no extra term in delta F because
F=0 and its derivative along the reference trajectory are zero. This
justifies (14) without replacing the physical endpoint convention.

The identity K_2=2U_3 gives

    K/U=2U'/U+[-2+O(nu)+O(nu)*s+O(nu^3)*s^2]/U.       (15)

Using (13), C>1 compact, and |s_+-| comparable to rho/nu, integration yields

    integral[s_-,s_+] (K-2U')/U ds
      =-4*pi/sqrt(C-1)+O_rho(nu*log(2/nu)).             (16)

For completeness: the -2/q integral differs from its full-line value by
O_rho(nu). The constant O(nu) numerator integrates to O(nu), the linear
O(nu)*s numerator to O(nu*log(2/nu)), and the quadratic O(nu^3)*s^2
numerator to O_rho(nu^2). The change from 1/U to 1/q in the constant
term contributes O(nu+nu^3*log(2/nu)). Every integration estimate is uniform;
there is no expansion assumed convergent on a growing s interval.

It remains to cancel the exact U endpoint factors against the signed
coordinate factors. At the endpoints,

    t_y=2*[(nu*x+1)*y-x^2]/(nu^2*y^3),
    F_y=1+2*a*k_p*s.

Use x=s-k_p*y and the exact conic equation y=-a*s^2+f0*m*x-f0. Then

    [(nu*x+1)*y-x^2]+y*F_y
      =2y-s^2+[nu+2*k_p*(1+a)]*s*y-k_p*(nu+k_p)*y^2.

From (9), nu+k_p=O(nu^2),
nu+2*k_p*(1+a)=O(nu^2), and k_p*(nu+k_p)=O(nu^3).
At either endpoint y is comparable to s^2, |s| comparable to rho/nu,
and F_y is bounded positively away from zero. Thus the last identity
has magnitude O_rho(nu)*y. Also U/y=1+O_rho(nu), by (9), (10), (13).
Consequently the exact endpoint factor satisfies

    (t_y/F_y)*U(s)^2=-2/nu^2*[1+O_rho(nu)].             (17)

There is no O(rho) remainder in (17). Combining (14), (16), and (17)
cancels the U(s_+)^2/U(s_-)^2 factor and proves

    log H_p'=log c(C)+O_rho(nu*log(2/nu)).               (18)

## 5. Reference endpoint labels and comparison

The signed label on the reference is exactly

    t_p=[(x-nu*y)^2-2y]/(nu^2*y^2).

Since x-nu*y=s-(k_p+nu)*y, its numerator is

    s^2-2y-2*(k_p+nu)*s*y+(k_p+nu)^2*y^2.

At the endpoints, (9)–(10) bound this expression after division by
nu^2*y^2 by K_rho*nu. Hence

    |t_p,-|+|t_p,+|<=K_rho*nu.                          (19)

Compare M_signed at the actual parameters/endpoints to the reference
with the same nu,C,mu2,mu3. The entire straight mu1 interpolation has
mu1/nu^2=O(nu), so it lies in the previously fixed regular direction box.
The signed endpoints can also be interpolated inside [-1,1]; their
normalized values remain strictly inside zeta=8rho. BVP uniqueness
identifies the endpoints with the actual and conic trajectories, respectively.
By (8), (9), and (19),

    |log H_reg'(ti)-log H_p'|
       <=K*rho^2*(|ti|+|to|+K_rho*nu)
                          +K*R^2*|mu1-mu1_p|
       <=K_rho*(|ti|+|to|+nu+nu^2).                    (20)

Equations (18) and (20) prove (T). The matching alpha changes along this
comparison and is controlled by the BVP derivatives. Treating it as an
uncompensated O(nu) perturbation would lose the result.

## Scope and checking

The proof uses existing bounded-endpoint capture and the actual canonical
slow embedding, together with the exact invariant-parabola surface. It is
not asserted for arbitrary O(nu) values of mu2 or arbitrary O(nu^3)
coefficients of mu1. In those larger families (12) or mu1-mu1_p=O(nu^4)
can fail. It does not require the actual field itself to have an invariant
conic or the actual trajectory to have zero signed endpoints.

The exact coefficient identity (11), K_2=2U_3, the affine conic parameter,
and the endpoint numerator cancellation were checked by symbolic
substitution. The analytic estimates, compact uniformity, and BVP
variation argument are written mathematics, not a Lean-certified theorem.

## Finite reproduction

```bash
python benchmarks/hilbert16_regular_anchor.py --output artifacts/hilbert16/regular_anchor.json
```

This replay checks the exact conic, speed/cofactor and endpoint identities.
It does not formally verify the Green estimates, parameter-uniform
multiplier bound or the resulting physical cycle count.


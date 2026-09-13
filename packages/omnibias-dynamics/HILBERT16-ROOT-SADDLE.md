# Actual first-root saddle tail for negative lambda1

This is an independent written derivation, not a numerical or formal analytic certificate. It concerns the existing selected small-label physical itinerary. Sections 9-11 add an at-most-two bound across every positive center height admitted by this itinerary. They do not assert that every height is admitted, cover other sections or itineraries, or settle full graphic cyclicity. No external normal-form theorem is used for the new saddle tail: its estimates use an invariant rectangle and the exact planar event-derivative identity.

## 1. Parameters, exact field, and conclusion

Fix the canonical embedding and r=-1, C in a compact subset of (1,infinity), A=1+nu*Abar with bounded Abar, epsilon=nu*k with k uniformly positive analytic. Restrict the compact slow parameters to

    c <= L=-lambda0 <= Lmax,
    lambda1 < 0,
    delta=lambda1^2-4L >= delta0 > 0.

The exact normal field is

    Vdot=f(V)+h*g(V,h), hdot=-V*h,
    f(V)=-L*epsilon^3+lambda1*epsilon^2*V
          +epsilon*V^2*zeta(V,epsilon),
    zeta(0,epsilon)=-1, zeta(V,0)=-1+V/3.

Write q=-f and kappa_field=-g, abbreviated k below only inside formulas for the vector field. On each fixed physical-coordinate rectangle, k=1+O(epsilon), k_V=O(epsilon), k_h,k_Vh=O(epsilon^2), and k+h*k_h>0 for small epsilon. These are the exact jointly analytic coefficient bounds already derived in the [grazing](HILBERT16-GRAZING-PASSAGE.md) and [exponential](HILBERT16-EXPONENTIAL-PASSAGE.md) notes.

The center height is H=epsilon^3*exp(-kappa/epsilon). All derivatives in kappa keep every physical parameter and epsilon fixed.

Choose one fixed u in (0,2), and a sufficiently small physical incoming signed-label interval |t_i|<=tbox<u^2/2. The ordinary positive-height incoming fast connector must lie in the same normal chart; at epsilon=0 its height at V=u is (u^2-t_i)/2. Its continuation therefore has uniform positive height and a uniformly nonzero derivative between that height and t_i. This is a checkable common compact connector hypothesis, already of the kind used in [singular transport](HILBERT16-SINGULAR-TRANSPORT.md). Fix the selected large outgoing physical section near its zero label, with the existing uniform transversality margins.

The conclusion derived below is: there are positive constants epsilon0, kappa_b, C_tail and gamma, depending only on these fixed compacts and sections, such that every admitted passage with

    0<epsilon<epsilon0, kappa>=kappa_b, |t_i|<=tbox

satisfies

    0<D_epsilon'(t_i(kappa)) <= C_tail*exp(-gamma*kappa).

No upper bound on kappa is imposed in this estimate. The input-label restriction is essential. For any positive lower regular-slope bound m_reg on the captured tube, choosing kappa_tail>=kappa_b with C_tail*exp(-gamma*kappa_tail)<m_reg gives D'<m_reg<=Hreg' throughout this admitted tail. Its displacement slope is positive. Sections 9-11 supply the connected admitted domain, a uniform positive regular lower slope and an overlap argument for the stated small-label itinerary; no separate layer counts are added.

## 2. The exact slow-line saddle and its eigenvalue ratio

For the outgoing branch set V=-epsilon*x. Then

    f(-epsilon*x)=-epsilon^3*B_epsilon(x),
    B_epsilon(x)=L+lambda1*x-x^2*zeta(-epsilon*x,epsilon),
    B_epsilon -> B_-(x)=L+lambda1*x+x^2

in C2 on every fixed radial interval, uniformly over the coefficient compact. Let

    s=sqrt(delta), r1=(-lambda1-s)/2, r2=(-lambda1+s)/2.

Both roots are positive; r1, r2 and their separation have uniform positive lower/upper bounds as appropriate. The implicit function theorem supplies an exact simple first root r_epsilon=r1+O(epsilon), with B_epsilon'(r_epsilon)=-s+O(epsilon). At the actual equilibrium (V_o,h)=(-epsilon*r_epsilon,0), the Jacobian is triangular:

    [ epsilon^2*B_epsilon'(r_epsilon)   g(V_o,0) ]
    [                   0                  epsilon*r_epsilon ].

Its eigenvalues are exactly

    lambda_s=epsilon^2*B_epsilon'(r_epsilon)<0,
    lambda_u=epsilon*r_epsilon>0.

The stable manifold h=0 is exact. The linearized saddle power exponent is

    eta_epsilon=|lambda_s|/lambda_u
               =epsilon*(-B_epsilon'(r_epsilon))/r_epsilon
               =epsilon*(s/r1+O(epsilon)).

This identifies the relevant joint scale: a transverse entry of size exp(-kappa/epsilon) produces a stable-direction attenuation on the scale exp[-(s/r1)*kappa]. The uniform estimate below obtains a weaker positive exponent gamma without invoking a uniform linearization theorem.

## 3. A uniform outgoing saddle rectangle

Use h=epsilon^3*y and rescaled time tau=epsilon*(normal time). Exactly,

    x_tau=epsilon*D(x,y), y_tau=x*y,
    D=B_epsilon(x)+k(-epsilon*x,epsilon^3*y)*y.

In radial notation, partial_x k=O(epsilon^2), partial_y k=O(epsilon^5). Choose d>0 fixed and small compared with the lower bounds on r1 and r2-r1. There are uniform a_min>0, X<infinity, b>0 and s0>0 such that, for small epsilon,

    a=r_epsilon-d >= a_min,
    bnd=r_epsilon+d <= X,
    B_epsilon(a)>=b, B_epsilon(bnd)<=-b,
    B_epsilon'(x)+y*k_x(x,y)<=-s0

on a<=x<=bnd and 0<=y<=y0, after choosing a fixed sufficiently small y0>0. Choose also kmax*y0<b/2. At x=a the x velocity points right; at x=bnd it points left. Thus the closed x interval is invariant until the exit y=y0. Since y_tau>=a_min*y, that exit occurs in finite time for each positive entry height, however small.

For the passage starting at x=a with entry y_e, x_tau is positive initially. At a hypothetical first D=0 point,

    D_tau=D_y*x*y>0,

because x_tau=0 there and D_y=k+y*k_y>0. A first crossing from D>0 is impossible. Hence the actual x coordinate increases throughout this passage, even though the right side of the trapping rectangle points inward. This also establishes the selected outgoing direction.

Before x=a, B_epsilon is uniformly positive on [0,a]. Define

    S_epsilon(a)=integral_0^a x/B_epsilon(x) dx <= Smax<infinity.

The exact scalar radial equation gives

    0<y_e<=exp((Smax-kappa)/epsilon).

Choose fixed kappa_b>Smax+1 and epsilon0 sufficiently small that y_e<y0. The pre-rectangle compensated exponent

    Psi_pre=integral_0^a [B_epsilon'(x)+y*k_x(x,y)]/[B_epsilon(x)+k*y] dx

is bounded in absolute value by a uniform constant. This follows directly from the positive B lower bound, bounded B', and k_x=O(epsilon^2); it is not a compact-kappa limit. Therefore the preceding assertions hold for every kappa>=kappa_b.

## 4. Exact event derivative and exponential attenuation

For a planar flow z'=F(z), the derivative of a transition between transverse sections follows from the determinant of its fundamental matrix: det M=exp(integral div F). Applied to an entry section x=a, parameter y_e, and exit section y=y0, parameter x_e, it gives

    dx_e/dy_e = -epsilon*D(a,y_e)/(x_e*y0)
                  *exp integral [epsilon*(B_epsilon'+y*k_x)+x] d tau.

Here the sign is fixed by det(F_entry,(0,1))=epsilon*D(a,y_e) and det(F_exit,(dx_e/dy_e,0))=-x_e*y0*(dx_e/dy_e). Since integral x d tau=log(y0/y_e), this is exactly

    dx_e/dy_e = -epsilon*D(a,y_e)/(x_e*y_e)
                  *exp integral epsilon*(B_epsilon'+y*k_x) d tau.

The already-derived exact compensated initial-height variation, evaluated at x=a, gives

    dy_e/dkappa = -A_epsilon/epsilon
                    *y_e/D(a,y_e)*exp(Psi_pre),
    A_epsilon=L+k(0,H)*exp(-kappa/epsilon).

Multiplication cancels the small entry height, the entry denominator and epsilon:

    dx_e/dkappa = A_epsilon/x_e *exp(Psi_pre)
                   *exp integral epsilon*(B_epsilon'+y*k_x) d tau >0.

The residence time T obeys T>=log(y0/y_e)/X. Thus, with gamma=s0/X>0,

    exp integral epsilon*(B_epsilon'+y*k_x) d tau
      <=exp[-epsilon*gamma*log(y0/y_e)]
      <=C*exp(-gamma*kappa).

All remaining factors have uniform upper bounds and x_e>=a_min. Consequently

    0<dx_e/dkappa<=C*exp(-gamma*kappa)

for all kappa>=kappa_b, with one epsilon0. This supplies the required derivative estimate, not just a value power-law heuristic. It allows the exact root and exponent to vary with epsilon and the passive parameters.

## 5. Outgoing continuation and derivative transport

At exit, h_e=epsilon^3*y0 and T_e=V_e^2/2=epsilon^2*x_e^2/2. Thus

    c1*epsilon^2<=T_e<=C1*epsilon^2,
    0<T_e,kappa<=C*epsilon^2*exp(-gamma*kappa).

Set T=V^2/2 with V<0 and use h as independent variable. Exactly,

    T_h=q(V)/h+k(V,h)>0.

Positivity persists: Vdot=f+h*g begins negative and at a hypothetical zero its normal-time derivative is F_h*(-V*h)<0, so it cannot first cross from negative to positive. Therefore |V|>=epsilon*a_min as h increases.

On a fixed bounded negative-V rectangle, joint analyticity and the explicit cubic limit yield

    q(V)>=-C*epsilon^3,
    q(V)<=C*epsilon*(epsilon^2+T),
    |q(V)|<=C*epsilon*(epsilon^2+T),
    k=1+O(epsilon).

For example q(-w)=L*epsilon^3+lambda1*epsilon^2*w+epsilon*w^2*(1+w/3+O(epsilon)); minimizing its quadratic lower bound gives the first inequality, while epsilon^2*w<=C*epsilon*(epsilon^2+w^2) gives the absolute bound. These estimates are uniform for bounded w>=0.

The scalar differential inequality

    T_h<=C+C*epsilon*T/h+C*epsilon^3/h

and T_e=O(epsilon^2), h_e=epsilon^3*y0 imply T(h)<=C*(epsilon^2+h) on a fixed height interval, by an integrating factor h^(-C*epsilon). The factor (h/h_e)^(C*epsilon) remains uniformly bounded. Also T>=T_e and T_h>=1/2 once h>=C2*epsilon^3 for a fixed large C2. Hence, after that short initial height interval,

    c*(epsilon^2+h)<=T(h)<=C*(epsilon^2+h).

On a proposed enlarged fixed rectangle, integrating

    (T-h)_h=q/h+k-1

gives T-h=O(epsilon) uniformly there (the epsilon^3*log(1/epsilon) term is harmless). This closes the continuation bootstrap: choose the negative-V boundary outside the curve V=-sqrt(2h), for 0<=h<=hmax, by a fixed margin; the last estimate precludes reaching that boundary before hmax when epsilon is small. Positivity and |V|>=epsilon*a_min preclude the other V boundary. Choose hmax past the selected outgoing event, again with a margin. The orbit therefore reaches the selected large outgoing physical section near its zero signed label by the same transversal event argument as the previous passage notes. This supplies actual outgoing continuation uniformly in kappa; it is not inferred from the compact-kappa theorem.

At fixed h the variation P=partial T(h)/partial T_e satisfies

    P_h=-F_V/(h*V)*P, P(h_e)=1.

The exact coefficient decomposition used by [height comparison](HILBERT16-HEIGHT-COMPARISON.md) is

    F_V/V=-2*epsilon+epsilon^2*lambda1/V
                  +epsilon*V*Bcal(V,epsilon)+h*g_V/V,

where Bcal is uniformly bounded. On this outgoing branch lambda1/V>0, |V|>=epsilon*a_min and g_V=O(epsilon). Therefore

    F_V/V>=-C*epsilon-C*h,
    P(h)<=exp(C*h)*(h/h_e)^(C*epsilon)<=C.

The physical endpoint correction is exact. For the section equation E_sigma(V,h)=0 and its signed label t=T_sigma(h),

    t_kappa=-T_sigma'(h)*T_kappa/(T_h+V*(E_sigma)_h).

The already fixed large-branch transversality makes this factor positive and uniformly bounded. Consequently

    0<t_o,kappa<=C*epsilon^2*exp(-gamma*kappa).

## 6. Incoming lower sensitivity on the selected small-label tube

By the fixed fast-connector hypothesis in section 1, every admitted incoming physical label |t_i|<=tbox reaches V=u with height h_u in a common positive compact interval. The map from h_u to the physical t_i has derivative bounded strictly negatively, and tends to -2 as epsilon tends to zero.

On 0<=V<=u<2, q>=0 and q_V>=0 for sufficiently small epsilon. Indeed write exactly zeta=-1+V*beta(V,epsilon), where beta=1/3+O(epsilon) in C1 on this compact interval; then q_V=-epsilon^2*lambda1+epsilon*V*(2-V+O(epsilon*V)). The positive gap 2-u controls the entire interval, including V=0. Since k_V=O(epsilon) and q+kh>=kh,

    (q_V+h*k_V)/(q+kh)>=-C*epsilon.

The incoming compensated exponent therefore satisfies Psi_i(u)>=-C*epsilon*u, independently of kappa. The exact moving-center variation at this fixed V is

    h_u,kappa=-epsilon^2*A_epsilon
                  *h_u/(q(u)+k(u,h_u)*h_u)*exp(Psi_i(u)).

The positive compact endpoint height, A_epsilon>=c, and q(u)=O(epsilon) give |h_u,kappa|>=c2*epsilon^2. Transfer through the fast connector then yields

    t_i,kappa>=c3*epsilon^2>0.

Together with section 5 this proves the tail slope bound stated in section 1. The incoming connector condition cannot be omitted or replaced by an unspecified bounded passage.

## 7. Why unrestricted fixed-epsilon H->0 is a different regime

The exact slow line also has a positive equilibrium near V=3: f(V)/epsilon tends to V^2*(-1+V/3), with simple root 3. At that root f_V=3*epsilon+O(epsilon^2) and the transverse eigenvalue is -3+O(epsilon). In reversed normal time its saddle ratio is eta_i=epsilon+O(epsilon^2). If passages through both slow-line saddles and their outer sections are included, the two fixed-epsilon local power exponents suggest a multiplier proportional to H^(eta_o-eta_i), not necessarily tending to zero. Indeed s/r1<1 when 4L<lambda1^2<(9/2)L. This observation is a warning about domain, not a proof of a full global two-saddle asymptotic or a counterexample in the selected small-label tube.

The tube |t_i|<=tbox<u^2/2, u<2, avoids this positive-equilibrium endpoint regime: its connector at V=u has height bounded away from zero. The present theorem covers every sufficiently small H that is actually admitted by that tube, without claiming every H is admitted. In a joint epsilon->0 scaling the transition from x=O(1) to physical V=O(1) occurs at kappa of order log(1/epsilon); the proof above remains valid there because its outgoing rectangle and incoming fixed-V connector do not require one common bounded radial x cut.

## 8. Existing implementation reuse

Exact interval coefficient/derivative premises can be checked using `omnibias.dynamics.quadratic_unfolding`, core Interval/Taylor arithmetic, and the existing Bell/jet primitives. Root isolation can certify r_epsilon and the separating rectangle using `verified.rootfind.interval_newton`; `turning_geometry` already checks related sign conditions. The determinant/event formula is the analytic mechanism behind existing monodromy and Poincare code, but no finite-time integration of exponentially long saddle residence is required here. The existing `omnibias.dynamics.continuation` module also certifies a finite parameter segment, a finite event, and a join via endpoint uniqueness. Those tools can validate finite premises; the uniform rectangle/variation argument and connected physical capture interface provide the analytic control used here.

## 9. Connected singular admission across all positive heights

Fix Hmax>0 sufficiently small for the old grazing theorem and fixed incoming/output signed-label intervals I_i,I_o within [-1,1]. Let I_i be a small interval about zero with sup I_i<u^2/2, for the fixed u in (0,2). The large outer sections and the fast connector from V=u have the common-chart and strict transversality margins of section 1. These choices precede the smallness choice for epsilon.

For every 0<H<=Hmax, the incoming branch reaches V=u. On 0<=V<=u, q>=c*epsilon^3>0 and k>=1/2. In reversed normal time V increases with velocity q+kh. Exactly,

    h_V=V*h/(q+k*h), h(0)=H,
    0<=h_V<=2V, hence H<=h(V)<=H+V^2.

Choose the chart to contain this segment with a margin. No height loss or blow-up occurs; q's positive lower bound gives finite travel time for each fixed epsilon. Continuous ODE dependence gives a smooth h_u(H)=h(u;H) on (0,Hmax], with

    partial_H h_u=exp integral_0^u partial_h[V*h/(q+k*h)] dV >0.

No uniform bound on this exponential as H->0 is needed. In fact h_u(H)<=H*exp(integral_0^u V/q dV), so h_u(H)->0 at fixed epsilon as H->0. Thus the proof does not assert that arbitrarily tiny heights reach the selected small-label section at a fixed epsilon.

The positive-height fast connector identifies I_i with one height interval J_i(epsilon) at V=u, in a fixed positive compact range. Its map chi_epsilon:J_i(epsilon)->I_i is strictly decreasing with derivative bounded away from zero and infinity in magnitude; chi_0(h)=u^2-2h. Therefore

    J_in={H in (0,Hmax]: h_u(H) in J_i(epsilon)}

is an interval, possibly empty. On it t_i(H)=chi_epsilon(h_u(H)) is smooth and strictly decreasing.

The outgoing branch exists at its selected large section for every 0<H<=Hmax after imposing one common smallness cutoff. For kappa>=kappa_b this follows from sections 3-5. For H>=epsilon^3 use the existing grazing proof. Cover the remaining heights by the initial thin-height theorem and the existing exponential theorem on one fixed finite band [kappa_a,kappa_b+1], with kappa_a inside the initial-sector overlap. These ranges overlap and the trajectories agree by uniqueness. Only finitely many cutoffs are intersected; existence is not inferred from infinitely many compact-kappa results.

Choose Hmax sufficiently small that all outgoing labels lie in one small section interval I_o with a margin. The old grazing limit is t_o=-2H; on a fixed compact exponential band the limit is zero; and section 5 gives uniform t_o=O(epsilon) in the saddle tail. Alternatively, restricting to a smaller connected I_o still preserves connectedness: the exact square-root initialization and scalar variational formula in [height comparison](HILBERT16-HEIGHT-COMPARISON.md), section 3, gives Xi=partial_H T<0 on the whole outgoing branch. That argument uses F(0,H)<0, not the sign of lambda1. The large-branch endpoint factor is positive, hence t_o,H<0 and t_o^{-1}(I_o) is an interval.

The singular admitted set J_sing=J_in intersect t_o^{-1}(I_o) is therefore an interval, and its t_i image is an interval. A further nonrectangular or nonmonotone itinerary restriction requires its own connectedness proof; it is not included silently.

## 10. A uniform positive regular multiplier

Use the captured fixed-alpha regular map G_R and the constants of [regular jets](HILBERT16-REGULAR-JETS.md). Write kappa_q>0 for its bound q(x)>=kappa_q*(1+|x|)^2, and Q for the upper constant; kappa_q is unrelated to logarithmic height. Its exact first-variation factorization gives

    G_R'=[D(-R)/D(R)]*[q(R)/q(-R)]*E(R),
    D'/D=2/q, |log E|<=2*K_w*delta_reg*R<=1/2.

The last bound uses its existing sector delta_reg*(R+1)<=eta_p<=1/(4*K_w). Since q(R)/q(-R)>=kappa_q/Q and

    integral[-R,R] 2/q dx
      <=integral[-infinity,infinity] 4/((x+1)^2+C_min-1) dx
      =4*pi/sqrt(C_min-1),

define c_star=exp[-4*pi/sqrt(C_min-1)]>0. Then

    G_R'>=m_G:=(kappa_q/Q)*c_star*exp(-1/2)>0.

This is uniform in R, epsilon and the fixed admitted alpha; it uses actual variations and no zero-anchor assumption.

Include the exact finite-nu signed-coordinate maps

    Z_sigma,nu(t)=(rho^2*Z_sigma(t)+C*nu^2)/Delta_sigma,
    Delta_sigma=rho^2+2*sigma*nu*rho+C*nu^2,
    Hreg=Z_+,nu^{-1} composed with G_R composed with Z_-,nu.

For r=-1, rho<=1/4 and |t|<=1 put s_sigma=sqrt(1+2*sigma*rho+rho^2*t) and D_sigma=1+sigma*rho+s_sigma. Then 1/2<=s_sigma<=5/4, 1<=D_sigma<=5/2, so

    |Z_sigma'|=rho^2/(s_sigma*D_sigma^2),
    (16/125)*rho^2<=|Z_sigma'|<=2*rho^2.

For small nu/rho, each affine factor rho^2/Delta_sigma lies in [1/2,2]. When the physical input and output labels both lie in [-1,1],

    |Z_-,nu'(t)|/|Z_+,nu'(Hreg(t))|
       >=(1/4)*(8/125)=2/125>1/64.

Hence Hreg'>=m_H:=m_G/64>0. Together with the existing upper and curvature bounds, after the prescribed rho-then-nu choices,

    m_H<=Hreg'<=theta_H<1, |Hreg''|<=Kreg.

For the domain, take the signed-coordinate preimage of the fixed-alpha matching interval I_alpha from [regular jets](HILBERT16-REGULAR-JETS.md), equation (16). This is an interval since Z_-,nu decreases. Hreg increases there. Intersect with I_i and Hreg^{-1}(I_o); these restrictions preserve intervals and contain every cycle under consideration. Call this domain I_reg. The complete displacement domain

    I=t_i(J_sing) intersect I_reg

is an interval. Its intermediate regular trajectories need only satisfy the fine endpoint tube, not a stronger coarse selection condition imposed on the cycles themselves.

## 11. One two-cycle bound on every admitted positive height

Choose a fixed kappa_tail>max(kappa_b,kappa_star) such that C_tail*exp(-gamma*kappa_tail)<m_H, where kappa_star is the initial-sector cutoff. It is one finite choice depending only on the fixed compacts and sections, not epsilon. On I wherever kappa>=kappa_tail, the new tail gives

    F'=Hreg'-D'>0, F=Hreg-D.

On the initial band H>=epsilon^3*exp(-kappa_star/epsilon), the old thin-height plus grazing result gives F'<0. Since lambda1<=-2*sqrt(c)<0 throughout the first-root parameter compact, the existing exponential C1 theorem gives F''>0 on the one fixed finite band

    kappa in [kappa_a,kappa_tail+1], 0<kappa_a<kappa_star.

Its negative R_kappa margin is uniform on this finite band and dominates Kreg for small epsilon, as in [exponential passage](HILBERT16-EXPONENTIAL-PASSAGE.md), section 10. Intersect its cutoff with the finitely many cutoffs above.

The input label increases with kappa because t_i,H<0 and H_kappa<0. Thus initial, middle and tail portions occur in that order on the one interval I, with overlaps. No zero of F' lies in the initial or tail portion. Any two zeros of F' would lie in the middle portion, and their intervening interval would lie there and in I, where F''>0. This is impossible. Hence F' has at most one zero, and Rolle's theorem bounds the distinct zeros of F by two over all positive center heights admitted by this itinerary.

This proves one derivative pattern across overlaps, not a sum of separate cycle counts. It also excludes an interval of cycles. Any cycle in the outer tail is hyperbolic and repelling in physical time because its return multiplier Hreg'/D'>1.

The resulting bound has one epsilon0 for the strict first-root coefficient compact and the fixed small-label tube. It excludes delta->0, L->0, unbounded coefficients, loss of section transversality, different itineraries and an unproved assertion that every nearby cycle enters this tube. In particular it is not full graphic cyclicity, the full quadratic Hilbert problem, or all-degree Hilbert XVI. Its analytic implications have not been formally verified.

## Finite verification

```bash
python benchmarks/hilbert16_root_saddle.py --output artifacts/hilbert16/root_saddle.json
```

The benchmark checks exact rescaling, saddle and variation identities and
uses the existing interval and root-isolation primitives for a declared
coefficient rectangle. Its perturbation envelope is an explicit premise;
it does not compute the physical epsilon cutoff or formally verify the
analytic continuation, connectedness, or cycle-count arguments above.

The separate formal replay

```bash
python -m benchmarks.hilbert16_formal_replay --output artifacts/hilbert16/formal_margin_replay.json
```

passes the declared rectangle's six positive interval-margin signs through
the existing minimal Lean kernel. This checks their finite rational sign
obligations conditional on interval membership. The actual interval
evaluation and canonical-family envelope remain outside that replay.
The Mathlib [Rolle module](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean)
also checks the abstract inference from one derivative zero at most to
two function zeros at most. The physical premises in this note remain
separate analytic obligations.

The complementary [joint matching theorem](HILBERT16-JOINT-MATCHING.md)
concerns strict positive `4L-lambda1^2` and a shrinking physical escape
scale. The present first-root theorem has the opposite discriminant sign.
The two results leave the coalescing-root limit to a separate analysis.
That analysis is now the [χ-atlas companion](HILBERT16-COALESCING-CAPTURE.md)
together with the [saddle-node](HILBERT16-SADDLE-NODE.md) and
[shrinking-root](HILBERT16-SHRINKING-ROOT.md) notes; they do not pass
G1 or G4.

# Actual singular passage on the regular-passage sections

This companion to [the Hilbert-16 program](HILBERT16.md) gives an analytic coordinate/connector argument, relying on the published entry-exit theorem and its parameter-dependent proof. It proves a singular-passage approximation on fixed compact interior sectors. It does not treat the other blow-up charts, endpoint fibers, poles, zero discriminant, or the full graphic. No novelty or formal verification claim is made.

Primary source: Huzak and Kristiansen, *On entry-exit formulas for degenerate turning point problems in planar slow-fast systems*, Nonlinearity 39 (2026), 085026, DOI 10.1088/1361-6544/ae9443. The [author-hosted published PDF](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf) has Theorem 2.4 on PDF page 8, its proof in Sections 3–4, the actual family/chart calculations (6.2)–(6.7) on pages 28–29, and the explicit limiting map (6.8)–(6.12) and Theorem 6.1 on page 30. Page numbers here count the PDF's first page as page 1; printed page labels differ by one in portions of the manuscript.

The argument below is our derivation from that source and the exact polynomial family. The source's theorem prints the small-parameter/section variables but suppresses passive parameter arguments. Section 5 below explains the parameter extension being used; it is not asserted to be an extra explicitly printed theorem.

## 1. Parameters and exact physical-to-family transport

Use the positive global scale nu, distinct from the canonical epsilon of the entry-exit theorem. Write the physical quadratic family as

    xdot = A*x - y + x^2 + nu*(p+r)*x*y + nu^2*m*y^2,
    ydot = C*x + x^2 + x*y + nu*r*y^2.

Thus (mu1,mu2,mu3)=(nu^2*m,nu*p,nu*r). The slow-fast corner considered here further has m=nu*mtilde and p=nu*ptilde, while r lies in a fixed compact subset of (-infinity,0). One may fix r=-1 as an affine weighted chart; allowing r to vary merely gives a redundant positive rescaling of these physical parameters.

For the regular passage use

    q(x)=(x^2+2*x+C)/2, w=y-(x^2-C)/2, z=w/q(x), R=rho/nu.

Here C is in a fixed compact subset of (1,infinity), rho>0 is fixed first, and the physical sections are x=sigma*R for sigma=+-1. Set X=nu*x and Y=nu^2*y. On these sections the exact height is

\[
Y_\sigma(\nu,z)=\frac{\rho^2}{2}(1+z)
  +\sigma\nu\rho z+\frac{C\nu^2}{2}(z-1).                 \tag{1}
\]

On Y>0 the family coordinates are

\[
v=\frac X Y=\frac{\sigma\rho}{Y_\sigma},\qquad
h_f=\frac1Y=\frac1{Y_\sigma}.                              \tag{2}
\]

The symbol h_f is the family chart's z2 variable, and becomes the paper normal form's y variable. It is not the regular coordinate z=w/q. The sections satisfy v=sigma*rho*h_f exactly. All coordinate and time changes up to this point preserve time orientation: with tau=t/nu and d sigma_time/d tau=Y>0, the family field is

\[
\dot v=F=m+pv-\nu v^3-h_f+A\nu vh_f-C\nu^2v^2h_f,
\quad
\dot h_f=-h_fN,
\]
\[
N=r+v+\nu v^2+C\nu^2vh_f.                                 \tag{3}
\]

Define the exact slow coordinate V=-N. Now reverse time. If
ell=1+2*nu*v+C*nu^2*h_f, direct differentiation gives

\[
\dot V=\ell F+C\nu^2vVh_f,\qquad \dot h_f=-Vh_f.           \tag{4}
\]

This is the actual normal time orientation used in the paper. The inverse coordinate is

\[
v=-\frac{2(r+V)}{1+C\nu^2h_f+
 \sqrt{(1+C\nu^2h_f)^2-4\nu(r+V)}}.                        \tag{5}
\]

It is analytic jointly with parameters on every fixed compact (V,h_f) set for sufficiently small nu. Thus even the large, but fixed-rho, section coordinates below cause no obstruction to this coordinate map. No assertion uniform in rho->0 is being made at this stage.

## 2. Exact canonical slow normalization

The line h_f=0 is invariant. Let v0 be the root of

    nu*v0^2+v0+r=0

near -r, and l=1+2*nu*v0>0. Let lambda=(lambda0,lambda1) range over a compact set with

\[
4\lambda_0+\lambda_1^2\le-\chi<0.                          \tag{6}
\]

There is a unique analytic k near -3*r>0 solving

\[
k=\frac{3v_0}{l}
  +\frac{\nu^2k^2\lambda_1}{l^2}
  +\frac{4\nu^4k^3\lambda_0}{l^4}.                        \tag{7}
\]

Indeed, at nu=0 the right side is -3*r and the derivative with respect to k of the equation's left-minus-right side is 1. The analytic implicit function theorem is uniform on the fixed compact r/lambda set after shrinking nu.

Set

\[
F_0=\frac{\nu^2k^3\lambda_0}{l},\qquad
F_1=-\nu k^2\lambda_1-
       \frac{2\nu^3k^3\lambda_0}{l^2},
\]
\[
\widetilde p=3v_0^2+F_1,\qquad
\widetilde m=F_0-\widetilde p\,v_0+v_0^3,
\quad p=\nu\widetilde p,\quad m=\nu\widetilde m.           \tag{8}
\]

For the scalar slow-line field Vdot=l(v)*(m+p*v-nu*v^3), the Taylor coefficients c_j in V at V=0 satisfy exactly

\[
c_0=(\nu k)^3\lambda_0,\quad
c_1=(\nu k)^2\lambda_1,\quad c_2=-\nu k,
\]
\[
\frac{c_3}{\nu k}=\beta_\nu
=\frac1{kl^2}-\frac{2\nu^3k\lambda_1}{l^4}
               -\frac{8\nu^5k^2\lambda_0}{l^6}.           \tag{9}
\]

The last expression tends to beta0=-1/(3*r)>0. Let epsilon=nu*k. Since k stays positive and d(epsilon)/d(nu) stays positive, nu is analytic in epsilon near zero, with passive parameters. The entire field (4) takes the exact general-Theorem-2.4 form

\[
\dot V=\epsilon\big[\epsilon^2\lambda_0+
 \epsilon\lambda_1 V+V^2\zeta(V,\epsilon)\big]
 +h_f g(V,h_f,\epsilon),\qquad \dot h_f=-Vh_f,             \tag{10}
\]

where zeta and g are analytic jointly with the passive parameters,

\[
\zeta(V,0)=-1+\beta_0 V,\qquad g(V,h_f,0)=-1.             \tag{11}
\]

Analytic divisibility by epsilon in the slow line follows because (4) vanishes on h_f=0 at nu=0. The constant and linear coefficients are fixed by (9); the quotient after removing them is divisible by V^2. Formula (11) follows also by setting nu=0 after division by nu: the limiting cubic before division by k is

    V^3 + 3*r*V^2 + (3*r^2-ptilde)*V
        + (r^3-ptilde*r+mtilde).

In particular the normalized cubic coefficient beta_nu varies with nu. In general this does NOT give the more restricted exact formula (6.1) with a constant beta and only epsilon*V^2 in zeta's remainder. Theorem 2.4 is the correct theorem to invoke; its leading function is nevertheless exactly the one evaluated in (6.8).

The parameters A and C enter only the coefficient g and the off-line coordinate transformation. They can be carried as passive compact parameters. In particular, a free splitting alpha=A-1 may be retained; no zero-endpoint compensation is imposed in the singular argument.

## 3. Exact fiber labels on the limiting sections

At nu=0, (4) becomes Vdot=-h_f, h_fdot=-V*h_f. Its fast fibers have the first integral

\[
B^2=V^2-2h_f.                                             \tag{12}
\]

On sigma's limiting section, this is

\[
K_\sigma(z)=r^2+\frac{4\sigma r}{\rho(1+z)}
                 -\frac{4z}{\rho^2(1+z)^2}.              \tag{13}
\]

The inverse branch near z=0, using the square label t=B^2, is exactly

\[
Z_\sigma(t)=\frac{2}{1-\sigma r\rho+
              \sqrt{1-2\sigma r\rho+\rho^2t}}-1.         \tag{14}
\]

On fixed bounded t-sets and compact r-sets, for sufficiently small rho the radicand and denominator stay positive and K_sigma(Z_sigma(t))=t. The expansion is

\[
Z_\sigma(t)=\sigma r\rho+\frac{5r^2-t}{4}\rho^2
  +O(\rho^3).                                             \tag{15}
\]

In particular, the two z section centers have opposite order-rho offsets. A regular map centered only at z=0 cannot be equated directly to the singular entry formula.

Define the ACTUAL moving curves Gamma_sigma^nu(B) in (V,h_f) by inserting z=Z_sigma(B^2) into (1),(2),(4)'s coordinate V=-N. These curves depend analytically on nu, B and all fixed passive parameters. They are exactly the physical sections, in the chosen B labels. At nu=0,

\[
V\big(\Gamma_\sigma^0(B)\big)
=-\frac{\sigma}{\rho}
 [1+\sqrt{1-2\sigma r\rho+\rho^2B^2}].                    \tag{16}
\]

Thus the physical LEFT section (sigma=-1) is the POSITIVE-V incoming section in normal-forward time. The physical RIGHT section (sigma=+1) is the NEGATIVE-V outgoing section. Both have h_f of order rho^-2, so neither is directly the paper's small-height section.

## 4. Ordinary connectors to the paper sections

Fix common compact input labels I=[Bmin,Bmax] with Bmin>0, and compact passive parameters such that

\[
a=\exp\!\left(\frac{\pi\lambda_1}{
          \sqrt{-4\lambda_0-\lambda_1^2}}\right),\qquad
b=\beta_0(a+1),\qquad 1-bB\ge\omega>0\quad(B\in I).       \tag{17}
\]

Let M(B)=a*B/(1-b*B). Select a slightly enlarged common compact output interval J in (0,infinity) containing all M(I) with a positive margin. Choose a positive paper section height h sufficiently small that the positive incoming section coordinates sqrt(B^2+2*h), including a small enlargement of I, lie below 1/beta0 uniformly. Such an h exists from (17), compactness and b>beta0. Choose rho>0 sufficiently small that both Gamma curves have h_f>h and the signs in (16) hold on I and J.

For the incoming connector, follow normal-forward flow from Gamma_-^nu(B) to h_f=h. At nu=0 the entire arc has V>=B>=Bmin and h_f decreases. If Ymax is an upper bound for h_f on the incoming curves, its travel time is at most

\[
T_-\le B_{\min}^{-1}\log(Y_{\max}/h),                    \tag{18}
\]

and its endpoint is transverse because |h_fdot|>=h*Bmin. The arc belongs to a fixed compact set for fixed rho,h. For the outgoing connector, follow BACKWARD normal flow from Gamma_+^nu(B), B in J, to h_f=h. The same proof applies with min(J)>0 and -V in place of V.

Analytic dependence of finite-time ODE solutions, the uniform endpoint transversality, and the implicit function theorem therefore give connector maps

\[
s_{\rm in}^\nu(B)=\sqrt{V_{\rm in}^\nu(B)^2-2h},\qquad
s_{\rm out}^\nu(B)=\sqrt{V_{\rm out}^\nu(B)^2-2h},          \tag{19}
\]

with

\[
s_{\rm in}^\nu=\mathrm{id}+O_{C^k,\rho,h}(\nu),\quad
s_{\rm out}^\nu=\mathrm{id}+O_{C^k,\rho,h}(\nu)           \tag{20}
\]

for every fixed finite k, uniformly on the designated compact parameter sets. Here the square roots are label definitions at the paper section, not claims of an invariant at positive nu. At zero they are exactly B by (12). For small nu, s_out is an increasing local diffeomorphism; on slightly smaller compact ranges its inverse is id+O_C^k(nu). The enlargements avoid endpoint domain loss under these compositions.

No exponential-duration estimate is required for these connectors. The actual singular delay occurs only inside the small-height transition handled by Theorem 2.4. The constants in (18)–(20) may diverge as rho->0 or Bmin->0; rho is fixed before shrinking nu.

## 5. Parameter dependence of the entry-exit remainder

The precise extension used is as follows. Let theta denote finitely many passive parameters in a compact set, with the functions f_lambda,g analytic (or sufficiently C^N) jointly with theta on common neighborhoods. Suppose the negative-polynomial bound (6), negative-zeta bound on the slow interval, compact nonzero input/output base labels and section transversality all hold with common strict margins. Then the proof of Theorem 2.4 gives, for each finite k, a jointly C^k function

    phi(Vin,epsilon,eta;theta), phi(Vin,0,0;theta)=0,

with eta=epsilon*log(1/epsilon), on finitely many parameter neighborhoods covering the compact set. Consequently the resulting actual maps have uniform C^(k-1) section-coordinate remainder bounds of order epsilon*log(1/epsilon). Only this forward implication is used; a rate bound alone would not imply the smooth extension.

Fix the desired final differentiability order k first, and run the input and normal-form constructions at an order N sufficiently larger to absorb their finite derivative losses. The original coefficients are analytic, so this is available. In particular, Hadamard division

    alpha0(epsilon,theta)=epsilon*A0(epsilon,theta),
    A0(epsilon,theta)=integral[0,1] partial_epsilon(alpha0)(t*epsilon,theta) dt

loses one derivative if alpha0 is given only with finite regularity. Choosing the normal-form order above k accounts for that loss; using an order-k normal form alone would not.

This is obtained by carrying passive parameters through the proof, not by deducing uniformity from pointwise convergence:

1. Theorem 2.4 already assumes compact Lambda and the uniform bound P_lambda<=-c in (2.7). Here P_lambda(s)=-(s-lambda1/2)^2+(4*lambda0+lambda1^2)/4<=-chi/4, explicitly. The normal slow coefficient is negative on the chosen interval because its limiting value is -1+beta0*V, with a strict upper-end margin.
2. In the outer blow-up chart, Lemma 3.1 uses the entry-exit normal form from De Maesschalck–Schecter (2016), Proposition 2.1. **Remark 1 on page 5 of that paper explicitly allows a finite-dimensional additional parameter in f and g.** Its Section 3 proof consists of finite coordinate normalizations by integration followed by regular-ODE dependence in the variables epsilon and epsilon*log(epsilon); a passive coefficient vector is carried through those operations. The spectrum transverse to the line has magnitudes bounded away from zero on the compact nonzero-base interval. See Remark 1 and Section 3 in the [author PDF](https://schecter.math.ncsu.edu/entry-exit.pdf).
3. For the inner saddle, the explicit transformations (4.6) integrate (Q_lambda+1)/(s*Q_lambda) and (zeta+1)/(s*zeta). Both integrands have removable singularities at zero; the denominators away from zero have uniform strict bounds after choosing the local sections. All coefficient derivatives in theta are therefore bounded. Lemma 4.2 uses finitely smooth normal forms for the FIXED hyperbolic spectrum (-1,-1,+1), with no change of resonance pattern as theta varies. The directly cited family statement is Zhu–Rousseau (2002), Proposition 4.6, printed pages 348–349, in [Finite Cyclicity of Graphics with a Nilpotent Singularity of Saddle or Elliptic Type](https://yorkspace.library.yorku.ca/bitstreams/fc2121d3-1b87-4e03-bbae-5f01fe202378/download). It supplies a finitely smooth normalizing family preserving the two saddle coordinates, with smooth parameter coefficients. The underlying parameter-family normal-form setting is also developed in Huzak–Kristiansen's reference [13], Ilyashenko–Yakovenko, *Finitely-smooth normal forms of local families of diffeomorphisms and vector fields*, Russian Math. Surveys 46 (1991), 1–43, DOI 10.1070/RM1991v046n01ABEH002733. Parameters remain coefficients; one does not append zero eigenvalues and call the enlarged system hyperbolic. Locally the normalizing maps and retained coefficient alpha0(epsilon;theta) can be chosen C^k jointly. Compactness gives finitely many neighborhoods and a common smaller cutoff. The integration in Lemma 4.3 then has its only resonant logarithm multiplied by alpha0(epsilon;theta)=epsilon*A0(epsilon;theta), so it is smooth in the independent pair (epsilon,eta), uniformly in theta.
4. The middle transition in Lemma 4.5 is regular because P_lambda is uniformly negative. Its ODE maps are jointly smooth. The tail integrals in (4.12) are smoothly parameter dependent: after the cancellation displayed there, their integrands and each finite parameter derivative are integrable uniformly on compact Lambda.
5. The closing implicit equation is (4.13). In positive outgoing magnitude u, the derivative of its leading relation has magnitude 1/[u*(1+beta0*u)]. It is bounded below on the chosen compact u/beta0 set. Thus the parameter-dependent implicit function theorem applies uniformly; uniqueness identifies the actual transition maps on overlaps along eta=epsilon*log(1/epsilon). The off-curve C^k extensions need not agree. The finite local cover already suffices for uniform norm bounds; if a single extension is wanted, a smooth parameter partition of unity combines those extensions while preserving their common on-curve value and their zero value at (epsilon,eta)=(0,0).

The flat terms entering the inner normal form also remain uniform with passive parameters. In the paper's inner chart, take 0<=y2<=mu<1 and t=r1*rho21. Terms of the form r1^(-3)*rho21^(-1)*y2^(1/t)*g(...) extend C^N to the boundary on a sufficiently small fixed t-interval. Each derivative through total order N, including passive-parameter derivatives, is a finite sum bounded by a constant times t^(-M)*|log(y2)|^L*y2^(1/t-j), with j<=N. Choose the upper t cutoff eta_N so 1/(2*eta_N)>N+1. Splitting the power into y2^(1/(2*t)-j)*y2^(1/(2*t)) gives vanishing at y2=0; the remaining exponential decay dominates every inverse power and logarithm as t->0. Passive derivatives hit uniformly smooth coefficients, while the exponent and normalized spectrum do not vary with those parameters. This supplies the common finite-order chart regularity required before applying the parameter-family normal form.

These facts justify the uniform compact-parameter extension in this setting using the same normal-form ingredients as the published proof. They do not supply a computable numerical value for the singular remainder constant or a bound uniform as one of the strict margins tends to zero. A formal proof would have to formalize those normal-form/ODE results; the repository's finite rational checker does not discharge them.

For a C^4 rate take k>=5. Since all section derivatives up to order 4 of phi vanish at (epsilon,eta)=(0,0), the fundamental theorem of calculus in those two variables, and a compact bound for the corresponding mixed derivatives of phi, give

\[
\|\Delta_\epsilon-\Delta_0\|_{C^4}
 \le L[\epsilon+\epsilon\log(1/\epsilon)].                \tag{21}
\]

Only k>=3 is required for a C^2 rate. The singular theorem may be applied with alpha=A-1 as a passive parameter in a compact interval, then evaluated at a nonanalytic compensation alpha(nu,d). No derivative of that compensation with respect to nu is needed for the uniform section-coordinate estimate. Alpha is held fixed when differentiating the return displacement with respect to its section coordinate.

## 6. Actual normal-forward map on our curves

On the paper sections, transform the actual signed section map into positive base-magnitude labels:

\[
E_\nu(s)=\sqrt{\Delta_\epsilon(\sqrt{s^2+2h})^2-2h}.
\]

The limiting relation obtained from (11) and the principal-value equation in Theorem 2.4 is M(s)=a*s/(1-b*s), exactly as in (6.8). All square-root arguments stay bounded away from zero on the chosen compact input/output sets. Therefore (21) and smooth composition give

\[
E_\nu=M+O_{C^4}(\nu\log(1/\nu)),                         \tag{22}
\]

uniformly in the compact passive parameters. We used epsilon=nu*k with k positive and analytic.

The actual normal-forward singular transition Gamma_-^nu -> Gamma_+^nu in the B label is exactly

\[
S_\nu=(s_{\rm out}^\nu)^{-1}\circ E_\nu\circ
      s_{\rm in}^\nu.
\]

Combining (20),(22) proves

\[
\boxed{S_\nu(B)=\frac{aB}{1-bB}
       +O_{C^4,\rho,h}(\nu\log(1/\nu)).}                 \tag{23}
\]

The same conclusion holds in C^2. All statements concern a common compact input interval, a common output neighborhood and fixed rho,h, with the small-nu cutoff allowed to depend on those data. The physical closing map runs in the reverse direction and is S_nu^{-1}.

## 7. Correct composition and the weighted two-zero gate

Let P_{nu,alpha} be the actual regular normalized-z passage from physical left to right. In square fiber labels set

\[
H_{\nu,\alpha}(t)=K_+\big(P_{\nu,\alpha}(Z_-(t))\big).
\]

Where both output base labels are positive, its unsquared map is sqrt(H(v^2)). A physical cycle in the specified itinerary satisfies

\[
\sqrt{H_{\nu,\alpha}(v^2)}=S_\nu(v),\quad\text{equivalently}\quad
F_\nu(v)=H_{\nu,\alpha}(v^2)-S_\nu(v)^2=0.               \tag{24}
\]

It is NOT the equation regular(v)=M^{-1}(v): normal-forward singular passage and physical regular passage both run left-to-right; physical closing is their inverse-direction counterpart.

For the limiting M, an exact identity is

\[
\left(\frac1v\frac d{dv}[H(v^2)-M(v)^2]\right)'
  =4vH''(v^2)-\frac{6a^2b}{(1-bv)^4}.                    \tag{25}
\]

Consequently, if the independently proved regular-passage estimates give |H''|<=Kreg*rho^2 on a common interval, then for small enough rho the right side is strictly negative uniformly on the compact data a,b>0 and v>0. Its leading negative magnitude is at least 6*amin^2*bmin since 0<1-bv<1. The actual singular C^2 error preserves this sign after nu is small enough for this fixed rho: for r_entry=S_nu^2-M^2,

\[
\left|\left(r_{\rm entry}'/v\right)'\right|
 \le \|r_{\rm entry}''\|/v_{\min}
       +\|r_{\rm entry}'\|/v_{\min}^2.                   \tag{26}
\]

Three distinct zeros of F_nu would give two zeros of F_nu' by Rolle, hence two zeros of F_nu'/v, contradicting the strict derivative sign. Thus (25)–(26), combined with the actual common-domain regular H'' estimate, give at most TWO distinct cycles for this interior itinerary. This algebra is independent of the free additive square displacement d and of any limiting linear multiplier c; an affine H=d+c*t is annihilated by the weighted operator. The singular proof here supplies its actual C^2 input. It does not independently prove the required regular common-domain H'' estimate.

The bound counts roots on a connected interval where the same regular and singular maps are defined. If the matching domain were split into several components, the argument would apply separately and would not by itself bound the number of components. Domain connectedness must be supplied by the common-section/capture argument.

## 8. Checks and remaining scope

The [symbolic audit](../../benchmarks/hilbert16_singular_transport.py) verifies 15 identities: the reversed V field, limiting fast invariant, exact slow normalization coefficients, section inverse/expansion/orientation for each side, both moving physical heights, and (25). It rejects unresolved or nonzero residuals with explicit exceptions, including under optimized Python. Its normal and optimized runs each pass all 15 identities. This is a finite algebra audit, not a validated long-delay integration or a formal analytic proof.

Reproduce from the repository root:

```bash
uv run python benchmarks/hilbert16_singular_transport.py --output artifacts/hilbert16/singular_transport.json
uv run python -O benchmarks/hilbert16_singular_transport.py --output artifacts/hilbert16/singular_transport_optimized.json
```

Without `--output`, the benchmark writes beneath `$OMNIBIAS_SCRATCH`, defaulting to `artifacts/`. The [normal report](../../artifacts/hilbert16/singular_transport.json) and [optimized report](../../artifacts/hilbert16/singular_transport_optimized.json) are regenerable artifacts. The separate [common-section argument](HILBERT16-ENDPOINT-PASSAGE.md) supplies the regular-passage premises referenced above.

The nested choice is essential: choose a compact interior parameter/base sector; choose paper height h; choose rho sufficiently small for the regular curvature and section geometry; then choose nu sufficiently small depending on rho,h for both actual passages and the singular remainder. None of the arguments allow exchanging these limits without additional estimates.

Not covered: B->0, 1-bB->0, r->0, 4*lambda0+lambda1^2->0, C->1, other parameter blow-up charts, possible itineraries outside the common capture tube, cyclicity of the full nilpotent graphic, a global quadratic bound, arbitrary polynomial degree, or the algebraic part of Hilbert 16. No full-Hilbert conclusion follows.

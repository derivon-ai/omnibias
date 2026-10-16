# Quantitative neighborhoods for Gaussian optimizer paths

The [exact polyhedral path compiler](gaussian-polyhedral-path.md)
classifies signs for sufficiently small positive parameters. This
extension certifies a concrete interval and encloses the true analytic
face path there, including its Taylor remainder. Bias and target paths
remain the exact finite polynomials stored by that compiler.

A successful certificate for a `valid` path identifies the unique global
polyhedral optimizer on the certified interval. For an `invalid` path it
quantifies failure of the selected KKT representation; a negative
selected multiplier still need not rule out another representation using
omitted tight inequalities.

## A complex disk that includes bias variation

Use the selected affine face's projected centers \(a'_j\), tangent
projector \(P\), center diameter \(D\), and
\(q=D^2/(8\nu s)<1\). The physical heat age \(s\) remains fixed;
\(t\) is the parameter of the bias/target path. Set

\[
V(R)=\sum_{n\ge1}\|Pv_n\|_2R^n,\qquad
B(R)=\sum_{n\ge1}
 \left(\max_j b_{jn}-\min_j b_{jn}\right)R^n.
\]

Only the finitely many supplied polynomial coefficients enter these sums.
The coefficientwise ranges preserve common bias gauges exactly. For
complex \(|t|\le R\) and a displacement \(\|h\|_2\le r\), the
relative Gaussian phases span an arc of length at most

\[
\frac{Dr}{2\nu s}+B(R)=2\eta,\qquad
\eta=\frac{Dr}{4\nu s}+\frac{B(R)}2.
\]

The following sufficient inequalities are checked with rational
arithmetic and outward upper bounds for Euclidean norms:

\[
\boxed{\quad
\eta^2<1-q,\qquad
\kappa=\frac{q}{1-\eta^2}<1,\qquad
sV(R)+\frac{DB(R)}{4(1-\eta^2)}
 \le(1-\kappa)r.
\quad}
\]

An upper bound for \(D\) may be used in \(\eta\) and the forcing
term; \(q\) retains the exact squared-diameter formula. The bias forcing
term is essential: applying a fixed-bias radius theorem unchanged would
not control this joint path.

## The dimension-independent covariance estimate

Here is an explicit proof of the factor \(1/4\) in the contraction and
mixed-moment bounds. Absorb real logit shifts into positive normalized
weights \(w_j\), and write their remaining phases as \(\phi_j\),
with \(|\phi_j|=1\). Let

\[
S=\sum_jw_j\phi_j,\qquad
u_j=\sqrt{w_j},\qquad D_\phi=\operatorname{diag}(\phi_j),
\qquad v=D_\phi u.
\]

Rotating the common phase to center its arc gives
\(|S|\ge\cos\eta>0\). Define the oblique projection

\[
Q=I-\frac{uv^T}{S}.
\]

Transposes in this formula are ordinary transposes, while norms and
orthogonality use the Hermitian Euclidean inner product. For any complex
vector \(z=\alpha u+y\) with \(u^*y=0\),

\[
Qz=y-u\frac{v^Ty}{S},\qquad
\|Qz\|_2^2=\|y\|_2^2+\frac{|v^Ty|^2}{|S|^2}.
\]

Because \(\|u\|=\|v\|=1\) and \(v^Tu=S\), the functional
\(v^T\) restricted to \(u^\perp\) has norm
\(\sqrt{1-|S|^2}\). It follows that
\(\|Q\|_2\le1/|S|\), including the one-mode case where \(Q=0\).

Let \(X\) have real rows
\(\sqrt{w_j}(a'_j-\sum_kw_ka'_k)^T\). The covariance of the
complex normalized weights has the exact representation

\[
C_{\mathbb C}=\frac{X^TD_\phi QX}{S}.
\]

The real covariance \(X^TX\) has norm at most \(D^2/4\): every
real unit projection of the centers lies in an interval of length at most
\(D\), and its variance is at most that length squared divided by four.
This real symmetric operator has the same norm on complex vectors.
Therefore

\[
\boxed{\quad
\|C_{\mathbb C}\|_2
\le\frac{D^2}{4|S|^2}
\le\frac{D^2}{4\cos^2\eta}
\le\frac{D^2}{4(1-\eta^2)}.
\quad}
\]

For any real scalar mode values \(c_j\), replace the second matrix
\(X\) by the real vector
\(y_j=\sqrt{w_j}(c_j-\sum_kw_kc_k)\). Its norm is at most
\(\operatorname{range}(c)/2\), giving

\[
\|\operatorname{Cov}_{\mathbb C}(a',c)\|_2
\le\frac{D\operatorname{range}(c)}{4(1-\eta^2)}.
\]

Apply this separately to each real bias coefficient vector and multiply
by \(|t|^n\) before summing. A complex polynomial bias increment is
not itself a real scalar coefficient vector; treating it as one would
not justify the same variance bound.

## Holomorphic contraction and primal remainder

With \(\mu'\) the projected-center mean, use the fixed-point map

\[
T_t(h)=sP(v(t)-v_0)
 +\mu'(x_0+h,b(t))-\mu'(x_0,b_0).
\]

The covariance estimate bounds its spatial derivative by \(\kappa\).
At \(h=0\), integrate along the bias segment
\(b_0+\theta(b(t)-b_0)\), \(0\le\theta\le1\). The mixed
estimate bounds the resulting mean change by
\(DB(R)/[4(1-\eta^2)]\). The displayed forcing inequality thus makes
the closed spatial ball invariant. Banach contraction gives its unique
fixed point for every \(|t|\le R\). Uniform convergence of holomorphic
fixed-point iterates gives a holomorphic branch in the open disk.

Consequently \(\|x(t)-x_0\|_2\le r\), and Cauchy's coefficient
bound gives, for \(0\le\rho<R\),

\[
\left\|x(t)-\sum_{n=0}^Nx_nt^n\right\|_2
\le r\frac{(\rho/R)^{N+1}}{1-\rho/R},
\qquad |t|\le\rho.
\]

Taking circles of radius approaching \(R\) from below justifies this
bound even without assuming holomorphy beyond the closed disk. No complex
root search is involved.

## Original KKT tails and strict sign margins

Each sign quantity needs its own analytic bound; a primal remainder
alone does not control original normal multipliers. An omitted slack has
change bound

\[
M_i=\|PG_i^T\|_2r.
\]

For a selected multiplier, let \(\ell\) be its row of
\(L_E=(EE^T)^{-1}E\), and use the **original** centers. Since
\(L_Ex(t)\) is constant, integration of the mixed covariance along the
joint spatial/bias segment gives

\[
M_i=\sum_{n\ge1}|\ell v_n|R^n
 +\frac{\operatorname{range}_j(\ell a_j)\eta}
        {2s(1-\eta^2)}
\quad\text{with}\quad
|\lambda_i(t)-\lambda_i(0)|\le M_i.
\]

Along this segment, apply the mixed estimate to the projected spatial
increment and separately to every real bias coefficient row. Their
combined bound is
\(\operatorname{range}(\ell a)[Dr/(2\nu s)+B(R)]/[4s(1-\eta^2)]\),
which is the displayed term because the bracket equals \(2\eta\).
This also avoids bounding a large common normal offset that cancels
exactly; no real-range assumption is made about an aggregate complex
logit increment.

For either quantity, the remainder after order \(N\) is bounded by

\[
T_i(\rho)=M_i\frac{(\rho/R)^{N+1}}{1-\rho/R}.
\]

If \(c_k\ne0\) is its first nonzero coefficient, verify the strict
rational inequality

\[
\boxed{\quad
\sum_{n=k+1}^N|c_n|\rho^{n-k}
 +\frac{T_i(\rho)}{\rho^k}<|c_k|.
\quad}
\]

The left side bounds all later terms after division by \(t^k\),
uniformly for \(0<t\le\rho\), because each bound is nondecreasing
in positive \(t\). Thus the leading sign holds throughout that interval.
Structural zero identities need no sign margin; an unexplained zero
prefix remains inconclusive. A `valid` path requires every sign test to
be certified. For an `invalid` path, one certified negative sign suffices
to reject the selected KKT representation throughout the interval. Other
tests remain diagnostic and need not have certified signs there.

Identical projected centers make the primal inverse a polynomial despite
changing biases. Original normal multipliers can still vary with those
biases, so their denominator and tail obligations do not disappear.

## API and interpretation

`certify_gaussian_polyhedral_path_radius(path, *, parameter_radius,
spatial_radius, radius)` in
`omnibias.core.verified.gaussian_polyhedral_radius` accepts the exact
path artifact and rational radii \(R,r,\rho\). It checks the complex
disk and the sign margins required by the path's status. An inconclusive
path cannot receive this certificate. All three radii must be strictly
positive, with \(\rho<R\); entries are exact integers or `Fraction`
values, excluding booleans.

For a `valid` path, the resulting `GaussianPolyhedralPathRadius` certifies
the unique global optimizer on \(0\le t\le\rho\), including its
already-certified base point. For an `invalid` path, negative signs are
certified on \(0<t\le\rho\); the base equilibrium remains valid.
The same artifact's `.enclose(t)` returns an outward enclosure of the
true analytic face point for exact \(0\le t\le\rho\). It encloses
an optimizer only when the path is `valid`. `.tail_bound(t)` returns the
Euclidean Taylor error at that parameter. The record retains `path`,
`analytic`, `radius`, `sign_bounds`, `status`, `eventual_active_indices`,
and `certificate`. Its active-set description concerns \(0<t\le\rho\);
the base point can have additional tight inequalities.

Each sign-bound record retains the original `sign`, `change_bound`,
`polynomial_remainder_bound`, `scaled_tail_bound`, `margin`, and
`certified` flag. Unexplained zero prefixes have no polynomial remainder,
scaled tail or margin value. Structural zero identities have zero bounds
and are certified directly. A known leading coefficient is certified
exactly when its computed margin is positive. For an `invalid` result,
inspect these flags before interpreting any sign other than the certified
negative rejection evidence.

The standalone `certify_gaussian_bias_inverse_series(series, *,
parameter_radius, spatial_radius)` in
`omnibias.core.verified.gaussian_bias_analytic` returns the nested
`GaussianBiasSeriesCertificate` without polyhedral sign claims. It
retains the parameter and spatial radii, `diameter_bound`,
`phase_half_width_bound`, `complex_contraction_bound`,
`target_growth_bound`, `bias_growth_bound`, `forcing_bound`, and
`available_displacement`. Here the target growth is \(V(R)\), without
the age factor. Its `.tail_bound(rho)` accepts \(0\le\rho<R\), and
its `.enclose(t)` accepts exact real \(|t|<R\).

The supplied radii are sufficient proof parameters, not estimates of the
nearest singularity or largest active-set interval. Refusal does not show
that the optimizer changes face within the requested interval. There is
no automatic radius optimization, additional active-set search or
arbitrary-posterior support. The calculation inherits exact order,
arithmetic, storage and proof-dependency budgets; interval conversion at
evaluation is outward. Certificate sealing is not a formal proof-checker
verification.

## A certified interval for cubic contact

Use the scalar example from the path compiler: centers \(-1,1\),
\(\nu=s=1\), \(x\le0\), target \(v(t)=-t\), and biases
\((1/4,1/4+2t)\). The free branch's leading slack is
\(2t^3/3\); the boundary branch's leading selected multiplier is
\(-t^3/3\).

For the free branch, \(r=1/4\) and \(R=1/20\) give
\(\eta=7/40\), \(\kappa=800/1551\), and positive self-map
margin \(151/7755\). At order seven, both sign tests pass through
\(\rho=1/125\). Order three also suffices, with the smaller certified
choice \(\rho=1/100000\). These are sufficient intervals, not maximal
ones.

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_path import (
    compile_gaussian_polyhedral_path,
)
from omnibias.core.verified.gaussian_polyhedral_radius import (
    certify_gaussian_polyhedral_path_radius,
)

source = GaussianHeatMap(
    ((-1,), (1,)), (Fraction(1, 4), Fraction(1, 4)), viscosity=1, time=1,
)
base = restrict_gaussian_heat(source, (), ())
assert base is not None

def contact_path(active, order):
    face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), active)
    assert face is not None
    equilibrium = certify_gaussian_polyhedral_equilibrium(face, (0,), (0,))
    assert equilibrium is not None
    result = compile_gaussian_polyhedral_path(
        equilibrium,
        ((-1,),) + ((0,),) * (order - 1),
        ((0, 2),) + ((0, 0),) * (order - 1),
    )
    assert result is not None
    return result

free_path = contact_path((), 7)
bound_path = contact_path((0,), 7)
assert free_path.status == "valid" and bound_path.status == "invalid"
rho = Fraction(1, 125)
free_radius = certify_gaussian_polyhedral_path_radius(
    free_path, parameter_radius=Fraction(1, 20),
    spatial_radius=Fraction(1, 4), radius=rho,
)
bound_radius = certify_gaussian_polyhedral_path_radius(
    bound_path, parameter_radius=Fraction(1, 20),
    spatial_radius=Fraction(1, 4), radius=rho,
)
assert free_radius is not None and bound_radius is not None
assert free_radius.enclose(rho)[0].hi < 0
bound_point = bound_radius.enclose(rho)[0]
assert bound_point.lo <= 0 <= bound_point.hi

for active in ((), (0,)):
    low_order = certify_gaussian_polyhedral_path_radius(
        contact_path(active, 3), parameter_radius=Fraction(1, 20),
        spatial_radius=Fraction(1, 4), radius=Fraction(1, 100000),
    )
    assert low_order is not None
```

The free enclosure contains the true optimizer. The boundary enclosure
contains the analytic face point \(x=0\), whose selected multiplier is
certifiably negative for positive parameters in the interval.

## Prior work

The common-phase rotation that excludes partition zeros is the elementary
half-plane argument recorded in
[Sao, *Ghosts of Softmax*, Lemma 4.5 and Theorem 4.6](https://arxiv.org/html/2603.13552v1#S4.SS3).
Only that geometric fact is used here. The covariance estimate above,
holomorphic contraction and Cauchy bounds supply the additional
obligations for an inverse path and its KKT signs. Taylor differentiation
of implicit functions is established prior work, including
[Wagner, Walther and Schaefer](https://arxiv.org/abs/0912.2208).
This specialized certificate carries no claim of a new general theorem
or universal efficient optimizer.

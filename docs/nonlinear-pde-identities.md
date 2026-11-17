# Exact nonlinear PDE identities

Sparse exponential polynomials provide an exact representation that supports
pointwise multiplication as well as differentiation. Their rational quotients
can represent nonconstant Cole–Hopf fields and check coupled nonlinear PDEs by
coefficient arithmetic. This extends the kinds of identities represented in the
library; it does not supply a general nonlinear PDE solver or a new
linearization theorem.

The [profile quotient](profile-pde.md) represents differential operators acting
on a profile family. It is not an algebra of pointwise products of those
profiles. Here the input is a **concrete function**, represented explicitly as

\[
P(x)=\sum_{(\lambda,\alpha)}c_{\lambda,\alpha}
x^\alpha e^{\lambda\cdot x},
\qquad c_{\lambda,\alpha}\in\mathbb Q,
\quad \lambda\in\mathbb Q^d,
\quad \alpha\in\mathbb N^d.
\]

The sum is finite. Canonicalization combines equal frequency/multi-index pairs
and removes zero coefficients. Coefficients and frequencies use exact rational
arithmetic; no small coefficient is discarded by a numerical tolerance.

## Closure and exact coefficient comparison

Products add frequencies and polynomial powers:

\[
(x^\alpha e^{\lambda\cdot x})(x^\beta e^{\mu\cdot x})
=x^{\alpha+\beta}e^{(\lambda+\mu)\cdot x}.
\]

Differentiation has two contributions:

\[
\partial_i(x^\alpha e^{\lambda\cdot x})
=\alpha_i x^{\alpha-e_i}e^{\lambda\cdot x}
+\lambda_i x^\alpha e^{\lambda\cdot x}.
\]

The first term is omitted when \(\alpha_i=0\). These identities prove closure
under addition, multiplication and all finite partial derivatives.

Coefficient comparison is also necessary for an identity on a nonempty open
set. Group a putative zero as \(\sum_\lambda P_\lambda(x)e^{\lambda\cdot x}\).
Choose a direction \(v\) so that all \(\lambda\cdot v\) are distinct. For a fixed
\(\lambda\), apply

\[
\prod_{\mu\ne\lambda}
(D_v-\mu\cdot v)^{\deg P_\mu+1}.
\]

Every other exponential mode vanishes. On the remaining polynomial this acts
as a product of operators \(D_v+(\lambda-\mu)\cdot v\), times the nonzero
exponential. Each factor is invertible on polynomials of bounded degree:
\(D_v\) is nilpotent there and the added constant is nonzero. The remaining
polynomial therefore vanishes if and only if \(P_\lambda=0\). Repeating for all
frequencies proves independence and the exact zero criterion.

This reasoning concerns functional identities on open sets. Vanishing only on
a sampled trajectory, a boundary plane or a finite point set does not imply
that every coefficient vanishes.

## Rational fields retain their domains

For \(u=P/Q\), differentiation uses

\[
\partial_i u=\frac{(\partial_iP)Q-P(\partial_iQ)}{Q^2}.
\]

The numerator and denominator remain exponential polynomials. Addition,
multiplication and division are handled by the ordinary fraction identities.
An identically zero denominator or divisor is refused.

A rational field must retain all nonzero conditions introduced while building
it. Denote this finite set of guards by \(\Gamma\), and its regular domain by

\[
\Omega_\Gamma=\{x:g(x)\ne0\text{ for every }g\in\Gamma\}.
\]

Addition and multiplication inherit the guards of both operands.
Differentiation inherits the operand's guards. Reciprocation additionally
requires its numerator to be nonzero. Division inherits both operands' guards
and requires the divisor's numerator to be nonzero. Algebraic simplification
must preserve these conditions: \(P/P=1\) only where the original quotient was
defined. The same applies to multiplying a singular expression by zero or
taking its zeroth power.

Thus a zero numerator proves that the field is zero **on its regular domain**.
Cross multiplication proves equality on the common regular domain. Neither
claim establishes equality of partial-function domains or removes a pole. A
global PDE solution requires a separate nonvanishing argument for its guards.
For example, a nonempty sum of real exponentials with positive coefficients is
strictly positive everywhere. General signed denominators do not have this
automatic guarantee.

## Cole–Hopf with several rates

For positive rational \(\nu,c_j\) and rational \(k_j\), let

\[
\tau(x,t)=\sum_j c_j e^{k_jx+\nu k_j^2t},
\qquad u=-2\nu\frac{\tau_x}{\tau}.
\]

Every mode solves \(\tau_t=\nu\tau_{xx}\), and positivity gives
\(\tau>0\) everywhere. Direct differentiation gives the stronger residual
identity

\[
u_t+uu_x-\nu u_{xx}
=-2\nu\,\partial_x\left(\frac{\tau_t-\nu\tau_{xx}}{\tau}\right).
\]

Consequently the field solves viscous Burgers' equation globally. The converse
does not require \(\tau_t-\nu\tau_{xx}=0\): the quotient on the right may depend
only on time. That distinction matters when inferring an equation for \(\tau\)
from one for \(u\).

A useful nonconstant example is
\(\tau=1+e^{x+t}+e^{2x+4t}\), with \(\nu=1\). This uses different temporal
rates and is not one travelling-wave profile. The construction is the classical
[Cole–Hopf transformation](https://doi.org/10.1090/qam/42889), and multiple-front
families have established [prior work](https://doi.org/10.1016/j.amc.2007.02.003).

The example uses coordinate order \((t,x)\). All constants below are exact
integers; noninteger exact values can be supplied as `Fraction` objects.

```python
from omnibias.core.exp_polynomial import ExpPolynomial
from omnibias.core.rational_field import RationalField, discover_rational_relations

tau = (ExpPolynomial.constant(1, 2)
       + ExpPolynomial.exponential((1, 1))
       + ExpPolynomial.exponential((4, 2)))
u = RationalField(tau.differentiate(1).scale(-2), tau)
one = RationalField.from_polynomial(ExpPolynomial.constant(1, 2))
features = (u.differentiate(0), u*u.differentiate(1),
            u.differentiate(1, 2), u, one)
residual = features[0] + features[1] - features[2]
assert residual.is_zero()
assert all(g.known_global_sign() in (-1, 1)
           for g in residual.nonzero_conditions)

relations = discover_rational_relations((features,))
assert relations.rank == 4
assert len(relations.relations) == 1
a = relations.relations[0]
assert a[0] == a[1] == -a[2]
assert a[3] == a[4] == 0
```

The result contains every rational relation in this five-feature library. Its
one-dimensional nullspace identifies Burgers' equation up to nonzero scale.
The separate sign checks establish that these particular denominator guards
never vanish; `is_zero()` alone would not establish that fact.

The interval evaluator checks those same domain conditions before using an
algebraic zero:

```python
from omnibias.core.verified.exp_polynomial import evaluate_rational_field
from omnibias.core.verified.interval import Interval

enclosure = evaluate_rational_field(
    residual, (Interval(-1.0, 1.0), Interval(-2.0, 2.0)))
assert enclosure == Interval.point(0.0)
```

This exact zero with globally signed guards needs no transcendental backend.
Evaluating a nonzero exponential field requires the rigorous backend. General
guards must have interval enclosures excluding zero on the requested box;
failure is inconclusive and raises an exception. Finite-range overflow also
refuses evaluation. Exact identity checking and numerical enclosure remain
separate operations.

## A coupled system in two spatial dimensions

The same construction gives a meaningful system test. For
\(x\in\mathbb R^d\), choose

\[
\tau(x,t)=\sum_j c_j e^{k_j\cdot x+\nu\lVert k_j\rVert^2t},
\qquad U=-2\nu\frac{\nabla\tau}{\tau}.
\]

It satisfies

\[
\partial_tU_i+\sum_j U_j\partial_jU_i-\nu\sum_j\partial_{jj}U_i=0,
\qquad
\partial_iU_j-\partial_jU_i=0.
\]

To verify the nonlinear equation, write \(q=\log\tau\). Then
\((U\cdot\nabla)U=2\nu^2\nabla\lVert\nabla q\rVert^2\), and the vector
residual becomes

\[
-2\nu\nabla\left(
q_t-\nu\Delta q-\nu\lVert\nabla q\rVert^2\right)
=-2\nu\nabla\left(\frac{\tau_t-\nu\Delta\tau}{\tau}\right)=0.
\]

The curl identity follows from commuting mixed derivatives. The irrotational
condition is substantive: this construction does not parameterize every
solution of the vector Burgers system. See [*The Hopf–Cole transformation,
topological solitons and multiple fusion solutions for the n-dimensional
Burgers system*](https://doi.org/10.1016/j.physleta.2015.09.033).

With coordinates \((t,x,y)\), a nonfactorizing test is

\[
\tau=1+e^{x+t}+e^{2y+4t}+2e^{x+y+2t},
\qquad (u,v)=-2(\tau_x,\tau_y)/\tau.
\]

Both cross-advection terms \(v u_y\) and \(u v_x\) are nonzero. Checking
\(u_t+u u_x+v u_y-u_{xx}-u_{yy}=0\), the corresponding equation for \(v\), and
\(u_y-v_x=0\) therefore exercises genuine coupling rather than two independent
scalar equations.

```python
def velocity(tau):
    return (RationalField(tau.differentiate(1).scale(-2), tau),
            RationalField(tau.differentiate(2).scale(-2), tau))

def features_for(field, u, v):
    one = RationalField.from_polynomial(
        ExpPolynomial.constant(1, 3, budget=field.budget))
    return (field.differentiate(0), u*field.differentiate(1),
            v*field.differentiate(2), field.differentiate(1, 2),
            field.differentiate(2, 2), u, v, one)

def combine(features, coefficients):
    first = features[0]
    result = RationalField.from_polynomial(
        ExpPolynomial.constant(0, first.dimension, budget=first.budget))
    for field, coefficient in zip(features, coefficients, strict=True):
        result = result + field.scale(coefficient)
    return result

tau = (ExpPolynomial.constant(1, 3)
       + ExpPolynomial.exponential((1, 1, 0))
       + ExpPolynomial.exponential((4, 0, 2))
       + ExpPolynomial.exponential((2, 1, 1), 2))
u, v = velocity(tau)
rows = (features_for(u, u, v), features_for(v, u, v))
relations = discover_rational_relations(rows)
assert relations.rank == 7
assert len(relations.relations) == 1
coefficients = relations.relations[0]
assert all(combine(row, coefficients).is_zero() for row in rows)
assert (u.differentiate(2) - v.differentiate(1)).is_zero()
assert not (v*u.differentiate(2)).is_zero()
assert not (u*v.differentiate(1)).is_zero()

# Validate the same discovered coefficients on different rates and directions.
held_out_tau = (ExpPolynomial.constant(1, 3)
                + ExpPolynomial.exponential((5, 2, -1), 2)
                + ExpPolynomial.exponential((2, -1, 1), 3)
                + ExpPolynomial.exponential((10, 1, 3)))
held_out_u, held_out_v = velocity(held_out_tau)
for field in (held_out_u, held_out_v):
    row = features_for(field, held_out_u, held_out_v)
    assert combine(row, coefficients).is_zero()
```

The eight columns are \((w_t,uw_x,vw_y,w_{xx},w_{yy},u,v,1)\), with one family
for \(w=u\) and one for \(w=v\). The shared relation supplies both PDE rows.
The held-out fields use different spatial directions and temporal rates, and
the coefficients are applied without refitting. Positivity of both heat
potentials makes the constructed velocities regular everywhere.

Polynomial parameters can also be represented as extra coordinates. For
\((t,x,y,a,b,c)\), use
\(\tau=1+a e^{x+t}+b e^{2y+4t}+c e^{x+y+2t}\), with zero exponential frequency
in the three parameter coordinates. Differentiating only physical axes
\(t,x,y\) holds the amplitudes fixed. The regression checks both nonlinear PDE
rows and recovers the same rank-seven relation by exact coefficient arithmetic
in all six coordinates. This proves the identities for every amplitude choice
on its regular domain, without sampling or fitting amplitudes. It does not
prove global regularity for arbitrary amplitudes: negative choices can make
\(\tau\) vanish, so its global-sign test is inconclusive over all of
\(\mathbb R^6\). Positive amplitudes give a separate global positivity proof.

A second regression represents the parameters as exponential biases instead:

\[
\tau=1+e^{t+x+a}+e^{4t+2y+b}+e^{2t+x+y+c}.
\]

Each term is constructed with `ExpPolynomial.exponential` in the six
coordinates. Both coupled PDE identities and the same discovered relation now
hold for **all real biases** \(a,b,c\), and `tau.known_global_sign()` returns
`1` over all of \(\mathbb R^6\). Physical differentiation still holds the
biases fixed. Changing a bias changes a positive modal amplitude while
preserving the heat-dispersion relation between spatial and temporal weights.

More generally, with scores
\(s_j=k_j\cdot x+\nu\lVert k_j\rVert^2t+b_j\), the classical Cole–Hopf
readout has the softmax form

\[
U=-2\nu\,\frac{\nabla\tau}{\tau}
=-2\nu\sum_j k_j\frac{e^{s_j}}{\sum_\ell e^{s_\ell}}.
\]

Include the constant term as \(k_0=0\), \(b_0=0\). This gives an exact nonlinear
PDE solution family with free biases and fixed dispersion weights. Its
guarantee uses exponential modes and the logarithmic derivative readout; other
activations or arbitrary network architectures need their own identity proof.

## Exact discovery in a finite feature library

Suppose the proposed features are concrete rational fields
\(F_j=P_j/Q_j\), for example \(u_t\), \(uu_x\) and \(u_{xx}\). On their common
regular domain let \(D=\prod_jQ_j\). Then

\[
\sum_j a_jF_j=0
\quad\Longleftrightarrow\quad
\sum_j a_jP_j\prod_{k\ne j}Q_k=0.
\]

Expanding the right-hand side gives a finite rational matrix: a column for each
feature and a row for each frequency/multi-index pair. Its nullspace contains
exactly the rational coefficient relations in the supplied library. Stacking
the coefficient constraints for independently specified fields requires the
same relation to hold for every example. All denominator obligations remain
part of the result's interpretation. For nonlinear PDEs, linear combinations
of these solutions generally fail to be solutions; stacking imposes separate
identities on each supplied field.

`discover_rational_relations` returns `relations`, `rank`, `candidate_count`,
`constraint_count`, and the `nonzero_conditions` for each supplied family. Its
linear-algebra budget and the fields' exponential-polynomial budgets are
separate checks. Polynomial operations require matching dimensions and budget
values; resource refusal does not imply that an identity is false.

Cancellation does not remove domain obligations:

```python
denominator = (ExpPolynomial.exponential((1,))
               - ExpPolynomial.constant(1, 1))
singular = RationalField(ExpPolynomial.constant(1, 1), denominator)
zero = singular - singular
assert zero.is_zero()
assert zero.nonzero_conditions == singular.nonzero_conditions
assert (singular**0).nonzero_conditions == singular.nonzero_conditions
assert denominator.known_global_sign() is None
```

This zero expression still excludes \(x=0\), where its original denominator
vanishes. A `None` sign result is inconclusive; it does not itself prove a zero
exists. In this example, substitution at zero supplies that separate argument.

This is identification inside a finite library using complete structured
functions. It does not recover a unique physical law from arbitrary data. A
single field can satisfy additional identities, and an incomplete library can
miss the desired equation. Coefficients in a discovered PDE can be solved by
linear algebra because the features are fixed; simultaneously solving for
unknown solution parameters generally produces nonlinear algebraic equations.

## Limits and relation to earlier methods

For prescribed periodic initial velocities, the
[Fourier heat construction](periodic-burgers.md) supplies a complementary
representation with rigorous exponential-series errors. It supports scalar
trigonometric data and coupled potential vector data, including spatial slope
sign changes excluded by positive sums of real exponential modes.

For the positive Cole–Hopf family, a prescribed velocity also defines a
[convex inverse problem](inverse-pde-map.md). Exact affine rank and an interval
root enclosure can certify the unique spatial input at a fixed time, and an
entropy witness gives a global objective bound. This is a separate inverse
query within the established solution family.

The representation initially uses rational coefficients and rational linear
exponents. It does not automatically convert every annihilator/profile pair,
admit arbitrary transcendental constants, or handle nested activations. A shift
of an exponential can introduce an irrational amplitude, so exact closure under
arbitrary bias translations must not be inferred from derivative closure.
The symbolic bias coordinates above keep these exponentials explicit; they do
not replace \(e^b\) with a rational constant.

Pointwise multiplication can grow the support, and a fixed finite profile space
cannot contain all such products. Indeed, a finite-dimensional unital algebra
of continuous real functions on a connected
domain contains only constants: the powers of any member \(f\) satisfy a
nonzero polynomial relation \(p(f)=0\), while a nonconstant continuous image
would contain an interval, contradicting the finite root set of \(p\). Exact
nonconstant multiplication therefore cannot be contained in a fixed finite
profile space. Term counts, arithmetic work and rational coefficient sizes need
explicit resource limits.

Rational exponential ansatz methods and coefficient matching have extensive
[prior art](https://doi.org/10.1016/j.chaos.2006.03.020). Their use can produce
invalid or redundant claims without residual and domain checks; see
[*Be careful with the Exp-function method*](https://arxiv.org/abs/1011.4265).
The reusable capability here is exact, bounded identity checking with retained
domain conditions. It provides neither general existence and uniqueness nor a
solution for arbitrary initial or boundary conditions, and it does not issue a
Lean-checked proof.

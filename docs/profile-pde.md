# Profile algebra for heat, Laplace, forcing, and PDE identification

The [arbitrary-profile ridge construction](bias-collapse-pde.md) has a necessary
limitation: heat and Laplace operators usually have no nonzero polynomial
multiplier that annihilates every scalar activation. Specifying an ODE for the
activation gives a larger useful class of PDE solutions. This page derives and
implements that extension, including forced equations and repeated roots.

The algebra uses established constant-coefficient differential-operator theory.
It is not a new general existence theorem. The broader exponential-polynomial
approach appears in [Ehrenpreis–Palamodov-based PDE
models](https://arxiv.org/abs/2212.14319). Here the reusable contribution is exact
coefficient construction combined with trainable forward evaluation, without
numerically finding or separating characteristic roots.

## An exact profile-dependent criterion

Let a scalar profile satisfy the monic, constant-coefficient ODE

\[
R(D_z)\phi=0,\qquad R(s)=s^m+\sum_{j=0}^{m-1}r_js^j,\quad m\geq1.
\]

The compiler takes rational coefficients for \(A\), \(R\), and \(F\), and
constructs rational coefficients for \(Q\). Choose rational vectors \(a,w\),
with \(w\ne0\). Consider

\[
u(x)=e^{a\cdot x}Q(D_z)\phi(z),\qquad z=w\cdot x+b.
\]

By the product and chain rules,

\[
D_i\bigl(e^{a\cdot x}\psi(z)\bigr)
=e^{a\cdot x}(a_i+w_iD_z)\psi(z).
\]

Thus, for a constant-coefficient PDE system \(A(D)\) and a vector forcing
polynomial \(F\),

\[
A(D)u=e^{a\cdot x}F(D_z)\phi(z)
\quad\text{for every }\phi\in\ker R(D_z)
\]

holds exactly when

\[
\boxed{A(a+ws)Q(s)\equiv F(s)\pmod{R(s)}.}
\]

**Proof.** Divide each component of \(A(a+ws)Q-F\) by the monic polynomial
\(R\). The divisible part acts as zero on \(\phi\). The remainder is a
linear combination of \(\phi,\ldots,\phi^{(m-1)}\). At any fixed argument,
these initial derivatives can be prescribed arbitrarily for the order-\(m\)
linear ODE. Therefore the remainder vanishes for all admissible profiles if and
only if every coefficient is zero. Repeated roots do not alter this argument.

Every operator class has a unique representative of degree below \(m\).
Coefficient matching therefore gives a rational affine system with
\(m\times\text{equations}\) rows and \(m\times\text{field components}\)
columns. The compiler returns a particular pack and the whole homogeneous
nullspace. Inconsistency excludes only the declared carrier, direction and
profile representation of the forcing.

This quotient concerns **differential operators acting on profiles**. It is
not an algebra of pointwise profile products: two functions in \(\ker R(D)\)
need not have a product in the same space. Nonlinear PDEs cannot be added by
silently multiplying these operator coefficient vectors.

## Heat modes that the arbitrary-profile compiler excludes

For \(A=\partial_t-\partial_x^2\), choose
\(a=(-1,0)\), \(w=(0,1)\) and \(R=s^2+1\). Then
\(A(a+ws)=-1-s^2\equiv0\). All fields
\(e^{-t}(c_0\cos x+c_1\sin x)\) solve heat.

```python
from fractions import Fraction
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.profile_system import ProfileSystem

heat = OperatorSpec.polynomial({(1, 0): 1, (0, 2): -1})
heat_system = ProfileSystem.from_operators(
    ((heat,),), (0, 1), carrier=(-1, 0), annihilator=(1, 0, 1)
)
family = heat_system.solve()
assert family.consistent and family.rank == 0
assert len(family.homogeneous_basis) == 2
heat_pack = family.homogeneous_basis[0]  # Q(s)=1
assert heat_system.residual_coefficients(heat_pack) == ((Fraction(0),),) * 2
```

The evaluator represents the complete profile state
\(y=(\phi,\phi',\ldots,\phi^{(m-1)})\) through its companion ODE

\[
y'=C_Ry,\qquad y(z)=\exp(zC_R)y(0),
\]

where \(C_R\) has ones on the superdiagonal and last row
\((-r_0,\ldots,-r_{m-1})\). Runtime `initial_jet` entries are raw derivatives
at zero, not Taylor coefficients. They and the scalar bias are trainable.

```python
import torch
from omnibias.torch.profile_system import profile_solution

tx = torch.tensor([[0.1, 0.3], [0.4, 0.7]], dtype=torch.float64)
initial_jet = torch.tensor([0., 1.], dtype=tx.dtype, requires_grad=True)
values = profile_solution(tx, heat_pack, initial_jet)
expected = torch.exp(-tx[:, 0]) * torch.sin(tx[:, 1])
torch.testing.assert_close(values[:, 0], expected, atol=2e-14, rtol=2e-13)
values.square().sum().backward()
assert initial_jet.grad is not None and torch.isfinite(initial_jet.grad).all()
```

The matching JAX entry point is `omnibias.jax.profile_system.profile_solution`.
Keep the pack as static metadata or close over it in a compiled function;
coordinates, initial derivatives and bias remain runtime arguments. The
evaluators currently support real float32 and float64. The companion state
retains oscillatory and repeated-root profiles using real arithmetic.
Trainable initial derivatives, biases and mixture weights must be independent
of the spatial coordinates; making them functions of the query point adds
product-rule terms that the compiled PDE identity does not include.

One particular solution plus a homogeneous correction also solves a forced
initial-boundary problem. For
\(u_t-u_{xx}=e^{-2t}\sin x\), \(u(0,x)=0\), and
\(u(t,0)=u(t,\pi)=0\), the compiler constructs

\[
u(t,x)=(e^{-t}-e^{-2t})\sin x.
\]

```python
forced_heat = ProfileSystem.from_operators(
    ((heat,),), (0, 1), carrier=(-2, 0), annihilator=(1, 0, 1)
).solve(((1,),))
assert forced_heat.consistent and forced_heat.particular is not None
solution = (
    profile_solution(tx, forced_heat.particular, [0., 1.])
    + profile_solution(tx, heat_pack, [0., 1.])
)[:, 0]
expected = (torch.exp(-tx[:, 0]) - torch.exp(-2*tx[:, 0])) * torch.sin(tx[:, 1])
torch.testing.assert_close(solution, expected, atol=2e-14, rtol=2e-13)
```

Here the sine factor enforces both spatial boundaries, and the two carriers
cancel at time zero. Those analytic constraints, together with the exact PDE
coefficient identities, establish the construction; finite grid residuals are
additional numerical regressions. The boundary tests also construct the
harmonic field \(\sinh(x)\sin(y)/\sinh(1)\) on
\([0,1]\times[0,\pi]\) by combining opposite exponential carriers.

## Repeated roots supply a resonant Poisson solution

For Laplace in coordinates \((x,y)\), take \(a=(1,0)\), \(w=(0,1)\).
The restricted operator is \(1+s^2\). With only \(R=1+s^2\), every
represented field is harmonic, so nonzero forcing is impossible in that space.
For \(R=(1+s^2)^2\), polynomial-times-trigonometric profiles become available.

Choose initial derivatives \((0,0,0,1)\), giving

\[
\phi(y)=\frac{\sin y-y\cos y}{2},\qquad
(D_y^2+1)\phi=\sin y.
\]

Then \(u=e^x\phi(y)\) satisfies \(\Delta u=e^x\sin y\), with
\(u(x,0)=u_y(x,0)=0\).

```python
poisson = ProfileSystem.from_operators(
    ((OperatorSpec.laplacian(),),), (0, 1), carrier=(1, 0),
    annihilator=(1, 0, 2, 0, 1),
)
forcing = ((1,), (0,), (1,), (0,))  # (1+D_y**2)phi, not phi itself
forced = poisson.solve(forcing)
assert forced.consistent and forced.particular is not None
assert poisson.residual_coefficients(forced.particular) == forcing
xy = torch.tensor([[0.2, 0.4], [-0.1, 0.7]], dtype=torch.float64)
u = profile_solution(xy, forced.particular, [0., 0., 0., 1.])[:, 0]
expected = torch.exp(xy[:, 0]) * (torch.sin(xy[:, 1]) - xy[:, 1]*torch.cos(xy[:, 1])) / 2
torch.testing.assert_close(u, expected, atol=2e-14, rtol=2e-13)
```

Enlarging the annihilator alone does not resolve all incompatible forcing. In
this example, asking for \(F=1\) remains inconsistent: the requested forcing
would be \(e^x\phi(y)\) for **every** profile in the larger space. The physical
forcing \(e^x\sin y\) is instead represented by \(F=1+s^2\) applied to the
chosen generalized profile. Keeping this distinction prevents a false resonance
claim.

```python
assert not poisson.solve(((1,),)).consistent
```

Changing `initial_jet` or bias in a homogeneous pack preserves the PDE. A
homogeneous correction may therefore use its own freely chosen admissible
profile and bias while keeping a fixed source unchanged. Changing the profile
or bias of the particular inhomogeneous pack generally changes that source.

## Exact candidate-operator identification

Given several profile packs and a library of candidate scalar operators acting
on chosen field components, `discover_profile_relations` substitutes each
candidate and stacks the exact remainder coefficients. Its rational nullspace
contains every linear relation in that library that vanishes for every
independently chosen admissible profile in the supplied family.

This is an exact identity search over represented functions. It does not sample
the PDE, infer an arbitrary black-box network's symbolic form, select the
sparsest equation, or establish the true physical law from observations.
Candidate libraries and the need for diverse trajectories are established in
[PDE-FIND](https://arxiv.org/abs/1609.06401).

```python
from omnibias.core.profile_discovery import discover_profile_relations

packs = []
for k in (1, 2):
    system = ProfileSystem.from_operators(
        ((heat,),), (0, k), carrier=(-k*k, 0), annihilator=(1, 0, 1)
    )
    packs.append(system.solve().homogeneous_basis[0])
library = (
    (0, OperatorSpec.mixed((0, 0))),
    (0, OperatorSpec.mixed((1, 0))),
    (0, OperatorSpec.mixed((0, 2))),
)
ambiguous = discover_profile_relations(packs[:1], library)
identified = discover_profile_relations(packs, library)
assert len(ambiguous.relations) == 2
assert identified.rank == 2 and len(identified.relations) == 1
identity_weight, time_weight, space_weight = identified.relations[0]
assert identity_weight == 0 and time_weight == -space_weight != 0
```

One heat frequency makes \(u_t\), \(u_{xx}\) and \(u\) proportional, so it
cannot identify a unique relation in this library. Two distinct frequencies
remove the spurious zeroth-order relation. Tests validate the recovered heat
operator on a third frequency absent from construction. That held-out success
does not imply extrapolation to every unrelated model.

## Scope and numerical cost

`RidgeBudget` limits exact construction, quotient degree, matrix storage,
elimination work and coefficient growth. No numerical rank threshold is used.
The dense tensor matrix exponential uses quadratic state storage per point and
can lose accuracy for ill-conditioned companion matrices or large arguments.
The compiler's host-algebra budget does not bound backend batch memory or
reverse-mode tapes. No performance advantage over specialized Fourier or
closed-form solvers is claimed here.

PDE and forcing identities are exact over the rational metadata. Matrix
exponential evaluation and coordinate-to-tensor conversion are floating-point
operations, not certified enclosures. Manually constructed packs require an
independent residual check before being treated as PDE solutions.

Finding directions, carriers, annihilators and boundary-compatible combinations
remains part of the problem. These primitives expand the representable class;
they do not prove completeness for arbitrary domains, variable coefficients or
nonlinear systems, and do not establish a previously unknown PDE theorem.

For concrete functions with products and quotients, see
[exact nonlinear PDE identities](nonlinear-pde-identities.md). That representation
supports coupled nonlinear identities and retains the nonzero conditions of
every original denominator.

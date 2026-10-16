# Bias collapse and exact PDE solution families

Normalized bias collapse extracts activation derivatives. For a **linear,
constant-coefficient, homogeneous PDE system**, those derivatives can be combined
so that the equations hold identically. The new `ridge_system` primitives compute
these combinations using exact rational linear algebra, then evaluate them in a
Torch or JAX forward pass. Initial and boundary conditions remain separate
constraints.

This is an implementation of differential syzygies specialized to ridge profiles.
It is not a claim of a new general PDE theorem. Polynomial syzygies and arbitrary
potentials already appear in [Härkönen, Hirsch and Sturmfels, *Making Waves*,
Section 3](https://arxiv.org/abs/2111.14045). Broader solution representations underlie
[Härkönen, Lange-Hegermann and Raiţă, *Gaussian Process Priors for Systems of Linear
Partial Differential Equations with Constant Coefficients*](https://arxiv.org/abs/2212.14319).
The contribution here is a bounded compiler integrated with omnibias's raw
activation derivatives, including coupled equations of different derivative orders.
For composed networks, see [deep collapse and the chain rule](deep-bias-collapse.md).
For heat, Laplace and forcing, see [profile algebra and PDE identification](profile-pde.md).

## What actually collapses

For a continuous activation, the unnormalized expression

\[
\phi(z+b_1)-\phi(z+b_2)\longrightarrow 0
\qquad (b_1\longrightarrow b_2).
\]

It does not become a function supported on a hyperplane. For differentiable
activations the normalized difference converges to \(\phi'(z+b)\). More generally,
for \(r\geq1\), the centered difference

\[
C_{r,h}\phi(z)=\frac{1}{h^r}\sum_{j=0}^r
(-1)^{r-j}\binom rj\phi\bigl(z+(j-r/2)h\bigr)
\longrightarrow\phi^{(r)}(z).
\]

Thus \(r+1\) biases extract derivative order \(r\) **with these weights and
normalization**. The number of biases alone does not establish this limit.

Repeated application of the fundamental theorem of calculus gives the useful
error formula

\[
C_{r,h}\phi(z)=\mathbb E\left[\phi^{(r)}
\left(z+h\sum_{i=1}^r U_i\right)\right],
\qquad U_i\sim\operatorname{Uniform}[-1/2,1/2]
\]

with independent \(U_i\). Since their sum has mean zero and variance \(r/12\),
Taylor's theorem yields

\[
|C_{r,h}\phi(z)-\phi^{(r)}(z)|
\leq \frac{rh^2}{24}
\sup_{|s-z|\leq r|h|/2}|\phi^{(r+2)}(s)|.
\]

This bound assumes \(\phi\in C^{r+2}\) on that interval and concerns exact
arithmetic. Floating-point cancellation grows as the bias spacing shrinks. The
implementation below evaluates the collapsed derivative formulas directly,
without choosing a tiny bias spacing.

## The mapping from bias derivatives to a PDE system

Let \(u\) have \(n\) components and let a system of \(m\) equations be

\[
A(D)u=0,\qquad A(D)=\sum_\alpha A_\alpha D^\alpha,
\qquad A_\alpha\in\mathbb Q^{m\times n}.
\]

Here \(p\) is the maximum derivative order in the original PDE, before any
cancellation on restriction. The stored `RidgeSystem.order` can be smaller;
it does not by itself specify the smoothness needed to evaluate the original PDE.

For a fixed nonzero normal \(w\in\mathbb Q^d\), define

\[
B_k(w)=\sum_{|\alpha|=k}A_\alpha w^\alpha,
\qquad A_w(s)=\sum_{k=0}^{p}B_k(w)s^k.
\]

Take a vector polynomial \(Q(s)=\sum_{j=0}^{r}q_js^j\). It specifies a vector
of collapsed neurons sharing the same normal and bias:

\[
u_{Q,\phi,b}(x)=\sum_{j=0}^{r}q_j\phi^{(j)}(w\cdot x+b).
\]

**Exact criterion.** This field solves the system for every
\(\phi\in C^{p+r}(\mathbb R)\) and every bias \(b\) if and only if

\[
A_w(s)Q(s)=0.
\]

**Proof.** The chain rule gives
\(D^\alpha\phi^{(j)}(w\cdot x+b)=w^\alpha\phi^{(j+|\alpha|)}(w\cdot x+b)\).
Consequently,

\[
A(D)u_{Q,\phi,b}=
\sum_{\ell=0}^{p+r}
\underbrace{\left(\sum_{k+j=\ell}B_k(w)q_j\right)}_{R_\ell}
\phi^{(\ell)}(w\cdot x+b).
\]

If all \(R_\ell\) vanish, the residual vanishes. Conversely, fix an argument
\(z_0\) and choose \(\phi(z)=(z-z_0)^\ell/\ell!\). Its finite derivative
vector at \(z_0\) isolates \(R_\ell\), so validity for every smooth profile
forces each coefficient to vanish. This proof requires finite smoothness,
not analyticity.

The compiler constructs the finite matrix
\(T_{(\ell,e),(j,c)}=B_{\ell-j}(w)_{e,c}\), interpreting missing coefficients
as zero, and returns its rational nullspace. This is **all vector polynomials
of degree at most the requested bound** in the right kernel. It is a vector-space
basis, not a minimal generating basis over the polynomial ring: increasing the
bound can return derivative shifts of earlier solutions.

With restricted order \(p_w\), this matrix has \(m(p_w+r+1)\) rows and
\(n(r+1)\) columns. Once the normal has been substituted, its size does not
grow with the spatial dimension. Dense rational elimination can still be
expensive, especially when coefficient numerators and denominators grow.
Finding suitable normals remains a separate problem.

`RidgeBudget` limits orders, matrix allocation, elimination work and coefficient
bit growth. Refusal is distinct from an empty kernel. Compact Laplacian powers
are restricted using \((\sum_i w_i^2)^q s^{2q}\), without expanding a mixed jet.

## A coupled wave system

Use coordinates \((t,x)\) and fields displacement \(u\), velocity \(v\) and
strain \(r\):

\[
v-u_t=0,\qquad r-u_x=0,\qquad v_t-4r_x=0.
\]

For \(w=(2,1)\),

\[
A_w(s)=\begin{pmatrix}-2s&1&0\\-s&0&1\\0&2s&-4s\end{pmatrix},
\qquad Q(s)=\begin{pmatrix}1\\2s\\s\end{pmatrix}.
\]

There is no nonzero constant vector in this polynomial kernel. Allowing one
bias derivative produces
\((u,v,r)=(\phi,2\phi',\phi')(2t+x+b)\), which satisfies all three equations
for any sufficiently smooth activation. This is a precise sense in which
activation substitution preserves the equations; it does not preserve the
values of the fields or their boundary data.

```python
from fractions import Fraction
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_system import RidgeSystem

identity = OperatorSpec.mixed((0, 0))
dt = OperatorSpec.mixed((1, 0))
wave = (
    (OperatorSpec.polynomial({(1, 0): -1}), identity, None),
    (OperatorSpec.polynomial({(0, 1): -1}), None, identity),
    (None, dt, OperatorSpec.polynomial({(0, 1): -4})),
)
system = RidgeSystem.from_operators(wave, (2, 1))
assert system.kernel_basis(max_degree=0) == ()
pack, = system.kernel_basis(max_degree=1)
assert pack.coefficients == (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(2), Fraction(1)),
)
assert all(value == 0 for row in system.residual_coefficients(pack) for value in row)
# A noncharacteristic direction has no arbitrary-profile polynomial pack.
assert RidgeSystem.from_operators(wave, (3, 1)).kernel_basis(max_degree=3) == ()
```

Evaluate the field in a forward pass. Biases and scalar mixture coefficients
remain trainable; the rational normal is fixed by the algebraic construction.
The backend module has the same `ridge_solution` API in Torch and JAX.
It supports `sigmoid`, `tanh`, `softplus`, `gaussian`, `exp`, `sin`, `cos`, `sinh`
and `cosh`. Manually constructing a `RidgeSyzygy` validates its metadata only;
check its `residual_coefficients` against the system before treating it as a
solution. The tensor evaluator does not independently verify PDE membership.

```python
import torch
from omnibias.torch.ridge_system import ridge_solution

points = torch.tensor([[0.1, 0.2], [0.3, 0.7]], dtype=torch.float64)
bias = torch.tensor(0.2, dtype=points.dtype, requires_grad=True)
field = ridge_solution(points, pack, bias, activation="tanh")
assert field.shape == (2, 3)
field.square().sum().backward()
assert bias.grad is not None and torch.isfinite(bias.grad)
```

## Solving initial and boundary data in a supported family

A superposition of these atoms still satisfies the PDE. Opposite wave directions
with sine profiles can also enforce \(u(t,0)=u(t,\pi)=0\) and \(v(0,x)=0\)
identically. Only the initial displacement amplitudes then need to be fitted.
The following small manufactured example recovers

\[
u(t,x)=\sin(x)\cos(2t)+\tfrac13\sin(3x)\cos(6t).
\]

It solves the stated initial-boundary problem because these data lie in the
chosen finite span. There is no claim that an arbitrary boundary problem lies
in such a span, or that this improves on a Fourier solver for this example.

```python
def standing_feature(points, frequency):
    right, = RidgeSystem.from_operators(
        wave, (2 * frequency, frequency)
    ).kernel_basis(max_degree=1)
    left, = RidgeSystem.from_operators(
        wave, (-2 * frequency, frequency)
    ).kernel_basis(max_degree=1)
    # Normalize displacement amplitude independently of nullspace pivot choices.
    right_scale = float(right.coefficients[0][0])
    left_scale = float(left.coefficients[0][0])
    return 0.5 * (
        ridge_solution(points, right, activation="sin") / right_scale
        + ridge_solution(points, left, activation="sin") / left_scale
    )

xs = torch.linspace(0, torch.pi, 17, dtype=torch.float64)
initial_points = torch.stack((torch.zeros_like(xs), xs), dim=-1)
frequencies = (1, 3)
design = torch.stack(
    [standing_feature(initial_points, k)[:, 0] for k in frequencies], dim=-1
)
initial_displacement = torch.sin(xs) + torch.sin(3 * xs) / 3
amplitudes = torch.linalg.lstsq(design, initial_displacement).solution
assert torch.allclose(amplitudes, xs.new_tensor([1, 1/3]), atol=2e-14, rtol=0)

ts, grid_xs = torch.meshgrid(
    torch.linspace(0, 1, 9, dtype=xs.dtype), xs, indexing="ij"
)
grid = torch.stack((ts, grid_xs), dim=-1)
features = torch.stack([standing_feature(grid, k) for k in frequencies], dim=-1)
solution = features @ amplitudes
expected = torch.sin(grid_xs) * torch.cos(2 * ts)
expected += torch.sin(3 * grid_xs) * torch.cos(6 * ts) / 3
assert torch.max(torch.abs(solution[..., 0] - expected)) < 2e-14
assert torch.max(torch.abs(solution[:, [0, -1], 0])) < 2e-14
assert torch.max(torch.abs(solution[0, :, 1])) < 2e-14
```

Independent tests differentiate the resulting field to check all three PDE
residuals, compare with analytic fields, and test Torch/JAX values and parameter
gradients. The rational compiler proves finite coefficient identities; it is not
a Lean proof or a floating-point enclosure. Tensor conversion and derivative
evaluation introduce ordinary numerical error.

## Limits and other consequences

- **An empty kernel is a restricted conclusion.** For heat,
  \(\partial_t-\partial_x^2\) at \(w=(1,1)\) reduces to \(s-s^2\).
  Its polynomial right kernel is zero although \(e^{t+x}\) solves the PDE.
  Exponential-specific modes, polynomial profiles and complex characteristics
  require different searches; the compiler does not report them as impossible.
- **Elliptic equations expose the limitation of real normals.** The Laplacian
  reduces to \(\|w\|^2s^2\). For nonzero real \(w\) it has no nonzero
  polynomial multiplier annihilating every profile, although affine profiles
  are harmonic. Intersecting hyperplanes alone cannot supply general harmonic
  solutions.
- **Training must preserve the constraint.** Bias changes and scalar sums of
  constrained atoms preserve the system. Arbitrary changes to normals,
  componentwise readout matrices or nonlinear composition generally do not.
  Directions and operator coefficients currently accept rational real values;
  complex and floating-point rank decisions are outside this API.
- **Finite spacing is a separate approximation.** Replacing each derivative in
  a pack by its own finite-difference estimate usually breaks exact cancellation.
  Bias translations or differences of the entire constrained vector field do
  preserve the PDE, because translation in bias commutes with the operator.
- **Integration commutes under regularity assumptions.** A finite bias integral
  of constrained atoms remains constrained when differentiation under the
  integral is justified. In particular,
  \(\int_a^b\phi'(w\cdot x+s)\,ds=\phi(w\cdot x+b)-\phi(w\cdot x+a)\).
  This does not construct an inverse PDE operator or a universally faster optimizer.
- **Hyperplane intersections need rank conditions.** A square affine system
  \(Mx+c=0\) has the differentiable solution \(x=-M^{-1}c\) while \(M\)
  stays invertible. Inconsistent systems have no intersection, underdetermined
  systems need a selection rule, and rank changes can destroy smooth dependence.
  Differentiable constraint geometry alone does not establish PDE solvability.
- **Dirac localization needs a different limit.** For sigmoid,
  \(\beta\sigma'(\beta z)\to\delta(z)\) distributionally as
  \(\beta\to\infty\). Normalized bias collapse alone gives \(\sigma'(z)\).
  ReLU's first derivative is a step and its second derivative is a Dirac
  distribution; discarding that singular term is not classical differentiation.
  For nonzero \(w\), \(\delta(w\cdot x+b)\) integrates a test function over
  the hyperplane with factor \(1/\|w\|\). Distributional fields require a
  different evaluator from the smooth tensor API here.

Variable coefficients, nonlinear PDEs, finding suitable normals, completeness,
boundary solvability, conditioning and performance against established solvers
remain additional problems. The executable result establishes a specific exact
solution-family construction, not that previously unsolved PDEs become trivial.

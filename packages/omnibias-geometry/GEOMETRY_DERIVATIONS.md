# omnibias-geometry derivations

Math, index conventions, and numerical notes for the differential-geometry
operators. All quantities are computed in float64; cross-backend parity is
asserted to `rtol=1e-9` (the autodiff-of-metric path is exact for analytic
metrics but not bit-identical to the tighter `1e-12` used by the pure
sigma-tower ops).

## Index conventions

- `christoffel` returns `Gamma[..., k, i, j] = Gamma^k_{ij}`.
- `riemann_tensor` returns `R[..., rho, sigma, mu, nu] = R^rho_{sigma mu nu}`.
- `ricci_tensor` returns `R[..., sigma, nu] = R_{sigma nu}`.
- metric derivative array `dg[..., i, j, k] = d g_ij / d x_k`.

## Metric, inverse, volume element

The metric is supplied as a per-point callable `g_point(x): (d,) -> (d, d)`. The
batched metric is `vmap(g_point)`; the inverse is a closed-form linear solve
`g^{ij} = (g_{ij})^{-1}`; the volume element is `sqrt(|det g|)`.

## Metric derivatives (autodiff, exact for analytic metrics)

`dg = vmap(jacfwd(g_point))`. For an analytic `g_point`, forward-mode autodiff
returns the exact partial derivatives (to machine precision). This is the only
non-closed-form ingredient; the docstrings and the README label it as such.

## Christoffel symbols

\[
    \Gamma^k_{ij} = \tfrac12\,g^{kl}\big(\partial_i g_{lj}
        + \partial_j g_{li} - \partial_l g_{ij}\big).
\]

In code the bracket `partial_i g_lj + partial_j g_li - partial_l g_ij` is built
from `dg` by index permutation and contracted with `g^{kl}` (an einsum).

## Curvature

\[
    R^\rho{}_{\sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma}
        - \partial_\nu \Gamma^\rho_{\mu\sigma}
        + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}
        - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma},
\]
\[
    R_{\sigma\nu} = R^\rho{}_{\sigma\rho\nu}, \qquad
    R = g^{\sigma\nu} R_{\sigma\nu}.
\]

`partial Gamma` is obtained by nesting `jacfwd` over the per-point Christoffel
function (autodiff of the analytic metric, twice). Validated on the round
2-sphere: `R = 2/R^2`, `Ricci = g/R^2`, Gaussian curvature `K = 1/R^2`.

## Covariant derivative

\[
    \nabla_i f = \partial_i f, \quad
    \nabla_i V^k = \partial_i V^k + \Gamma^k_{il} V^l, \quad
    \nabla_i \omega_k = \partial_i \omega_k - \Gamma^l_{ik}\omega_l.
\]

The partial derivatives `partial_i V^k` come from the closed-form field gradient.

## Laplace-Beltrami operator

Using the identity
\[
    \Delta_g f = \tfrac{1}{\sqrt{|g|}}\,\partial_i\!\big(\sqrt{|g|}\,g^{ij}\,\partial_j f\big)
              = g^{ij}\big(\partial_i\partial_j f - \Gamma^k_{ij}\,\partial_k f\big),
\]
the field Hessian `partial_i partial_j f` and gradient `partial_k f` are exact
closed forms; the Christoffel correction uses the metric. For the flat Euclidean
metric `Gamma = 0` and `g^{ij} = delta^{ij}`, so `Delta_g f` reduces to the
ordinary Laplacian (a regression test asserts exact agreement). On the unit
sphere, `Delta_{S^2} cos(theta) = -2 cos(theta)` (the `l=1` eigenvalue), verified
to `atol=1e-9`.

## Geodesics

The geodesic equation `d^2 x^k/dt^2 + Gamma^k_{ij} (dx^i/dt)(dx^j/dt) = 0` is
exposed as the RHS `a^k = -Gamma^k_{ij} v^i v^j` for an ODE integrator. Sanity:
equatorial motion on the sphere has zero `theta`-acceleration.

## Pullback metric (learned charts)

An immersion (chart) \(\varphi:\mathbb{R}^d\to\mathbb{R}^n\), \(d\le n\), induces a
Riemannian metric on its domain by *pulling back* the ambient metric \(h\):

\[
    g_{ab}(x) = \sum_{i,j} h_{ij}\big(\varphi(x)\big)\,
        \frac{\partial \varphi^i}{\partial x^a}\,
        \frac{\partial \varphi^j}{\partial x^b}
    \;=\; J^\top h\, J, \qquad
    J = \frac{\partial \varphi}{\partial x}\in\mathbb{R}^{n\times d}.
\]

With the Euclidean ambient metric \(h = I_n\) this is simply \(g = J^\top J\). The
Jacobian \(J\) is obtained by forward-mode autodiff of `phi`, so the metric is
exact for analytic and neural-network charts alike (it is autodiff of the chart,
not a finite difference). `metric_spec_from_chart` wraps \(g\) into a `MetricSpec`,
so the connection and curvature operators above consume it unchanged -- they only
read `manifold.metric.g_point`, which makes "learned manifolds" a one-primitive
unlock. The Christoffel symbols then nest a second `jacfwd` over `g_point`, i.e. a
`jacfwd` of `jacfwd(phi)`.

Validated against the canonical unit-sphere embedding
\(\varphi(\theta,\phi) = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)\),
which reproduces the round metric \(\operatorname{diag}(1,\sin^2\theta)\), scalar
curvature \(R = 2\), and -- for a constant linear chart \(\varphi(x)=Ax\) -- the
flat SPD metric \(g = A^\top A\).

## General relativity: Einstein tensor and curvature invariants

Built directly on the curvature tensors above (same `jacfwd`-of-metric path, so
still autodiff-exact for analytic metrics). Index conventions extend the ones
above; the fully-lowered Riemann tensor is
`lowered_riemann` \(R_{\rho\sigma\mu\nu} = g_{\rho a} R^a{}_{\sigma\mu\nu}\).

- **Einstein tensor** \(G_{\mu\nu} = R_{\mu\nu} - \tfrac12 R\,g_{\mu\nu}\)
  (`einstein_tensor`). Symmetric; in \(d=2\) it is identically zero; its trace
  obeys \(g^{\mu\nu}G_{\mu\nu} = \tfrac{2-d}{2}R\), an independent contraction
  check back to `scalar_curvature`.
- **Einstein field-equation residual**
  \(G_{\mu\nu} + \Lambda g_{\mu\nu} - \kappa T_{\mu\nu}\)
  (`einstein_equation_residual`). `stress_energy=None` is the vacuum case; the
  cosmological constant \(\Lambda\) and coupling \(\kappa\) are keyword scalars.
- **Kretschmann scalar** \(K = R_{\rho\sigma\mu\nu}R^{\rho\sigma\mu\nu}\)
  (`kretschmann_scalar`). Raised with four inverse metrics. For a
  constant-curvature \(d\)-space \(K = 2d(d-1)/R^4\); in \(d=2\) it collapses to
  \(K = R^2\); Schwarzschild gives \(K = 48 M^2 / r^6\), finite where the metric
  components diverge -- the invariant that certifies a real (not coordinate)
  singularity.
- **Weyl conformal tensor**
  \(C_{abcd} = R_{abcd} - \tfrac{1}{d-2}(g_{ac}R_{bd}-g_{ad}R_{bc}-g_{bc}R_{ad}+g_{bd}R_{ac})
  + \tfrac{R}{(d-1)(d-2)}(g_{ac}g_{bd}-g_{ad}g_{bc})\) (`weyl_tensor`,
  \(d\ge 3\); zeros for \(d<3\)). Totally trace-free; vanishes on conformally
  flat / maximally symmetric spaces; equals the lowered Riemann tensor in vacuum.
- **Geodesic deviation (Jacobi) acceleration**
  \(a^\rho = -R^\rho{}_{\sigma\mu\nu}\,u^\sigma \xi^\mu u^\nu\)
  (`geodesic_deviation`), the tidal relative acceleration of neighbouring
  geodesics with tangent `u` and separation `xi`.

Validated against Schwarzschild (\(G=0\), \(K=48M^2/r^6\)), de Sitter FRW
(\(G_{00}=3H^2\), \(G+\Lambda g=0\) at \(\Lambda=3H^2\)), the round \(S^3\)
(\(G=-(1/R^2)g\), \(C=0\), \(K=12/R^4\)), the 2D identities \(G\equiv 0\) and
\(K=R^2\) on a non-trivial conformal metric, a `sympy` symbolic \(S^3\) reference,
and the contracted Bianchi identity \(\nabla^\mu G_{\mu\nu}=0\) by finite
differences on a non-Einstein FRW metric. **Honesty:** autodiff-exact (like the
rest of the package); no numerical-relativity / ADM 3+1 evolution is claimed.

## de Rham topology: Hodge Laplacian, Betti numbers, degree, Gauss-Bonnet

The de Rham slice of algebraic topology on the closed-form substrate.

### Hodge-de Rham Laplacian on a `k`-form

The Hodge Laplacian is \(\Delta = d\delta + \delta d\). On a **0-form** it is the
scalar Laplacian \(\Delta f = \delta d f = -\Delta_g f\) (`hodge_laplacian`
delegates to `hodge_laplacian_scalar`, valid on curved manifolds). On a
**`k \ge 1`-form** with a *constant* (flat / Cartesian, \(\Gamma^k_{ij}=0\))
metric the Hodge and Bochner Laplacians coincide and act componentwise,

\[
  (\Delta\omega)_{I} = -\,g^{mn}\,\partial_m\partial_n\,\omega_{I},
\]

which is exactly `hodge_laplacian_scalar` applied to each named component. This
is proved by expanding \(d\) and \(\delta\) with vanishing Christoffel symbols
(e.g. a flat 1-form on \(\mathbb R^2\): \((\Delta\omega)_1 = -(\partial_{xx}
+\partial_{yy})a\) after the \(d\delta\) and \(\delta d\) terms cancel the mixed
partials). A **curved** `k`-form Laplacian additionally carries the Weitzenböck
curvature term \(\Delta = \nabla^*\nabla + \mathrm{Ric}\)-type correction; this
is honestly a `NotImplementedError` (composing the name-based `d`/`δ` would
require re-naming the intermediate form). The constant-metric guard checks
`christoffel` — an independent cross-check of the connection.

### Betti numbers via harmonic forms (Hodge theorem)

By the Hodge theorem the `k`-th de Rham cohomology is isomorphic to the space of
harmonic `k`-forms \(\ker\Delta\). `hodge_laplacian_matrix` builds the Gram
matrix \(M_{ij} = \langle b_i, \Delta b_j\rangle_{L^2}\) in a finite Fourier form
basis (Euclidean \(L^2\) product over the quadrature domain), and
`betti_number` returns its nullity (singular values below a tolerance).
`harmonic_projection` returns the kernel component of a coefficient vector via
the symmetric eigendecomposition. On the **flat 2-torus** the harmonic `k`-forms
are the constant-coefficient forms, so \(b_k = \binom{d}{k}\) gives
\((b_0, b_1, b_2) = (1, 2, 1)\).

### Degree and winding

- **Winding number** of a circle map \(S^1\to S^1\):
  \(\deg = \tfrac{1}{2\pi}\oint \partial_{axis}\varphi\,dx\) (`winding_number`);
  for \(\varphi = q\,x\) it is exactly `q`.
- **Degree** of a map \(M^2\to S^2\): the normalised pullback of the target area
  form, \(\deg = \tfrac{1}{4\pi}\int_M n\cdot(\partial_0 n\times\partial_1 n)\,dx\)
  (`map_degree`). The identity
  \(n=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)\) integrates to `1`.

### Gauss-Bonnet Euler characteristic

For a closed surface, `gauss_bonnet_euler` computes
\(\chi = \tfrac{1}{2\pi}\int_M K\,dA\) with \(K = \tfrac12 R\) the Gaussian
curvature and \(dA = \sqrt{|g|}\,dx\), reusing `scalar_curvature` and
`sqrt_det_metric` — an independent tie-back to the curvature stack. The round
`S^2` integrates to \(\chi = 2\).

**Honesty.** Field-component derivatives are closed-form (sigma-tower); curvature
and \(\sqrt{|g|}\) are autodiff-exact; the Betti / degree / Euler integrals are
**numerical** (quadrature + nullity), certifiable by an
`omnibias.core.verified.Interval` enclosure that brackets exactly one integer
(the monopole/degree quantisation certificate). Combinatorial topology (homotopy
groups \(\pi_n\), persistent homology / TDA, simplicial \(\mathbb Z\)-homology,
Smith normal form) is **out of thesis** — see `docs/scope-and-guarantees.md`.

## Surface integration of differential forms

Integrating a `k`-form over a `k`-dimensional parametrized submanifold reduces to a
pullback plus a quadrature. For an immersion \(\varphi:\mathbb R^d\to\mathbb R^n\)
with Jacobian \(J = \partial\varphi/\partial x\in\mathbb R^{n\times d}\), the
pullback of a `k`-form has components (over strictly increasing ambient index sets)

\[
    (\varphi^*\omega)_{a_1\dots a_k}
        = \sum_{i_1<\dots<i_k}
          \omega_{i_1\dots i_k}\!\big(\varphi(x)\big)\,
          \det\!\Big(\frac{\partial\varphi^{i_p}}{\partial x^{a_q}}\Big)_{p,q},
\]

and the integral is the change-of-variables identity
\(\int_M\omega=\int_{\text{box}}\varphi^*\omega\), taking the top-degree component
\(a=(0,\dots,d-1)\) when `k = d`. Each \(k\times k\) minor determinant is expanded
by the **Leibniz permutation sum** \(\det = \sum_\pi \operatorname{sgn}(\pi)\prod_p
J_{i_p,\,a_{\pi(p)}}\) in pure Python (`pullback_form_components`), so only
`+`/`*`/indexing touch the tensors and torch / jax agree bit-for-bit. The metric
volume element is \(\sqrt{|\det g|}\) with the pullback metric \(g=J^\top hJ\), and
`surface_integral` integrates \(\int_M f\,dA = \sum_q w_q\,f(y_q)\sqrt{|\det g(x_q)|}\).

**Honesty.** The integrand is exact (closed-form field derivatives + exact
forward-mode chart Jacobian); the integral is Gauss-Legendre **quadrature** — exact
for polynomial integrands up to the rule degree, convergent otherwise. Validated by:
the unit-square forms \(dx\wedge dy\to 1\), \(x\,dx\wedge dy\to\tfrac12\); the closed
1-form circulation \(\oint(-y\,dx+x\,dy)=2\pi R^2\); the unit-sphere area \(4\pi\)
(and \(\int_{S^2}1\,dA=\text{area}\)); and **Green's theorem** as a Stokes self-test
— with \(\omega=-y\,dx+x\,dy\), \(d\omega=2\,dx\wedge dy\), the boundary line integral
(`integrate_form`) equals the interior area integral of \(d\omega\)
(`exterior_derivative` then `integrate_form_values`), both \(=2\pi R^2\).

## References

- do Carmo, *Riemannian Geometry*.
- Lee, *Introduction to Smooth Manifolds*.
- Wald, *General Relativity*, appendix on the curvature tensor index conventions.
- Misner, Thorne & Wheeler, *Gravitation* (Einstein / Weyl / Kretschmann).
- Bott & Tu, *Differential Forms in Algebraic Topology* (de Rham, degree,
  Chern-Weil).
- Warner, *Foundations of Differentiable Manifolds and Lie Groups* (Hodge theory).

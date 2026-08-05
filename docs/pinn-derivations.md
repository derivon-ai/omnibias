# omnibias-pinn v0.1 — Math derivations

This note collects the four derivations that v0.1 ships in its
implementation: 3D vector-potential incompressibility (with the
Coulomb-gauge constraint), skew-symmetric advection (the kernel for the
``EnergyConserving`` / ``EnstrophyConserving`` cages), Sobolev
preconditioning generalised away from the 2D NS special case, and the
``p``-Laplacian via composition of the K=2 and K=3 omnibias bias-collapse
units.

Notation. Inner products and divergences are over a periodic box
$\Omega = [0, L]^3$ unless stated otherwise. The shorthand $\partial_i =
\partial / \partial x_i$, the Einstein summation convention applies, and
$\nabla \cdot$ / $\nabla \times$ / $\Delta$ are the standard 3D
divergence / curl / Laplacian. Tensor fields use
$(B, C, D, ...)$ shape conventions consistent with the implementation.

## 1. 3D vector potential and the Coulomb gauge

### 1.1 The setup

Let $u: \Omega \to \mathbb{R}^3$ be a velocity field. We want a typed
neural-network architecture whose output is *strictly* divergence-free,
i.e.
$$\nabla \cdot u(x, t) \equiv 0, \quad \forall (x, t).$$

The classical hardware-level construction is the vector potential:
introduce a vector field $A: \Omega \to \mathbb{R}^3$ and define
$$u := \nabla \times A. \tag{1}$$

Then because $\nabla \cdot (\nabla \times A) = \partial_i \epsilon_{ijk}
\partial_j A_k = \epsilon_{ijk} \partial_i \partial_j A_k = 0$ (the
contraction of a symmetric and antisymmetric pair vanishes), $u$ is
divergence-free *exactly*, modulo floating-point round-off.

### 1.2 Closed-form $u$ from the omnibias one-layer field

Let $A$ be parameterised as a 3-output one-layer omnibias field:
$$A_a(x, t) = b_a + \sum_h c_{ah}\, \sigma\!\left(W_{h\bullet}\cdot
(x, t) + \beta_h\right), \quad a = 1, 2, 3.$$

Pre-activations $z_h = W_{hi} x_i + W_{h, D+1} t + \beta_h$, written
compactly as $W_h \cdot \xi + \beta_h$ where $\xi = (x_1, x_2, x_3, t)$.
For a Riccati-class $\sigma$, every order $n$ derivative is closed-form
via the omnibias fast-path: $\sigma^{(n)}(z)$ is one polynomial in
$\sigma(z)$.

Then
$$\partial_j A_a = \sum_h c_{ah}\, \sigma'(z_h)\, W_{hj},$$
so
$$u_i = (\nabla \times A)_i = \epsilon_{ijk}\, \partial_j A_k =
\epsilon_{ijk}\, \sum_h c_{kh}\, W_{hj}\, \sigma'(z_h). \tag{2}$$

This is the closed-form expression every backend's curl op uses; it
needs only $\sigma'$, never $\sigma''$ or higher. Every additional
derivative of $u$ is one more order of $\sigma$:
$$\partial_l u_i = \epsilon_{ijk}\, \sum_h c_{kh}\, W_{hj}\, W_{hl}\,
\sigma''(z_h),$$
and the Laplacian of $u$ is
$$(\Delta u)_i = \epsilon_{ijk}\, \sum_h c_{kh}\, W_{hj}\, |W_h|^2\,
\sigma'''(z_h),$$
where $|W_h|^2 = \sum_l W_{hl}^2$ contracts over the spatial axes only
(the time slot does not appear in the spatial Laplacian).

### 1.3 The gauge ambiguity and the Coulomb fix

The vector potential is *not* unique: $A$ and $A + \nabla \chi$ produce
the same $u$ for any scalar $\chi$. This is the gauge freedom of (1).
For PINN training the freedom is harmless, but it inflates the
parameter space and makes optimisation slower. The standard cure is the
**Coulomb gauge**:
$$\nabla \cdot A = 0. \tag{3}$$

Adding (3) as an auxiliary penalty
$$\mathcal{L}_\text{gauge} = \lambda_\text{gauge}\, \langle (\nabla \cdot
A)^2 \rangle$$
removes the freedom *and* leaves $\nabla \cdot u \equiv 0$ untouched.
The closed-form expression for $\nabla \cdot A$ is one Riccati pass:
$$\nabla \cdot A = \sum_a \partial_a A_a = \sum_h \sigma'(z_h)\,
\sum_a c_{ah}\, W_{ha}. \tag{4}$$

Numerical sanity check (from the implementation tests): for a freshly
initialised 3D ``OneLayerVectorField`` with random $W, c$, the residual
$(\nabla \cdot u)^2 \langle \rangle \le 10^{-25}$ in float64, and the
gauge residual $(\nabla \cdot A)^2$ is $O(1)$ until the auxiliary
penalty is trained against.

### 1.4 Spectral basis

For ``SpectralVectorField`` the vector potential lives in the same
$(2K+1)^3$ Fourier basis. Curl is one *signed shift* per axis (the
$\sin/\cos$ blocks swap), so equation (1) is closed-form in coefficient
space with cost $O(K^3)$. The Coulomb gauge becomes
$$(i k_x)\hat A_x + (i k_y)\hat A_y + (i k_z)\hat A_z = 0$$
which can be enforced by projecting $\hat A$ onto the plane normal to
$k$ at every collocation point — a one-line per-mode operation.

Either route yields $\nabla \cdot u = 0$ at machine precision. The
implementation calls this branch via
``cage.IncompressibleProjection(backend="vector_potential", gauge="coulomb")``.

## 2. Skew-symmetric advection

### 2.1 Three equivalent forms

The advection of a velocity field $u$ has three standard formulations:
$$
\begin{aligned}
\text{(advective)} && (u \cdot \nabla) u_i &= u_j \partial_j u_i, \\
\text{(divergence)} && \partial_j (u_j u_i) &= u_j \partial_j u_i + u_i
\partial_j u_j, \\
\text{(skew-symmetric)} && \tfrac{1}{2}\!\left[(u \cdot \nabla) u_i +
\partial_j (u_j u_i)\right] &= u_j \partial_j u_i + \tfrac{1}{2} u_i
\partial_j u_j.
\end{aligned}
$$
For *exactly* divergence-free $u$ (i.e. when ``IncompressibleProjection``
is in the loop) the three are identical. For a $u$ produced by an
unconstrained network the three give *different* discrete derivatives,
and only the skew-symmetric one preserves the energy
$E = \tfrac{1}{2} \int |u|^2$ to bit precision.

### 2.2 Energy-conservation proof

Let $A_\text{ss}(u)_i = u_j \partial_j u_i + \tfrac{1}{2} u_i
\partial_j u_j$ be the skew-symmetric form. Compute the energy rate
under the inviscid, unforced equation $\partial_t u = -A_\text{ss}(u)$:
$$
\frac{dE}{dt} = \int u_i \partial_t u_i = -\int u_i A_\text{ss}(u)_i
= -\int u_i u_j \partial_j u_i - \tfrac{1}{2} \int u_i u_i \partial_j
u_j.
$$
Use $u_i u_j \partial_j u_i = \tfrac{1}{2}\, u_j \partial_j (u_i u_i)$
on the first term and integrate by parts (periodic $\Omega$):
$$
\int u_j \partial_j (u_i u_i) = -\int (u_i u_i)\, \partial_j u_j.
$$
Substituting back:
$$
\frac{dE}{dt} = \tfrac{1}{2} \int (u_i u_i)\, \partial_j u_j -
\tfrac{1}{2} \int (u_i u_i)\, \partial_j u_j = 0.
$$

The cancellation is *symbolic*: it does not need $\nabla \cdot u = 0$.
That is the statement "skew-symmetric advection conserves energy
identically, even off the divergence-free manifold". The two other
forms each contribute a non-vanishing term $\pm \tfrac{1}{2} \int u_i
u_i \partial_j u_j$ which only cancels modulo $\nabla \cdot u$.

### 2.3 Closed-form on the omnibias field

For ``OneLayerVectorField`` with components $u_a = b_a + \sum_h c_{ah}
\sigma(z_h)$:
$$
\partial_j u_a = \sum_h c_{ah}\, \sigma'(z_h)\, W_{hj},
$$
$$
A_\text{ss}(u)_i = u_j(x)\, \partial_j u_i + \tfrac{1}{2} u_i(x)\,
\partial_j u_j,
$$
which is one $(B, C)$ tensor combining $u$ (closed form via
$\sigma$), $\partial u$ (closed form via $\sigma'$, plus the $W$
contraction), and a single contraction over the spatial axes. The cost
is $O(BHD)$ — same as the vanilla advective form — so the
energy-conservation property is *free*.

The implementation puts this behind ``cage.EnergyConserving(field)``
which wraps the field's evaluation so that any subsequent
``state.velocity.advect()`` call returns $A_\text{ss}(u)$ instead of
$(u \cdot \nabla) u$. The ``EnstrophyConserving`` variant is the same
trick on the vorticity equation.

## 3. Sobolev preconditioning, generalised

### 3.1 The 2D NS special case (lifted from the existing solver)

The 2D NS vorticity-streamfunction residual is
$$R(x, t) = \omega_t + \psi_y \omega_x - \psi_x \omega_y -
\nu\, \Delta \omega - f_\omega,$$
and the existing ``omnibias_spectral_pinn`` solver replaces the naive
MSE $\|R\|^2$ with a *Sobolev-preconditioned* loss
$$\mathcal{L}_\text{Sob}(R) = \frac{1}{(2K+1)^2}\, \sum_{k}
\frac{|\hat R(k)|^2}{(1 + k_\text{stiff}^2)^p},$$
where $\hat R$ is the spatial Fourier transform of $R$ and
$k_\text{stiff}^2 = (k_x^2 + k_y^2)^2$. The denominator
*decreases* the weight of high-wavenumber modes, which absorbs the
spectral-bias direction along which the residual is hardest to
minimise.

### 3.2 The general statement

The Sobolev preconditioner is **equation-agnostic**: it depends only on
the *stiffness operator* $S$ associated with the residual, not on the
residual's specific form. For a general PDE residual
$R = \mathcal{N}(u)$ where $\mathcal{N}$ has linear part $L$, the
preconditioned loss is
$$\mathcal{L}_p(R) = \langle R,\, (1 + S)^{-p}\, R \rangle, \tag{5}$$
where $S$ is the *positive-semidefinite* part of $L^* L$ and $\langle
\cdot, \cdot \rangle$ is the chosen inner product (typically $L^2$ on
the periodic box, computed via Parseval = a per-mode reduction).

### 3.3 Why this is the right thing to do

Suppose $R$'s spectrum is dominated by high-stiffness modes, e.g.
biharmonic-class equations have $S \sim (k^2)^2$. Then ordinary MSE
fails to penalise the residual proportionally to the mode's *physical
importance* (a mode with high $S$ value drives instability); the
Sobolev-$p$ weight $(1+S)^{-p}$ is precisely the inverse-stiffness
projector that flattens the residual landscape, so SGD spends its
gradient steps where they matter.

### 3.4 Per-equation $S$ tables

| equation | linear part $L$ | stiffness $S = L^* L / k_\text{ref}^2$ | recommended $p$ |
|---|---|---|---|
| Heat | $\partial_t - \alpha \Delta$ | $(\alpha k^2)^2$ | 0.5 |
| Burgers | $\partial_t + u \partial_x - \nu \partial_x^2$ | $\nu^2 k^4$ | 0.5 |
| 2D NS (vort.-stream) | $\omega_t - \nu \Delta\omega$ | $\nu^2 k^4$ | 1.0 |
| 3D NS (primitive) | $\partial_t - \mu \Delta$ (per-component) | $\mu^2 k^4$ | 1.0 |
| KS | $\partial_t + \partial_x^2 + \partial_x^4$ | $(k^2 + k^4)^2$ | 1.0 |
| CH | $\partial_t + \Delta(\Delta - 1)$ | $((k^2)^2 + k^2)^2$ | 1.0 |
| Biharmonic | $\Delta^2$ | $(k^2)^4$ | 0.5 |

For $p = 0$ the preconditioner is identity (vanilla MSE). For $p > 0$
high-frequency modes are downweighted; for $p < 0$ they are
*upweighted*, which is appropriate when training a low-frequency
initialisation that needs to grow into the high-frequency physics.

### 3.5 Implementation contract

The function ``omnibias.pinn.torch.losses.sobolev_residual_loss(residual, *,
L, sobolev_p, spatial_axes=None)`` receives:

- ``residual``: the residual tensor of shape ``(..., n_1, ..., n_D)`` where
  the *trailing* axes are spatial and the leading axis is time / batch.
- ``L``: the period of the spatial domain -- a scalar broadcasts, a tuple is
  per spatial axis.
- ``sobolev_p``: the Sobolev exponent.
- ``spatial_axes``: which axes are spatial (i.e. periodic and FFT-reducible);
  defaults to "all but axis 0".

It returns
```text
mean_{t, b} sum_k |R_hat(k, t)|^2 / (1 + |k|^4)**sobolev_p,
```
so the stiffness is the canonical biharmonic $S(k) = |k|^4$ inferred from the
residual's *axes only* -- the linear part of the PDE is not introspected. The
companion ``sobolev_weight(...)`` returns the per-mode weight
$1 / (1 + |k|^4)^p$ alone, which is the hook for a non-Riccati /
non-spectral stiffness: build the weight yourself and apply it in Fourier
space. By Parseval, ``sobolev_p = 0`` is exactly plain MSE.

## 4. p-Laplacian via composition of K=2 and K=3 collapses

### 4.1 The operator

The p-Laplacian is
$$\Delta_p u := \nabla \cdot \left(|\nabla u|^{p - 2}\, \nabla u\right),
\quad p \ge 1.$$

For $p = 2$ it reduces to the ordinary Laplacian; for $p \ne 2$ it is
quasilinear. The omnibias closed-form derivative tower handles the
linear pieces; the nonlinear factor $|\nabla u|^{p-2}$ has to be folded
through carefully.

### 4.2 Closed form via composition

Expand the divergence:
$$\Delta_p u = |\nabla u|^{p-2} \Delta u + (p - 2)\, |\nabla u|^{p-4}\,
(\nabla u)_i (\nabla u)_j (\nabla\nabla u)_{ij}, \tag{6}$$
where $(\nabla\nabla u)_{ij} = \partial_i \partial_j u$ is the Hessian.

Each of the three factors $\nabla u$, $\Delta u$, $\nabla \nabla u$ is
closed-form on the omnibias field via the standard fast-path. The
Hessian–gradient–gradient triple product is one $(B, D, D)$ matrix
multiplied by two $(B, D)$ vectors:
$$g_i g_j H_{ij} = g^T H g.$$

So the kernel is, schematically (single sample, single component):
<!-- docs-test: skip reason="function-body fragment shown schematically" -->
```python
g  = ops.gradient(state, "u")          # (D,)
H  = ops.hessian(state, "u")           # (D, D)
L  = ops.laplacian(state, "u")         # scalar
g2 = (g * g).sum()
gHg = (g.unsqueeze(0) @ H @ g.unsqueeze(-1)).squeeze()
return g2.pow(0.5*(p-2)) * L + (p-2) * g2.pow(0.5*(p-4)) * gHg
```

### 4.3 Why "composition of K=2 and K=3 collapses"

The omnibias bias-collapse units of order $K$ produce closed-form
derivatives up to order $K - 1$. The Laplacian uses $K = 2$
($\sigma''$); the Hessian uses $K = 2$ ($\sigma''$); the
gradient-of-gradient inside a chain rule needs $K = 3$ ($\sigma'''$)
when composed through a *parameterised activation gate* like the
``ReLU^p`$ patch. For pure Riccati activations ($\tanh$, $\sigma$,
softplus, gaussian, exp) the K=2 collapse is sufficient because the
Hessian is already closed-form via $\sigma''$.

The IDEAS_BACKLOG C5 entry conjectured that p-Laplacian could be
realised by composing two K=2 collapses; the proof is (6) — the second
$|\nabla u|^{p-4}$ factor is a scalar gate, not a new derivative order,
so K=2 is enough for any $p \ge 1$.

### 4.4 Numerical regularisation near $|\nabla u| = 0$

Equation (6) has a removable singularity at $|\nabla u| = 0$ when
$p < 2$ (the $(p-2)$ exponent in the prefactor goes negative). The
implementation uses the standard trick: replace $|\nabla u|$ with
$\sqrt{|\nabla u|^2 + \epsilon^2}$ for a small $\epsilon$ (default
$10^{-8}$). This is exposed via ``ops.p_laplacian(state, name, p,
*, eps=1e-8)``.

### 4.5 Test contract

The implementation cross-checks:

- $p = 2$: $\Delta_2 u = \Delta u$ (returns ``ops.laplacian(state, name)``
  to bit precision).
- $p = 1$: total variation flow; check sign matches the known formula
  $\Delta_1 u = (1 / |\nabla u|) (\Delta u - g^T H g / |\nabla u|^2)$
  on a manufactured solution.
- General $p$: gradient-via-autograd cross-check on a small batch
  (``rtol=1e-9``).

## 5. Chebyshev-T basis for non-periodic domains

### 5.1 Why a second basis

The Fourier basis used by ``SpectralVectorField`` is the right tool for
spatially periodic problems (channel flow with periodic walls, the 2D
NS box, the CH lattice, etc). Many physical PINN problems instead live
on a non-periodic interval $[a, b]$ -- e.g. confined droplet
simulations, beam bending, domain-decomposed solvers stitching
multi-block geometries -- where a Fourier-cosine basis acquires a
Gibbs-like aliasing penalty at the boundary.

For these we ship ``ChebyshevVectorField``, which uses the
Chebyshev-T basis $\{T_n(\xi)\}_{n=0}^{K}$ on $\xi \in [-1, 1]$:
$$T_n(\cos\theta) = \cos(n\theta).$$

User inputs $x \in [a, b]$ are rescaled internally:
$$\xi = \frac{2(x - a)}{b - a} - 1, \quad
\frac{d}{dx} = \frac{2}{b - a}\,\frac{d}{d\xi},$$
so the chain-rule factor for an $m$-th derivative along the $d$-th
spatial axis is $\big(2 / (b_d - a_d)\big)^m$.

### 5.2 Closed-form derivatives

Differentiation in Chebyshev coefficient space is the linear map
$D \in \mathbb{R}^{(K+1) \times (K+1)}$ defined by
$$D[n, m] = \begin{cases}
    2m / c_n & \text{if } m > n \text{ and } (m + n) \text{ is odd},\\
    0 & \text{otherwise},
\end{cases}$$
with $c_0 = 2$ and $c_n = 1$ for $n \ge 1$. That is, if
$f(\xi) = \sum_n a_n T_n(\xi)$ then $f'(\xi) = \sum_n (Da)_n T_n(\xi)$.

Higher-order spatial derivatives use $D^k$. To evaluate at a point,
note that the *derivative basis* $\big(D^\top\big)^k\, T(\xi)$ produces a
length-$(K+1)$ vector whose dot product with the original coefficient
vector $a$ equals $f^{(k)}(\xi)$. The implementation pre-builds $D$
once as a buffer (torch) / leaf (jax), composes powers as needed, and
einsums the result into the multi-axis coefficient block exactly the
same way ``SpectralVectorField`` einsums Fourier derivative tables.

### 5.3 Multi-D Laplacian and biharmonic

Unlike the Fourier case, the Chebyshev Laplacian is *not* diagonal in
coefficient space (the second-derivative matrix $D^2$ is upper
triangular but not a scalar multiple of the identity). For the
biharmonic and higher polylaplacians, the implementation uses the
multinomial expansion
$$\Delta^k = \sum_{m_1 + \cdots + m_D = k} \binom{k}{m_1\,\cdots\,m_D}
\prod_d \frac{\partial^{2 m_d}}{\partial x_d^{2 m_d}},$$
and each pure spatial derivative is closed-form via $D^{2 m_d}$. This is
a $\binom{k + D - 1}{D - 1}$-term sum, $O(1)$ in $k$ for fixed $D$, and
each term is one einsum over a precomputed $D^{2 m_d}$. No autograd is
involved.

### 5.4 Test contract

- ``test_laplacian_matches_d2x_plus_d2y_2d`` (both backends): the v0.1
  ``Delta`` op equals $\partial_x^2 + \partial_y^2$ to ``rtol=1e-12``.
- ``test_biharmonic_via_multinomial_2d`` (torch, jax): the ``biharmonic``
  op equals ``polylaplacian(k=2)`` and the explicit $\partial_x^4 + 2
  \partial_x^2 \partial_y^2 + \partial_y^4$ expansion to ``rtol=1e-9``.
- ``test_chebyshev_parity.py`` (cross-backend, 5 activations × 3
  spatial dims): Torch and JAX agree to ``rtol=1e-12`` on values, first
  partials, gradients, divergence, Laplacian; ``rtol=1e-11`` on
  biharmonic and ``polylaplacian(k=2)``.

## 6. References to existing code

- Closed-form Laplacian on the one-layer field: ``omnibias.jax.laplacian.neural_field_laplacian``.
- Closed-form Hessian (used by Section 4): ``omnibias.jax.laplacian.neural_field_hessian``.
- Closed-form polylaplacian (Section 1.2 base case for $\Delta u$, $\Delta^2 u$): ``omnibias.jax.laplacian.neural_field_polylaplacian``.
- Spectral 2D NS solver that ships the original Sobolev preconditioner (Section 3): the archived spectral 2D Navier-Stokes solver.
- Spectral 2D field (used in Section 1.4): the archived spectral 2D Cahn-Hilliard field.
- Reference PINN architectures (used by Section 2.3): package architecture modules under ``omnibias.torch.architectures``.

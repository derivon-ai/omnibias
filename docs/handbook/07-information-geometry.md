# Chapter 7 — Information geometry & natural gradient

> Modules: `omnibias.jax.information` / `omnibias.torch.information`
> (exponential families & Fisher), `omnibias.curvature.natural_gradient`
> (Fisher-scoring optimisation).

## What this chapter gives you

The previous chapters put a metric on *space*. Information geometry puts a metric
on the space of **model parameters**: the Fisher information

\[
  F_{ij}(\theta) = \mathbb E\Bigl[\partial_i \ln p_\theta\,\partial_j \ln p_\theta\Bigr]
\]

is the natural Riemannian metric on a statistical manifold. Two payoffs:

1. For an **exponential family** \(p_\theta(x) \propto e^{\theta x - A(\theta)}\),
   the cumulants are just the derivative tower of the log-partition `A`: the mean
   is \(A'(\theta)\), the variance / Fisher information is \(A''(\theta)\). Since
   omnibias evaluates \(A^{(k)}\) in closed form, the whole cumulant tower is one
   activation-tower call.
2. The **natural gradient** \(\tilde\nabla = F^{-1}\nabla L\) takes steps that are
   invariant to reparametrisation and — with the canonical link — coincide with
   Fisher scoring / IRLS / Newton.

```
 log-partition A ─tower─▶ [A, A', A'', …] = [_, mean, Fisher, …]
                                  │
        loss grad ∇L ──────▶ damped_solve(F, ∇L) ──▶ natural_gradient_step
```

---

## Exponential families & Fisher (JAX / Torch)

`base` selects the log-partition `A` by activation name (default `"softplus"`,
the Bernoulli/logistic family). Examples in JAX; the Torch twin is identical.

### `exponential_family_cumulants`

<!-- docs-test: signature -->
```python
exponential_family_cumulants(theta, *, base="softplus", order=2) -> Array
```
The cumulant tower \([\kappa_0,\dots,\kappa_{\text{order}}]\) with
\(\kappa_k = A^{(k)}(\theta)\): \(\kappa_1\) mean, \(\kappa_2\) variance / Fisher,
\(\kappa_3\) skewness scale, … stacked on a leading axis.

```python
import jax.numpy as jnp
from omnibias.jax.information import exponential_family_cumulants
print([round(float(v), 4) for v in exponential_family_cumulants(0.0, order=3)])
# [0.6931, 0.5, 0.25, 0.0]  ->  A=ln2, mean=1/2, var=1/4, third cumulant 0 at θ=0
```

### `glm_mean`, `glm_variance`, `fisher_information`

<!-- docs-test: signature -->
```python
glm_mean(theta, *, base="softplus") -> Array          # κ1 = A'(θ)  (the link inverse)
glm_variance(theta, *, base="softplus") -> Array      # κ2 = A''(θ)
fisher_information(theta, *, base="softplus") -> Array # = glm_variance (1-D Fisher–Rao)
```
For a 1-parameter exponential family the variance *is* the Fisher information.
For Bernoulli (`softplus`), `glm_mean` is the logistic sigmoid and the Fisher
information is `μ(1−μ)`.

```python
import jax.numpy as jnp
from omnibias.jax.information import glm_mean, fisher_information
theta = jnp.array([-1.0, 0.0, 1.0])
print([round(float(m), 4) for m in glm_mean(theta)])            # sigmoid: [0.2689, 0.5, 0.7311]
print([round(float(f), 4) for f in fisher_information(theta)])  # μ(1-μ): [0.1966, 0.25, 0.1966]
```

### `moment_match` and `fit_natural_parameter`

<!-- docs-test: signature -->
```python
moment_match(mean, *, base="softplus", max_iter=100, tol=1e-12) -> Array
fit_natural_parameter(samples, *, base="softplus", dim=-1, ...) -> Array
```
`moment_match` inverts the link — the natural parameter `θ` with `A'(θ) = mean` —
by closed-form Fisher-scoring (Newton with the cumulant-ratio step). For an
exponential family this is the MLE (MLE ≡ moment matching). `fit_natural_parameter`
reduces `samples` to their mean and calls it.

```python
import jax.numpy as jnp
from omnibias.jax.information import moment_match, glm_mean, fit_natural_parameter
theta_hat = moment_match(jnp.array(0.6))             # logit(0.6)
print(round(float(theta_hat), 4), round(float(glm_mean(theta_hat)), 4))   # 0.4055 0.6
samples = jnp.array([0.0, 1.0, 1.0, 0.0, 1.0])       # Bernoulli draws, mean 0.6
print(round(float(fit_natural_parameter(samples)), 4))                    # 0.4055
```

---

## Natural-gradient optimisation

`omnibias.curvature.natural_gradient` wires the Fisher metric to parameter
updates. The two generic helpers work on *any* `(P, P)` Fisher and `(P,)`
gradient; the two GLM drivers specialise to the closed-form one-layer field
Fisher.

### `damped_solve`

<!-- docs-test: signature -->
```python
damped_solve(fisher, grad, *, damping=1e-3) -> Array
```
The natural-gradient direction \(\delta = (F + \lambda I)^{-1}\nabla L\)
(Tikhonov-regularised so it stays well-posed when `F` is singular).

```python
import jax.numpy as jnp
from omnibias.curvature.natural_gradient import damped_solve
F = jnp.array([[4.0, 1.0], [1.0, 3.0]])
g = jnp.array([1.0, 2.0])
print([round(float(v), 4) for v in damped_solve(F, g, damping=1e-6)])
```

### `natural_gradient_step`

<!-- docs-test: signature -->
```python
natural_gradient_step(theta, grad, fisher, *, learning_rate=1.0, damping=1e-3) -> Array
```
The preconditioned update \(\theta - \text{lr}\,(F+\lambda I)^{-1}\nabla L\) —
metric-aware gradient descent, invariant to smooth reparametrisations.

```python
import jax.numpy as jnp
from omnibias.curvature.natural_gradient import natural_gradient_step
theta = jnp.array([0.0, 0.0])
F = jnp.array([[4.0, 1.0], [1.0, 3.0]]); g = jnp.array([1.0, 2.0])
print([round(float(v), 4) for v in natural_gradient_step(theta, g, F, learning_rate=0.5)])
```

### `glm_loss_gradient`

<!-- docs-test: signature -->
```python
glm_loss_gradient(X, Y, W, beta, c, b, *, activation="tanh", family="bernoulli") -> Array
```
The flat `(P,)` gradient of the GLM negative log-likelihood for the one-layer
natural parameter \(\eta_n = b + \sigma(Wx_n+\beta)\cdot c\):
\(\tfrac1B\sum_n (A'(\eta_n) - y_n)\,g_n\), with `A'` the GLM mean and `g_n` the
closed-form per-sample parameter gradient — no autodiff backward pass.
Families: `"bernoulli"`, `"poisson"`, `"gaussian"`.

### `glm_natural_gradient_step`

<!-- docs-test: signature -->
```python
glm_natural_gradient_step(X, Y, W, beta, c, b, *, activation="tanh", family="bernoulli",
                          learning_rate=1.0, damping=1e-3) -> (b_new, c_new, beta_new, W_new)
```
One Fisher-scoring / IRLS step on a one-layer GLM field: updates `(b, c, β, W)` via
`θ ← θ − lr (F + λI)⁻¹ g` with the closed-form GLM Fisher and the GLM NLL
gradient. For `family="gaussian"` this is Gauss–Newton (and, at `damping=0`,
equals the one-layer MSE Newton step). Returns the parameters in their input
shapes.

```python
import jax.numpy as jnp
from omnibias.curvature.natural_gradient import glm_loss_gradient, glm_natural_gradient_step

rng = __import__("numpy").random.default_rng(0)
B, H, D = 32, 4, 3
X = jnp.asarray(rng.normal(size=(B, D)))
Y = jnp.asarray((rng.uniform(size=B) < 0.5).astype(float))      # Bernoulli labels
W = jnp.asarray(rng.normal(size=(H, D)) * 0.3)
beta = jnp.asarray(rng.normal(size=H) * 0.1)
c = jnp.asarray(rng.normal(size=H) * 0.3)
b = jnp.asarray(0.0)

g = glm_loss_gradient(X, Y, W, beta, c, b, family="bernoulli")
print("gradient length P =", int(g.shape[0]))                  # 1 + 2H + H*D = 21
b1, c1, beta1, W1 = glm_natural_gradient_step(X, Y, W, beta, c, b, family="bernoulli", damping=1e-2)
print(W1.shape, c1.shape, beta1.shape, b1.shape)               # (4,3) (4,) (4,) ()
```

!!! tip "Why natural gradient?"
    Vanilla gradient descent depends on how you happen to *parametrise* the model;
    the Fisher metric quotients that arbitrariness out. On a well-specified GLM the
    natural-gradient step is Newton's method, so it converges in far fewer
    iterations — and here the Fisher is closed form, so each step is cheap.

---

## You've finished the handbook

You now have the full closed-form stack:

- **Chapters 1–2** — the neural jet: exact derivatives → ODE/PDE discovery.
- **Chapters 3–4** — geometry: curvature on manifolds; the metric-free de Rham `d`.
- **Chapters 5–6** — probability between distributions: divergences and transport,
  differentiable *and* certified.
- **Chapter 7** — the geometry of the model itself: Fisher metric, natural gradient.

For precise signatures and parameter tables, every function above is rendered in
the [API reference](../api/symbolic.md). For the broader omnibias framework
(PyTorch / JAX / Keras backends, PINNs, gauge theory), see the package docs from
the top navigation.

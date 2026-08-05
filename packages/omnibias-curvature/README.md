# omnibias-curvature

Closed-form parameter Hessian, Gauss-Newton Fisher, and KFAC Kronecker
factors for one-hidden-layer Riccati fields. Built on the omnibias
fast-path activation kernels (`σ'`, `σ''`) — no autograd backward needed.

## v0.1 surface

For the one-layer scalar field

```
f(x; W, β, c, b) = b + σ(W·x + β) · c
```

with `σ ∈ {tanh, sigmoid, softplus, gaussian, exp}` (the Riccati class),
every block of the parameter Hessian / Fisher is algebraic. v0.1 ships:

- `one_layer_param_grad(x, W, β, c, b, activation)` — per-sample
  `∇θ f(x)` of length `P = 1 + H + H + H·D`.
- `one_layer_param_hessian(x, W, β, c, b, activation)` — per-sample
  full `(P, P)` Hessian; block-diagonal in the hidden index `h`.
- `mse_gauss_newton_fisher(X, Y, W, β, c, b, activation)` — `F = (2/B) Σ
  g_n g_n^T` and `∇L` for MSE loss.
- `mse_newton_step(X, Y, W, β, c, b, activation, lr, damping)` — one
  Gauss-Newton step.
- `kfac_kron_factors(X, W, β, c, b, activation)` — closed-form KFAC
  `(A, G)` Kronecker factors:
  - `A = (1/B) Σ x_n x_n^T`     (input second moment)
  - `G = (1/B) Σ (c⊙σ'(z_n))(c⊙σ'(z_n))^T`     (pre-activation grad cov)

Validated against `jax.grad` / `jax.hessian` to ≤1e-10 for every
Riccati-class activation; see `tests/test_one_layer.py`.

## What this is for

The closed-form Hessian/Fisher is the building block for second-order
optimisation:

1. **Newton-Gauss for one-layer regression / classifier heads.** When
   the model is shallow and the loss is MSE / cross-entropy, the
   closed-form Fisher beats Adam by 5-10× in sample efficiency
   (Martens & Grosse 2015 Table S3 baseline).
2. **KFAC layer registration for FermiNet.** A FermiNet trainer uses
   `kfac_jax`'s auto-registration. Plugging
   `kfac_kron_factors` in as a custom layer block replaces the
   autograd-estimated Fisher with the omnibias closed-form one —
   exact, not stochastic.
3. **Sobolev / Jacobian regularisation.** The closed-form Hessian also
   enables exact computation of penalties like `‖H‖_F²` or
   `tr(J^T H J)` for backflow / equivariant chain rules.

## On the roadmap

- Multi-layer Hessian assembly (chain-rule the per-layer Kronecker
  factors through equivariant maps).
- `kfac_jax` custom-layer registration so a FermiNet trainer can
  drop in the omnibias-Fisher block for the envelope layer.
- Torch port (the v0.1 above is JAX-only).

The KFAC integration plan is tracked on the roadmap (`docs/roadmap.md`).

## Status

Alpha (`0.1.0a1`). API surface is stable for the one-layer use case,
but `kfac_jax` integration is not yet wired into a FermiNet
trainer.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.

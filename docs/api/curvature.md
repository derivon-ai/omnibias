# omnibias-curvature

**Closed-form curvature primitives for one-layer Riccati fields.**

`omnibias-curvature` is an alpha extension package. It is useful for research into second-order optimization, Fisher/KFAC factors, and loss-landscape analysis, but it is not part of the curated-core stability contract.

## Public alpha surface

::: omnibias.curvature.one_layer
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Certified conditioning & the ε→0 rank/regularization collapse

Tikhonov-regularized solving collapses onto the Moore–Penrose / minimum-norm solution as the damping vanishes, \((A+\varepsilon I)^{-1}b \to A^{+}b\) as \(\varepsilon\to 0\). Taken naively that limit **blows up** on a rank-deficient / ill-conditioned \(A\); omnibias does it *stably* and — the actual contribution — pairs it with a **rigorous certified conditioning bound**.

- **Differentiable register** (`omnibias.curvature.regularize`, JAX + a bit-identical `omnibias.curvature.torch` twin): [`regularized_solve`][omnibias.curvature.regularize.regularized_solve] — the generic \((A+\varepsilon I)^{-1}b\) that `damped_solve` and `mse_newton_step` now delegate to (bit-identical at the same damping); [`min_norm_solve`][omnibias.curvature.regularize.min_norm_solve] — eigen-truncated \(A^{+}b\), no blow-up; [`regularization_path`][omnibias.curvature.regularize.regularization_path] — the measured homotopy; and [`rank_collapse`][omnibias.curvature.regularize.rank_collapse] — certified-damping or min-norm entry point that attaches a sealed certificate.
- **Rigorous register** (`omnibias.core.verified.conditioning`): `certified_min_eigenvalue` / `certified_max_eigenvalue`, `certified_condition_number` (upper endpoint \(+\infty\) when positive-definiteness cannot be certified — the honest rank-deficient signal), `certified_damping` (smallest \(\varepsilon\) provably giving \(\kappa(A+\varepsilon I)\le T\)), `certified_regularization_error`, and `conditioning_certificate` (a sealed v1 certificate carrying the \(\lambda_{\min}>0\) `LDLᵀ` pivots for the Lean bridge).
- **Optimizer option**: `NaturalGradient` / `GaussNewton` gain `target_condition` — on the dense-metric path they floor the damping at `certified_damping` and stash the sealed certificate on `last_certificate`.

!!! note "The `eps -> 0` collapse (a distinct limit)"
    \((A+\varepsilon I)^{-1}b \to A^{+}b\) as `eps -> 0` is a **distinct** limit from the founding multi-bias `delta -> 0` derivative collapse and the `beta -> inf` feasibility penalty — same spirit, different parameter, never conflated. It does **not** ride the σ-derivative tower: the solves (`solve` / `eigh`) are labelled *numerical* and only the conditioning enclosure is *verified* (never "closed-form"). `certified_regularization_error` is sound only on `range(A)` (a null-space component of the right-hand side makes the naive Tikhonov solve diverge — exactly the blow-up the collapse avoids).

::: omnibias.curvature.regularize
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Exact-curvature sharpness ("SAM done right")

Sharpness-Aware Minimization (Foret et al. 2021) minimizes the worst-case loss in an \(\ell_2\)-ball, \(\max_{\lVert\varepsilon\rVert\le\rho} L(\theta+\varepsilon)\), and approximates the inner max by a **single gradient-ascent step** — keeping only the *linear* Taylor term and paying an extra forward/backward pass. It never sees the curvature that actually makes a basin sharp or flat.

For the one-layer Riccati field, omnibias gives the closed-form \(\sigma', \sigma''\) (and, through autodiff of the closed form, \(\sigma'''\) — the third derivative the *gradient* of a curvature penalty needs). That lets `omnibias.curvature.sharpness` measure and regularize sharpness **exactly**:

- [`mse_loss_hessian`][omnibias.curvature.sharpness.mse_loss_hessian] — the exact **full** loss Hessian (Gauss–Newton term *plus* the \(r_n\,\nabla^2_\theta f\) residual term); equals `jax.hessian` of the loss to ~1e-9 for every Riccati activation.
- `hessian_trace` / `hessian_frobenius_sq` / `hessian_top_eigenvalue` — the sharpness functionals \(\sum_i\lambda_i\), \(\sum_i\lambda_i^2\), \(\lambda_{\max}\).
- [`sharpness_aware_loss`][omnibias.curvature.sharpness.sharpness_aware_loss] — a differentiable curvature regularizer \(L + \lambda\,\mathcal S(\nabla^2 L)\).
- [`sam_objective`][omnibias.curvature.sharpness.sam_objective] — the **ascent-free, exact second-order** SAM surrogate \(L + \rho\lVert\nabla L\rVert + \tfrac12\rho^2\lambda_{\max}\), which upper-bounds the true ball worst-case and converges to it at \(O(\rho^3)\) (SAM's linear estimate errs at \(O(\rho^2)\)).

On a realizable teacher/student regression with label noise (12 seeds), curvature-regularized training reaches minima with **~2× smaller** top-eigenvalue curvature than plain MSE and **generalizes better**, beating classic (ascent-step) SAM on the same task.

!!! note "Honesty caveats"
    The JAX module is a **one-layer** primitive. The claim is *not* "flat minima always generalize better" — that is problem-dependent (No Free Lunch); the claim is that omnibias can measure and regularize curvature *exactly and cheaply* where SAM only estimates its linear shadow. The only approximations are the ones you opt into: the SAM surrogate keeps the second-order term (error \(O(\rho^3)\)) and clamps \(\lambda_{\max}\) at 0 so negative curvature is never rewarded.

::: omnibias.curvature.sharpness
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Multi-layer torch curvature (`omnibias.curvature.torch`)

The arbitrary-depth, PyTorch counterpart. For a deep network the full parameter Hessian is too large to form, so the curvature functionals are computed **matrix-free** from exact **Hessian-vector products** (reverse-over-reverse autograd). Because the omnibias torch activations are plain differentiable ops (no custom `autograd.Function` with a hand-written backward), this double-backward *is* the exact Hessian action, and the differentiable penalty gradient (a reverse-over-reverse-over-reverse pass) rides on the closed-form \(\sigma'''\).

The API takes a scalar `loss` (with an autograd graph) and the `params` it depends on (e.g. `list(model.parameters())`), so it drops into any training loop — an MLP, a `JetMLP` PINN, or the `omnibias.score.flow` CNF velocity field:

```python
import torch
from omnibias.torch.architectures import JetMLP
from omnibias.curvature.torch import sharpness_aware_loss, sam_objective

net = JetMLP(in_dim=3, hidden=32, out_dim=1, depth=4, base="tanh").double()
params = [p for p in net.parameters() if p.requires_grad]

X = torch.randn(128, 3, dtype=torch.float64)
Y = torch.randn(128, dtype=torch.float64)

pred = net(X).squeeze(-1)
loss = ((pred - Y) ** 2).mean()

# curvature-regularized objective (matrix-free, differentiable end-to-end)
obj = sharpness_aware_loss(loss, params, lam=3e-3, measure="frobenius", n_samples=1)
obj.backward()
```

- `hvp` / `dense_hessian` — one exact HVP, or the full dense Hessian by HVPs on the basis (small nets only). `dense_hessian` matches `torch.func.hessian` to floating point, and its spectrum matches the JAX closed-form one-layer Hessian across backends.
- `top_eigenvalue` / `hessian_eigenvalue_extremes` — \(\lambda_{\max}\) (and \(\lambda_{\min}\)) by two-phase power iteration on the HVP; correct even for an indefinite Hessian away from a minimum.
- `hutchinson_trace` / `hutchinson_frobenius_sq` — unbiased Rademacher estimators of \(\operatorname{tr}(H)\) and \(\lVert H\rVert_F^2\) (exact in expectation).
- `sharpness_aware_loss` / `sam_objective` — the same differentiable objectives as the JAX module, now at any depth.

Requires the optional `torch` extra: `pip install "omnibias-curvature[torch]"`.

!!! note "Exactness ledger (torch)"
    `hvp` / `dense_hessian` are **exact** (autograd). `top_eigenvalue` is exact in the power-iteration limit. `hutchinson_*` are **unbiased stochastic** (variance \(\sim 1/n\)) — for a reported number on a small net, prefer `dense_hessian` + the matrix helpers.

::: omnibias.curvature.torch.sharpness
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## `ExactSAM` — the exact-sharpness penalty as an optimizer

[`ExactSAM`][omnibias.curvature.torch.ExactSAM] packages the functionals above as a drop-in `torch.optim.Optimizer` that targets **test error / flat minima** — the axis where Adam is *not* optimal — instead of per-step training loss.

Its design follows a validated result: in the `examples/mnist1d_double_descent` study, finding **H4** shows that an *exact sharpness penalty* (`sharpness_aware_loss`) suppresses the model-wise double-descent test-error peak more cleanly than Adam *or* an exact SAM ascent step. Classic SAM pays **2×** (an ascent forward/backward plus the real step) to *estimate* the sharpness direction; omnibias has that direction in closed form, so `ExactSAM` instead **adds the exact sharpness gradient to the step** and **amortizes** the expensive curvature probe over `probe_every` steps — targeting `≤ SAM`'s cost:

```python
from omnibias.curvature.torch import ExactSAM

opt = ExactSAM(net.parameters(), lr=1e-2, lam=1e-3, measure="frobenius", probe_every=4)
for _ in range(3):
    # a scalar-loss closure: graph intact, and no .backward() of your own
    opt.step(lambda: ((net(X).squeeze(-1) - Y) ** 2).mean())
```

Every step uses the cheap loss gradient (one backward); every `probe_every` steps it refreshes the exact sharpness gradient `∇S` (`penalty` mode) or the exact second-order SAM-gap gradient (`ascent` mode) and reuses the stale direction in between.

The **base** optimizer that descends the sharpness-augmented gradient `g = ∇L + λ∇S` is selectable:

- `base="sgd"` (default) — self-contained heavy-ball SGD, mirroring how SAM wraps SGD. Best peak-suppression on the classification (`ce_relu`) register, but a plain-SGD base can *underfit* least-squares / `tanh` regression.
- `base="adam"` — Adam-preconditions `g` (adds a second-moment buffer `v`). Decouples *fit* (per-coordinate adaptivity) from *flatten* (the penalty); with `lam=0` it is **bit-for-bit Adam**, so it degrades gracefully to Adam where the penalty would hurt.
- `base="frugal"` — a memory-lean adaptive base: one momentum buffer plus a per-tensor RMS scalar (`O(#tensors)`), the [`FrugalCurvature`][omnibias.torch.optim.FrugalCurvature] philosophy applied to `g`.

**Auto-`λ` (`lam_auto=True`, penalty mode).** A *fit-preservation cap*: each step, `λ` is set to the largest value keeping the combined step first-order loss-decreasing, `λ_eff = min(λ, lam_safety·‖∇L‖²/|⟨∇L,∇S⟩|)` (uncapped when the penalty does not oppose the loss gradient), so `λ` becomes an *upper bound*. It is exact for `base="sgd"`, a raw-coordinate heuristic for the adaptive bases, and costs one extra dot product (no extra backward). On `ce_relu` it is the best arm in the `examples/mnist1d_double_descent` study; see the caveat for its (informative) limit on `mse_tanh`.

!!! note "Honesty caveats"
    `ExactSAM` is **amortized packaging** of the already-validated exact-sharpness penalty (H4), not new math. The "`≤ SAM` cost" claim is only credible with wall-clock telemetry, since each refresh costs several double-backward HVPs — the `examples/mnist1d_double_descent` harness records exactly that (`update_time_s`). It regularizes curvature; it does **not** assert that flatter always generalizes: in the `examples/mnist1d_double_descent` grid the penalty **helps** classification (`ce_relu`) monotonically in `λ` but **hurts** `tanh`/MSE regression. `lam_auto` (the fit-preservation cap) lifts `ce_relu` to the study's best generalization, but it **cannot** fix `mse_tanh`: the money plot shows `λ_eff` stays pinned at the bound there because the penalty does *not* fight the fit (train loss is low) — the regression damage is a **generalization / metric-misalignment** effect (flat MSE-to-one-hot basins ≠ good accuracy), invisible to any fit-based signal. Practical rule: **apply sharpness where flatness aligns with the target metric (classification / CE); on MSE-to-one-hot regression prefer Adam (`lam_auto` off).** Torch-only for now.

::: omnibias.curvature.torch.optim
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

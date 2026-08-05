# omnibias-torch

PyTorch backend for omnibias.

## Top-level API

::: omnibias.torch
    options:
      show_root_heading: false
      heading_level: 3

## Activation registry

::: omnibias.torch.activations.registry
    options:
      show_root_heading: false
      heading_level: 3

## Closed-form integral transforms

The other side of the derivative tower: for the activations whose Laplace,
Fourier or Mellin transform is itself elementary, `omnibias.torch.transforms`
evaluates it in one shot -- no quadrature, no series truncation, no iteration --
as a differentiable tensor op. Which pairs ship, and why the others do not, is
decided in [`omnibias.core.transforms`](core.md#integral-transform-identities);
`omnibias.jax.transforms` is the bit-identical twin.

Nine Laplace kernels (`exp`, `relu`, `sin`, `cos`, `sinh`, `cosh`, `gaussian`,
`sigmoid`, `tanh`, `sech`), two Fourier kernels (`gaussian` and `sech`, the two
self-reciprocal profiles -- the saturating activations are not `L^1` and their
transforms are distributional), and one Mellin kernel (`gaussian`). The
Fermi-Dirac integral `Gamma(s) eta(s)` ships as the separately named
`fermi_dirac_mellin` because it is the Mellin transform of `1 - sigmoid`, not of
`sigmoid`, whose own Mellin integral diverges; it is restricted to `Re(s) > 1`,
the same scope wall `omnibias.core.verified.dirichlet` enforces on `zeta`.

Two numerical notes worth knowing. The Gaussian Laplace kernel routes through
`erfcx`, because `exp(s^2/2)` overflows and `erfc(s/sqrt 2)` underflows past
`s ~ 38` while their product decays like `1/s` -- the naive form returns `nan`
exactly where the answer is small and well conditioned. And
`fermi_dirac_mellin` is **not** differentiable on torch: `torch.special.zeta`
ships no derivative rule, where `jax.scipy.special.zeta` does. That asymmetry
lives in the special-function libraries, not in omnibias; every other kernel is
differentiable on both backends.

`TransformBlock` (and its `LaplaceTransform` / `FourierTransform` /
`MellinTransform` aliases) makes a transform a trainable layer, evaluating
`T[sigma](scale * x + shift)` with both parameters learnable. It reparameterises
the offset through a softplus onto the open region of convergence, so no
optimizer step can drive a learnable spectral variable into the half-plane where
the closed form is meaningless.

::: omnibias.torch.transforms
    options:
      show_root_heading: false
      heading_level: 3

## OperatorMultiBiasUnit

::: omnibias.torch.unit
    options:
      show_root_heading: false
      heading_level: 3

## Blocks

::: omnibias.torch.blocks
    options:
      show_root_heading: false
      heading_level: 3

## Piecewise & tempered activations

The hard almost-everywhere family and the smooth beta-tempered surrogate
family (see the [activation dictionary](../activations.md)).

::: omnibias.torch.activations.piecewise
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.torch.activations.tempered
    options:
      show_root_heading: false
      heading_level: 3

## Learnable-temperature blocks

::: omnibias.torch.tempered_blocks
    options:
      show_root_heading: false
      heading_level: 3

## Architectures

::: omnibias.torch.architectures
    options:
      show_root_heading: false
      heading_level: 3

## Optimisers (Gauss-Newton, matrix-free CG/CGLS, trust-region & stochastic Newton-CG, cubic-regularised Newton, K-FAC, conformal-symplectic descent, memory-lean per-tensor curvature, exact-jet L-BFGS, natural / Riemannian gradient, integral / energy natural gradient)

The exact-curvature methods come in two flavours. The **functional** cores
(`GaussNewton`, `CubicRegularizedNewton`, `CubicRegularizedGaussNewton`, `JetLBFGS`)
act on a flat parameter vector plus a `torch.func` closure -- the scientific-computing
interface. The **drop-in** front ends implement the standard `torch.optim.Optimizer`
contract, so any of them is a one-line, universal replacement for Adam in an existing
`model.parameters()` / `opt.step(closure)` loop (Adam is never restricted; these simply
join the pool):

- `CubicNewton` / `CubicGaussNewton` -- cubic-regularised (Newton / Gauss-Newton) steps;
- `TrustRegionNewtonCG` -- matrix-free full-Hessian Newton via the Steihaug-Toint
  truncated-CG trust-region subproblem solver (`steihaug_cg`), robust to indefinite
  curvature (stops on negative-curvature / at the trust-region boundary);
- `StochasticNewtonCG` -- the subsampled-Newton split (curvature HVP on a subset,
  gradient on the batch) with Levenberg damping and an optional `resample` data hook;
- `DiagonalCurvature` -- the scalable exact-diagonal (Gauss-Newton / Hutchinson) "Adam
  substitute" with relative flooring, Adam-style bias correction, and subsampled refresh;
- `FrugalCurvature` -- the **memory-lean** relative of `DiagonalCurvature`: one `O(P)` momentum
  buffer plus **per-tensor** exact-curvature scalars (`O(#tensors)`), so its optimiser state is
  roughly *half* Adam's two `O(P)` buffers. The full exact diagonal is computed transiently and
  immediately **reduced** to per-tensor scalars (never stored), trading per-coordinate granularity
  for memory; optional Lion-like `sign_momentum`, `clip`, and monotone `safeguard`;
- `ConformalSymplectic` -- **Conformal-Symplectic Descent (CSD)**: optimisation as a
  *dissipative Hamiltonian flow* integrated by a structure-preserving conformal-symplectic map
  (friction as the exact contraction `mu = exp(-gamma * lr)`, the gradient as a symplectic kick),
  with the exact curvature diagonal reused as a physical **mass** `M` (reparametrisation-aware
  preconditioning), an optional exact directional-jet line search, and an optional Langevin
  thermostat -- a physics-grounded, first-order-cost Adam alternative;
- `KFAC` -- a generic hook-based Kronecker-factored (`nn.Linear`) preconditioner;
- `JetLBFGSOptimizer` -- the exact-jet L-BFGS front end;
- `NaturalGradient` -- the metric-preconditioned (natural-gradient / Riemannian) step
  `delta = (M(theta) + damping I)^{-1} g` with a **pluggable metric** `M`. Two closed-form
  metrics drop straight in: the Gauss-Newton **Fisher** `(1/N) J^T J`
  (`gauss_newton_fisher` / matrix-free `gauss_newton_fisher_matvec`; with `F = H` this is
  Newton and recovers a quadratic's minimiser in one step) and the **geometry pullback**
  `g = J^T h J` of a learned chart (`omnibias.geometry.torch.ops.pullback_metric`). The
  functional core `natural_gradient_direction` accepts either a dense `(P, P)` metric or a
  matrix-free `v |-> M v` operator (solved by conjugate gradient); it is decoupled from the
  loss gradient, so *any* SPD Riemannian metric is admissible. Bit-identical jax twin:
  `omnibias.jax.optim.natural_gradient_step`.

Curvature is obtained matrix-free by closure-based double-backward autograd, so the
drop-ins work on any differentiable loss -- the omnibias closed-form jet is what makes
that curvature *exact*. The full-Hessian Newton-CG and K-FAC methods are small/medium
scale here (no LLM-scale claim).

::: omnibias.torch.optim
    options:
      show_root_heading: false
      heading_level: 3

## Fastpath kernels

::: omnibias.torch.fastpath
    options:
      show_root_heading: false
      heading_level: 3

## Faà di Bruno jets

::: omnibias.torch.jet
    options:
      show_root_heading: false
      heading_level: 3

## Multivariate (multi-index) jets

::: omnibias.torch.jet_mv
    options:
      show_root_heading: false
      heading_level: 3

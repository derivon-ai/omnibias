# Math validation matrix

This matrix lists every mathematical primitive the audit covered, the
file/symbol where it lives, the analytic reference used to validate it,
and where the proof is locked down. Each row carries:

- **Formula / invariant**: the mathematical identity the implementation
  satisfies.
- **File / symbol**: where the implementation lives.
- **Reference**: the analytic / closed-form source of truth.
- **Local coverage**: which test suite pins the primitive on a
  constrained CPU-only development host.
- **GPU job**: whether the production-fidelity validation needs a GPU
  job (rare; only for primitives whose closed-form parity matters at
  ferminet-class problem sizes). See [`benchmarks.md`](benchmarks.md).
- **Status**: ``proven locally``, ``needs GPU job``, or ``blocked``.

## omnibias-core

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| sigmoid polynomial recurrence | `P_{n+1}(s) = s(1-s) P_n'(s)`, `P_0 = s` | `sigmoid_polynomial_coeffs` in [`packages/omnibias-core/src/omnibias/core/polynomials.py`](../packages/omnibias-core/src/omnibias/core/polynomials.py) | Eulerian numbers; `sigma^(n)(0)` Taylor series of `1/(1+e^{-z})` | `test_polynomials*.py` (Phase 1a, 142 cases) | none | proven locally |
| tanh polynomial recurrence | `T_{n+1}(t) = (1-t^2) T_n'(t)`, `T_0 = t` | `tanh_polynomial_coeffs` in [`packages/omnibias-core/src/omnibias/core/polynomials.py`](../packages/omnibias-core/src/omnibias/core/polynomials.py) | Bernoulli/tangent-number Taylor series of `tanh(z)` | `test_polynomials*.py` | none | proven locally |
| Hermite recurrence | `He_{n+1} = z He_n - n He_{n-1}`, `He_0=1`, `He_1=z` | `hermite_coeffs` in [`packages/omnibias-core/src/omnibias/core/polynomials.py`](../packages/omnibias-core/src/omnibias/core/polynomials.py) | Probabilist's Hermite polynomial recurrence; `He_{2k}(0) = (-1)^k (2k)! / (2^k k!)` | `test_polynomials*.py` | none | proven locally |
| Asymptotic vanishing | `P_n(0) = P_n(1) = 0` for `n>=1`; `T_n(+/-1) = 0` for `n>=1` | same | activation derivative `-> 0` at `+inf` / `-inf` | `test_polynomials_invariants.py` | none | proven locally |

## omnibias-torch

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| sigmoid n-th derivative | `sigma^(n)(z) = P_n(sigma(z))` | `sigmoid_nth_derivative` in [`packages/omnibias-torch/src/omnibias/torch/fastpath/eulerian.py`](../packages/omnibias-torch/src/omnibias/torch/fastpath/eulerian.py) | shared `omnibias.core.polynomials` | `tests/test_jax_parity.py` + `test_fastpath_correctness.py` | none | proven locally |
| tanh n-th derivative | `tanh^(n)(z) = T_n(tanh(z))` | `tanh_nth_derivative` in [`.../fastpath/legendre.py`](../packages/omnibias-torch/src/omnibias/torch/fastpath/legendre.py) | same | same | none | proven locally |
| Gaussian n-th derivative | `g^(n)(z) = (-1)^n He_n(z) g(z)` | `gaussian_nth_derivative` in [`.../fastpath/hermite.py`](../packages/omnibias-torch/src/omnibias/torch/fastpath/hermite.py) | same | same | none | proven locally |
| OMBU literal forward | `f_K(z) = sum_k s_k sigma(z + b_k)` | `multibias_literal_forward` in [`.../fastpath/dispatch.py`](../packages/omnibias-torch/src/omnibias/torch/fastpath/dispatch.py) | direct construction | `test_unit.py`, `test_blocks.py` | none | proven locally |
| OMBU collapsed forward | `sigma^(K-1)(z + bar b)` via fastpath | `multibias_collapsed_forward` in same | derivative tower from `omnibias.core.polynomials` | `test_unit.py`, `test_fastpath_*` | none | proven locally |
| OMBU integral window | `S(z + b_hi) - S(z + b_lo)` with small-width Taylor fallback | `multibias_integral_window_forward` in same | antiderivative table per spec | `test_integral_primitives.py` | none | proven locally |
| Riccati identity | `sigma'(z) - P(sigma(z)) == 0` for every Riccati spec | `tests/test_jax_parity_invariants.py` | analytic | 12 cases pass | none | proven locally |
| Integral round-trip | `(d/dz) integral(z) == forward(z)` | same test | autograd | 18 cases pass | none | proven locally |
| Default-dtype propagation | `OMBU` / OperatorBlock parameters follow `torch.get_default_dtype()` | [`.../identity_init.py`](../packages/omnibias-torch/src/omnibias/torch/identity_init.py), [`.../stencil.py`](../packages/omnibias-torch/src/omnibias/torch/stencil.py) | torch convention | `test_audit_regressions.py` 27 cases | none | proven locally |

## omnibias-jax

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| Closed-form Laplacian | `Laplacian f = sum_h c_h sigma''(z_h) ||W_h||^2` | `neural_field_laplacian` in [`packages/omnibias-jax/src/omnibias/jax/laplacian.py`](../packages/omnibias-jax/src/omnibias/jax/laplacian.py) | derivative chain rule on `f(x) = b + sum_h c_h sigma(W_h.x + beta_h)` | `test_audit_regressions.py` (parity vs `jax.hessian`) + `test_jax_parity.py` | none | proven locally |
| Closed-form Hessian | `H_x f = W^T diag(sigma''(z) c) W` | `neural_field_hessian` in same | symmetric rank-H outer product | `test_audit_regressions.py` (parity vs `jax.hessian`) | none | proven locally |
| Closed-form gradient | `grad f = (sigma' c) W` | `neural_field_value_grad_*` in same | chain rule | `test_audit_regressions.py` | none | proven locally |
| Polylaplacian | `Delta^k f = sum_h c_h sigma^(2k)(z_h) ||W_h||^{2k}` | `neural_field_polylaplacian` in same | iterated trace identity (see module docstring) | `test_polylaplacian.py` (k=1,2,3 vs trace-of-Hessian) | none | proven locally |
| jit/vmap closure | every helper is jit-able / vmap-able | same | XLA tracing | `test_audit_regressions.py` 38 cases | none | proven locally |

## omnibias-ferminet

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| Envelope `(value, grad, Hessian)` | closed-form per-(orbital, electron) for the FermiNet PRE_DETERMINANT envelope | `envelope_value_grad_hessian` in [`packages/omnibias-ferminet/src/omnibias/ferminet/integration.py`](../packages/omnibias-ferminet/src/omnibias/ferminet/integration.py) | `omnibias.jax.neural_field_value_grad_hessian` per orbital, vmap'd over electrons | `tests/test_integration.py` (smoke) | full-fidelity at `n_e in {8..32}, H in {64..256}` needs a GPU job | needs GPU job for production fidelity |
| Tier-2 LKE | matrix identity `Delta log\|det M\| = trace(M^{-1} Delta M) - trace((M^{-1} grad M)^2)` | `make_omnibias_tier2_local_kinetic_energy` in same | textbook LU-aware kinetic energy | smoke via mock LKE in `tests/test_integration.py` | full FermiNet end-to-end needs a GPU job | needs GPU job for production fidelity |
| Tier-2 restricted ansatz | closed-form Laplacian / `\|grad\|^2` of `log\|det M\|` | [`.../restricted.py`](../packages/omnibias-ferminet/src/omnibias/ferminet/restricted.py) | autograd reference (`jax.hessian` of `tier2_log_abs_psi`) | `tests/test_restricted.py` | as above | needs GPU job for production fidelity |
| Folx-compatible Laplacian | `(value, gradient, Laplacian)` shape match | `forward_laplacian` in [`.../folx_compat.py`](../packages/omnibias-ferminet/src/omnibias/ferminet/folx_compat.py) | folx public API | `tests/test_folx_compat.py` | none | proven locally |

## omnibias-pinn

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| `value` / `derivative` / `mixed_partial` ops | chain rule on one-layer field | [`.../jax/ops/basic.py`](../packages/omnibias-pinn/src/omnibias/pinn/jax/ops/basic.py), torch twin | autograd reference | `tests/jax`, `tests/torch`, `tests/cross_backend` | none | proven locally |
| `gradient`, `divergence`, `curl`, `laplacian` | textbook vector calculus | `.../jax/ops/basic.py`, `.../jax/ops/vector.py` | analytic + autograd | unit tests + parity tests | none | proven locally |
| `hessian`, `biharmonic`, `polylaplacian` | iterated derivatives | `.../jax/ops/high_order.py` | analytic + autograd | unit + parity | none | proven locally |
| `advection`, `material_derivative`, `p_laplacian` | `(u . nabla) u`, `Du/Dt`, `nabla.(\|nabla u\|^{p-2} nabla u)` | `.../jax/ops/nonlinear.py` | analytic | unit + parity | none | proven locally |
| Heat / Burgers / CH / KS / NS / biharmonic residuals | textbook PDE residuals | `.../jax/equations/`, torch twin | shipped analytic-solution tests in `test_*_equations.py` | unit + cross-backend | full integration needs a GPU job | needs GPU job for production fidelity |
| jit-compatibility of equation residuals | every residual call jit-traceable | same files (after F6.1 fix) | XLA tracing | `test_jax_equations_jit.py` (new, 6 cases) | none | proven locally |
| Cross-backend bit-parity | `rtol=1e-9, atol=1e-12` (float64) | `tests/cross_backend/` (pinn) | shared coefficient module | 620 / 620 passed (excluding research-deps) | none | proven locally |

## omnibias-qpinn

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| TISE residual | `(H psi - E psi)` split-real | [`.../jax/equations/tise.py`](../packages/omnibias-qpinn/src/omnibias/qpinn/jax/equations/tise.py), torch twin | textbook stationary Schrodinger | unit + cross-backend | training to convergence needs a GPU job | needs GPU job |
| TDSE residual | `(i hbar d psi/dt - H psi)` split-real | `.../tdse.py`, torch twin | textbook | unit + cross-backend | training needs a GPU job | needs GPU job |
| NLS / Gross-Pitaevskii residual | `(i d psi/dt - (-1/2 nabla^2 psi + V psi + g \|psi\|^2 psi))` | `.../nls.py`, torch twin | textbook + private rotating-NLS regression | unit + cross-backend | full GP training needs a GPU job | needs GPU job |
| Helmholtz residual | `(nabla^2 + k^2) psi` | `.../helmholtz.py`, torch twin | textbook | unit + cross-backend | none | proven locally |
| Klein-Gordon residual | `(box + m^2) phi` | `.../klein_gordon.py`, torch twin | textbook + phi^4 kink reference | unit + cross-backend | full kink training needs a GPU job | needs GPU job |
| Dirac residual | `(i gamma^mu d_mu - m) psi` mostly-minus metric | `.../dirac.py`, torch twin | Peskin-Schroeder | unit + cross-backend | full Dirac plane-wave training needs a GPU job | needs GPU job |
| Gamma matrix anticommutator | `{gamma^mu, gamma^nu} = 2 eta^{mu nu} I`, `eta = diag(+1,-1,-1,-1)` | [`.../qpinn/_core/spinor.py`](../packages/omnibias-qpinn/src/omnibias/qpinn/_core/spinor.py) | textbook | `test_audit_regressions.py` (9 cases) + existing `_core/test_spinor_pauli.py` | none | proven locally |
| `gamma_5` properties | hermitian, idempotent, anticommutes with `gamma^mu` | same | textbook | same regression test | none | proven locally |
| Norm cage projection | `psi -> psi / sqrt(<psi, psi>)` | `.../jax/cage/norm.py`, torch twin | analytic | `test_cage_norm.py` (jax + torch) | none | proven locally |
| Bloch periodicity cage | `psi(x + L) = e^{i k L} psi(x)` | `.../jax/cage/bloch.py`, torch twin | textbook Bloch theorem | `test_cage_bloch.py` | none | proven locally |
| Hermitian projection | `psi -> (psi + conj(psi)) / 2` | `.../jax/cage/hermitian.py`, torch twin | analytic | `test_cage_hermitian.py` | none | proven locally |
| Cross-backend bit-parity | `rtol=1e-9, atol=1e-12` between jax and torch twins | `tests/cross_backend/` | shared spinor / complex DSL | 30+ tests | none | proven locally |

## omnibias-curvature

| Primitive | Formula / invariant | File / symbol | Reference | Local coverage | GPU job | Status |
|---|---|---|---|---|---|---|
| Per-sample parameter gradient | `\partial f/\partial \theta` for one-layer field | `one_layer_param_grad` in [`packages/omnibias-curvature/src/omnibias/curvature/one_layer.py`](../packages/omnibias-curvature/src/omnibias/curvature/one_layer.py) | chain rule (see module docstring) | `tests/test_one_layer.py` (vs `jax.grad`) | none | proven locally |
| Per-sample parameter Hessian | block-diagonal-in-h structure | `one_layer_param_hessian` in same | analytic + symmetry | same (vs `jax.hessian`) | none | proven locally |
| Gauss-Newton Fisher | `F = (2/B) sum_n g_n g_n^T` | `mse_gauss_newton_fisher` in same | MSE + Gaussian noise model | same | none | proven locally |
| Newton step | `theta -> theta - eta (F + lambda I)^{-1} grad L` | `mse_newton_step` in same | analytic | same | none | proven locally |
| KFAC factors | `F_W ~= A (x) G`, `A = (1/B) X^T X`, `G = (1/B) (c sigma') (c sigma')^T` | `kfac_kron_factors` in same | Martens-Grosse 2015 | same | none | proven locally |

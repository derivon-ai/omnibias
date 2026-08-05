# Backend parity matrix (torch <-> jax)

For each shared API surface this matrix records:

- **Torch symbol** and **JAX symbol**
- **Dtype assumption** (float64 strict / float32 relaxed)
- **Device assumption** (CPU on the development host; GPU only via a GPU job)
- **Parity tolerance** at float64
- **Test file** that pins the parity contract
- **Status**: ``proven locally``, ``needs GPU job``, ``blocked``

## Activations registry

| Surface | Torch symbol | JAX symbol | Dtype | Device | Tolerance (float64) | Test file | Status |
|---|---|---|---|---|---|---|---|
| spec lookup | `omnibias.torch.activations.registry.get_activation` | `omnibias.jax.activations.get_activation` | n/a | CPU | string match | `tests/test_jax_parity.py::test_list_activations_matches_torch` | proven locally |
| polynomial coefficients (sigmoid / tanh / hermite) | `omnibias.torch.fastpath.{eulerian,legendre,hermite}.*_coeffs` (re-export of core) | `omnibias.jax._fastpath.*_coeffs` (re-export of core) | n/a | CPU | bit-identical (shared core module) | `tests/test_jax_parity.py::test_polynomial_coeffs_shared` | proven locally |
| every activation `forward` | `spec.forward` | `spec.forward` | float64 | CPU | `rtol=1e-6, atol=1e-6` | `tests/test_jax_parity.py::test_forward_matches_torch` | proven locally |
| every activation `derivative` (n=1) | `spec.derivative` | `spec.derivative` | float64 | CPU | `rtol=1e-6, atol=1e-6` | `tests/test_jax_parity.py::test_derivative_matches_torch` | proven locally |
| every activation `fastpath(z, n)` for n in `[0..max_n]` | `spec.fastpath` | `spec.fastpath` | float64 | CPU | `rtol=1e-6, atol=1e-6` (loosened to `rtol=1e-5` for n>=5 / `gelu`) | `tests/test_jax_parity.py::test_fastpath_matches_torch_all_orders` | proven locally |
| Riccati identity `sigma'(z) - P(sigma(z)) == 0` | spec.riccati_polynomial | same | float64 | CPU | `rtol=1e-12, atol=1e-13` | `tests/test_jax_parity_invariants.py::test_riccati_identity_*` | proven locally |
| Integral round-trip `d/dz integral(z) == forward(z)` | autograd of `spec.integral` | same | float64 | CPU | `rtol=1e-10, atol=1e-12` | `tests/test_jax_parity_invariants.py::test_integral_round_trip_*` | proven locally |
| Negative-order error class | both raise `ValueError` | same | n/a | CPU | error class match | `tests/test_jax_parity_invariants.py::test_negative_order_parity_*` | proven locally |
| Tight float64 fastpath parity at n=3..6 (bit-stable activations) | `spec.fastpath` for `sigmoid/tanh/softplus/gaussian/exp/sin/cos/sinh/cosh` | same | float64 | CPU | `rtol=1e-13, atol=1e-14` | `tests/test_jax_parity_invariants.py::test_fastpath_tight_parity` | proven locally |
| GPU parity smoke (CUDA tensor) | torch CUDA path | n/a (CPU only) | float32 | GPU | `rtol=1e-5, atol=1e-6` | GPU job (>= 8 GB) | needs GPU job |

## One-layer field closed-form vector calculus (omnibias.jax.laplacian)

| Surface | Torch symbol | JAX symbol | Dtype | Device | Tolerance (float64) | Test file | Status |
|---|---|---|---|---|---|---|---|
| `value` | n/a (field-level only on JAX side) | `neural_field_value` | float64 | CPU | exact | `tests/test_jax_parity.py` | proven locally |
| `Laplacian == trace(jax.hessian)` | n/a | `neural_field_laplacian` | float64 | CPU | `rtol=1e-10, atol=1e-12` | `tests/test_jax_parity.py` + `packages/omnibias-jax/tests/test_audit_regressions.py` | proven locally |
| `Hessian == jax.hessian` | n/a | `neural_field_hessian` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `packages/omnibias-jax/tests/test_audit_regressions.py` | proven locally |
| `polylaplacian` for k in {1, 2, 3} | n/a | `neural_field_polylaplacian` | float64 | CPU | `rtol=1e-7, atol=1e-9` | `packages/omnibias-jax/tests/test_polylaplacian.py` | proven locally |
| jit / vmap closure | n/a | every helper | float64 | CPU | within `rtol=1e-10` of eager | `packages/omnibias-jax/tests/test_audit_regressions.py` | proven locally |

## omnibias-pinn (T2)

| Surface | Torch symbol | JAX symbol | Dtype | Device | Tolerance (float64) | Test file | Status |
|---|---|---|---|---|---|---|---|
| `value` / `derivative` / `mixed_partial` | `omnibias.pinn.torch.ops.basic.*` | `omnibias.pinn.jax.ops.basic.*` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `packages/omnibias-pinn/tests/cross_backend/test_chebyshev_parity.py`, `test_one_layer_parity.py`, `test_spectral_parity.py` | proven locally |
| `gradient` / `divergence` / `curl` / `laplacian` | `.../torch/ops/{basic,vector}.py` | `.../jax/ops/{basic,vector}.py` | float64 | CPU | `rtol=1e-9, atol=1e-12` | same | proven locally |
| `hessian` / `biharmonic` / `polylaplacian` | `.../torch/ops/high_order.py` | `.../jax/ops/high_order.py` | float64 | CPU | `rtol=1e-9, atol=1e-12` | same | proven locally |
| `advection` / `material_derivative` / `p_laplacian` | `.../torch/ops/nonlinear.py` | `.../jax/ops/nonlinear.py` | float64 | CPU | `rtol=1e-9, atol=1e-12` | same | proven locally |
| Heat / Burgers / CH / KS / NS / biharmonic residuals | `.../torch/equations/*.py` | `.../jax/equations/*.py` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `packages/omnibias-pinn/tests/cross_backend/test_equations_parity.py` | proven locally |
| Cages (conservation, incompressible) | `.../torch/cage/` | `.../jax/cage/` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `packages/omnibias-pinn/tests/cross_backend/test_*_parity.py` | proven locally |
| Sobolev / NTK / causal / entropy losses | `.../torch/losses/*.py` | `.../jax/losses/*.py` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `packages/omnibias-pinn/tests/cross_backend/test_losses_parity.py` | proven locally |
| jit-compatibility of JAX residuals | n/a | every equation residual | float64 | CPU | bit-stable | `packages/omnibias-pinn/tests/jax/test_jax_equations_jit.py` (new) | proven locally |
| `needs_research` integration tests | torch + jax research-parity | research not shipped | float64 | CPU | `rtol=1e-7, atol=1e-9` | self-skip in public CI; GPU job for internal validation | needs GPU job |

## omnibias-qpinn (T3)

| Surface | Torch symbol | JAX symbol | Dtype | Device | Tolerance (float64) | Test file | Status |
|---|---|---|---|---|---|---|---|
| TISE / TDSE / NLS / Helmholtz / KleinGordon / Dirac residuals | `omnibias.qpinn.torch.equations.*` | `omnibias.qpinn.jax.equations.*` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `packages/omnibias-qpinn/tests/cross_backend/test_equations_parity.py`, `test_relativistic_parity.py` | proven locally |
| RotatingNLS residual | `omnibias.qpinn.torch.equations.RotatingNLS` | `omnibias.qpinn.jax.equations.RotatingNLS` | float64 | CPU | `rtol=1e-9, atol=1e-12` (cross-backend); `rtol=1e-12, atol=1e-14` against internal benchmark | cross-backend test + internal regression | needs GPU job for internal regression |
| Norm / Bloch / Hermitian / Parity cages | `omnibias.qpinn.torch.cage.*` | `omnibias.qpinn.jax.cage.*` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `tests/cross_backend/test_cage_parity*.py` | proven locally |
| Diagnostics (energy, current, norm, vortex) | `omnibias.qpinn.torch.diagnostics.*` | `omnibias.qpinn.jax.diagnostics.*` | float64 | CPU | `rtol=1e-9, atol=1e-12` | `tests/torch/test_diagnostics.py`, `tests/jax/test_diagnostics.py` | proven locally |
| Gamma anticommutator / gamma_5 | shared `omnibias.qpinn._core.spinor` | same | complex128 | CPU | `atol=1e-13` | `packages/omnibias-qpinn/tests/test_audit_regressions.py` (9 cases) | proven locally |
| jit-compatibility of JAX residuals | n/a | every equation residual | float64 | CPU | `rtol=1e-12` | `packages/omnibias-qpinn/tests/test_audit_regressions.py` (5 cases) | proven locally |
| QPINN smoke integration (QHO / wavepacket / dark soliton / Helmholtz / KG kink / Dirac plane wave) | torch + jax | torch | float64 | CPU smoke; GPU job for full | small problem size pass | full GPU run pending | needs GPU job |

## omnibias-curvature (T3)

| Surface | Torch symbol | JAX symbol | Dtype | Device | Tolerance (float64) | Test file | Status |
|---|---|---|---|---|---|---|---|
| `pack_params` / `unpack_params` | n/a (alpha JAX-only) | `omnibias.curvature.one_layer.*` | float64 | CPU | exact round-trip | `packages/omnibias-curvature/tests/test_one_layer.py` | proven locally |
| Per-sample gradient | n/a | `one_layer_param_grad` | float64 | CPU | `rtol<=1e-12` vs `jax.grad` | same | proven locally |
| Per-sample Hessian | n/a | `one_layer_param_hessian` | float64 | CPU | `rtol<=1e-12` vs `jax.hessian`, symmetric | same | proven locally |
| MSE Gauss-Newton Fisher | n/a | `mse_gauss_newton_fisher` | float64 | CPU | analytic | same | proven locally |
| MSE Newton step | n/a | `mse_newton_step` | float64 | CPU | analytic | same | proven locally |
| KFAC factors | n/a | `kfac_kron_factors` | float64 | CPU | analytic | same | proven locally |
| Torch port (planned) | not yet shipped | n/a | n/a | n/a | n/a | docs/roadmap.md | blocked |

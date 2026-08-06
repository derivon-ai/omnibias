# Examples gallery

Runnable, self-contained scripts. Each lives in
[`docs/examples/`](https://github.com/derivon-ai/omnibias/tree/main/docs/examples)
and runs on CPU in seconds (the QPINN demo in well under a minute).

| Example | Backend | What it shows |
|---|---|---|
| [`quickstart_torch.py`](examples/quickstart_torch.py) | PyTorch | OMBU, OperatorBlock, `cmbLinear`, one training step |
| [`quickstart_jax.py`](examples/quickstart_jax.py) | JAX | closed-form value / grad / Hessian; Laplacian vs autodiff |
| [`quickstart_keras.py`](examples/quickstart_keras.py) | Keras 3 (TF / JAX / torch) | OMBU, OperatorBlock, `cmbDense` on the unified backend |
| [`pinn_heat.py`](examples/pinn_heat.py) | PyTorch | closed-form-Laplacian PINN for the 1D heat equation |
| [`pinn_solver_curvature.py`](examples/pinn_solver_curvature.py) | PyTorch | second-order PINN training on the solver: Gauss-Newton / cubic-Newton vs Adam / L-BFGS, grad-norm balancing |
| [`pinn_solver_inverse.py`](examples/pinn_solver_inverse.py) | PyTorch | recover an unknown PDE coefficient from point observations, and residual-adaptive collocation at equal point budget |
| [`pinn_fourier_features.py`](examples/pinn_fourier_features.py) | PyTorch | spectral bias on two-scale Poisson: single layer vs deep vs Fourier-feature field, all on exact jets |
| [`pinn_multiscale_feedback.py`](examples/pinn_multiscale_feedback.py) | PyTorch | measure the band scales with `suggest_frequency_bands` instead of guessing the ladder; MscaleDNN mixture and trainable activation slopes |
| [`pinn_causal_marching.py`](examples/pinn_causal_marching.py) | PyTorch | one loss weight separating failure from success, `GradNormWeighter` finding it by measurement, `TimeMarcher` warm-started window by window, self-adaptive pointwise attention |
| [`pinn_conservation_attention.py`](examples/pinn_conservation_attention.py) | PyTorch | conservation as structure rather than penalty: the flux-form and integral cages, plus a non-local attention field with closed-form `d/dx` |
| [`pinn_xpinn_stiff.py`](examples/pinn_xpinn_stiff.py) | PyTorch | two-material seam where value continuity alone gets the wrong answer, heterogeneous per-patch subfields, and ETDRK4 / Rosenbrock where explicit stepping blows up |
| [`transforms_closed_form.py`](examples/transforms_closed_form.py) | PyTorch + JAX | closed-form Laplace / Fourier / Mellin transforms of activations: quadrature oracle, the honest gaps, and a trainable transform layer |
| [`qpinn_tise_qho.py`](examples/qpinn_tise_qho.py) | PyTorch | quantum harmonic-oscillator ground state (TISE) |
| [`variational_harmonic_oscillator.py`](examples/variational_harmonic_oscillator.py) | PyTorch | least action: EL residual, energy conservation, action is minimized at the true path |
| [`variational_brachistochrone.py`](examples/variational_brachistochrone.py) | PyTorch | the cycloid extremizes and minimizes the descent-time functional |
| [`variational_geodesic.py`](examples/variational_geodesic.py) | PyTorch + geometry | geodesics as least action: EL == `geodesic_rhs`, equator is the shortest path |
| [`variational_klein_gordon.py`](examples/variational_klein_gordon.py) | PyTorch | classical field theory: the Klein-Gordon plane wave is a least-action solution |
| [`variational_higher_order.py`](examples/variational_higher_order.py) | PyTorch | Euler-Poisson functional derivative: Pais-Uhlenbeck oscillator + Euler-Bernoulli beam |
| [`examples/proof_carrying_pde`](../examples/proof_carrying_pde/) | Pure Python | sealed a-posteriori PDE certificate + proof-machine verdict |
| [`examples/certified_fluid_dynamics`](../examples/certified_fluid_dynamics/) | Pure Python | residual-only Navier-Stokes certificate (Taylor-Green / Kolmogorov) + proof-machine verdict |

## Running

```bash
# PyTorch quickstart
pip install omnibias-torch
python docs/examples/quickstart_torch.py

# Keras 3 unified backend (pick a backend)
pip install omnibias-keras[jax]
KERAS_BACKEND=jax python docs/examples/quickstart_keras.py

# Least action (harmonic oscillator, brachistochrone, Klein-Gordon, higher order)
pip install omnibias-variational[torch]
python docs/examples/variational_harmonic_oscillator.py
python docs/examples/variational_higher_order.py
# geodesics also need omnibias-geometry:
pip install "omnibias-variational[torch,geometry]"
python docs/examples/variational_geodesic.py

# Certified lattice mass gap (fixed transfer matrix; not a continuum claim)
pip install omnibias-geometry
python docs/examples/gauge_certified_mass_gap.py

# Proof-carrying PDE certificate
python -m examples.proof_carrying_pde.run_demo

# Certified fluid dynamics (Taylor-Green / Kolmogorov)
python -m examples.certified_fluid_dynamics.run_demo
```

Every example is deterministic (fixed seeds) so you can use the printed
numbers as a smoke check.

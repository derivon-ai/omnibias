# Primitive packages

Shared infrastructure belongs here; solver front-ends and application experiments
live in standalone repositories under `../omnibias_projects/`.

<!-- BEGIN GENERATED PACKAGE INVENTORY -->

| Distribution | Version | Python | Maturity | License | Responsibility |
| --- | --- | --- | --- | --- | --- |
| [omnibias-binary](api/binary.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | Apache-2.0 | Closed-form quantization gradients for binary/ternary/k-bit neural-network training via the omnibias tanh-beta Riccati derivative (torch + jax). |
| [omnibias-boolean](api/boolean.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | Apache-2.0 | Differentiable Boolean algebra: exact ANF/Reed-Muller and Walsh spectra, Boolean differential calculus, reproductive equation solving, and a beta-annealed soft-gate system solver built on the omnibias closed-form derivative towers (torch + jax). |
| [omnibias-convex](api/convex.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Differentiable + certified convex optimization (LP/QP) for omnibias: closed-form-Hessian log-barrier interior-point solver, KKT implicit-function gradients (argmin as a differentiable op), and verified optimality enclosures (jax + torch). |
| [omnibias-core](api/core.md) | 0.5.0rc1 | >=3.10 | 4 - Beta | Apache-2.0 | Numerically-stable closed-form n-th derivative forward-pass framework: pure-Python core with polynomial coefficient generators (Eulerian / Legendre / Hermite) and the backend-agnostic ActivationSpec protocol. |
| [omnibias-curvature](api/curvature.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Closed-form parameter Hessian / Gauss-Newton Fisher / KFAC factors for one-hidden-layer Riccati fields, built on the omnibias closed-form σ' / σ'' primitives. |
| [omnibias-difference](api/difference.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | Apache-2.0 | The founding delta->0 (multi-bias collapse) register: certified finite-difference -> derivative extraction, umbral / Sheffer sequence calculus, and asymptotic-coefficient reading (Stirling / Bernoulli / Euler numbers) read straight off the closed-form omnibias towers. Pure-Python core + rigorous interval-tower certificates from omnibias.core.verified, with bit-identical torch/jax twins for the finite-difference stencil operator. |
| [omnibias-discrete](api/discrete.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Shared differentiable + certified discrete-optimization substrate for omnibias: the DiscreteProblem seam, the annealed temperature-collapse (beta -> inf) sigmoid relaxation solved by unrolled descent (torch + jax twins), a rounding + k-flip local-search decoder with a brute-force oracle, and a rigorous optimality-gap certificate (Lasserre / moment-SOS lower bound over the Boolean hypercube). |
| [omnibias-fields](api/fields.md) | 0.2.0rc1 | >=3.10 | 4 - Beta | Apache-2.0 | Backend-agnostic field substrate for omnibias: the FieldState value object, the attribute-DSL views, the lazy sigma^(n) cache, and the cross-backend (torch + jax) closed-form differential-operator surface (gradient, divergence, curl, laplacian, hessian, jacobian, integration, inner products, Sobolev norms, tensor divergence, Wirtinger). |
| [omnibias-graph](api/graph.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Differentiable spectral graph operators (Laplacians, spectral embedding, heat kernel) and continuous combinatorial relaxations (Gumbel-Sinkhorn, SoftSort, soft top-k) with torch + jax bit-parity. |
| [omnibias-jax](api/jax.md) | 0.5.0rc1 | >=3.10 | 4 - Beta | Apache-2.0 | JAX backend for omnibias: closed-form n-th derivative activation kernels (sigmoid via Eulerian polynomials, tanh via Legendre, Gaussian via Hermite), neural-field Laplacian / Hessian primitives, and Born-Oppenheimer derivative tools for variational quantum Monte Carlo. |
| [omnibias-keras](api/keras.md) | 0.0.2a1 | >=3.10 | 3 - Alpha | Apache-2.0 | Keras 3 unified backend for omnibias: closed-form n-th derivative scalar operators (OMBU), operator-typed blocks, and drop-in cmbDense / cmbConv layers that run on TensorFlow, JAX, or PyTorch via keras.ops. |
| [omnibias-partition](api/partition.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Trainable soft partitions of unity: oblique, axis-aligned and sparse gates, regional expert composition, temperature hardening and scoped membership-gap certificates, with NumPy, PyTorch and JAX implementations. |
| [omnibias-qcalculus](api/qcalculus.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | Apache-2.0 | Quantum / q-calculus: exact q-numbers, q-factorials, Gaussian (q-)binomials and q-Pochhammer symbols, the Jackson q-derivative and q-integral, q-exponentials and q-deformed Bernoulli / Euler numbers, and basic hypergeometric series with certified geometric tails. The q -> 1 limit recovers ordinary calculus (a distinct limit, never conflated with the delta -> 0 founding collapse). Built on omnibias-core and omnibias-difference, with bit-identical torch/jax Jackson-derivative twins. |
| [omnibias-sos](api/sos.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Certified universal positivity by optimization: sound Sum-of-Squares / Positivstellensatz decompositions. A floating-point SDP proposes a Gram matrix; the proof is a rigorous interval LDL^T positive-definiteness certificate that reuses omnibias.core.verified and the Mathlib-free Lean kernel obligation, so certificates can earn theorem_prover_verified. |
| [omnibias-struct](api/struct.md) | 0.1.0a2 | >=3.10 | 3 - Alpha | AGPL-3.0-or-later **or commercial** | Certified differentiable dynamic programming: soft Viterbi / shortest-path / CTC layers whose logsumexp_beta relaxation (the beta->inf temperature axis) is differentiated exactly by the closed-form softplus/sigmoid derivative tower (the delta->0 axis) via omnibias.{torch,jax}.jet, with a closed-form logsumexp_beta >= max gap certificate validated against brute-force hard DP; bit-identical torch + jax twins. |
| [omnibias-torch](api/torch.md) | 0.5.0rc1 | >=3.10 | 4 - Beta | Apache-2.0 | PyTorch backend for omnibias: trainable scalar operators (OMBU), operator-typed blocks, closed-form activation derivative kernels, and reference PINN / CmbNet / CvxLayer architectures. |

<!-- END GENERATED PACKAGE INVENTORY -->

For a PINN, start with `omnibias-torch` or `omnibias-jax`. Add `fields` when
integrating the field-state API, `partition` for region models, and `curvature`
for supported parameter-curvature operations. Install only what the application
uses. The existing PINN solver is the external `omnibias-pinn` repository.

Versions, maturity, Python requirements, licenses and summaries above come from
package metadata. API inventories come from public module paths and explicit
exports; they do not import optional numerical backends. Regenerate these facts
with `python scripts/generate_inventory.py`; CI rejects stale output.
Package-specific requirements and optional extras live in each `pyproject.toml`.

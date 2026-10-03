# Primitive packages

The repository contains 16 independently installable distributions. Shared
infrastructure belongs here; solver front-ends and application experiments
live in standalone repositories under `../omnibias_projects/`.

| Distribution | Responsibility |
| --- | --- |
| [omnibias-core](api/core.md) | Shared algebra and verified numerics |
| [omnibias-torch](api/torch.md) | PyTorch derivatives and trainable operators |
| [omnibias-jax](api/jax.md) | JAX derivatives and trainable operators |
| [omnibias-keras](api/keras.md) | Keras 3 activation and operator layers |
| [omnibias-fields](api/fields.md) | Field calculus for PINN integration |
| [omnibias-difference](api/difference.md) | Finite-difference primitives |
| [omnibias-partition](api/partition.md) | Soft partitions of unity |
| [omnibias-discrete](api/discrete.md) | Shared discrete optimization seam |
| [omnibias-struct](api/struct.md) | Differentiable structured computation |
| [omnibias-binary](api/binary.md) | Differentiable quantization primitives |
| [omnibias-boolean](api/boolean.md) | Boolean algebra primitives |
| [omnibias-qcalculus](api/qcalculus.md) | q-calculus primitives |
| [omnibias-convex](api/convex.md) | Convex optimization and certificate backend |
| [omnibias-sos](api/sos.md) | Polynomial positivity certificates |
| [omnibias-curvature](api/curvature.md) | Parameter curvature primitives |
| [omnibias-graph](api/graph.md) | Differentiable graph primitives |

For a PINN, start with `omnibias-torch` or `omnibias-jax`. Add `fields` when
integrating the field-state API, `partition` for region models, and `curvature`
for supported parameter-curvature operations. Install only what the application
uses. The existing PINN solver is the external `omnibias-pinn` repository.

Core, PyTorch and JAX form the stable derivative layer. Fields is beta;
Keras and the other extensions retain their existing alpha maturity.
Package-specific requirements and optional extras live in each `pyproject.toml`.
License terms differ by distribution; consult its `LICENSE` before adoption.

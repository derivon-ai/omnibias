# omnibias

<picture>
  <source media="(max-width: 600px)" srcset="docs/img/omnibias-hero-mobile.svg">
  <img src="docs/img/omnibias-hero.svg" width="1280" alt="omnibias — high-order derivatives, direct and trainable. Riccati derivative polynomials and tanh derivative curves.">
</picture>

[![CI](https://github.com/derivon-ai/omnibias/actions/workflows/ci.yml/badge.svg)](https://github.com/derivon-ai/omnibias/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/read-the_docs-087f72)](https://omnibias.ai/)
[![PyPI](https://img.shields.io/pypi/v/omnibias-torch?label=PyPI%20%C2%B7%20torch)](https://pypi.org/project/omnibias-torch/)
[![Open core](https://img.shields.io/badge/core-Apache--2.0-087f72)](LICENSING.md)
[![Commercial licensing](https://img.shields.io/badge/certified_tier-AGPL_or_commercial-5964b4)](COMMERCIAL-LICENSE.md)

**High-order derivatives without nested spatial autodiff.**

Train physics-informed networks with derivative towers—and extend them with
smooth, trainable decisions.

omnibias is a mathematical foundation for **Physics-Informed Neural Networks
(PINNs)** and differentiable models. It turns activation identities into
reusable derivative towers, propagates Taylor jets through supported networks,
and provides smooth routing primitives for models with trainable decisions.
Spatial derivatives remain differentiable with respect to model parameters:
build a physics residual, a regional model or a soft tree, then train it with
the optimizer that fits your problem.

**[Start with a PINN →](docs/pinn.md)** ·
**[Choose an API →](docs/derivatives.md)** ·
**[Explore the packages →](docs/packages.md)** ·
**[Commercial support →](mailto:info@derivon.ai)**

## The expensive derivative does not have to be another backward pass

A fourth-order PDE residual needs fourth spatial derivatives. Building those
by repeatedly differentiating a reverse-mode graph can make graph construction,
memory and evaluation dominate the actual learning problem. Computing many
mixed partials compounds the cost.

omnibias takes a direct route. For supported activations, polynomial recurrences
evaluate `σ⁽ⁿ⁾(z)` without recursive autodiff. Sigmoid and tanh need **one base
activation evaluation plus a derivative polynomial**, regardless of the number
of backward passes a nested approach would build. Taylor composition carries these
derivatives through affine layers and nonlinearities. A single propagation
returns the requested directional jet; a multivariate jet returns mixed
partials. The spatial derivative is an ordinary differentiable tensor
expression, so parameter training still uses the framework you already know.

The result is a practical separation: **forward propagation for spatial
derivatives, ordinary autodiff for parameter learning**. It is useful wherever
a loss depends on derivatives of a learned function—not only PINNs, but also
curvature-sensitive objectives and derivative-based scientific models.

| Build with | What omnibias provides |
| --- | --- |
| **High-order physics residuals** | Directional jets, mixed partials, specialized Laplacian and repeated-Laplacian paths |
| **Field calculus** | Gradients, divergence, curl, Hessians and reusable field state |
| **Trainable decisions** | Soft partitions, differentiable selection and structured computation |
| **Curvature-aware learning** | Parameter-curvature primitives and optimizers for supported objectives |
| **Scoped numerical guarantees** | Outward-rounded intervals, Taylor models and checker-backed certificates |

PyTorch and JAX provide the network-jet APIs. Keras 3 provides activation and
operator layers across its supported backends. These are complementary
surfaces; choose the backend and primitive your application actually needs.

## Derivative performance, by workload

Activation derivatives, direct Laplacians and general deep-network jets use
separate algorithms. The specialized paths exploit activation identities and
operator contractions; composing every derivative through a deep MLP does
more work. See the [full measurements](docs/performance.md) for all baselines,
accuracy checks, compilation costs and reproduction commands.

| Workload | omnibias | Baseline | Speedup |
| --- | ---: | ---: | ---: |
| Activation derivative · `n = 8` | 0.1405 ms | 30.9908 ms · Torch nested autograd | **220×** |
| Laplacian · `D = 60` | 0.0227 ms | 0.5544 ms · JAX dense Hessian | **24.4×** |
| Repeated Laplacian · `Δ³` | 0.0134 ms | 66.4122 ms · JAX dense nested | **4,974×** |
| Repeated Laplacian · `Δ⁴` | 0.0129 ms | 64.8343 ms · folx nested | **5,036×** |

![Activation derivatives, Laplacians and repeated Laplacians compared with autodiff baselines on their respective workloads.](docs/img/specialized-derivatives.svg)

Float64 CPU, nine timed repeats. Activation: 20,000 tanh inputs, Torch eager.
Laplacian: 64 points, 32 hidden units. Repeated Laplacian: 32 points, 16 hidden
units, 16 dimensions. Operator comparisons use JAX JIT with **runtime inputs
and weights**; compilation is recorded separately. All successful methods
pass independent 80-digit accuracy checks at sampled inputs. Dense `Δ⁴` reached
the 3 GiB process budget; no speedup is claimed for that unfinished run.

[Activation data](docs/benchmarks/derivative_order.json) ·
[Laplacian data](docs/benchmarks/laplacian_scaling.json) ·
[Repeated-Laplacian data](docs/benchmarks/polylaplacian_order.json)

The independent deep-MLP comparison remains available in the
[performance guide](docs/performance.md#general-deep-network-jets), including
JAX Taylor-mode AD and cases where it wins.

## High-dimensional physics without a full derivative tensor

The one-layer identity is direct:

$$
\Delta^k f(x) = \sum_h c_h\,\sigma^{(2k)}(w_h\cdot x+\beta_h)\,\lVert w_h\rVert^{2k}.
$$

No dense Hessian or order-`2k` spatial tensor is needed. Deep MLPs use
`deep_field_laplacian` to propagate the Laplacian directly, with **no fixed
input-dimension ceiling** and retained parameter gradients. Both backends
exercise this path at **5,000 dimensions**, beyond the full mixed-jet budget.
For fixed layer widths and depth, work and memory grow linearly with dimension.

In automatic mode, deep repeated Laplacians use exact support enumeration
while it fits the configured budget, then a reported directional estimator. Full
mixed jets still have combinatorial output size. The
[5,000-dimensional training example](docs/derivatives.md#a-deep-laplacian-in-5000-dimensions)
and [operator guarantees](docs/guarantees.md#laplacians-without-the-mixed-jet-dimension-ceiling)
explain which path to choose.

## Install, then differentiate

The published backend distributions are independently installable:

```bash
pip install omnibias-torch
# Alternatives: omnibias-jax or omnibias-keras
```

This README describes the current source tree. Published packages can lag the
branch, especially alpha extensions. To run both examples against this exact
checkout, install the workspace; no application repository is required:

```bash
git clone https://github.com/derivon-ai/omnibias.git
cd omnibias
uv sync --all-packages --group docs
# Run the examples with uv run python.
```

**A fourth spatial derivative, still trainable.** The jet stores Taylor
coefficients; `jet_to_tower` restores ordinary derivatives before the loss.

```python
import torch
from omnibias.torch.jet import jet_to_tower, mlp_jet

weight = torch.tensor([[0.7], [-0.4]], requires_grad=True)
bias = torch.tensor([0.1, 0.2], requires_grad=True)
readout = torch.tensor([[1.0, -0.5]], requires_grad=True)
layers = [(weight, bias, "tanh"), (readout, None, None)]
x = torch.linspace(-1.0, 1.0, 32).reshape(-1, 1)

tower = jet_to_tower(mlp_jet(x, torch.ones_like(x), layers, order=4))
u, du, d2u, d3u, d4u = tower.unbind(0)
loss = (d4u - 24.0).square().mean()
loss.backward()
assert weight.grad is not None
```

This is a derivative-and-gradient demonstration. A PDE solution also needs
appropriate boundary conditions, collocation and independent validation.
The [PINN guide](docs/pinn.md) adds those ingredients and an optimizer step;
the [derivative guide](docs/derivatives.md) covers mixed partials and JAX.

## Give your network a smooth if/else

A hard split routes an input to one branch. A soft split learns the routing:
`g(x) = sigmoid(β(w·x − t))`. Two branches become
`(1 − g)·f₀(x) + g·f₁(x)`, keeping gradients into the threshold, split direction
and branch models. Multiple splits form nonnegative regional weights that sum
to one. This lets conditional structure participate in learning.

**A two-leaf soft tree with trainable regional models:**

```python
import torch
from omnibias.partition.torch import combine, partition_weights_arrays

features = torch.tensor([[-0.8], [0.4], [1.2]])
split = torch.tensor([[1.0]], requires_grad=True)
threshold = torch.tensor([0.0], requires_grad=True)
experts = torch.nn.Linear(1, 2)

memberships = partition_weights_arrays(
    split, threshold, features, beta=4.0, depth=1
)
prediction = combine(memberships, experts(features))
prediction.square().mean().backward()
assert split.grad is not None and experts.weight.grad is not None
```

Use the same mechanism for neural feature encoders, mixtures of regional
models, and soft-tree heads. The external `omnibias-tab` consumer builds
GBM-style Newton boosting and trainable neural/tree compositions on these
primitives; those are distinct training paths, not an automatic conversion
of an existing hard GBM into a differentiable network.

Increasing `β` sharpens routing toward a hard partition away from split ties.
The finite-temperature model is the differentiable one; an exact hard if/else
does not acquire a derivative at its jump. See the
[partition API](docs/api/partition.md) for hardening, regional models and
scoped soft-to-hard certificates.

## Attach a guarantee to the quantity you actually checked

`omnibias.core.verified` supplies outward-rounded interval arithmetic and
Taylor-model enclosures. Optimization primitives add certificates for their
supported problems. These tools can bound a stated quantity over a stated
domain under explicit assumptions; they do not turn a sampled training loss
into a proof about every point in a continuous domain.

Certificate digests protect integrity. Formal-verification flags require the
corresponding checker to pass. This distinction makes the result inspectable:
read the model, domain, assumptions and earned checks alongside the numerical
bound. The [guarantees](docs/guarantees.md) define that contract.

## A small foundation, room for your application

The core owns shared coefficients and combinatorics; backend packages realize
them as tensors; field and decision primitives compose them. The same
Riccati identities behind sigmoid and tanh feed higher derivatives without
duplicating coefficient implementations across frameworks.

Two useful limits remain distinct: shrinking a bias spacing yields a
derivative, while increasing inverse temperature sharpens a soft decision.
The [derivative guide](docs/derivatives.md) explains the normalization and
the [guarantees](docs/guarantees.md) spell out precision and cost.

<!-- BEGIN GENERATED PACKAGE INVENTORY -->

| Distribution | Version | License |
| --- | --- | --- |
| [omnibias-binary](docs/api/binary.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-boolean](docs/api/boolean.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-convex](docs/api/convex.md) | 0.1.0a1 | AGPL-3.0-or-later **or commercial** |
| [omnibias-core](docs/api/core.md) | 0.4.0 | Apache-2.0 |
| [omnibias-curvature](docs/api/curvature.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-difference](docs/api/difference.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-discrete](docs/api/discrete.md) | 0.1.0a1 | AGPL-3.0-or-later **or commercial** |
| [omnibias-fields](docs/api/fields.md) | 0.1.0 | Apache-2.0 |
| [omnibias-graph](docs/api/graph.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-jax](docs/api/jax.md) | 0.4.0 | Apache-2.0 |
| [omnibias-keras](docs/api/keras.md) | 0.0.1a1 | Apache-2.0 |
| [omnibias-partition](docs/api/partition.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-qcalculus](docs/api/qcalculus.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-sos](docs/api/sos.md) | 0.1.0a1 | AGPL-3.0-or-later **or commercial** |
| [omnibias-struct](docs/api/struct.md) | 0.1.0a1 | Apache-2.0 |
| [omnibias-torch](docs/api/torch.md) | 0.4.0 | Apache-2.0 |

<!-- END GENERATED PACKAGE INVENTORY -->

The table is generated from package metadata. Solvers, products and research
applications live in separate repositories; they do not expand the primitive
API by default. A PINN integration can begin with a backend, then add `fields`,
`partition` or `curvature` as its requirements grow.

High-order derivatives still have floating-point limits. Full mixed jets
still contain combinatorially many coefficients. Exact differentiation does
not guarantee good optimization or convergence to a PDE solution. Shared
coefficients support tested backend parity, not universal bit identity across
every device and compiler.

## Open core. A commercial path when you need one.

The derivative and field packages are **Apache-2.0**, including commercial and
closed-source use under its terms. The certified optimization tier offers
**AGPL-3.0-or-later or a commercial agreement**. Package boundaries and license
metadata are checked in CI; see [Licensing](LICENSING.md) for the exact grants.

## Contact

omnibias is built and maintained by **[Derivon](https://derivon.ai/)**.
See [Governance](GOVERNANCE.md) and [Maintainers](MAINTAINERS.md) for how
project decisions are made.

| Topic | Contact |
| --- | --- |
| Commercial licensing, integration and support | [info@derivon.ai](mailto:info@derivon.ai) |
| Technical questions and partnerships | [info@derivon.ai](mailto:info@derivon.ai) |
| Bug reports and feature requests | [GitHub Issues](https://github.com/derivon-ai/omnibias/issues) |
| Security reports | [Security policy](SECURITY.md) |

Contributions are welcome under the [CLA](CLA.md). Read
[Contributing](CONTRIBUTING.md), the [Code of Conduct](CODE_OF_CONDUCT.md),
and the [agent guide](AGENTS.md) before making a change.

## Citation

```bibtex
@software{omnibias,
  title   = {omnibias: closed-form n-th derivatives of activations},
  author  = {Grigoryants, Vardan},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/derivon-ai/omnibias}
}
```

GitHub's “Cite this repository” button reads [CITATION.cff](CITATION.cff).

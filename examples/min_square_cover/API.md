# `omnibias-shape` API specification

Status: proposed (blueprint). This is a **specification**, not shipped code. It defines
a new alpha package `packages/omnibias-shape` that provides differentiable shape /
occupancy fields and soft-coverage operators with a **closed-form derivative tower**, so
downstream code (the [min_square_cover example](PLAN.md)) can build coverage energies
whose gradient and Hessian are exact.

The package is the "missing primitive" identified in the analysis: omnibias has the
sigmoid derivative tower, a field substrate, a log-sum-exp smooth-max, `beta` annealing,
cardinality relaxations, second-order optimizers, differentiable LP, and certified
enclosures, but nothing that exposes a **soft shape / occupancy** or a **soft union /
coverage** as a first-class op. `omnibias-shape` fills exactly that gap.

## Design principles

- Mirror the shape of an existing alpha package,
  [omnibias-graph](../../packages/omnibias-graph): the root
  `omnibias.shape.__init__` exports only `__version__`; all operators live under
  `omnibias.shape.torch.ops` and `omnibias.shape.jax.ops` as **bit-identical twins**
  (same `__all__`, same algorithm, coefficients shared from `omnibias-core`).
- Occupancy derivatives are **closed form**, not autodiff: every center-derivative of a
  soft box is a polynomial in `sigmoid` evaluated by
  [riccati_sigmoid_derivative](../../packages/omnibias-binary/src/omnibias/binary/torch/ops/quantize.py)
  (built on
  [sigmoid_polynomial_coeffs](../../packages/omnibias-core/src/omnibias/core/polynomials.py)).
- Default tensors use the framework default dtype
  (`torch.get_default_dtype()` / the JAX default), never a hardcoded `float32`.
- Honest labels: "closed form" is reserved for the sigmoid-built occupancy and coverage;
  anything that composes a user callable by autodiff is labelled as such.

## Package layout

```
packages/omnibias-shape/
  pyproject.toml           # setuptools; version = "0.1.0a1"; Alpha classifier
  README.md                # **Status: Alpha (0.1.0a1).** + capabilities + honest scope
  LICENSE                  # copied from a sibling package
  src/omnibias/shape/
    __init__.py            # exports __version__ only
    py.typed
    torch/
      __init__.py          # re-exports omnibias.shape.torch.ops surface
      ops/
        __init__.py        # sorted __all__ re-export
        occupancy.py       # soft shapes + closed-form derivatives
        coverage.py        # soft-OR / LSE union + coverage energy / residual / grad / hessian
        cardinality.py     # gate L0 surrogate + anneal helpers
    jax/
      __init__.py
      ops/
        __init__.py
        occupancy.py       # bit-identical twin
        coverage.py
        cardinality.py
  tests/                   # torch <-> jax parity + closed-form vs autodiff
```

Dependencies (in `pyproject.toml`): `omnibias-core` (polynomials), `omnibias-binary`
(Riccati derivative evaluation + `BetaAnnealScheduler`), and `omnibias-hopfield`
(log-sum-exp smooth-max). Optional extras `torch` and `jax`, matching
[packages/omnibias-graph/pyproject.toml](../../packages/omnibias-graph/pyproject.toml).

## Conventions

- A pixel grid of shape `(n_1, ..., n_D)` is described by a tuple of `D` one-dimensional
  coordinate vectors `axes = (ax_0, ..., ax_{D-1})` (`ax_d` has length `n_d`); a 2-D image
  is `axes = (rows, cols)` with `rows` of length `M` and `cols` of length `N`.
- `centers` is `(K, D)` for `K` shapes; `side` is a scalar or `(D,)`; `beta > 0` is the
  sharpness; `gates` is `(K,)` in `[0, 1]`.
- Occupancy tensors are `(K, n_1, ..., n_D)`; a coverage field is `(n_1, ..., n_D)`.
- All ops are pure functions (no in-place mutation of inputs) and are `vmap` / `jit`
  friendly on the JAX side.

## `occupancy.py`

```python
def soft_interval(t, center, side, beta):
    """1-D soft interval indicator (a difference of two sigmoids), in (0, 1).

    box(t) = sigmoid(beta (t - center + side/2)) - sigmoid(beta (t - center - side/2)).
    Broadcasts over ``t`` and ``center``. As ``beta -> inf`` it converges pointwise to
    the hard indicator of ``[center - side/2, center + side/2]``.
    """

def soft_box(axes, centers, side, beta):
    """Separable soft box / hyper-rectangle occupancy.

    Returns ``(K, n_1, ..., n_D)`` with ``m[k] = prod_d soft_interval(axes[d], centers[k, d], side_d, beta)``.
    Axis-aligned square when ``D == 2`` and ``side`` is scalar.
    """

def soft_box_grad(axes, centers, side, beta):
    """Closed-form gradient of the box occupancy w.r.t. each center coordinate.

    Returns ``(K, D, n_1, ..., n_D)`` where ``out[k, d] = d m[k] / d centers[k, d]``.
    Uses the order-1 Riccati derivative of ``sigmoid`` per axis; no autodiff.
    """

def soft_box_hessian(axes, centers, side, beta):
    """Closed-form per-shape Hessian of the box occupancy w.r.t. its own center.

    Returns ``(K, D, D, n_1, ..., n_D)`` where ``out[k, a, b] = d^2 m[k] / d centers[k, a] d centers[k, b]``.
    Diagonal ``a == b`` uses the order-2 Riccati derivative; off-diagonal ``a != b`` is the
    product of two order-1 axis derivatives (separability). See ``HESSIAN.md``.
    """

# Extensions (P4 in PLAN.md), same closed-form-derivative contract:
def soft_disk(axes, centers, radius, beta):
    """Soft disk / ball occupancy: sigmoid(beta (radius^2 - ||x - center||^2))."""

def soft_polytope(axes, halfplanes, beta):
    """Soft convex polytope: a smooth AND (min-of-sigmoids via -logsumexp of -a_i) over
    half-plane constraints ``n_i . x <= b_i``. Reuses the hopfield log-sum-exp kernel."""
```

Implementation note: `soft_box_hessian` and `soft_box_grad` evaluate the one-dimensional
factors `b, b', b''` once per axis via `riccati_sigmoid_derivative(s, order)` for
`order in {1, 2}`, then assemble the separable outer products (the exact expressions in
[HESSIAN.md](HESSIAN.md)).

## `coverage.py`

```python
@dataclass
class CoverageCache:
    """Reusable products for the soft-OR gradient / Hessian (avoids recompute)."""
    coverage: Tensor        # C, shape (n_1, ..., n_D)
    product: Tensor         # P = prod_k (1 - alpha_k m_k)
    leave_one_out: Tensor   # P_{\k}, shape (K, n_1, ..., n_D)

def soft_or_coverage(occupancy, gates):
    """Probabilistic-OR union coverage.

    C = 1 - prod_k (1 - gates_k * occupancy_k). Returns ``(C, CoverageCache)``.
    Multilinear in each ``occupancy_k`` (so d^2 C / d m_k^2 = 0), which is what makes the
    coverage Hessian clean; see ``HESSIAN.md``.
    """

def lse_coverage(occupancy, gates, beta):
    """Log-sum-exp smooth-max union (alternative to soft-OR), reusing
    ``omnibias.hopfield`` ``logsumexp_value`` / ``logsumexp_hessian``."""

def coverage_energy(occupancy, gates, ones_mask, *, loss="softplus", kappa=1.0,
                    lam=0.0, bg_mask=None, mu=0.0):
    """Scalar coverage energy for a scalar-loss optimizer closure.

    E = sum_{ones} L(1 - C) + lam * sum_k gates_k + mu * sum_{bg} (coverage of background),
    where ``L`` is ``"softplus"`` (smooth hinge) or ``"sq_hinge"`` (squared hinge). The
    ``lam`` term is the L0 count surrogate; the optional ``mu`` term is off by default
    (covering background is allowed by the problem).
    """

def coverage_residual(occupancy, gates, ones_mask, *, weight=1.0):
    """Under-coverage residual vector r = sqrt(weight) * (1 - C) restricted to the
    1-pixels, for the Gauss-Newton residual closure (objective 1/2 ||r||^2)."""

def coverage_energy_grad(params, axes, side, beta, ones_mask, *, loss="softplus",
                         kappa=1.0, lam=0.0, wrt="centers"):
    """Closed-form gradient of ``coverage_energy`` w.r.t. packed ``params``.

    ``params`` packs ``centers`` (K, D) and, when ``wrt="all"``, the gate logits (K,).
    Returns a flat gradient of length ``P_dim``.
    """

def coverage_energy_hessian(params, axes, side, beta, ones_mask, *, loss="softplus",
                            kappa=1.0, lam=0.0, wrt="centers", gauss_newton=False):
    """Closed-form dense Hessian of ``coverage_energy`` (the deliverable-c implementation).

    Returns ``(P_dim, P_dim)``. ``gauss_newton=True`` drops the residual-curvature terms
    (keeps only the PSD J^T J part). The full derivation and the exact block formulas are
    in ``HESSIAN.md``; this function is validated against ``torch.func.hessian`` and
    central finite differences (a package test).
    """
```

## `cardinality.py`

```python
def l0_surrogate(gates, *, kind="sum", eps=1e-3):
    """Differentiable "number of active shapes" surrogate. ``"sum"`` is sum(gates);
    ``"concave"`` is a sharper L0 surrogate ``sum(gates / (gates + eps))``."""

def anneal_lambda(step, *, lam_start, lam_end, num_steps, schedule="linear"):
    """Count-penalty schedule companion to ``BetaAnnealScheduler`` (grow ``lam`` as the
    coverage constraint is satisfied, so squares are removed only once cover is feasible)."""

def prune_inactive(centers, gate_logits, *, threshold=0.5):
    """Drop shapes whose gate has collapsed below ``threshold``; returns the pruned
    ``(centers, gate_logits)`` for a smaller subsequent solve."""
```

`beta` itself is scheduled with the shipped
[BetaAnnealScheduler](../../packages/omnibias-binary/src/omnibias/binary/schedule.py);
target counts can be relaxed with
[soft_top_k](../../packages/omnibias-graph/src/omnibias/graph/torch/ops/relaxation.py)
when a fixed budget is preferred over an `L0` penalty.

## `torch/ops/__init__.py` (public surface)

```python
from omnibias.shape.torch.ops.occupancy import (
    soft_box, soft_box_grad, soft_box_hessian, soft_disk, soft_interval, soft_polytope,
)
from omnibias.shape.torch.ops.coverage import (
    CoverageCache, coverage_energy, coverage_energy_grad, coverage_energy_hessian,
    coverage_residual, lse_coverage, soft_or_coverage,
)
from omnibias.shape.torch.ops.cardinality import anneal_lambda, l0_surrogate, prune_inactive

__all__ = [
    "CoverageCache", "anneal_lambda", "coverage_energy", "coverage_energy_grad",
    "coverage_energy_hessian", "coverage_residual", "l0_surrogate", "lse_coverage",
    "prune_inactive", "soft_box", "soft_box_grad", "soft_box_hessian", "soft_disk",
    "soft_interval", "soft_or_coverage", "soft_polytope",
]
```

The `omnibias.shape.jax.ops` twin re-exports the identical `__all__`.

## How the example consumes it

Optimize the continuous-energy register (scalar-loss closure; the exact drop-in
optimizers apply the Hessian matrix-free, while `coverage_energy_hessian` provides the
closed-form cross-check):

```python
from omnibias.shape.torch import ops as shape
from omnibias.torch.optim import CubicNewton   # optim.py L1747

opt = CubicNewton([centers, gate_logits])
def closure():
    occ = shape.soft_box(axes, centers, side, beta.value(step))
    return shape.coverage_energy(occ, gate_logits.sigmoid(), ones_mask, lam=lam)
opt.step(closure)                               # scalar loss, graph intact, no .backward()
```

Gauss-Newton register (residual-vector closure) via
[GaussNewton](../../packages/omnibias-torch/src/omnibias/torch/optim.py) (L584) /
`CubicGaussNewton` (L1839) with `functional_residual_fn`:

```python
def residual(params):
    occ = shape.soft_box(axes, unpack_centers(params), side, beta.value(step))
    return shape.coverage_residual(occ, unpack_gates(params).sigmoid(), ones_mask)
```

Certified optimality gap (LP-relaxation register) via
[omnibias-convex](../../packages/omnibias-convex/src/omnibias/convex/torch/__init__.py):
build the pixel-aligned candidate cover matrix `C` (`C[p, j] = 1` iff candidate square
`j` covers 1-pixel `p`) and solve fractional cover `min 1^T x s.t. C x >= 1, 0 <= x <= 1`
as `A x <= b` with `solve_lp`; the LP value is a certified lower bound `L`, optionally
enclosed by `certify_qp_optimum`
([certify.py](../../packages/omnibias-convex/src/omnibias/convex/certify.py)).

Certified cover / robustness via
[omnibias-verify](../../packages/omnibias-verify/src/omnibias/verify/_core/global_opt.py):
`certified_minimize(f, box, ...)` (L216) to prove `max over the 1-cell of the
under-coverage is <= 0` for the rounded solution, and `reachable_box` / `lipschitz_bound`
([certificates.py](../../packages/omnibias-verify/src/omnibias/verify/_core/certificates.py)
L230 / L167) for the position-jitter robustness margin.

## Monorepo wiring checklist

From [.claude/skills/omnibias-dev-new-package/SKILL.md](../../.claude/skills/omnibias-dev-new-package/SKILL.md):

- SPDX header on every `.py`:
  `# SPDX-License-Identifier: Apache-2.0`
  and the copyright line.
- `pyproject.toml`: setuptools backend, `version = "0.1.0a1"`,
  `Development Status :: 3 - Alpha`, `[tool.setuptools.packages.find] where=["src"]
  include=["omnibias.*"]`; a `py.typed` marker; a sorted `__all__` including
  `"__version__"`.
- Add `packages/omnibias-shape` to `[tool.uv.workspace].exclude` in the root
  [pyproject.toml](../../pyproject.toml) (extension packages install per package).
- Add a CI job in [.github/workflows/ci.yml](../../.github/workflows/ci.yml) modeled on a
  backend job (needs torch / jax).
- Add `docs/api/shape.md` (`::: omnibias.shape` + `Status: Alpha`), register it under
  `Packages` in [mkdocs.yml](../../mkdocs.yml) nav, add `packages/omnibias-shape/src` to
  the mkdocstrings `paths` and the docs-job install loop.
- Add entries to [llms.txt](../../llms.txt) and a bullet under `## [Unreleased]` in
  [CHANGELOG.md](../../CHANGELOG.md).
- Vendor-neutral language only (GPU job / GPU cluster; no scheduler, vendor, hostname, or
  absolute local path).
- Typing tier: extension-tier (not on the shared `mypy --strict` gate), but author modules
  strict-clean; do not bump versions unless the task says so.

## Verify

```bash
python -m pytest packages/omnibias-shape/tests -q
uv run ruff check packages tests
mkdocs build --strict
```

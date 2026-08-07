# Release blockers and publish readiness

This document tracks every blocker, caveat, and follow-up that must be resolved
(or explicitly accepted) before a public release. The verified go / no-go
sign-off is [`release-readiness.md`](release-readiness.md); reproduction
is [`reproducibility.md`](reproducibility.md).

## Release tracks

The 42 distributions publish on two tracks (maturity is the package's own
`Development Status` classifier; **track** is a separate release decision):

- **Curated public core** -- published first, held to the
  [API-stability contract](stability.md): `omnibias-core`, `omnibias-torch`,
  `omnibias-jax`, `omnibias-ferminet`, `omnibias-keras`, `omnibias-fields`,
  `omnibias-pinn`, `omnibias-geometry`.
- **Extended set** -- Alpha, on TestPyPI / private until each package's paper
  lands: the remaining 34 distributions (see [`packages.md`](packages.md) for the
  full grouped inventory). Each ships real, tested torch/jax math with its own CI job but is
  **not** under the API-stability contract.

`omnibias-flow`, `omnibias-pde`, and `omnibias-gauge` are **not** separate
distributions -- they are folded into `omnibias.score.flow`,
`omnibias.pinn.solver`, and `omnibias.geometry.gauge`, guarded by
`packages/omnibias-core/tests/test_package_registry.py`.

## Release-gating guards (must be green to tag)

These static/regression guards gate every public tag and run in CI:

- **Placeholders** -- no residual `<...>` legal-identity placeholder in any
  tracked file (`packages/omnibias-core/tests/test_no_placeholders.py`).
- **Version single-source-of-truth** -- `pyproject` metadata matches every
  in-code `__version__`; no stray submodule markers
  (`packages/omnibias-core/tests/test_version_consistency.py`).
- **Leakage / secrets** -- no local paths (`/u/`, `/home/`), scheduler/vendor
  names, private-tree references, or tokens in shipped files
  (`packages/omnibias-core/tests/test_no_leakage.py`).
- **Concept terminology** -- no over-claim vocabulary regressions
  (`test_concept_terminology.py`).
- **Packaging hygiene** -- sorted `__all__`, `py.typed` present, and a clean
  wheel build+import for every publishable distribution.

## Needs GPU-cluster validation before the "production-fidelity" label

These are **not** blockers for an alpha / beta / stable publish at the versions
shipped today, but a release pinned as "validated against production-fidelity
benchmarks" needs each to run green on the GPU cluster. Vendor-neutral run
instructions are in [`benchmarks.md`](benchmarks.md); exact selectors below.

- G1 [core] CPU full parity matrix (~20 min, 16 GB):
  ```bash
  python -m pytest \
    packages/omnibias-core/tests packages/omnibias-torch/tests \
    packages/omnibias-jax/tests packages/omnibias-ferminet/tests \
    tests -m 'not gpu and not slow_gpu' -q
  ```

- G2 [torch] GPU smoke (>= 8 GB, ~5 min): CUDA tensor parity for OMBU +
  OperatorBlock + GrowableOMBU.
  ```bash
  python -m pytest packages/omnibias-torch/tests -m gpu -q
  ```

- G3 [ferminet] local-kinetic-energy regression (>= 20 GB, ~30 min):
  full-fidelity envelope `(value, grad, Hessian)` parity at
  `n_e in {8, 12, 16, 32}`, `H in {64, 128, 256}` against `jax.hessian`.
  ```bash
  python -m pytest packages/omnibias-ferminet/tests -m 'slow_gpu or gpu' -q
  ```

- G4 [pinn] integration against the internal reference (4 h, CPU 8 cores):
  ```bash
  python -m pytest packages/omnibias-pinn/tests \
    -m 'needs_research and not gpu' -q
  ```

- G5 [qpinn] unit + integration (4 h, CPU 8 cores):
  ```bash
  python -m pytest packages/omnibias-qpinn/tests -m 'not gpu' -q
  ```

- G6 [qpinn] GPU integration (Galerkin eigensolver, >= 20 GB, ~45 min):
  ```bash
  python -m pytest packages/omnibias-qpinn/tests/integration -m gpu -q
  ```

- G7 [curvature] smoke (1 h, CPU 4 cores):
  ```bash
  python -m pytest packages/omnibias-curvature/tests -q
  ```

## Accepted limitations (documented, not blockers)

- **Curated Beta packages** (`omnibias-pinn`, `omnibias-fields`,
  `omnibias-geometry`) ship under `Development Status :: 4 - Beta` with a frozen
  public surface. Production-fidelity NS / KS / CH numbers live in the internal
  benchmark archive and are reproduced via G4.
- **`omnibias-keras`** ships in the curated track at `Alpha (0.0.1a1)`: the
  activation-level math is bit-identical to the torch/jax core by construction
  (same `omnibias-core` coefficients), but the Keras 3 wrapper surface may still
  shift between alpha releases.
- **`omnibias-qpinn`** (Alpha): TISE / TDSE / NLS / Helmholtz / Klein-Gordon /
  Dirac residuals are bit-stable cross-backend; the Bloch cage handles only
  2nd-order single-axis derivatives (parity-projected cage on the roadmap).
- **`omnibias-curvature`** (Alpha): one-layer gradient / Hessian / Fisher /
  Newton / KFAC factors; multi-layer assembly and the full torch port are on
  the roadmap.
- **Extended-set API instability**: the 34 alpha distributions may change their
  public surface between alpha releases; only the curated core is frozen.
- **`mypy --strict` scope**: enforced on the T1 workspace
  (`core/torch/jax/ferminet`) and extended to the curated field packages;
  full-alpha strict typing is tracked per package.
- **Regenerated build artifacts**: `packages/*/build/`, `*.egg-info/`, and
  `.pytest_cache/` are git-ignored and regenerated by `pip install -e` /
  `python -m build` -- cosmetic, no fix needed.

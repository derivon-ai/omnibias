# Working on omnibias

omnibias is reusable derivative and optimization infrastructure, with PINNs
as its primary integration target. Read only the files needed for the task.

## Find the right home

- `packages/omnibias-core`: pure-Python coefficients, combinatorics and verified numerics.
- `packages/omnibias-{torch,jax,keras}`: framework implementations.
- `packages/omnibias-fields`: field state, caching and differential operators.
- `packages/omnibias-{curvature,partition}`: parameter curvature and soft partitions.
- Other reusable primitives are listed in `docs/packages.md`.
- Solvers, products and application experiments belong in sibling repositories
  under `../omnibias_projects/`, including `omnibias-pinn`.
- Do not create a distribution unless it has a distinct dependency tier or a
  reusable API with an actual consumer. Prefer a submodule for small additions.

## Numerical contracts

- Activation polynomial coefficients live only in `omnibias.core.polynomials`.
  Core must not import torch, JAX, TensorFlow or Keras.
- Keep torch/JAX jet kernels paired. Shared coefficients alone do not guarantee
  cross-device floating-point identity; preserve existing parity tolerances.
- A directional jet stores `f^(k)/k!`; a multivariate jet stores `D^alpha f/alpha!`.
  Use conversion helpers before treating coefficients as derivatives.
- Negative derivative orders raise `ValueError`; unsupported orders raise
  `NotImplementedError`. Use framework default dtype for new tensors.
- Retain parameter gradients and JAX tracing. Avoid host conversions in kernels.
- Preserve outward rounding and certificate scope. A digest is integrity, not
  a proof; verification flags require their actual checker to pass.
- Never imply that an exact derivative proves convergence of a trained PDE solution.

## Make and verify a change

- Add a regression test for each behavioral change; run the affected package
  tests and corresponding cross-backend parity tests.
- Keep public exports explicit and sorted. Do not bump versions unless requested.
- New modules should be strict-clean. Core, torch and JAX are strict-gated;
  run focused typing checks for extension changes.
- Documentation Python blocks must run, with definitions in the same document.
  Verify real signatures. A necessary opt-out must include a reason.
- Keep tracked files vendor-neutral. No local user paths, hostnames or scheduler
  commands. Put generated artifacts in `$OMNIBIAS_SCRATCH` or `artifacts/`.

```bash
uv sync --all-packages --group docs
uv run pytest packages/omnibias-<name>/tests -q
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src packages/omnibias-jax/src
uv run pytest tests/test_docs_snippets.py -q
uv run mkdocs build --strict
```

## Load context on demand

- Public usage: `docs/pinn.md`, `docs/derivatives.md`, then the relevant `docs/api/` page.
- Changing numerical kernels: `.cursor/skills/omnibias-backends/SKILL.md`.
- Building a PINN consumer: `.cursor/skills/omnibias-pinn/SKILL.md`.
- `AGENTS.md` is the sole general instruction source; adapter files only point here.

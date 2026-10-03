# Contributing

Develop reusable primitives here; build solver products and application
experiments in sibling repositories under `../omnibias_projects/`.
The package map is in [docs/packages.md](docs/packages.md).

```bash
uv sync --all-packages --group docs
uv run pytest packages/omnibias-core/tests -q
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src packages/omnibias-jax/src
uv run pytest tests/test_docs_snippets.py -q
uv run mkdocs build --strict
```

Run the tests for every changed package and any affected cross-backend parity
tests. For Keras, choose `KERAS_BACKEND` before importing it. JAX 64-bit parity
requires `JAX_ENABLE_X64=1` before arrays are created.

A contribution should include a focused implementation, a regression test for
each behavior change, and executable documentation for a new public API.
Use shared core polynomial coefficients, preserve parameter gradients, and
state numerical and approximation limits. Do not change versions unless the
release task asks for it.

[AGENTS.md](AGENTS.md) routes repository maintenance; the
[numerical contracts](docs/development/numerical-contracts.md) cover shared math invariants.
Review [LICENSING.md](LICENSING.md) before adding cross-package dependencies.

Contributions require acceptance of the existing [CLA](CLA.md); the PR bot
records the signature.

Maintainers: [RELEASE.md](RELEASE.md) describes independent package releases and
Trusted Publishing setup.

## Maintaining CLA authentication

CLA Assistant stores signatures in the private `derivon-ai/cla-signatures`
repository on its initialized `main` branch. Its fine-grained token needs
that repository selected under resource owner `derivon-ai`, with **Contents:
read and write** and any required organization approval. The normal GitHub CLI
login separately needs permission to manage secrets in `derivon-ai/omnibias`.

Create `.local/github/cla.yml` (git-ignored) containing paths and settings only:

```yaml
token_file: ~/.config/omnibias/cla-token
repository: derivon-ai/omnibias
signatures_repository: derivon-ai/cla-signatures
branch: main
```

Point `token_file` to the existing file containing one token. Keep that file
outside the checkout. Check access, then upload directly to the Actions secret:

```bash
uv run --group docs python scripts/setup_cla_secret.py
uv run --group docs python scripts/setup_cla_secret.py --apply
```

The helper never displays or copies the token into YAML; upload uses standard
input with command output captured. A 404 can mean missing repository access
or a missing branch. A successful preflight checks read access; the token still
needs write permission for signature persistence. Let CLA Assistant create its
signature file, then rerun the failed workflow. An unsigned contributor must
personally accept the CLA; authentication setup does not provide a signature.

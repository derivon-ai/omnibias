# Contributing to omnibias

Thanks for your interest in improving omnibias. This project is a
numerically-stable, closed-form n-th derivative forward-pass framework,
so correctness and bit-stable cross-backend parity matter more here than
in a typical library. Please read this guide before opening a pull
request.

## License & Contributor License Agreement (read first)

omnibias ships in **two licence tiers**, and which one your change lands in
depends on the package you are touching:

- **Tier P — `Apache-2.0`** (28 packages): the derivative tower and everything
  built directly on it.
- **Tier C — `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`**
  (14 packages): the certified-decision layer.

The authoritative mapping is `[tool.omnibias.license_tiers]` in the root
[`pyproject.toml`](pyproject.toml); [`LICENSING.md`](LICENSING.md) explains the
split. **Never hand-edit an SPDX header** — run

```bash
python scripts/license_headers.py          # stamp every file from the tier table
python scripts/license_headers.py --check  # what CI runs
```

and it will apply the right header for the file's package:

```python
# Tier P
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
```

```python
# Tier C
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
```

The copyright line names Derivon as the **project steward** so the SPDX and
packaging surface stays uniform across every distribution; it does not change
the ownership point below.

**One structural rule.** A permissive package must never depend on a copyleft
package, through `dependencies` or through any `optional-dependencies` extra —
that would silently subject the whole adoption tier to the AGPL.
`packages/omnibias-core/tests/test_license_consistency.py` fails the build if
you add such an edge. If you need certified functionality inside a permissive
package, the answer is to move the *consumer* to Tier C, not to add the extra.

Because of the dual license, **every contributor must sign the
[Contributor License Agreement](docs/CLA.md)** before a pull request can be
merged. An automated check on your PR will prompt you to sign by commenting:

```
I have read the CLA Document and I hereby sign the CLA
```

The CLA is repository-wide and lets the project license your contribution under
Apache-2.0, the AGPL, **and** commercial terms — including moving it between
tiers as packages evolve. **You keep the copyright** to your work; you are
granting a license, not assigning ownership. The
[CLA](docs/CLA.md), not the file header, is what grants the project its license.

## Ground rules

- Be respectful. This project follows the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- You have signed the [CLA](docs/CLA.md) and your contribution is your own
  original work (or you have the right to contribute it).
- Every behavioural change ships with a regression test.
- Public APIs are bit-identical across backends *by construction*
  (the polynomial coefficients come from the shared
  `omnibias.core.polynomials` module). Do not fork the math per backend.
- `omnibias.core` must never import `torch`, `jax`, or `keras`.

## Development setup

The repository is a [uv](https://github.com/astral-sh/uv) workspace.

```bash
git clone https://github.com/derivon-ai/omnibias && cd omnibias
uv sync --all-extras --dev
```

The stable workspace is `omnibias-core` + `omnibias-torch` +
`omnibias-jax` + `omnibias-ferminet`. The extension packages
(`omnibias-pinn`, `omnibias-qpinn`, `omnibias-curvature`) and the
`omnibias-keras` unified backend ship their own `pyproject.toml` and are
installed (and tested) per package:

```bash
pip install -e packages/omnibias-core
pip install -e packages/omnibias-keras[test]
```

## Agent tooling (skills & rules)

If you use an AI assistant (Cursor, Claude Code), the repo ships skills and rules
that encode these conventions so changes stay correct:

- **Maintainer skills** live in `.cursor/skills/omnibias-dev-*` (canonical) and
  are mirrored to `.claude/skills/` by `python scripts/sync_skills.py`. Edit the
  `.cursor` copy and re-run the sync (`--check` runs in CI).
- **Consumer skills** (`omnibias-*`) are the `omnibias-skills` package; the repo's
  committed copies under `.cursor/skills` / `.claude/skills` are produced by
  `omnibias-skills install` and drift-checked in CI against
  `packages/omnibias-skills/src/omnibias/skills/_bundled`. Edit the bundle, not
  the copies.
- **Rules** in `.cursor/rules/` (the universal `omnibias.md` plus path-scoped
  `.mdc` rules) auto-attach by file path.

## Running the checks

```bash
# Tests for the installed workspace
uv run pytest

# A single package
python -m pytest packages/omnibias-core/tests -q

# Lint and type-check (T1 packages are under mypy --strict)
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src \
  packages/omnibias-jax/src packages/omnibias-ferminet/src

# Docs must build clean in strict mode
mkdocs build --strict
```

For the Keras 3 unified backend, select the backend explicitly:

```bash
KERAS_BACKEND=jax        python -m pytest packages/omnibias-keras/tests -q
KERAS_BACKEND=tensorflow python -m pytest packages/omnibias-keras/tests -q
KERAS_BACKEND=torch      python -m pytest packages/omnibias-keras/tests -q
```

### Slow / compute-heavy tests

Some tests (lattice Monte-Carlo, multi-sweep stochastic-quantisation, and the
JAX-eager lattice paths) are memory- and time-heavy and are marked
`@pytest.mark.slow`. The default `pytest` configuration **deselects** them
(`addopts = -m "not slow"`), so a plain `pytest` run stays light on a laptop or
login node. Run them explicitly when you touch that code:

```bash
# run only the heavy tests
python -m pytest packages/omnibias-geometry/tests/gauge -q -m slow
# run everything
python -m pytest packages/omnibias-geometry/tests/gauge -q -m "slow or not slow"
```

These heavy suites are intended to run on a compute grid rather than an
interactive node; off-node submission helpers live in the separate, private
`omnibias_experiments` project (extracted from the formerly gitignored
`internal/` tree).

### The documentation is executed

Every fenced ` ```python ` block in the README, the root guides, the whole `docs/`
tree, and each package README is **run** by `tests/test_docs_snippets.py` (the
`docs_snippets` CI job). Docs otherwise rot silently: a renamed symbol keeps
rendering perfectly while every reader who copies the snippet hits a traceback.

```bash
# run the documentation as a test suite
python -m pytest tests/test_docs_snippets.py -q

# triage report: per-document outcome, with every failure in a document
python tests/test_docs_snippets.py --all
python tests/test_docs_snippets.py --all --only docs/cookbook   # one subtree
```

A document's blocks execute **in order, sharing one namespace**, because the docs
are narratives — the handbook fits a field in one block and reads jets off it three
blocks later. So a snippet may rely on names defined earlier in the same page, but
never on names that appear nowhere.

New blocks are executed by default; each opt-out is an HTML comment directly above
the fence (invisible in the rendered page) and, except for `signature`, must state
a reason:

```markdown
<!-- docs-test: signature -->                     an API signature, not runnable code
<!-- docs-test: skip reason="needs a GPU" -->     not executable; reason is mandatory
<!-- docs-test: slow -->                          real code, too heavy for per-PR CI
<!-- docs-test: raises=ValueError -->             the block is *meant* to raise
<!-- docs-test: file-skip reason="design doc" --> the whole document opts out
```

`slow` is a document-level property (skipping one block of a chain would only
produce cascading `NameError`s), and those documents run on the weekly schedule.
A missing third-party import *skips* a document. A missing `omnibias.*` module is a
hard failure in the CI job, which installs all 42 distributions and sets
`OMNIBIAS_DOCS_SNIPPETS_STRICT=1` — that is exactly the "documented import no longer
exists" bug this suite exists to catch. A plain `uv run pytest` installs only the
four workspace members, so there the same import is an honest "not verified here"
skip rather than a false alarm.

The 23 notebooks in [`notebooks/`](notebooks/) are executed on the same weekly
schedule (the `notebooks` job), which holds the gallery's "CPU in under ~2 minutes,
fixed seeds, no external data" promise to account. To run one locally:

```bash
cd notebooks   # `_style` / `_fields` are sibling modules
PYTHONPATH=.. MPLBACKEND=Agg jupyter nbconvert --to notebook --execute \
  --stdout 01_closed_form_derivatives.ipynb > /dev/null
```

If that fails with unrelated version skew, the notebook's default `python3`
kernelspec is resolving to another interpreter; register your environment's kernel
(`python -m ipykernel install --user --name mine`) and pass
`--ExecutePreprocessor.kernel_name=mine`.

## Pull-request checklist

- [ ] A regression test covers every behavioural change.
- [ ] `ruff check` is clean.
- [ ] `mypy --strict` is clean for the T1 packages you touched.
- [ ] `mkdocs build --strict` is clean if you changed docs or public docstrings.
- [ ] Documented snippets still execute (`python -m pytest tests/test_docs_snippets.py -q`), and any new opt-out states a reason.
- [ ] Cross-backend numerics stay bit-identical (run the parity tests).
- [ ] No GPU-only, cluster-specific, or local-path details leak into tracked files.
- [ ] `CHANGELOG.md` has an entry under the unreleased section.
- [ ] Agent skill / rule drift checks pass if you touched them (`omnibias-skills install --check`, `python scripts/sync_skills.py --check`).
- [ ] If you changed a public benchmark script or its claim, regenerate the
      matching `docs/benchmarks/*.json` (smoke and/or `--full` as appropriate)
      and keep any `gates` block green.
- [ ] You have signed the [CLA](docs/CLA.md) (the PR bot will confirm).
- [ ] `python scripts/license_headers.py --check` is clean, and no permissive package gained a copyleft dependency.

## Commit messages

Use a short imperative summary line (<= 72 chars), then a blank line and
a body that explains the *why*. Reference the finding or issue id when
relevant.

## Reporting bugs

Open an issue using the
[bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include a
minimal reproducer, the backend (torch / jax / keras), dtype, and the
expected vs. observed numerics.

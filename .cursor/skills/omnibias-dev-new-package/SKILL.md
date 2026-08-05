---
name: omnibias-dev-new-package
description: Scaffold a new omnibias extension package the repo way -- a pyproject mirroring an existing alpha package, SPDX headers, the workspace exclude, a sorted __all__, a CI job, a docs/api page plus mkdocs nav, and llms.txt / CHANGELOG entries. Use when creating a new packages/omnibias-* package or wiring an extension into the monorepo. For contributors modifying omnibias itself, not for consumers using it.
---

# Scaffolding a new omnibias package

This is a uv workspace monorepo. A new extension package must follow the same
shape as the existing ones or CI, docs, and packaging will drift.

## Earn independent existence first

Before scaffolding, confirm the package **earns its own distribution**. It should
have at least one of: a **distinct domain** (a shared *mechanism* -- e.g. a
`beta -> inf` relaxation or a closed-form backward -- is not a shared domain), a
**distinct dependency / maturity tier**, or a **distinct audience**. If it only
re-composes another package's ops, or its top-level module would be a bare
`__version__` shim with the real code one module deep, ship it as a **submodule**
of an existing package and promote it to its own distribution only once it earns
independence.

- Cautionary stubs: `omnibias-score` (`0.0.0a1`) and `omnibias-qpinn` were promoted
  too early. `pde`, `gauge`, and `flow` were later folded back into `pinn` /
  `geometry` / `score` -- a thin solver layer, a sub-domain of geometry, and a
  premature `fields`-only stub, respectively.
- Folding a submodule back *out* into its own package later is cheap (move the
  subtree, add the wiring); un-shipping a premature distribution is not. Prefer the
  submodule until the evidence for independence is real.
- The `test_package_registry` guard (`packages/omnibias-core/tests`) enforces that
  every `packages/omnibias-*` distribution is registered exactly once in the root
  `[tool.uv.workspace]`, so a fold or a new package that skips the wiring fails CI.

## Copy an existing package as the template

`packages/omnibias-dynamics` (pure-Python alpha) and `packages/omnibias-skills`
(stdlib-only alpha) are clean, minimal references. Mirror their layout:

- `pyproject.toml`: `setuptools` backend, `version = "0.1.0a1"`, `Development Status :: 3 - Alpha`, `[tool.setuptools.packages.find] where=["src"] include=["omnibias.*"]`.
- src layout `src/omnibias/<pkg>/__init__.py`, with a module docstring and a sorted `__all__` (include `"__version__"`).
- `README.md` (`**Status: Alpha (0.1.0a1).**` + capabilities + honest scope) and a `LICENSE` (copy from a sibling package).

## Pick a licence tier before writing code

omnibias is open-core, and the tier is a *design* decision made up front, not a
header you paste. Record the new distribution in
`[tool.omnibias.license_tiers]` in the root `pyproject.toml`:

- **`permissive`** (`Apache-2.0`) -- the default. Anything that composes the
  derivative tower belongs here: this is the adoption engine.
- **`copyleft`** (`AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`) -- the
  certified-decision layer, whose output is a *guarantee* (verify / formal /
  sos / dynamics and the optimisation front-ends built on them).

**The invariant: a permissive package must never depend on a copyleft package**,
through `dependencies` or any `optional-dependencies` extra.
`packages/omnibias-core/tests/test_license_consistency.py` fails the build
otherwise. In practice this decides the tier for you: if your package needs a
`verify` or `sos` extra, it is copyleft. Tier C is the upward closure of that
seed set.

Then set the PEP 639 metadata to match -- `license = "<the tier's SPDX
expression>"`, `license-files = ["LICENSE"]`, `requires = ["setuptools>=77",
"wheel"]`, and **no** `License ::` classifier (setuptools>=77 rejects one next
to an SPDX expression) -- and copy the tier's `LICENSE` text from a sibling in
the same tier.

## Required headers and language

- **Never hand-write an SPDX header.** Run `python scripts/license_headers.py`,
  which reads the tier table and stamps every file correctly; `--check` is the
  CI dry run. The two shapes it produces:

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

- Pin a sibling omnibias dependency at a floor the sibling's version actually
  satisfies. An alpha sibling needs the alpha floor (`>=0.1.0a1`), because
  `0.1.0a1 < 0.1.0` under PEP 440 and `>=0.1.0` would make your package
  uninstallable once published. `test_version_consistency` catches this.
- Use vendor-neutral language in tracked files: "GPU job" / "GPU cluster", never a specific scheduler, vendor, internal hostname, or absolute local path. Reproduction scripts live in the separate, private `omnibias_experiments` project.

## Wire it into the monorepo (all of these)

1. Add `packages/omnibias-<pkg>` to `[tool.uv.workspace].exclude` in the root `pyproject.toml` (extension packages install per package, like the others). The `test_package_registry` guard checks this line exists and points at a real directory.
2. Add a CI job in `.github/workflows/ci.yml` modeled on the `dynamics` job (pure venv) or a backend job if it needs torch / jax.
3. Add `docs/api/<pkg>.md` (prose + `::: omnibias.<pkg>` + `Status: Alpha`), register it under `Packages` in `mkdocs.yml` nav, and add `packages/omnibias-<pkg>/src` to the mkdocstrings `paths` so the docs job can import it (also add the package to the docs-job install loop).
4. Add entries to `llms.txt` and a `CHANGELOG.md` bullet under `## [Unreleased]`.

## Typing tier and versions

- Extension packages are **not** on the shared `mypy --strict` CI gate or `[tool.mypy].mypy_path`, but author new modules strict-clean anyway (a self-scoped `[tool.mypy]` in the package pyproject is a clean way to keep it checkable).
- **Do not bump versions** unless the task explicitly says to.

## Verify

```bash
python -m pytest packages/omnibias-<pkg>/tests -q
python -m pytest packages/omnibias-core/tests/test_package_registry.py -q  # wiring guard
uv run ruff check packages tests
mkdocs build --strict
```

## Folding a package back (the reverse operation)

Consolidation is the mirror image of scaffolding. To fold `omnibias-<pkg>` into a
home package as a submodule:

1. Move `packages/omnibias-<pkg>/src/omnibias/<pkg>/` to
   `packages/<home>/src/omnibias/<home>/<pkg>/` and its `tests/` under
   `packages/<home>/tests/<pkg>/`; rewrite `omnibias.<pkg>` -> `omnibias.<home>.<pkg>`
   across the moved trees **and every external importer** (keep numerics
   bit-identical -- no math edits).
2. Label the absorbed submodule's maturity in its `__init__` docstring (an alpha
   submodule folded into a Beta home stays alpha until it earns otherwise) and add a
   one-line pointer in the home package's `__init__`.
3. De-wire the old distribution **everywhere at once**: delete
   `packages/omnibias-<pkg>/`, its `[tool.uv.workspace].exclude` line, its CI job (and
   any `for pkg in ...` docs-loop token), its `docs/api/<pkg>.md` + `mkdocs.yml` nav /
   `paths` entries (rename the api page, e.g. `api/<home>-<pkg>.md`, and repoint any
   `api/<pkg>.md` links so `mkdocs build --strict` stays green), and its `llms.txt` /
   `CHANGELOG.md` / `AGENTS.md` entries.
4. Verify per the block above; the `test_package_registry` guard will fail if any
   dangling wiring remains.


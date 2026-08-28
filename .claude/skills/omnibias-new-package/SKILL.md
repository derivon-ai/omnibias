---
name: omnibias-new-package
description: Scaffold a new omnibias extension package the repo way — pyproject mirroring an existing alpha package, SPDX headers, workspace exclude, sorted __all__, CI job, docs/api page plus mkdocs nav, and llms.txt / CHANGELOG entries. Use when creating a packages/omnibias-* distribution or wiring an extension into the monorepo.
---

# Scaffolding a new omnibias package

This is a uv workspace monorepo. A new extension package follows the same
shape as the existing ones or CI, docs, and packaging drift.

## Why nested AD fails

A new distribution is not "another autodiff wrapper." Premature packages that
only re-compose another package's ops (`score`, `qpinn` as cautionary stubs;
`pde` / `gauge` / `flow` folded back) inflate the tree without a new domain.
Generic project templates miss the license-tier invariant, the skill catalog,
and the operator-surface wiring.

## What only this tower unlocks

A package that **earns independent existence**: distinct domain, distinct
dependency / maturity tier, or distinct audience. Ambition stays inside
existing packages as alpha submodules until that test passes. Folding a
submodule out later is cheap; un-shipping a premature distribution is not.

## Use

Before scaffolding, confirm independence. Cautionary stubs: `omnibias-score`
and `omnibias-qpinn` promoted early; `pde`, `gauge`, and `flow` folded into
`pinn` / `geometry` / `score`. `test_package_registry` requires every
`packages/omnibias-*` distribution in the root workspace table.

Templates: `packages/omnibias-dynamics` (pure-Python alpha) and
`packages/omnibias-skills` (stdlib-only alpha).

- `pyproject.toml`: setuptools, `version = "0.1.0a1"`, `Development Status :: 3 - Alpha`,
  `[tool.setuptools.packages.find] where=["src"] include=["omnibias.*"]`.
- `src/omnibias/<pkg>/__init__.py` with sorted `__all__` (include `"__version__"`).
- `README.md` + `LICENSE` from a sibling in the same licence tier.

Licence tier is a design decision, recorded in `[tool.omnibias.license_tiers]`:

- **permissive** (`Apache-2.0`) — default; derivative-tower composition.
- **copyleft** (`AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`) —
  certified-decision layer (verify / formal / sos / dynamics and front-ends).

A permissive package never depends on a copyleft package
(`test_license_consistency.py`). Never hand-write SPDX; run
`python scripts/license_headers.py`. Vendor-neutral tracked files (no
scheduler names, no absolute local paths). Artifacts go to `$OMNIBIAS_SCRATCH`.

Then: CI job, `docs/api/<pkg>.md`, mkdocs nav, `llms.txt`, `CHANGELOG.md`,
and an `omnibias-<pkg>` skill (this catalog). `python scripts/check_package_skills.py`.

## Extend

Compose with `omnibias-skills` (catalog + bundle), `omnibias-empirical-validation`
(gates), `omnibias-core-concepts` (vocabulary). Theory 06-03 G1–G5 already
ship; new homes register in `benchmarks/theory_homes.py`.

## Next invention

A distribution that passes the independence test on a genuinely new domain
(not a re-compose), with a skill, a CI job, and an absolute gate in the same
change.

## Further references

- `AGENTS.md` (Don't / new package)
- `docs/packages.md`

---
name: omnibias-skills
description: Ship the capability skill subset via omnibias-skills while the full catalog lives in the repo. Use when adding a bundled skill, changing the installer, or keeping Cursor / Claude / pip copies byte-identical.
---

# omnibias-skills

Agent-skill library for building on omnibias: bundled Cursor / Claude Code Agent Skills
plus an idempotent installer CLI that places them into a project's .cursor and .claude
directories.

## Why nested AD fails

Generic assistant docs drift from the code. Two skill libraries (consumer vs maintainer)
doubled every invention. Nested tools cannot see OperatorBlock's six roles or the shared
polynomial contract.

## What only this tower unlocks

Agent-skill library for building on omnibias: bundled Cursor / Claude Code Agent Skills
plus an idempotent installer CLI that places them into a project's .cursor and .claude
directories.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Canonical authoring: `.cursor/skills/omnibias-<stem>/SKILL.md`.
Claude mirror: `python scripts/sync_skills.py`.
Pip subset (nine capability skills):
`packages/omnibias-skills/src/omnibias/skills/_bundled/skills/`.

```bash
python scripts/check_package_skills.py
python scripts/sync_skills.py --check
omnibias-skills install --check --tool all --dest .
```

Edit `.cursor/skills`, copy the nine into the bundle, then sync.

## Extend

- Source: [`packages/omnibias-skills`](../../../packages/omnibias-skills).
- Namespace: `omnibias.skills`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-skills/tests -q`.
- Compose with `omnibias-backends`, `omnibias-new-package`, `omnibias-core-concepts` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A tenth capability skill that earns the bundle by unlocking a new operator-surface role
with a runnable import map and a bakeoff pointer.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)

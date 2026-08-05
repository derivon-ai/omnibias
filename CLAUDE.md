# CLAUDE.md

This file orients Claude Code in the **omnibias** repository.

**Read [AGENTS.md](AGENTS.md) first.** It is the machine-oriented guide:
repository layout, build / test / lint commands, the derivative-tower contract,
and the do / don't list. Everything there applies to Claude Code too.

**Before claiming a capability, check [docs/operator-surface.md](docs/operator-surface.md)** --
the canonical capability matrix. In particular, `OperatorBlock` has six roles
(`identity | grad | laplacian | derivative | band | integral`), and omnibias has
a **closed-form integral operator** (the `integral` role: an antiderivative
window `S(z+b_hi)-S(z+b_lo)`, `S'=sigma`), not only closed-form derivatives.
Do not state otherwise.

## Agent skills

Skills live in `.claude/skills/` (mirrored from Cursor's `.cursor/skills/`):

- **Maintainer skills** (`omnibias-dev-*`) cover developing omnibias itself --
  the closed-form derivative tower, field operators, verified primitives, the
  certificate / Lean loop, and scaffolding a new package.
- **Consumer skills** (`omnibias-*`) cover *using* omnibias to build things.

## Keeping skills in sync (canonical sources)

- Maintainer skills are hand-authored **canonically in `.cursor/skills/`** and
  mirrored here by `python scripts/sync_skills.py` (CI runs it with `--check`).
  Edit the `.cursor` copy, then re-run the sync.
- Consumer skills come from the `omnibias-skills` package (canonical source in
  `packages/omnibias-skills/src/omnibias/skills/_bundled/skills/`); the repo's
  committed copies are produced by `omnibias-skills install` and drift-checked in
  CI. Edit the bundle, not the copies.

See the "Agent tooling" section of [AGENTS.md](AGENTS.md) for the full picture.

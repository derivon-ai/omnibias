# CLAUDE.md

This file orients Claude Code in the **omnibias** repository.

**Read [AGENTS.md](AGENTS.md) first.** Then load the concise capability map in
`.cursor/rules/omnibias.md` and the matching `omnibias-<package>` skill
before changing a package. They provide the repository layout, composition
paths, derivative-tower contract, and package-specific workflows.

**Before claiming a capability, check [docs/operator-surface.md](docs/operator-surface.md)** --
the canonical capability matrix. In particular, `OperatorBlock` has six roles
(`identity | grad | laplacian | derivative | band | integral`), and omnibias has
a **closed-form integral operator** (the `integral` role: an antiderivative
window `S(z+b_hi)-S(z+b_lo)`, `S'=sigma`), not only closed-form derivatives.
Do not state otherwise.

## Agent skills

Skills live in `.claude/skills/` (mirrored from Cursor's `.cursor/skills/`).
Every skill is `omnibias-<stem>/SKILL.md`: one skill per workspace
distribution, plus cross-cuts (`backends`, `frontier`, derivative-tower,
field, verified, formal, discovery, and research workflows).

## Keeping skills in sync (canonical sources)

- The full catalog is hand-authored **canonically in `.cursor/skills/`** and
  mirrored here by `python scripts/sync_skills.py` (CI runs it with `--check`).
  Edit the `.cursor` copy, then re-run the sync.
- The pip package `omnibias-skills` ships the capability nine, byte-identical
  to `.cursor/skills`. If you edit one of those nine, copy it into
  `packages/omnibias-skills/src/omnibias/skills/_bundled/skills/` so
  `omnibias-skills install --check` stays green.

See the "Agent tooling" section of [AGENTS.md](AGENTS.md) for the full picture.

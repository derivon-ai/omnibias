# omnibias-skills

The **consumer** agent-skill library for building on omnibias: bundled Cursor /
Claude Code *Agent Skills* that teach an AI assistant how to *use* omnibias
correctly, plus an idempotent installer CLI that places them into a project's
`.cursor/skills` and `.claude/skills` directories.

- **Six capability skills** (`omnibias-*`): `backends`, `fields-pinn`, `geometry`,
  `curvature-optim`, `verify`, `symbolic` — thin `SKILL.md` files that link the
  versioned docs and encode the gotchas.
- **Installer CLI**: `omnibias-skills install | update | uninstall | list`, with
  `--tool cursor|claude|all`, `--dest`, `--global`, `--dry-run`, and `--check`
  (a CI-friendly drift gate). Placement is always explicit — `pip install` has no
  side effects.

!!! note "Two libraries, two homes"
    This package is the **consumer** library (its canonical source is
    `_bundled/skills/`). Skills for *developing omnibias itself* are a separate,
    repo-only **maintainer** library (`.cursor/skills/omnibias-dev-*`) and are not
    shipped here. See the "Agent tooling" section of the repository `AGENTS.md`.

## Install

```bash
pip install omnibias-skills
omnibias-skills install            # ./.cursor/skills and ./.claude/skills
omnibias-skills install --global   # ~/.cursor and ~/.claude
omnibias-skills install --check    # exit non-zero on drift (CI)
```

The installer is idempotent, writes only `omnibias-*` skill directories, never
clobbers your own files, and records a `.omnibias-skills.manifest.json` so
`uninstall` is exact.

## Public API

::: omnibias.skills
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).

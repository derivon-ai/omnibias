# omnibias-skills

The **capability** agent-skill library for building on omnibias: bundled Cursor /
Claude Code *Agent Skills* plus an idempotent installer CLI that places them
into a project's `.cursor/skills` and `.claude/skills` directories.

The full catalog (every workspace package plus cross-cuts) lives in the
omnibias repository under `.cursor/skills/omnibias-*`. This package ships
the **capability nine**, byte-identical to those files:

- `omnibias-backends` — closed-form n-th derivatives and jets on torch / jax / keras
- `omnibias-fields` — field operators (grad / div / curl / laplacian / hessian)
- `omnibias-pinn` — physics-informed networks on the field substrate
- `omnibias-geometry` — metric, curvature, geodesics, exterior calculus
- `omnibias-curvature` — second-order optimizers, Fisher, and natural gradient
- `omnibias-verify` — certified enclosures, robustness certificates, validated dynamics
- `omnibias-symbolic` — neural-jet equation / PDE discovery
- `omnibias-frontier` — frontier routes (certified fluids, CAP, gauge/spectral)
- `omnibias-control` — exact jet-adjoint policy gradients and certified horizons

!!! note "One catalog"
    Author skills in `.cursor/skills/`. `python scripts/sync_skills.py` mirrors
    the full catalog to `.claude/skills/`. Copy the nine into
    `_bundled/skills/` so this package stays byte-identical. CI
    `omnibias-skills install --check` diffs only the bundle; repo-only skills
    are extra files and stay ignored. See the "Agent tooling" section of the
    repository `AGENTS.md`.

The always-on rule `.cursor/rules/omnibias.md` maps packages, AD bottlenecks,
and bakeoffs; `python scripts/check_package_skills.py` checks catalog coverage.

## Install

```bash
pip install omnibias-skills
omnibias-skills install            # ./.cursor/skills and ./.claude/skills
omnibias-skills install --global   # ~/.cursor and ~/.claude
omnibias-skills install --check    # exit non-zero on drift (CI)
```

The installer is idempotent, writes only the bundled `omnibias-*` skill
directories, never clobbers your own files, and records a
`.omnibias-skills.manifest.json` so `uninstall` is exact.

## Public API

::: omnibias.skills
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).

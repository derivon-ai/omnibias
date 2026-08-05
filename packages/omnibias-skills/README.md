# omnibias-skills

**Status: Alpha (0.1.0a1).**

The omnibias **consumer** agent-skill library: bundled Cursor / Claude Code
Agent Skills that teach an AI assistant how to *use* omnibias correctly, plus
an idempotent installer CLI that places them into your project.

This package is for developers **building on top of** omnibias. (Skills for
*developing omnibias itself* live in the repo under
`.cursor/skills/omnibias-dev-*` and are intentionally not shipped here.)

## Install

```bash
pip install omnibias-skills
omnibias-skills install            # writes ./.cursor/skills and ./.claude/skills
omnibias-skills install --tool cursor
omnibias-skills install --global   # ~/.cursor and ~/.claude
omnibias-skills list
omnibias-skills uninstall
```

`pip install` has **no side effects**; skills are placed only when you run
`omnibias-skills install`. The installer is idempotent, writes only
`omnibias-*` skill directories, never clobbers your own files, and records a
`.omnibias-skills.manifest.json` next to the target so `uninstall` is exact.
Run `omnibias-skills install --check` in CI to fail on drift.

## What you get

Six capability skills, each a thin `SKILL.md` that links the versioned docs:

- `omnibias-backends` -- closed-form n-th derivatives and jets on torch / jax / keras.
- `omnibias-fields-pinn` -- field operators (grad / div / curl / laplacian / hessian) and PINNs.
- `omnibias-geometry` -- metric, curvature, geodesics, exterior calculus (with honesty labels).
- `omnibias-curvature-optim` -- second-order optimizers, Fisher, and natural gradient.
- `omnibias-verify` -- certified enclosures, robustness certificates, validated dynamics.
- `omnibias-symbolic` -- neural-jet equation / PDE discovery.

## Tests

```bash
python -m pytest packages/omnibias-skills/tests -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.

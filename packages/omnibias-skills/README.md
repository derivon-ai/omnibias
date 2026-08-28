# omnibias-skills

**Status: Alpha (0.1.0a1).**

The omnibias **capability** agent-skill library: bundled Cursor / Claude Code
Agent Skills plus an idempotent installer CLI that places them into your
project.

The full catalog lives in the omnibias repo under `.cursor/skills/`. This
package ships the capability nine, byte-identical to those files.

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
`omnibias-skills install`. The installer is idempotent, writes only the
bundled `omnibias-*` skill directories, never clobbers your own files, and
records a `.omnibias-skills.manifest.json` next to the target so `uninstall`
is exact. Run `omnibias-skills install --check` in CI to fail on drift.

## What you get

Nine capability skills, each a `SKILL.md` with an import map and the next
invention on that seam:

- `omnibias-backends` -- closed-form n-th derivatives and jets on torch / jax / keras.
- `omnibias-fields` -- field operators (grad / div / curl / laplacian / hessian).
- `omnibias-pinn` -- physics-informed networks on the field substrate.
- `omnibias-frontier` -- frontier routes (certified fluids, CAP, gauge/spectral).
- `omnibias-geometry` -- metric, curvature, geodesics, exterior calculus.
- `omnibias-curvature` -- second-order optimizers, Fisher, and natural gradient.
- `omnibias-verify` -- certified enclosures, robustness certificates, validated dynamics.
- `omnibias-symbolic` -- neural-jet equation / PDE discovery.
- `omnibias-control` -- exact jet-adjoint policy gradients and certified horizons.

## Tests

```bash
python -m pytest packages/omnibias-skills/tests -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.

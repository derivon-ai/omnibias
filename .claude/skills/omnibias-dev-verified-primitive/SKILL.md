---
name: omnibias-dev-verified-primitive
description: Add or modify a rigorous primitive in omnibias.core.verified (Interval, affine zonotopes, TaylorModel(MV), enclosure engines, eig / dirichlet / lohner / kantorovich). Use when writing sound, outward-rounded numerics that must provably enclose the true value, with honest closed-form vs autodiff vs numerical labels. For contributors modifying omnibias itself, not for consumers using it.
---

# Writing a rigorous (verified) primitive

The rigorous register trades speed for **certainty**: an enclosure must provably
contain the true object. A silently-too-tight bound is a correctness bug, not a
performance win.

## Where it lives and what it may import

- `omnibias.core.verified`: `Interval` (outward-rounded), `affine` (zonotopes, dependency cancellation), `TaylorModel` / `TaylorModelMV`, `sequence_space`, `kantorovich`, `lohner`, `eig_operator`, `dirichlet`, plus `jet` / `jet_mv`, `transport`, `information`, `probability`, `ode`, `quadrature`.
- It is **pure Python** and never imports a backend (`torch` / `jax` / `keras`). It is the shared substrate for the differentiable and formal registers.

## Non-negotiable soundness rules

- Round **outward** on every operation; never let a bound tighten by accident.
- **Every enclosure must be tested to contain both a dense deterministic grid AND a random sample of true values.** This is the standard soundness test in this repo -- add it for any new enclosure.
- Report blow-up honestly (a too-large step / too-chaotic system). Never widen an enclosure silently to make a test pass, and never narrow one to look sharper.

## Honesty labels (say which register)

Label results **closed-form** (the sigma tower), **autodiff-exact** (forward-mode
autodiff of an analytic expression -- machine precision, but not closed form), or
**numerical** (grid / quadrature approximation). Carry a `scope` field
(e.g. `"local_box"`) in sealed payloads where the honesty convention expects it.

## Checklist

- Regression test + the grid-and-random soundness test.
- Regenerate the sorted `__all__`.
- Extension/verified modules are authored `mypy --strict`-clean.

```bash
python -m pytest packages/omnibias-core/tests -q -k verified
uv run ruff check packages tests
```

If the primitive produces a certificate, continue with the
`omnibias-dev-certificate-lean` skill.

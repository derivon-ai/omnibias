<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# Licensing

> **Short version.** Most of omnibias — the derivative tower and everything
> built directly on it — is **Apache-2.0**: use it anywhere, including in
> closed-source products, with no obligations beyond attribution. The
> **certified-decision** packages are **AGPL-3.0-or-later or commercial**. If
> the AGPL does not work for your product, contact **info@derivon.ai**.

omnibias ships as 42 separate distributions on PyPI, and they are **not all
under the same licence**. This page is authoritative; the per-package `LICENSE`
file and the SPDX header on every source file are generated from the same
table and always agree with it.

## The two tiers

| | Tier P — permissive | Tier C — copyleft |
|---|---|---|
| **SPDX** | `Apache-2.0` | `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial` |
| **Packages** | 28 | 14 |
| **Closed-source use** | yes, freely | needs a commercial licence |
| **SaaS / network use** | no obligation | AGPL §13 applies, or commercial |
| **Patent grant** | express (Apache §3) | via AGPL / by agreement |
| **Commercial licence needed?** | **never** | only to escape the AGPL |

### Tier P — Apache-2.0 (28 packages)

The primitive and everything that composes it. This is the part you are meant
to build on without talking to anyone.

`omnibias-core`, `omnibias-torch`, `omnibias-jax`, `omnibias-keras`,
`omnibias-fields`, `omnibias-ferminet`, `omnibias-pinn`, `omnibias-qpinn`,
`omnibias-geometry`, `omnibias-curvature`, `omnibias-symbolic`,
`omnibias-score`, `omnibias-measure`, `omnibias-fractional`,
`omnibias-variational`, `omnibias-difference`, `omnibias-qcalculus`,
`omnibias-timescale`, `omnibias-holonomic`, `omnibias-binary`,
`omnibias-boolean`, `omnibias-spiking`, `omnibias-hopfield`,
`omnibias-struct`, `omnibias-graph`, `omnibias-partition`, `omnibias-shape`,
`omnibias-skills`.

Apache-2.0 rather than MIT or BSD, deliberately: it carries an **express patent
grant** (§3) with a retaliation clause, which is what enterprise legal review
actually looks for and what MIT does not provide.

### Tier C — AGPL-3.0-or-later OR commercial (14 packages)

The certified-decision layer: verified enclosures, proof obligations, and the
optimisation front-ends that produce sealed optimality certificates. These are
the packages whose output is a *guarantee* you would put in front of a
regulator or a customer, and they are where the commercial offer lives.

`omnibias-verify`, `omnibias-formal`, `omnibias-sos`, `omnibias-dynamics`,
`omnibias-convex`, `omnibias-discrete`, `omnibias-qubo`, `omnibias-logic`,
`omnibias-nphard`, `omnibias-submodular`, `omnibias-combinatorics`,
`omnibias-routing`, `omnibias-tab`, `omnibias-control`.

Under the AGPL branch you may use, study, modify, and redistribute these for
free, subject to:

- **Copyleft.** Distributing the package or a derivative work obliges you to
  release the complete corresponding source under the AGPL.
- **Network use (§13).** Running a *modified* version to provide a service over
  a network obliges you to offer that service's users the corresponding source.
  This is the clause that distinguishes the AGPL from the GPL.
- **No warranty.** Provided "as is"; see `LICENSES/AGPL-3.0-or-later.txt`.

Under the commercial branch (`LicenseRef-omnibias-Commercial`) those two
obligations are removed, and support, warranty, and indemnity terms are
available by agreement. See [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md)
and contact **info@derivon.ai**.

## The invariant you can rely on

**No Apache-2.0 package depends on an AGPL package** — not through
`dependencies`, and not through any `optional-dependencies` extra. Installing
any Tier-P package, with any combination of its extras, never pulls a copyleft
distribution into your dependency tree.

This is enforced in CI by
[`packages/omnibias-core/tests/test_license_consistency.py`](packages/omnibias-core/tests/test_license_consistency.py),
which fails the build if anyone adds such an edge. It is not a convention; it
is a gate.

The dependency direction is one-way and intentional: Tier C sits *above*
Tier P and freely depends downward. `omnibias-difference` stays Apache-2.0 even
though the AGPL `omnibias-verify` requires it — permissive-below-copyleft is
fine, the reverse is not.

## Which tier is a given package in?

Three places, always in agreement:

1. **`[tool.omnibias.license_tiers]`** in the root [`pyproject.toml`](pyproject.toml)
   — the single source of truth.
2. The package's **`LICENSE`** file, which states its SPDX expression on line 4.
3. The **SPDX header** on every `.py` file in it.

Or read the published metadata: `pip show omnibias-<name>` reports the
`License-Expression` from the wheel.

## One consequence, stated plainly

`omnibias-core` is Apache-2.0, so `omnibias.core.verified` (interval
arithmetic, Taylor models, Lohner, Kantorovich) and `omnibias.core.proof`
(certificate format v1 and the Lean bridge) are permissively licensed.

That is deliberate. Interval arithmetic is decades-old published mathematics,
and a certificate *format* only becomes a standard if others can adopt it
without a licence conversation. The defensible surface is Tier C and the Lean
kernel, not the file format.

## Per-file metadata (SPDX / REUSE)

The repository follows the [REUSE Specification](https://reuse.software). Full
licence texts live in `LICENSES/`, and every source file carries a header:

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

Do not hand-write these. Run `python scripts/license_headers.py`, which reads
the tier table and stamps every file correctly; `--check` is the CI dry run.

The repository-root [`LICENSE`](LICENSE) is the AGPL, so GitHub's licence
detector reports the strongest licence in the tree rather than the weakest.
It does **not** override a package's own `LICENSE`.

## Contributions

Every contribution is accepted under a Contributor License Agreement
([`docs/CLA.md`](docs/CLA.md)) granting Derivon the right to license it under
Apache-2.0, the AGPL, **and** commercial terms. That grant is what makes both
tiers possible from one clean codebase. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Trademarks

The licences cover the **code**. Neither grants rights to the **"omnibias"
name or logo**. See [`TRADEMARKS.md`](TRADEMARKS.md).

---

*This page is a description of the licensing scheme, not legal advice. The
binding terms are the licence texts in `LICENSES/` and, for the commercial
branch, the signed agreement.*

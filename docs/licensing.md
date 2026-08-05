# Licensing

omnibias ships as **42 separate distributions**, and they are **not all under
the same licence**. Before you `pip install`, know which tier you are in.

!!! tip "The one-line answer"

    The derivative tower and everything built directly on it is **Apache-2.0**
    — use it anywhere, including closed-source and hosted products, with no
    obligations beyond attribution. Only the 14 **certified-decision** packages
    are AGPL-or-commercial.

## The two tiers

| | Tier P — permissive | Tier C — copyleft |
|---|---|---|
| **SPDX expression** | `Apache-2.0` | `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial` |
| **Packages** | 28 | 14 |
| **Closed-source use** | yes, freely | needs a commercial licence |
| **SaaS / network use** | no obligation | AGPL §13 applies, or commercial |
| **Express patent grant** | yes (Apache §3) | via AGPL / by agreement |
| **Commercial licence needed?** | **never** | only to escape the AGPL |

### Tier P — Apache-2.0

The primitive and its consumers. This is the part you are meant to build on
without a licence conversation.

`omnibias-core` · `omnibias-torch` · `omnibias-jax` · `omnibias-keras` ·
`omnibias-fields` · `omnibias-ferminet` · `omnibias-pinn` · `omnibias-qpinn` ·
`omnibias-geometry` · `omnibias-curvature` · `omnibias-symbolic` ·
`omnibias-score` · `omnibias-measure` · `omnibias-fractional` ·
`omnibias-variational` · `omnibias-difference` · `omnibias-qcalculus` ·
`omnibias-timescale` · `omnibias-holonomic` · `omnibias-binary` ·
`omnibias-boolean` · `omnibias-spiking` · `omnibias-hopfield` ·
`omnibias-struct` · `omnibias-graph` · `omnibias-partition` ·
`omnibias-shape` · `omnibias-skills`

Apache-2.0 rather than MIT or BSD, deliberately: it carries an **express patent
grant** with a retaliation clause, which is what enterprise legal review looks
for and what MIT does not provide.

### Tier C — AGPL-3.0-or-later **or** commercial

The certified-decision layer: verified enclosures, proof obligations, and the
optimisation front-ends that emit sealed optimality certificates. These produce
the *guarantees* you would put in front of a regulator or a customer.

`omnibias-verify` · `omnibias-formal` · `omnibias-sos` · `omnibias-dynamics` ·
`omnibias-convex` · `omnibias-discrete` · `omnibias-qubo` · `omnibias-logic` ·
`omnibias-nphard` · `omnibias-submodular` · `omnibias-combinatorics` ·
`omnibias-routing` · `omnibias-tab` · `omnibias-control`

Under the AGPL branch: free to use, study, modify, and redistribute, subject to
copyleft on distribution and the §13 source-disclosure obligation when you run
a *modified* version as a network service. Under the commercial branch: those
obligations are removed, with support / warranty / indemnity available by
agreement.

## The invariant you can rely on

!!! success "No Apache-2.0 package depends on an AGPL package"

    Not through `dependencies`, and not through any `optional-dependencies`
    extra. Installing any Tier-P package, with any combination of its extras,
    never pulls a copyleft distribution into your dependency tree.

This is a CI gate, not a convention:
`packages/omnibias-core/tests/test_license_consistency.py` fails the build if
anyone adds such an edge. The dependency direction is one-way by design — Tier C
sits above Tier P and depends downward freely. `omnibias-difference` stays
Apache-2.0 even though the AGPL `omnibias-verify` requires it.

## How to check a package's licence

```bash
pip show omnibias-fields | grep -i license
```

The wheel metadata carries a PEP 639 `License-Expression` field. Inside the
repository the same fact appears in three always-agreeing places: the
`[tool.omnibias.license_tiers]` table in the root `pyproject.toml` (the source
of truth), the package's own `LICENSE` file, and the SPDX header on every
`.py` file.

## One consequence, stated plainly

`omnibias-core` is Apache-2.0, so `omnibias.core.verified` (interval
arithmetic, Taylor models, Lohner, Kantorovich) and `omnibias.core.proof`
(certificate format v1 and the Lean bridge) are permissively licensed.

That is deliberate. Interval arithmetic is decades-old published mathematics,
and a certificate *format* only becomes a standard if others can adopt it
without asking permission. The defensible surface is Tier C and the Lean
kernel, not the file format.

## Contributing

Contributions are accepted under a CLA that grants Derivon the right to license
them under Apache-2.0, the AGPL, and commercial terms, and to move code between
tiers as packages evolve. See
[`CONTRIBUTING.md`](https://github.com/derivon-ai/omnibias/blob/main/CONTRIBUTING.md).

Never hand-write an SPDX header — run `python scripts/license_headers.py`,
which stamps every file from the tier table.

## Trademarks

The licences cover the **code**. Neither grants rights to the "omnibias" name
or logo. See
[`TRADEMARKS.md`](https://github.com/derivon-ai/omnibias/blob/main/TRADEMARKS.md).

---

*Descriptive, not legal advice. The binding terms are the licence texts in
`LICENSES/` and, for the commercial branch, the signed agreement. Commercial
enquiries: **info@derivon.ai**.*

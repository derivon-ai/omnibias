<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# Open core, with a commercial option

The derivative and field infrastructure is **Apache-2.0**. Use it in research,
commercial software, private deployments and closed-source products under the
Apache terms. A separate commercial license is not required for that tier.

The certified optimization tier is available under **AGPL-3.0-or-later OR
LicenseRef-omnibias-Commercial**. Follow the AGPL terms, or obtain a signed
commercial agreement from **[info@derivon.ai](mailto:info@derivon.ai)**. See the
[commercial offer](COMMERCIAL-LICENSE.md) for licensing, support and deployment
options.

## Package licenses

This inventory is generated from package metadata and the license-tier registry.
Each distribution ships its own license; repository separation changes neither
its existing grants nor its package license.

<!-- BEGIN GENERATED LICENSE INVENTORY -->

| License choice | Distributions |
| --- | --- |
| AGPL-3.0-or-later **or commercial** | [convex](packages/omnibias-convex/LICENSE), [discrete](packages/omnibias-discrete/LICENSE), [sos](packages/omnibias-sos/LICENSE) |
| Apache-2.0 | [binary](packages/omnibias-binary/LICENSE), [boolean](packages/omnibias-boolean/LICENSE), [core](packages/omnibias-core/LICENSE), [curvature](packages/omnibias-curvature/LICENSE), [difference](packages/omnibias-difference/LICENSE), [fields](packages/omnibias-fields/LICENSE), [graph](packages/omnibias-graph/LICENSE), [jax](packages/omnibias-jax/LICENSE), [keras](packages/omnibias-keras/LICENSE), [partition](packages/omnibias-partition/LICENSE), [qcalculus](packages/omnibias-qcalculus/LICENSE), [struct](packages/omnibias-struct/LICENSE), [torch](packages/omnibias-torch/LICENSE) |

<!-- END GENERATED LICENSE INVENTORY -->

## What the terms mean

Apache-2.0 permits commercial use, modification and redistribution, subject to
its conditions, including preservation of required notices. It includes an
express patent grant. The complete [Apache license](LICENSES/Apache-2.0.txt)
governs; it is not a promise of warranty or trademark rights.

The AGPL branch permits use, modification and redistribution under its terms.
Distribution of covered works carries corresponding-source obligations;
Section 13 additionally applies to a modified program's remote network users.
A signed commercial agreement provides the alternative grant for covered
modules. The [AGPL text](LICENSES/AGPL-3.0-or-later.txt) and
[commercial reference](LICENSES/LicenseRef-omnibias-Commercial.txt) explain those
branches. A license reference alone is not a commercial grant.

## Boundaries and contributions

Apache distributions must not require copyleft distributions through either
base dependencies or extras. Per-package metadata, bundled license texts and
source SPDX headers must agree. Release checks enforce the retained workspace;
consumer repositories also need their own dependency-boundary checks.

The repository-root [LICENSE](LICENSE) is AGPL so repository-level scanners
identify the strongest license present. It **does not override** each package's
Apache or dual-license grant. Read the package license when selecting a dependency.

Contributors retain their copyright and grant the rights described in the
[CLA](CLA.md), supporting both licensing tiers. Neither tier grants rights to
the omnibias name or logo; see [TRADEMARKS.md](TRADEMARKS.md).

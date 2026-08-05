<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# Getting help

## Start here

| You want to… | Go to |
|---|---|
| Learn what omnibias does | [Docs](https://omnibias.ai/) · [`README.md`](README.md) |
| Follow a worked example | [Handbook](https://omnibias.ai/handbook/) · [`docs/examples/`](docs/examples/) |
| Know whether a capability exists | [Operator surface](https://omnibias.ai/operator-surface/) — the canonical capability matrix |
| Understand what is *proved* vs *measured* | [Scope & guarantees](https://omnibias.ai/scope-and-guarantees/) |
| Ask a question | [GitHub Discussions](https://github.com/derivon-ai/omnibias/discussions) |
| Report a bug | [GitHub Issues](https://github.com/derivon-ai/omnibias/issues) |
| Report a vulnerability | [`SECURITY.md`](SECURITY.md) — **not** a public issue |
| Contribute a change | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Buy support or a commercial licence | <info@derivon.ai> · [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md) |

## What to include in a bug report

A numerical library needs more than "it's wrong". Please include:

- **package and version** — `pip show omnibias-<name>`;
- **backend and version** — PyTorch / JAX / Keras, plus `KERAS_BACKEND` if
  relevant, and CPU or GPU;
- **a minimal reproducer** that runs standalone, ideally under 20 lines;
- **expected vs observed values**, with the actual numbers. For a numerical
  discrepancy, state the magnitude: an absolute or relative error and the
  dtype. "Doesn't match" is not actionable; "differs by 3e-4 in float32 at
  order 6" is.

Cross-backend mismatches are treated as high priority: the backends are
bit-identical by construction, so any difference is a real defect.

## Level of support to expect

omnibias has **one maintainer** (see [`MAINTAINERS.md`](MAINTAINERS.md)).
Issues and discussions are answered as time allows, with no response-time
commitment. Security reports take priority.

If your organisation needs guaranteed response times, a support SLA, or
indemnity, that is a commercial arrangement rather than something to hope for
from a volunteer queue — email <info@derivon.ai>.

## Before filing

Two checks resolve a large share of reports:

1. **Is the capability actually claimed?** The
   [operator surface](https://omnibias.ai/operator-surface/)
   is canonical. `OperatorBlock` has six roles
   (`identity`, `grad`, `laplacian`, `derivative`, `band`, `integral`), and
   some things that look adjacent are deliberately out of scope.
2. **Is the method exact or approximate?** Closed-form derivatives, autodiff of
   an analytic quantity, and grid-based approximation are labelled differently
   throughout the docs on purpose — for example `omnibias-fractional` is
   non-local and grid-based, and is explicitly *not* closed form. Expecting
   exactness from an approximate register is the most common false report.

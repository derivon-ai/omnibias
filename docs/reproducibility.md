# Reproducibility

omnibias is built so that every numerical claim in the docs and papers is
reproducible from a clean checkout. This page pins the toolchain, the
determinism protocol, and the exact test selectors that back each claim.

## Toolchain

| component | pin | notes |
|---|---|---|
| Python | `3.14` development target, `3.10` supported floor | `requires-python = ">=3.10"`; the core / torch / jax CI jobs run the full `3.10`-`3.14` span, so backward compatibility is measured rather than assumed. `ruff` and `mypy` target the floor (`py310`), which is what catches a newer-only construct written on 3.14. |
| uv | `0.5.x` | workspace resolver / installer |
| OS (CI) | `ubuntu-latest` | math is OS-independent; GPU numbers come from the cluster |

Materialise the full workspace exactly as CI does:

```bash
uv sync --all-extras --dev
```

Each `pyproject.toml` pins dependency **ranges** (e.g. `torch>=2.0`,
`jax>=0.4.30`) so that consumers of the published wheels stay unconstrained. The
repository additionally **tracks `uv.lock`**, which resolves every one of those
ranges to an exact version for the development and CI workspace. To reproduce
that environment bit-exactly rather than re-resolving:

```bash
uv sync --frozen   # installs strictly from the committed lock, no re-resolution
```

Refresh the lock deliberately (and review the diff) when a dependency range
changes:

```bash
uv lock            # re-resolves; commit the result
```

The clean single-package installs (matching the per-package CI jobs) use a
plain virtualenv, e.g.:

```bash
python -m venv .venv-core
source .venv-core/bin/activate
pip install -e "packages/omnibias-core[test]"
```

## Determinism protocol

- **float64 everywhere in parity tests.** JAX runs with
  `jax.config.update("jax_enable_x64", True)` and torch tensors are created /
  compared in float64. The polynomial recursions accumulate float32 round-off,
  so the bit-identical contract is stated at float64 ULP tolerance.
- **Seeded inputs.** Randomised inputs use NumPy's
  `np.random.default_rng(seed)` with a fixed per-test seed (see the `_RNG_SEED`
  constant in the parity suites), so every run draws the same sample grid. Each
  enclosure test additionally covers a **dense deterministic grid** so a claim
  never rests on the random draw alone.
- **Hash seed.** For byte-stable ordering set `PYTHONHASHSEED=0` when comparing
  serialised structures (e.g. hash-sealed certificates).
- **Default dtype.** Library code defaults new tensors to the framework default
  dtype (`torch.get_default_dtype()` / `keras.config.floatx()`), never a
  hardcoded `float32`; set the framework default before constructing modules to
  reproduce a specific-precision run.

## Reproducing the claims

### Cross-backend bit-identical numerics

```bash
python -m pytest tests -q          # torch <-> jax <-> keras parity
python -m pytest packages/omnibias-fields/tests -q
```

### Closed-form derivative tower (core math)

```bash
python -m pytest packages/omnibias-core/tests -q
python -m pytest packages/omnibias-torch/tests packages/omnibias-jax/tests -q
```

### Per-package suites

```bash
python -m pytest packages/<distribution>/tests -q
```

`omnibias-keras` selects its backend explicitly:

```bash
KERAS_BACKEND=jax        python -m pytest packages/omnibias-keras/tests -q
KERAS_BACKEND=tensorflow python -m pytest packages/omnibias-keras/tests -q
KERAS_BACKEND=torch      python -m pytest packages/omnibias-keras/tests -q
```

### Verified enclosures and the formal loop

Rigorous primitives (`omnibias.core.verified`) and the certificate format
(`omnibias.core.proof`) are pure-Python and deterministic:

```bash
python -m pytest packages/omnibias-core/tests -k "verified or proof or certificate" -q
```

`theorem_prover_verified` is earned only by a genuine Lean kernel pass. With a
Lean toolchain present:

```bash
cd formal/omnibias-verified-kernel && lake build
```

Without a toolchain the bridge degrades gracefully and the flag stays `False`.

### Type and lint gates

```bash
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src \
  packages/omnibias-jax/src packages/omnibias-ferminet/src
mkdocs build --strict
```

## Production-fidelity GPU benchmarks

Headline FermiNet / PINN / QPINN numbers are produced on the GPU cluster, not in
default CI. The vendor-neutral run recipes, resource envelopes, and exact
`pytest` selectors are in [`benchmarks.md`](benchmarks.md) (the release-gating
subset is mirrored in `docs/release_blockers.md`). Those numbers are transcribed
into the docs; the closed-form math they exercise is reproduced bit-for-bit on
CPU by the suites above.

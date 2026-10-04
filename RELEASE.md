# Releasing distributions

The release workflow owns the **16 primitive distributions**. Extracted consumers
and both research distributions are validated locally and retain their own future
release lifecycle. Repository moves do not rename PyPI projects or revoke existing
grant terms. Never replace an artifact or reuse an occupied version.

## Prerelease policy

This migration prepares an **opt-in prerelease**, not a stable consumer upgrade:
`core`, `torch`, `jax` are `0.5.0rc1`; `fields` is `0.2.0rc1`; `keras` is
`0.0.2a1`; the other eleven primitives are `0.1.0a2`. Metadata and dependency
minimums are coordinated. Recheck index availability immediately before upload.

`[tool.omnibias.release].prerelease-only` blocks stable artifacts. Remove this guard
only after published-consumer feature profiles pass against the new primitives,
or validated successor consumer releases are available with correct constraints.
Microbenchmark wins are not a substitute for consumer compatibility.

After publication, select the migration explicitly:

```bash
python -m pip install --pre 'omnibias-torch==0.5.0rc1'
```

Ordinary installation without prerelease selection retains compatible published
stable packages. `scripts/check_prerelease_isolation.py` tests this with the new
primitive wheels available to the resolver. Avoid `--pre` in existing consumer
environments unless the whole selected dependency closure has been validated.

**Alpha-only exception:** Keras and first-release alpha projects have no stable
fallback. Resolvers can select their newest prerelease without `--pre`; Keras's
explicit dependency minimum can then pull the new core into the same environment.
The resolver check verifies this exception too. Use the historical constraints
below for existing installations; `--pre` is not a universal isolation switch.

## Published-consumer compatibility

The following affected features work with the historical primitives and fail with
the reduced primitive surface. This is the reason stable promotion remains blocked.

| Published consumer | Affected feature | Removed dependency surface |
| --- | --- | --- |
| FermiNet `0.2.0` | Hermite/oscillator energy entry point | `omnibias.core.ladder` |
| PINN `0.1.0` | Depth-residual boundary-mask/training entry point | `omnibias.core.local_jet`, backend local-training modules |
| Geometry `0.2.0` | Exponential-family Fisher metric | `omnibias.torch.information` |

Preserve the tested [historical constraints](scripts/constraints-published.txt):

```bash
python -m pip install -c scripts/constraints-published.txt \
  'omnibias-ferminet==0.2.0' 'omnibias-pinn==0.1.0' 'omnibias-geometry==0.2.0'
```

The feature probes execute outside checkouts. Their logs live in ignored artifacts:

```bash
uv run --no-sync python scripts/check_published_compatibility.py \
  --wheelhouse artifacts/wheel-validation/wheels
uv run --no-sync python scripts/check_prerelease_isolation.py \
  --wheelhouse artifacts/wheel-validation/wheels
```

Use `--require-compatible` on the first command for a stable-promotion gate.
The current prerelease audit records prepared-wheel failures explicitly while
requiring historical profiles to pass. It does not label incompatibility a success.

## Publisher configuration

Use OIDC Trusted Publishing only. Every identity uses owner `derivon-ai`, repository
`omnibias` and workflow `release.yml`. Existing production projects (`core`, `torch`,
`jax`, `keras`, `fields`) and the initial `binary` identity use environment `pypi`.
Other production identities use `pypi-<short-name>`. Test identities use
`testpypi-<short-name>`. `publisher_environment` in `scripts/release_preflight.py`
is the workflow's authoritative mapping.

Each environment must require a production-authorized reviewer and restrict release
refs. Protect new per-package environments before enabling their publishers. The
publishing matrix selects only that package's artifacts. No permanent PyPI token
belongs in a workflow or local YAML file.

PyPI limits simultaneous pending publishers and rejects identical pending identities
for different project names. Pending registration does not reserve a name. If the
account cannot register the complete first-release set, treat it as an external
release blocker: request an account-limit adjustment or explicitly plan staged
first publications and subsequent registrations. Do not bypass it with placeholder
releases. TestPyPI requires a separate authenticated account and separate mappings.

See [adding a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
[pending projects](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [filename reuse](https://pypi.org/help/#file-name-reuse).

## Prepare, review, then publish

1. Finish API, license and compatibility review. Update versions and dependency
   floors together; regenerate the lockfile and inventories. Preserve historical
   release artifacts and required license provenance.
2. Run the local gates below and CI on the exact candidate commit.
3. Dispatch `release` with target **prepare** (the default). An empty package input
   means all 16 primitives. The preparation path builds, executes installed-wheel
   README profiles, checks metadata and attests artifacts; it never uploads.
4. Verify every selected dependency, including optional feature closures, is in the
   selection or already available on the target index. An index error is not evidence
   of an unused version. Initial preparation should select the complete cohort.
5. After explicit release approval, dispatch the same reviewed commit with target
   `testpypi` or `pypi`, then approve the protected environment jobs. These are separate
   actions from merging a PR. Production artifacts are rebuilt and attested, not
   promoted byte-for-byte from TestPyPI.
6. Verify clean index installations and the uploaded artifact hashes/attestations.

Independent `omnibias-<package>-v<version>` tags must match package metadata.
Tag pushes prepare only; they never publish automatically. Aggregate tags are
unsupported. Partial selections fail when their dependency closure is incomplete.

Do not use `skip-existing`. If an upload partly succeeds, stop and compare the
existing filenames, hashes and attestations to the reviewed build. A rerun refuses
occupied versions. Select only remaining distributions when safe, or prepare a new
version after review; never silently overwrite a changed build with an old version.

## Installed-wheel gates

Build a coordinated wheelhouse from fresh source archives. The validator checks
44 distributions when the extracted ecosystem is present, with disjoint namespace
ownership, correct metadata and no editable/source-checkout leakage:

Use the uv version pinned by `[tool.uv].required-version`; CI reads the same pin.
If temporary storage has a small quota, set `TMPDIR` to a larger scratch directory
outside all source checkouts. Sequential profiles reuse the download cache.

```bash
uv run --no-sync python scripts/validate_wheels.py --projects-root ../omnibias_projects --readme
uv run --no-sync python scripts/validate_wheels.py --projects-root ../omnibias_projects \
  --no-build --numerical
```

`wheel-tests.toml` declares separate README and numerical feature profiles. README
Python examples come from the **wheel's long description**, not the current source
README. Missing imports fail. Apache profiles reject copyleft omnibias dependencies;
certified profiles opt into their documented integrations. The `--python` option
chooses the fresh environment interpreter. Run all primitive README profiles too:

```bash
uv run --no-sync python scripts/validate_wheels.py --artifact-only --readme \
  --python 3.12 --output artifacts/primitive-wheels
```

Place the 16 built wheels under that output directory's `wheels/` first. Use
`--only <distribution> ...` to select profiles. Artifact-only validation builds
nothing, reads examples from wheel metadata and disables source overrides. Rebuild
affected wheels whenever source, metadata or long descriptions change.

## README assets and rendering

Each distribution owns `docs/visuals/scene.py`, GIFs, PNG/SVG posters and a small
provenance record. Scenes compute package outputs; they are not performance data.
The common drawing code lives in `scripts/render_package_visuals.py` and
`scripts/visual_story.py`. Read the [benchmark reproduction guide](benchmarks/README.md)
for the separate measured workloads.

Use the reference CPU environment and the package wheels for byte-for-byte scene
generation. The reference assets use Python 3.12,
NumPy 2.5.3, Matplotlib 3.10.9, Pillow 12.2.0 and the bundled DejaVu Sans font;
backend versions are pinned in `scripts/visual-requirements.txt`. The locked
`presentation` dependency group supplies the renderer for ordinary development;
different backend or rendering versions can change the reference pixels.

```bash
uv venv artifacts/visuals-env --python 3.12
uv pip install --python artifacts/visuals-env/bin/python --torch-backend=cpu \
  -r scripts/visual-requirements.txt artifacts/wheel-validation/wheels/*.whl
export KERAS_BACKEND=torch JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
artifacts/visuals-env/bin/python scripts/render_package_visuals.py \
  packages/omnibias-core/docs/visuals/scene.py
artifacts/visuals-env/bin/python scripts/render_package_visuals.py --check \
  packages/omnibias-core/docs/visuals/scene.py
uv run --group presentation python scripts/check_package_presentation.py \
  --projects-root ../omnibias_projects --wheels artifacts/wheel-validation/wheels
```

`--check` reproduces GIF and static-poster bytes. Asset checks reject duplicate
animations, missing alternatives, oversized GIFs and flashing frame durations.
Wheel descriptions are rendered with PyPI's Markdown renderer. Published primitive
images use immutable commit URLs; commit assets first, then update README references.
Consumer-specific assets stay in their owning checkout. Before a consumer's future
release, assign an accessible immutable asset URL from its eventual repository;
local relative links are for its current unpublished source edition.

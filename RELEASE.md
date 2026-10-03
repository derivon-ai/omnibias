# Releasing distributions

Keep existing PyPI project names and historical releases. Repository moves do
not require new package names. Every changed distribution needs a fresh version;
the split removes APIs and requires a documented compatibility release, not an
overwrite of the existing artifacts. Review dependency ranges against the actual
API used, update module versions and metadata together, then regenerate `uv.lock`.

The main release workflow owns `core`, `torch`, `jax`, `keras`, and `fields` by
default. Other retained distributions can be selected explicitly. Extracted
projects own their future releases; this workflow rejects their package names.

## Configure once

For each existing PyPI project, add a Trusted Publisher with owner `derivon-ai`,
repository `omnibias`, workflow `release.yml`, environment `pypi`. Configure
TestPyPI separately with environment `testpypi`. For extracted repositories use
their actual GitHub repository name after it exists. Retire the obsolete
publisher only after the replacement is verified.

Protect the `pypi` GitHub environment with a required reviewer and permitted
release refs. OIDC credentials are requested only in publishing jobs. Never put
PyPI passwords or permanent upload tokens in workflow files.

## Release sequence

1. Audit removed public APIs, dependencies and licenses; choose new versions and
   record migration instructions. Run affected consumers against built wheels.
2. Commit metadata and lockfile changes. Run CI on that exact commit.
3. Push an independent tag such as `omnibias-core-v0.5.0rc1` matching its metadata,
   or manually dispatch `release` with explicit packages and `testpypi` target.
4. The workflow reruns CI, checks tag/version consistency and refuses occupied
   index versions or index lookup failures. It builds and attests artifacts.
5. Validate installation in a clean environment with only published dependencies.
   Then dispatch the same commit/tag with target `pypi` and approve its environment.
   Production artifacts are rebuilt and attested; they are not promoted byte-for-byte
   from TestPyPI. Never assume that an editable workspace proves wheel compatibility.

Aggregate tags are unsupported because distributions have independent versions.
Do not use `skip-existing`: it can hide an attempt to publish changed code with
an old version. A partial upload needs inspection before a retry; the preflight
intentionally stops if that version already exists.

## Current migration status

The existing published artifacts remain unchanged. The extracted consumers'
dependency and license boundaries are validated against local wheels. Public
release still requires backing up their repositories, assigning compatible new
versions and publishing their dependencies in order. Keep the last published
versions available while preparing those successor releases.

See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [PyPI filename reuse](https://pypi.org/help/#file-name-reuse).

## Installed-wheel integration gate

Run the shared validator against extracted local projects before preparing a
release. It builds each wheel from a fresh source archive, installs each consumer's declared dependencies
in isolation, and rejects editable/source leakage and overlapping wheel files:

```bash
uv run --no-sync python scripts/validate_wheels.py --projects-root ../omnibias_projects --numerical
```

Consumer `wheel-tests.toml` files select numerical `extras`, representative imports and test
paths. Optional `base_extras` select a required runtime choice for import-only
validation (Keras selects its Torch backend). Test files are copied into scratch directories; source checkouts are not
added to the import path. Detailed logs and the JSON report are written under
`artifacts/wheel-validation/`. Use `--no-build --only <distribution>` to retry a
failed environment after verifying its wheel is current. Rebuild after source
or metadata changes. These commands can run in consumer CI once its checkout
and dependency wheel artifacts are available; they do not create remote repos.

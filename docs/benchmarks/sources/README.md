# Measured source snapshots

Files here are immutable, content-addressed snapshots referenced by historical
benchmark JSON `source_sha256` values. The filename is the exact file's SHA-256
followed by `.py`. A snapshot is retained when subsequent library maintenance
changes a measured file; historical timings and hashes are not rewritten to
describe code that was not measured.

The one-layer and repeated-Laplacian measurements reference the archived
`omnibias.jax.laplacian` source. The general directional-jet measurement also
retains both backend `jet.py` versions from its measured run. Later reporting
fixes and kernel optimizations do not retroactively change those measurements.
Consult current package source for current API behavior.

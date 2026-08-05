# Security Policy

## Supported versions

omnibias is a set of versioned packages. Security and correctness fixes
are applied to the latest released minor of each package.

| Package | Supported |
|---|---|
| `omnibias-core` (0.4.x) | yes |
| `omnibias-torch` (0.4.x) | yes |
| `omnibias-jax` (0.4.x) | yes |
| `omnibias-ferminet` (0.2.x) | yes |
| `omnibias-pinn` (0.1.x, beta) | yes |
| `omnibias-qpinn` (0.0.x, alpha) | best-effort |
| `omnibias-curvature` (0.1.x, alpha) | best-effort |
| `omnibias-keras` (0.0.x, alpha) | best-effort |
| older releases | no |

## Reporting a vulnerability

Please **do not** open a public issue for security reports.

Use GitHub's private vulnerability reporting on this repository
("Security" tab -> "Report a vulnerability"). If that is unavailable,
email <info@derivon.ai> with the subject line `SECURITY:` (do not open a
public issue).

When reporting, include:

- the affected package and version,
- a minimal reproducer,
- the impact you observed.

## Response targets

- Acknowledgement: within 5 business days.
- Initial assessment: within 10 business days.
- Fix or mitigation plan: communicated after assessment, prioritized by
  severity.

Because omnibias is a numerical library, we also treat **silent
numerical-correctness regressions** in a shipped public API as
security-class issues and handle them through the same private channel.

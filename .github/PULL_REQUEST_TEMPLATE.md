## Summary

<!-- What does this PR change, and why? -->

## Related issues

<!-- e.g. Closes #123 -->

## Type of change

- [ ] Bug fix (numerical or behavioral)
- [ ] New feature (activation / operator / layer / backend)
- [ ] Documentation
- [ ] Refactor / internal
- [ ] CI / tooling

## Checklist

- [ ] A regression test covers every behavioral change.
- [ ] `ruff check packages tests` is clean.
- [ ] `mypy --strict` is clean for the T1 packages I touched.
- [ ] `mkdocs build --strict` is clean (if docs/docstrings changed).
- [ ] Cross-backend numerics stay bit-identical (parity tests pass).
- [ ] No GPU-only, cluster-specific, or local-path details leak into tracked files.
- [ ] `CHANGELOG.md` updated.
- [ ] I have signed the [CLA](../docs/CLA.md) (the PR bot will confirm).
- [ ] New source files carry the SPDX dual-license header.

## Notes for reviewers

<!-- Anything reviewers should focus on, especially math correctness. -->

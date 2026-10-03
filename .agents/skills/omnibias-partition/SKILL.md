---
name: omnibias-partition
description: Maintain soft partition weights, regional composition, hardening and membership certificates.
---

# Maintaining omnibias-partition

Owned implementation: [source](../../../packages/omnibias-partition/src/omnibias/partition);
public surface: [API](../../../docs/api/partition.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/` defines region topology and weight semantics; backend directories realize
them as tensors. `registry.py` combines regional models, while `certify.py` owns
bounds on the corresponding routing quantities. Keep region ordering consistent
between weights, experts, hard assignments and exported rules: permutations can
preserve a sum-to-one test while selecting the wrong expert.

For gate changes, test nonnegative weights and normalization, including ties,
saturated logits and nonuniform temperatures. Preserve gradients into thresholds,
directions and expert outputs. Sparse, axis-aligned and oblique parameterizations
must agree when they represent the same split. Hardening requires an explicit
tie convention and the stated temperature assumptions.

Treat a certificate as a claim about a specified routing quantity and input
domain. Do not extend it to arbitrary downstream experts without checking their
range and composition assumptions. Registry changes should cover heterogeneous
expert outputs and clear shape failures. Backend parity, registry and certificate
tests provide different checks; run all affected groups. Atlas, PINN and tabular
integrations stay in their consumer repositories.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-partition/tests/test_weights.py packages/omnibias-partition/tests/test_registry.py packages/omnibias-partition/tests/test_certify.py -q
```

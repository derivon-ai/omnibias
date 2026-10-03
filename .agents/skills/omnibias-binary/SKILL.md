---
name: omnibias-binary
description: Maintain hard quantizers, temperature schedules and surrogate-gradient contracts.
---

# Maintaining omnibias-binary

Owned implementation: [source](../../../packages/omnibias-binary/src/omnibias/binary);
public surface: [API](../../../docs/api/binary.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`torch/` and `jax/` own quantizer implementations; `schedule.py` owns temperature
progression. The public contract combines a discrete forward value with a chosen
backward rule. Test those two behaviors separately: matching hard outputs does
not validate the surrogate derivative, and a smooth backward rule does not make
the forward quantizer smooth.

Preserve threshold and tie conventions for binary, ternary and multilevel outputs.
Check broadcast shapes for per-channel thresholds or temperatures, dtype handling
for integer-like outputs, and optimizer registration for learnable beta parameters.
Schedule changes should be reproducible from explicit state rather than hidden
module-global counters.

Use `test_quantize.py` and `test_quantize01.py` for range and tie behavior,
`test_surrogate.py` for the chosen backward expression, and
`test_learnable_beta.py` for parameter updates. Keep Boolean algebra and equation
solving in the Boolean consumer of these differentiable primitives. When changing
the surrogate definition, document its mathematical choice and update both
backends; do not silently substitute a straight-through estimator because it has
the same forward values.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-binary/tests/test_quantize.py packages/omnibias-binary/tests/test_surrogate.py packages/omnibias-binary/tests/test_learnable_beta.py -q
```
